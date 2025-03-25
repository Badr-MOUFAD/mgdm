import torch
from torch.distributions import Distribution

import tqdm
from typing import Tuple, List

from utils.utils import load_yaml, fwd_mixture
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from diffusers import DDPMPipeline, StableDiffusionXLPipeline
from torch.func import grad
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from local_paths import REPO_PATH

from sound_model.misc import load_sound_model


class UNet(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, x, t):
        return self.unet(x, torch.tensor([t]))[:, :3]


class UnetHG(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, x, t):
        if torch.tensor(t).dim() == 0:
            t = torch.tensor([t])
        return self.unet(x, t).sample


class LDM(torch.nn.Module):

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, t):
        return self.net.model(x, torch.tensor([t]))

    def decode(self, z):
        if hasattr(self.net, "decode_first_stage"):
            return self.net.decode_first_stage(z)
        else:
            raise NotImplementedError

    def differentiable_decode(self, z):
        if hasattr(self.net, "differentiable_decode_first_stage"):
            return self.net.differentiable_decode_first_stage(z)
        else:
            raise NotImplementedError

    def differentiable_encode(self, x):
        z = self.net.differentiable_encode_first_stage(x)

        # NOTE: this operation involves scaling the encoding with factor
        # in our case this factor is 1., but perhaps it is not for other models
        # c.f. `get_first_stage_encoding` in ldm/models/diffusion/ddpm.py
        z = self.net.get_first_stage_encoding(z)

        return z

    def encode(self, x):
        z = self.net.encode_first_stage(x)
        # NOTE: see differentiable_encoder for info
        z = self.net.get_first_stage_encoding(z)
        return z


class EpsilonNetGM(torch.nn.Module):

    def __init__(self, means, weights, alphas_cumprod, cov=None):
        super().__init__()
        self.means = means
        self.weights = weights
        self.covs = cov
        self.alphas_cumprod = alphas_cumprod

    def forward(self, x, t):
        # if len(t) == 1 or t.dim() == 0:
        #     acp_t = self.alphas_cumprod[t.to(int)]
        # else:
        #     acp_t = self.alphas_cumprod[t.to(int)][0]
        acp_t = self.alphas_cumprod[t.to(int)]
        grad_logprob = grad(
            lambda x: fwd_mixture(
                self.means, self.weights, self.alphas_cumprod, t, self.covs
            )
            .log_prob(x)
            .sum()
        )
        return -((1 - acp_t) ** 0.5) * grad_logprob(x)


class EpsilonNetMCGD(torch.nn.Module):

    def __init__(self, H_funcs, unet, dim):
        super().__init__()
        self.unet = unet
        self.H_funcs = H_funcs
        self.dim = dim

    def forward(self, x, t):
        x_normal_basis = self.H_funcs.V(x).reshape(-1, *self.dim)
        # .repeat(x.shape[0]).to(x.device)
        t_emb = torch.tensor(t).to(x.device)
        eps = self.unet(x_normal_basis, t_emb)
        eps_svd_basis = self.H_funcs.Vt(eps)
        return eps_svd_basis


class EpsilonNet(torch.nn.Module):
    def __init__(self, net, alphas_cumprod, timesteps):
        super().__init__()
        self.net = net
        self.alphas_cumprod = alphas_cumprod
        self.acp_f8 = alphas_cumprod.to(torch.float64)
        self.timesteps = timesteps

    def forward(self, x, t):
        return self.net(x, torch.tensor(t))

    def predict_x0(self, x, t):
        acp_t = self.alphas_cumprod[t] / self.alphas_cumprod[self.timesteps[0].int()]
        return (x - (1 - acp_t) ** 0.5 * self.forward(x, t)) / (acp_t**0.5)

    def score(self, x, t):
        acp_t = self.alphas_cumprod[t] / self.alphas_cumprod[self.timesteps[0]]
        return -self.forward(x, t) / (1 - acp_t) ** 0.5

    def decode(self, z):
        return self.net.decode(z)

    def differentiable_decode(self, z):
        return self.net.differentiable_decode(z)


# TODO: fix shape handling
class EpsilonNetSVD(EpsilonNet):
    def __init__(self, net, alphas_cumprod, timesteps, H_func, shape, device="cuda"):
        super().__init__(net, alphas_cumprod, timesteps)
        self.net = net
        self.alphas_cumprod = alphas_cumprod
        self.H_func = H_func
        self.timesteps = timesteps
        self.device = device
        self.shape = shape

    def forward(self, x, t):
        # shape = (x.shape[0], 3, int(np.sqrt((x.shape[-1] // 3))), -1)
        x = self.H_func.V(x.to(self.device)).reshape(self.shape)
        return self.H_func.Vt(self.net(x, t))


class EpsilonNetSVDGM(EpsilonNet):
    def __init__(self, net, alphas_cumprod, timesteps, H_func, device="cuda"):
        super().__init__(net, alphas_cumprod, timesteps)
        self.net = net
        self.alphas_cumprod = alphas_cumprod
        self.H_func = H_func
        self.timesteps = timesteps
        self.device = device

    def forward(self, x, t):
        x = self.H_func.V(x.to(self.device))
        return self.H_func.Vt(self.net(x, t))


def load_ldm_model(config, ckpt, device):
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model


def load_gmm_epsilon_net(prior: Distribution, dim: int, n_steps: int):
    timesteps = torch.linspace(0, 999, n_steps).long()
    alphas_cumprod = torch.linspace(0.9999, 0.98, 1000)
    alphas_cumprod = torch.cumprod(alphas_cumprod, 0).clip(1e-10, 1)
    alphas_cumprod = torch.concatenate([torch.tensor([1.0]), alphas_cumprod])

    means, covs, weights = (
        prior.component_distribution.mean,
        prior.component_distribution.covariance_matrix,
        prior.mixture_distribution.probs,
    )

    epsilon_net = EpsilonNet(
        net=EpsilonNetGM(means, weights, alphas_cumprod, covs),
        alphas_cumprod=alphas_cumprod,
        timesteps=timesteps,
    )

    return epsilon_net


def load_epsilon_net(model_id: str, n_steps: int, device: str):
    hf_models = {
        "celebahq": "google/ddpm-celebahq-256",
    }
    pixelsp_models = {
        "ffhq": REPO_PATH / "configs/ffhq_model.yaml",
        "imagenet": REPO_PATH / "configs/imagenet_model.yaml",
    }
    ldm_models = {
        "ffhq_ldm": REPO_PATH / "configs/latent-diffusion/ffhq-ldm-vq-4.yaml",
    }

    timesteps = torch.linspace(0, 999, n_steps).long()

    if model_id in hf_models:
        hf_id = "google/ddpm-celebahq-256"
        pipeline = DDPMPipeline.from_pretrained(hf_id).to(device)
        model = pipeline.unet
        model = model.requires_grad_(False)
        model = model.eval()

        timesteps = torch.linspace(0, 999, n_steps).long()
        alphas_cumprod = pipeline.scheduler.alphas_cumprod.clip(1e-6, 1)
        alphas_cumprod = torch.concatenate([torch.tensor([1.0]), alphas_cumprod])

        return EpsilonNet(
            net=UnetHG(model), alphas_cumprod=alphas_cumprod, timesteps=timesteps
        )

    if model_id in pixelsp_models:

        # NOTE code verified at https://github.com/openai/guided-diffusion
        # and adapted from https://github.com/DPS2022/diffusion-posterior-sampling

        model_config = pixelsp_models[model_id]
        diffusion_config = REPO_PATH / "configs/diffusion_config.yaml"

        model_config = load_yaml(model_config)
        diffusion_config = load_yaml(diffusion_config)

        sampler = create_sampler(**diffusion_config)
        model = create_model(**model_config)

        # by default set model to eval mode and disable grad on model parameters
        model = model.eval()
        model.requires_grad_(False)

        alphas_cumprod = torch.tensor(sampler.alphas_cumprod).float().clip(1e-6, 1)
        alphas_cumprod = torch.concatenate([torch.tensor([1.0]), alphas_cumprod])

        net = UNet(model)
        return EpsilonNet(
            net=net,
            alphas_cumprod=alphas_cumprod,
            timesteps=timesteps,
        )

    if model_id in ldm_models:
        cfg_path = OmegaConf.load(ldm_models[model_id])
        ckpt_path = cfg_path.model.params.unet_config.ckpt_path
        model = load_ldm_model(cfg_path, ckpt_path, device)

        model = model.eval()
        model.requires_grad_(False)
        alphas_cumprod = torch.tensor(model.alphas_cumprod).float().clip(1e-6, 1)
        alphas_cumprod = torch.concatenate([torch.tensor([1.0]), alphas_cumprod])

        return EpsilonNet(
            net=LDM(model), alphas_cumprod=alphas_cumprod, timesteps=timesteps
        )

    if model_id == "sound_models":
        # NOTE refer to `SoundModelWrapper` and `load_sound_model` in module src.sound_model.misc
        # for detail about the sound model

        net, alphas_cumprod = load_sound_model(device)
        timesteps = torch.linspace(0, 999, n_steps, dtype=torch.int32)

        return EpsilonNet(net, alphas_cumprod, timesteps)

    if model_id == "sdxl1.0":
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float32
        )
        pipeline.to(device)

        acp = (1.0 - pipeline.scheduler.betas).cumprod(dim=0).to(device)
        net = SDWrapper(pipeline)

        return EpsilonNet(net, acp, timesteps)


def bridge_kernel_statistics(
    x_ell: torch.Tensor,
    x_s: torch.Tensor,
    epsilon_net: EpsilonNet,
    ell: int,
    t: int,
    s: int,
    eta: float = 1.0,
):
    """s < t < ell"""
    f8 = torch.float64

    alpha_cum_s_to_t = epsilon_net.acp_f8[t] / epsilon_net.acp_f8[s]
    alpha_cum_t_to_ell = epsilon_net.acp_f8[ell] / epsilon_net.acp_f8[t]
    alpha_cum_s_to_ell = epsilon_net.acp_f8[ell] / epsilon_net.acp_f8[s]
    std = (
        eta
        * ((1 - alpha_cum_t_to_ell) * (1 - alpha_cum_s_to_t) / (1 - alpha_cum_s_to_ell))
        ** 0.5
    )
    coeff_xell = ((1 - alpha_cum_s_to_t - std**2) / (1 - alpha_cum_s_to_ell)) ** 0.5
    coeff_xs = (alpha_cum_s_to_t**0.5) - coeff_xell * (alpha_cum_s_to_ell**0.5)

    coeff_xell, coeff_xs, std = (
        coeff_xell.to(dtype=f8),
        coeff_xs.to(dtype=f8),
        std.to(dtype=f8),
    )
    return coeff_xell * x_ell + coeff_xs * x_s, std


def bridge_kernel_all_stats(
    ell: int,
    t: int,
    s: int,
    epsilon_net: EpsilonNet,
    eta: float = 1.0,
) -> Tuple[float, float, float]:
    """s < t < ell

    Return
    ------
    coeff_xell, coeff_xs, std
    """
    f8 = torch.float64

    alpha_cum_s_to_t = epsilon_net.acp_f8[t] / epsilon_net.acp_f8[s]
    alpha_cum_t_to_ell = epsilon_net.acp_f8[ell] / epsilon_net.acp_f8[t]
    alpha_cum_s_to_ell = epsilon_net.acp_f8[ell] / epsilon_net.acp_f8[s]
    std = (
        eta
        * ((1 - alpha_cum_t_to_ell) * (1 - alpha_cum_s_to_t) / (1 - alpha_cum_s_to_ell))
        ** 0.5
    )
    coeff_xell = ((1 - alpha_cum_s_to_t - std**2) / (1 - alpha_cum_s_to_ell)) ** 0.5
    coeff_xs = (alpha_cum_s_to_t**0.5) - coeff_xell * (alpha_cum_s_to_ell**0.5)

    return coeff_xell.to(dtype=f8), coeff_xs.to(dtype=f8), std.to(dtype=f8)


def sample_bridge_kernel(
    x_ell: torch.Tensor,
    x_s: torch.Tensor,
    epsilon_net: EpsilonNet,
    ell: int,
    t: int,
    s: int,
    eta: float = 1.0,
):
    mean, std = bridge_kernel_statistics(x_ell, x_s, epsilon_net, ell, t, s, eta)
    return mean + std * torch.randn_like(mean)


def ddim_statistics(
    x: torch.Tensor,
    epsilon_net: EpsilonNet,
    t: float,
    t_prev: float,
    eta: float,
    e_t: torch.Tensor = None,
):
    t_0 = epsilon_net.timesteps[0]
    if e_t is None:
        e_t = epsilon_net.predict_x0(x, t)
    return bridge_kernel_statistics(
        x_ell=x, x_s=e_t, epsilon_net=epsilon_net, ell=t, t=t_prev, s=t_0, eta=eta
    )


def ddim_step(
    x: torch.Tensor,
    epsilon_net: EpsilonNet,
    t: float,
    t_prev: float,
    eta: float,
    e_t: torch.Tensor = None,
):
    t_0 = epsilon_net.timesteps[0]
    if e_t is None:
        e_t = epsilon_net.predict_x0(x, t)
    return sample_bridge_kernel(
        x_ell=x, x_s=e_t, epsilon_net=epsilon_net, ell=t, t=t_prev, s=t_0, eta=eta
    )


def ddim(
    initial_noise_sample: torch.Tensor, epsilon_net: EpsilonNet, eta: float = 1.0
) -> torch.Tensor:
    """
    This function implements the (subsampled) generation from https://arxiv.org/pdf/2010.02502.pdf (eqs 9,10, 12)
    :param initial_noise_sample: Initial "noise"
    :param timesteps: List containing the timesteps. Should start by 999 and end by 0
    :param score_model: The score model
    :param eta: the parameter eta from https://arxiv.org/pdf/2010.02502.pdf (eq 16)
    :return:
    """
    sample = initial_noise_sample
    for i in tqdm.tqdm(range(len(epsilon_net.timesteps) - 1, 1, -1)):
        t, t_prev = epsilon_net.timesteps[i], epsilon_net.timesteps[i - 1]
        sample = ddim_step(
            x=sample,
            epsilon_net=epsilon_net,
            t=t,
            t_prev=t_prev,
            eta=eta,
        )
    sample = epsilon_net.predict_x0(sample, epsilon_net.timesteps[1])

    return epsilon_net.decode(sample) if hasattr(epsilon_net.net, "decode") else sample


class VEPrecond(torch.nn.Module):
    """Wrapper to make Sound model compatible with EDM API."""

    def __init__(
        self,
        denoiser_fn: torch.nn.Module,
        sigmas: torch.Tensor,
    ):
        super().__init__()

        self.denoiser = denoiser_fn
        self.sigmas = sigmas
        self.sigma_max = sigmas.max().item()
        self.sigma_min = sigmas.min().item()

    def forward(self, x_t, sigma_t):
        return self.denoiser(x_t, sigma=sigma_t)

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


class VPPrecond(torch.nn.Module):
    """Variance Preserving Preconditioning as described in EDM [1].

    Convert a DDPM model with linear scheduler (as in DDPM paper) to EDM.
    Code adapted from https://github.com/NVlabs/edm/blob/main/training/networks.py

    References
    ----------
    .. [1] Karras, Tero, Miika Aittala, Timo Aila, and Samuli Laine.
    "Elucidating the design space of diffusion-based generative models."
    Advances in neural information processing systems 35 (2022): 26565-26577.
    """

    # fmt: off
    def __init__(
        self,
        model=None,         # Diffusion model.
        label_dim=0,        # Number of class labels, 0 = unconditional.
        use_fp16=False,     # Execute the underlying model at FP16 precision?
        beta_d=19.9,        # Extent of the noise level schedule.
        beta_min=0.1,       # Initial slope of the noise level schedule.
        M=1000,             # Original number of timesteps in the DDPM formulation.
        conditional=False,  # Use conditional model?
        learn_sigma=False,  # Learnable noise level?
        epsilon_t = 1e-3,   # Minimum t-value used during training.
    ):
    # fmt: on
        super().__init__()
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.M = M
        self.model = model
        self.conditional = conditional
        self.learn_sigma = learn_sigma
        self.sigma_min = float(self.sigma(epsilon_t))
        self.sigma_max = float(self.sigma(1))

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = (
            None
            if self.label_dim == 0
            else (
                torch.zeros([1, self.label_dim], device=x.device)
                if class_labels is None
                else class_labels.to(torch.float32).reshape(-1, self.label_dim)
            )
        )
        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma**2 + 1).sqrt()
        c_noise = (self.M - 1) * self.sigma_inv(sigma)

        if self.conditional:
            F_x = self.model(
                (c_in * x).to(dtype),
                c_noise.flatten(),
                class_labels=class_labels,
                **model_kwargs,
            )
        else:
            F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), **model_kwargs)
        if self.learn_sigma:
            F_x, _ = torch.split(F_x, x.shape[1], dim=1)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t**2) + self.beta_min * t).exp() - 1).sqrt()

    def sigma_inv(self, sigma):
        sigma = torch.as_tensor(sigma)
        return (
            (self.beta_min**2 + 2 * self.beta_d * (1 + sigma**2).log()).sqrt()
            - self.beta_min
        ) / self.beta_d

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


class SDWrapper(torch.nn.Module):
    def __init__(self, pipeline):
        super().__init__()

        self.pipeline = pipeline
        self.unet, self.vae = pipeline.unet, pipeline.vae
        self.prompt = "Pandas built with paper"
        self.use_cfg = False

        # these should be freezed as they aren't trainable
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def forward(self, x, t):
        prompt_embeds, uncond_prompt_embeds, *others = self.pipeline.encode_prompt(
            prompt=self.prompt,
            device=x.device,
            num_images_per_prompt=x.shape[0],
            do_classifier_free_guidance=self.use_cfg,
        )
        if self.use_cfg:
            prompt_embeds = torch.cat([uncond_prompt_embeds, prompt_embeds])
            x = torch.cat([x] * 2)

        added_cond_kwargs = SDWrapper._get_cond_kwargs(others, self.use_cfg)
        pred = self.unet(
            x,
            t,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
        ).sample

        return pred

    def differentiable_decode(self, z, force_float32: bool = True):
        # force decoding to be in float32 to avoid overflow
        vae = self.vae.to(torch.float32) if force_float32 else self.vae
        z = z.to(torch.float32) if force_float32 else z

        return vae.decode(z / vae.config.scaling_factor).sample

    @torch.no_grad()
    def decode(self, z, force_float32: bool = True):
        return self.differentiable_decode(z, force_float32)

    @staticmethod
    def _get_cond_kwargs(other_prompt_embeds: List[torch.Tensor], use_cfg: bool):
        pooled_prompt_embeds, negative_pooled_prompt_embeds = other_prompt_embeds

        # additional image-based embeddings
        add_time_ids = SDWrapper._get_time_ids(pooled_prompt_embeds)
        negative_add_time_ids = add_time_ids

        add_text_embeds = pooled_prompt_embeds
        if use_cfg:
            add_text_embeds = torch.cat(
                [negative_pooled_prompt_embeds, pooled_prompt_embeds]
            )
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids])

        added_cond_kwargs = {
            "text_embeds": add_text_embeds,
            "time_ids": add_time_ids,
        }
        return added_cond_kwargs

    @staticmethod
    def _get_time_ids(prompt_embeds: torch.Tensor):
        # NOTE these were deduced from pipeline.__call__ of diffuser v0.27.2
        # and are so far valid for sdxl1.0
        original_size = (1024, 1024)
        crops_coords_top_left = (0, 0)
        target_size = (1024, 1024)

        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor(
            prompt_embeds.shape[0] * [add_time_ids], dtype=prompt_embeds.dtype
        )
        return add_time_ids
