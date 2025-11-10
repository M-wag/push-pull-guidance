import tqdm
import numpy as np
import torch 
import os
from collections import defaultdict
from abc import ABC, abstractmethod
from torchvision.io import read_image, ImageReadMode

DISABLE_TQDM = False

#----------------------------------------------------------------------------
# Modified version torch.randn_like
def randn_like(tensor, generator=None):
    return torch.randn(
        tensor.shape,
        device=tensor.device,
        dtype=tensor.dtype,
        generator=generator
    )

#----------------------------------------------------------------------------
# Modified version of EDM sampler.
# Modular options for gradient and time step discretization.

class EDMSampler:
    def __init__(self, time_disc: str, gradient_kwargs={}):
        # self.init_gradient(gradient_kwargs)

        if time_disc == "edm":
            self.time_step_fn = time_steps_edm
        elif time_disc == "ddim":
            raise NotImplementedError()
        else:
            raise ValueError(f"Unknown time step method: {time_disc}")

    def init_gradient(self, gradient_kwargs, *, net=None):
        score_fn = ScoreDenoise(denoiser=net, scale=gradient_kwargs.scale_model_score)
        if "gvf" in gradient_kwargs.keys():
            score_fn = ScoreAdditive(score_fn, gradient_kwargs.gvf)

        self.gradient_fn = GradientEDM(score_fn)

    @torch.no_grad()
    def __call__(
        self                , 
        net                 , 
        noise               ,
        labels              ,
        device              ,
        *,
        num_steps           : int, 
        sigma_min           : float,
        sigma_max           : float, 
        rho                 : float, 
        S_churn             : float, 
        S_min               : float, 
        S_max               : float,
        S_noise             : float, 

        dtype               = torch.float32,
        apply_2nd_order     : bool = True,
        save_all_timesteps  : bool = True,
        correct_rgb         : bool = True,
        disable_tqdm        : bool = False,
        noise_seed          : int = None,
    ):
        # Adjust noise levels based on what's supported by the network
        sigma_min = max(sigma_min, net.sigma_min)
        sigma_max = min(sigma_max, net.sigma_max)
        
        # Time step discretization.
        t_steps = self.time_step_fn(num_steps, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho) # t_N=0
        # TODO : do net.round
        t_steps = t_steps.to(device=device, dtype=dtype)

        xs = None 
        # Intialize empty array to save intermediate timesteps
        if save_all_timesteps:
            xs = torch.empty((num_steps, noise.shape[0], net.img_channels, net.img_resolution, net.img_resolution))

        noise_generator = noise_seed
        if noise_seed is not None:
            noise_generator = torch.Generator(device=device).manual_seed(noise_seed)

        # Main sampling loop.
        x_next = noise.to(dtype) * t_steps[0]
        for i, (t_cur, t_next) in tqdm.tqdm(list(enumerate(zip(t_steps[:-1], t_steps[1:]))), unit='step', position=1, disable=disable_tqdm): # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur, noise_generator)

            # Euler step
            d_cur = self.gradient_fn(x_hat, t_hat, labels)
            x_next = x_hat + d_cur * (t_next - t_hat)

            # Apply 2nd order correction
            if apply_2nd_order and i < num_steps - 1:
                d_prime = self.gradient_fn(x_next, t_next, labels)
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
            
            if save_all_timesteps:
                xs[i] = x_next

        x_0 = xs if save_all_timesteps else x_next
        if correct_rgb:
            x_0 = (x_0 * 127.5 + 128) / 255

        return x_0, (t_steps.cpu().numpy(), )

    #----------------------------------------------------------------------------
    # Apply deterministic forward process of EDM sampler 

    @torch.no_grad()
    def edm_inversion(
        self,
        net,
        images,  
        labels,
        device,
        *,
        num_steps: int,
        sigma_min: float,
        sigma_max: float,
        rho: float,
        dtype=torch.float32,
        apply_2nd_order: bool = False,  
        disable_tqdm: bool = False,
        **kwargs,
    ):

        # Use denoising gradient 
        grad = DenoisingGradient(scale_model_score=1.0)
        # Adjust noise levels based on what's supported by the network
        sigma_min = max(sigma_min, net.sigma_min)
        sigma_max = min(sigma_max, net.sigma_max)

        # Reverse the time steps (from low noise to high noise)
        t_steps = self.time_step_fn(num_steps, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho)
        t_steps = torch.flip(t_steps, dims=[0]) 
        t_steps = t_steps.to(device=device, dtype=dtype)
        t_steps = torch.clamp(t_steps, min=1e-9) # Make sure t_steps is never 0

        # Register bottleneck activation
        name_bottleneck = f"skip_{net.num_skips - 1}"
        net.register_injection((name_bottleneck, "skip"))
        net.enable_injection_saving(True)

        # Main inversion loop (go from low noise to high noise)
        x_cur = images.to(dtype)
        activations_per_t = defaultdict(dict)
        for i, (t_low, t_high) in tqdm.tqdm( list(enumerate(zip(t_steps[:-1], t_steps[1:]))), unit='step', position=1, disable=disable_tqdm):
            x_next = x_cur + (t_high - t_low) * grad(x_cur, t_low, labels, net)
            x_cur = x_next
            if i == 1:
                return x_next, activations_per_t
                
            # Save bottleneck
            activations_per_t["bottleneck"][t_low] = net._injection_manager.load(name_bottleneck, "skip")

        return x_next, activations_per_t
#----------------------------------------------------------------------------
# Time step discretization functions 

def time_steps_edm(num_steps,  net=None, *, sigma_min, sigma_max, rho):
        if num_steps == 1:
            t_steps = torch.tensor([sigma_max, 0 ])
        else:
            step_indices = torch.arange(num_steps, dtype=torch.float64)
            t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
            t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0
        return t_steps

def time_steps_ddim(num_steps, device, net=None):
    raise NotImplementedError

#----------------------------------------------------------------------------
# Different gradients useable by the EDM sampler
# dx/dt = gradient(x, t)

class GradientEDM(torch.nn.Module):
    def __init__(self, score_fn):
        super().__init__()
        self.noise = lambda t : t
        self.noise_dot = lambda t : 1
        self.score_fn = score_fn

    def forward(self, x, t, labels):
        return -self.noise(t) * self.noise_dot(t) * self.score_fn(x, t, labels)

class ScoreAdditive(torch.nn.Module):
    def __init__(self, score_a, score_b):
        super().__init__()
        self.score_a = score_a
        self.score_b = score_b

    def forward(self, x, noise, labels):
        score = torch.zeros_like(x)
        score += self.score_a(x, noise, labels)
        score += self.score_b(x, noise)
        return score.to(noise.dtype)

class ScoreDenoise(torch.nn.Module):
    def __init__(self, denoiser, *, scale=1.0):
        super().__init__()
        self.denoiser = denoiser
        self.scale = scale

    def forward(self, x, noise, labels):
        denoised = self.denoiser(x, noise, labels).to(noise.dtype)
        return self.scale * (denoised - x) / noise ** 2

#----------------------------------------------------------------------------
# Helper functions for loading in templates from a path

def load_templates(path, device=None, dtype=None, for_torch=True, rescale=False):
    # Load templates data
    if isinstance(path, type(None)):
        return torch.tensor([0])
    elif os.path.isfile(path):
        templates = torch.unsqueeze(read_image(path, mode=ImageReadMode.RGB), 0)

    elif os.path.isdir(path):
        imgs = []
        for fname in sorted(os.listdir(path)): # iterate through each file in directory
            fpath = os.path.join(path, fname)
            if not os.path.isfile(fpath): 
                continue
            imgs.append(read_image(fpath))
        templates = torch.stack(imgs) if imgs else None

    else:
        raise ValueError(
            f"Template path must be an existing file, directory, or None; "
            f"got {path!r} (type {type(path).__name__})"
    )

    if device:
        templates = templates.to(device=device)
    if dtype:
        templates = templates.to(dtype=dtype)
    if rescale:
        templates = (templates - 128) / 127.5 
    return templates

def load_templates_batch(batch_template_info, device=None, dtype=None, for_torch=True, rescale=False):
    """
    batch_template_info: list of either paths, or list of filenames/indices to load from `template_dir`
    template_dir: if `batch_template_info` contains filenames or indices
    """

    batch_templates = []
    for entry in batch_template_info:
        batch_templates.append(load_templates(entry, device, dtype, for_torch))
    if for_torch:
        batch_templates = torch.concat(batch_templates) 

    if batch_templates == [] : 
        batch_templates = None

    return batch_templates

