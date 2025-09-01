import tqdm
import numpy as np
import torch 
import os
from torchvision.io import read_image, ImageReadMode
from abc import ABC, abstractmethod

DISABLE_TQDM = False

#----------------------------------------------------------------------------
# Modified version of EDM sampler.
# Modular options for gradient and time step discretization.

class EDMSampler:
    def __init__(self, time_disc: str, gradient_kwargs={}):
        self.gradient_fn = create_gradient_fn(gradient_kwargs)

        if time_disc == "edm":
            self.time_step_fn = time_steps_edm
        elif time_disc == "ddim":
            raise NotImplementedError()
        else:
            raise ValueError(f"Unknown time step method: {time_disc}")

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
    ):
        # Adjust noise levels based on what's supported by the network
        sigma_min = max(sigma_min, net.sigma_min)
        sigma_max = min(sigma_max, net.sigma_max)
        
        # Time step discretization.
        t_steps = self.time_step_fn(num_steps, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho) # t_N=0
        t_steps = t_steps.to(device=device, dtype=dtype)

        xs = None 
        # Intialize empty array to save intermediate timesteps
        if save_all_timesteps:
            xs = torch.empty((num_steps, noise.shape[0], net.img_channels, net.img_resolution, net.img_resolution))

        # Main sampling loop.
        x_next = noise.to(dtype) * t_steps[0]
        for i, (t_cur, t_next) in tqdm.tqdm(list(enumerate(zip(t_steps[:-1], t_steps[1:]))), unit='step', position=1, disable=disable_tqdm): # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

            # Euler step
            d_cur = self.gradient_fn(x_hat, t_hat, labels, net)
            x_next = x_hat + d_cur * (t_next - t_hat)

            # Apply 2nd order correction
            if apply_2nd_order and i < num_steps - 1:
                d_prime = self.gradient_fn(x_next, t_next, labels, net) 
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
        save_all_timesteps: bool = True,
        disable_tqdm: bool = False,
        **kwargs,
    ):

        # Adjust noise levels based on what's supported by the network
        sigma_min = max(sigma_min, net.sigma_min)
        sigma_max = min(sigma_max, net.sigma_max)

        # Reverse the time steps (from low noise to high noise)
        t_steps = self.time_step_fn(num_steps, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho)
        t_steps = torch.flip(t_steps, [0]) 
        t_steps = t_steps.to(device=device, dtype=dtype)

        # Initialize empty array arraty
        if save_all_timesteps:
            xs = torch.empty((num_steps, images.shape[0], net.img_channels, net.img_resolution, net.img_resolution))

        # Main inversion loop (go from low noise to high noise)
        x_cur = images.to(dtype)
        for i, (t_cur, t_next) in tqdm.tqdm( list(enumerate(zip(t_steps[:-1], t_steps[1:]))), unit='step', position=1, disable=disable_tqdm):
            print(t_cur.item(), t_next.item())
            # Reverse Euler step
            denoised = net(x_cur, t_cur, labels).to(dtype)
            d_cur = (x_cur - denoised) / t_cur
            x_next = x_cur - (t_next - t_cur) * d_cur
            
            if save_all_timesteps:
                xs[i] = x_next
                
            x_cur = x_next

        x_T = xs[-1] if save_all_timesteps else x_next
        return x_T, xs if save_all_timesteps else None

#----------------------------------------------------------------------------
# Time step discretization functions 

def time_steps_edm(num_steps,  net=None, *, sigma_min, sigma_max, rho):
        step_indices = torch.arange(num_steps, dtype=torch.float64)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        if net:
            t_steps = net.round_sigma(t_steps)
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0
        return t_steps

def time_steps_ddim(num_steps, device, net=None):
    raise NotImplementedError

#----------------------------------------------------------------------------
# Different gradients useable by the EDM sampler
# dx/dt = gradient(x, t)

class GradientMethod(ABC):
    def __init__(self, scale_model_score):
        self.scale_model_score = scale_model_score

    @abstractmethod
    def __call__(self, x, t, labels, net):
        pass

    def _gradient_denoise(self, x, t, labels, net):
        """Standard gradient computation that can be reused by subclasses."""
        denoised = net(x, t, labels).to(t.dtype)
        grad = self.scale_model_score * (x - denoised) / t
        return grad.to(t.dtype)

class DenoisingGradient(GradientMethod):
    def __init__(self, scale_model_score):
        self.scale_model_score = scale_model_score

    def __call__(self, x, t, labels, net):
        return self._gradient_denoise(x, t, labels, net)

class GVFGradient(GradientMethod):
    def __init__(self, scale_model_score, gvf):
        self.scale_model_score = scale_model_score
        self.gvf = gvf

    def __call__(self, x, t, labels, net):
        grad = torch.zeros_like(x)
        grad += self._gradient_denoise(x, t, labels, net)
        grad += self.gvf(x, t)
        return grad.to(t.dtype)

class ThresholdedGradient(GradientMethod):
    def __init__(self, scale_model_score, t0):
        self.scale_model_score = scale_model_score
        self.t0 = t0

    def __call__(self, x, t, labels, net):
        if t > self.t0: 
            return torch.zeros_like(x)
        return self._gradient_denoise(x, t,  labels, net)

def create_gradient_fn(grad_kwargs : dict) -> GradientMethod:
    keys = set(grad_kwargs.keys())

    if {"gvf"}.issubset(keys):
         return GVFGradient(**grad_kwargs)
    elif {"t0"}.issubset(keys):
         return ThresholdedGradient(**grad_kwargs)
    else:
         return DenoisingGradient(**grad_kwargs)

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

