import tqdm
import time
import pickle
import numpy as np
import torch 
import os
import itertools
import dnnlib
from PIL import Image
from einops import rearrange, repeat
from torchvision.io import read_image, ImageReadMode
from torch.autograd.functional import jvp
from dataclasses import dataclass
from typing import List, Any, Literal, Callable, Optional
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange

DISABLE_TQDM = False

@torch.no_grad()
def edm_sampler(
    net                 , 
    noise               ,
    labels              ,
    gvf                 ,       
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
    scale_model_score   : float = 1.0, 
    save_all_timesteps  : bool = True,
    correct_rgb         : bool = True,
    disable_tqdm        : bool = False,
):
    # Gradient of denoiser
    def gradient(x, t):
        grad = torch.zeros_like(x)
        denoised = net(x, t, labels).to(dtype)
        grad += scale_model_score * (x - denoised) / t
        if gvf:
            grad += gvf(x, t)
        return grad.to(dtype)

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    t_steps = edm_time_step_discrretization(num_steps, sigma_min, sigma_max, rho) # t_N=0
    t_steps = t_steps.to(device=device, dtype=dtype)

    xs = None 
    # Intialize empty array to save intermediate timestaps
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
        d_cur = gradient(x_hat, t_hat) 
        x_next = x_hat + d_cur * (t_next - t_hat)

        # Apply 2nd order correction
        if apply_2nd_order and i < num_steps - 1:
            d_prime = gradient(x_next, t_next) 
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        
        if save_all_timesteps:
            xs[i] = x_next

    y = xs if save_all_timesteps else x_next
    if correct_rgb:
        y = (y * 127.5 + 128) / 255

    return y, (t_steps.cpu().numpy(), )

def edm_time_step_discrretization(num_steps, sigma_min, sigma_max, rho):
    step_indices = torch.arange(num_steps)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
    return t_steps

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

