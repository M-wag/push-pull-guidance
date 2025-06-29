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
from torchvision.io import read_image, read_file
from torch.autograd.functional import jvp
from dataclasses import dataclass
from typing import List, Any, Literal, Callable
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from functools import partial
from .helpers import Config

### Samplers ###
def edm_sampler(
    net, 
    vf_template,         # Vector field induced by tempaplate and features      
    seed                : int , 
    device              ,
    dtype               ,
    *,
    class_idx           : int , 
    batch_size          : int,
    num_steps           : int, 
    sigma_min           : float,
    sigma_max           : float, 
    rho                 : float, 
    S_churn             : float, 
    S_min               : float, 
    S_max               : float,
    S_noise             : float, 
    scale_model_score   : float, 
    save_all_timesteps  : bool = True,
    latents             : Any = None,
):
    if seed is not None:
        torch.manual_seed(seed)

    # Pick latents and labels.
    if latents is None:
        g = torch.Generator(device=device).manual_seed(seed)
        latents = torch.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device, generator=g)

    # Handle class labels
    if net.label_dim:
        if class_idx is None:
            # Use zeros (no class)
            class_labels = torch.zeros([batch_size, net.label_dim], device=device)
        elif isinstance(class_idx, int):
            # Use one-hot encoded specified class
            class_labels = torch.eye(net.label_dim, device=device)[batch_size * [class_idx]]
        elif torch.is_tensor(class_idx):
            # Use provided class labels directly
            assert class_idx.shape == (batch_size, net.label_dim), \
                f"class_labels must have shape [{batch_size}, {net.label_dim}]"
            class_labels = class_idx.to(device)
        else:
            raise ValueError("class_idx must be None, int, or tensor")
    else:
        class_labels = None

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=dtype, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    def gradient(x, t):
        denoised = net(x, t, class_labels).to(dtype)
        grad_template = vf_template(x, t)
        grad_model = scale_model_score * (x - denoised) / t
        return grad_template + grad_model

    xs = None 
    # Intialize empty array to save intermediate timestaps
    print("Running network")
    if save_all_timesteps:
        xs = torch.empty((num_steps, batch_size, net.img_channels, net.img_resolution, net.img_resolution))
        metrics = torch.empty((5, num_steps))

    # Main sampling loop.
    x_next = latents.to(dtype) * t_steps[0]
    for i, (t_cur, t_next) in tqdm.tqdm(list(enumerate(zip(t_steps[:-1], t_steps[1:]))), unit='step', position=1): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

        dx = gradient(x_hat, t_hat) * (t_next - t_hat)
        x_next = x_hat + dx

        # Save intermediate timsteps
        if save_all_timesteps:
            xs[i] = x_next
            metrics[0, i] = torch.norm(x_next)

    xs = (xs * 127.5 + 128) / 255
    return xs, (t_steps, )

def load_templates(path, device=None, dtype=None, for_torch=True):
    # Load templates data
    if isinstance(path, type(None)):
        return torch.tensor([0])
    elif os.path.isfile(path):
        templates = torch.unsqueeze(read_image(path), 0)

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
    if for_torch:
        templates = (templates - 128) / 127.5 
    return templates

def load_templates_batch(batch_template_info, device=None, dtype=None, for_torch=True):
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

def schedule_diffusion(cnfg : ConfigSimulation):
    # Set seed
    if cnfg.seed is not None:
        torch.manual_seed(cnfg.seed)
        print(f"Setting config seed {cnfg.seed}")

    # Load network and update network 
    print(f'Loading network from "{cnfg.network_pkl}"...')
    with dnnlib.util.open_url(cnfg.network_pkl) as f:
        net_old = pickle.load(f)['ema'].to(cnfg.device)
    net = EDMPrecond(*net_old.init_args, **net_old.init_kwargs).to(cnfg.device)
    net.model.save_skips = True
    net.eval()
    misc.copy_params_and_buffers(net_old, net, require_all=True)

    # Iterate through combinations of parameters
    raw_data = np.empty((len(cnfg.split()), cnfg.diffusion.num_steps, cnfg.diffusion.batch_size, *cnfg.input_shape)) # (N_combs, t, B, C, H, W)
    assert len(raw_data.shape) == 6, f"raw_data should have rank 6, got shape : {raw_data.shape}"
    start_time = time.time()
    for idx, cnfg_split in enumerate(cnfg.split()):
        # Create guidance vectorfield
        templates = load_templates_batch([cnfg_split.guidance_vf.template_path] * cnfg_split.diffusion.batch_size, device=cnfg_split.device, dtype=cnfg.dtype)
        vf = create_vf(cnfg_split.guidance_vf, templates, net=net, cnfg_split_sim=cnfg_split)

        with torch.no_grad():
            xs, _ = edm_sampler(net, vf, seed=cnfg_split.seed, device=cnfg_split.device, dtype=cnfg.dtype, **cnfg_split.diffusion.to_dict())
        raw_data[idx] = xs
        
    total_time = time.time() - start_time
    print(f"Total schedule_diffusion time: {total_time:.2f} s")
    
    return raw_data

