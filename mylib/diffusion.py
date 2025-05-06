# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/
import tqdm
import pickle
import numpy as np
import torch
import os
import itertools
import dnnlib
from PIL import Image
from einops import rearrange, repeat
from torchvision.io import read_image
from torch.autograd.functional import jvp
from dataclasses import dataclass, asdict, replace, fields
from typing import List, Any
import torch
from einops import rearrange

#----------------------------------------------------------------------------
def generate_image_grid(
    net, 
    vf_template,         # Vector field induced by temlate and features      
    seed                : int , 
    device              ,
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
):
    if seed is not None:
        torch.manual_seed(seed)

    # Pick latents and labels.
    latents = torch.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
    class_labels = None
    if net.label_dim:
        class_labels = torch.eye(net.label_dim, device=device)[batch_size * [282]]

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    xs = None 
    # Intialize empty array to save intermediate timestaps
    if save_all_timesteps:
        xs = torch.empty((num_steps, batch_size, net.img_channels, net.img_resolution, net.img_resolution))
        metrics = torch.empty((5, num_steps))

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]

    for i, (t_cur, t_next) in tqdm.tqdm(list(enumerate(zip(t_steps[:-1], t_steps[1:]))), unit='step', position=1): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

        # template score
        d_template = vf_template(x_hat, t_hat)

        # model score
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_model = scale_model_score * (x_hat - denoised) / t_hat
        d_cur = (d_template + d_model) * (t_next - t_hat)
        x_next = x_hat + d_cur

        # Save intermediate timsteps
        if save_all_timesteps:
            xs[i] = x_next
            metrics[0, i] = torch.norm(x_next)

    return xs.numpy(), t_steps.cpu().numpy()

#----------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    def to_dict(self):
        return asdict(self)

    def __str__(self):
        lines = []
        max_key_len = max(len(k) for k in self.__dataclass_fields__)

        for key in self.__dataclass_fields__:
            value = getattr(self, key)
            if value is None:
                continue
            formatted_value = (
                f"[{', '.join(map(str, value))}]" if isinstance(value, list)
                else repr(value)
            )
            lines.append(f"{key:<{max_key_len}} = {formatted_value}")
        return "\n".join(lines)
    
    def __call__(self, **kwargs) -> 'Config':
        """  eturn a new instance of this Config with specified fields replaced."""
        invalid  = set(kwargs) - set(self.__dataclass_fields__)
        if invalid:
            raise AttributeError(f"Unknown fields for {type(self).__name__}: {invalid}")
        return replace(self, **kwargs)

    def split(self):
        # Split fields based on whether they contains lists
        fields_list = {}
        fields_no_list = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, Config):
                fields_list[field.name] = value.split()
            elif isinstance(value, list):
                fields_list[field.name] = value
            else:
                fields_no_list[field.name] = value

        # Make every combination of values and make new configs
        combinations = [dict(zip(fields_list.keys(), vals)) for vals in itertools.product(*fields_list.values())]
        cnfgs_split = []
        for combo in combinations:
            cnfgs_split.append(type(self)(**combo, **fields_no_list))

        return cnfgs_split 

    @property
    def shape_combination(self):
        return tuple(
                len(getattr(self, field.name))
                for field in fields(self)
                if isinstance(getattr(self, field.name), list)
        )

@dataclass(frozen=True)
class ConfigGuidanceVF(Config):
    scale_template_score:   float | list[float] | None  = None
    decay_rate:             float | list[float] | None = None
    v_0:                    float | list[float] | None = None
    n_projectors:           float | list[int] | None = None
    dim_projector:          float | list[int] | None = None
    template_path:          str | None = None

@dataclass(frozen=True)
class ConfigDiffusion(Config):
    scale_model_score:  float = 1.0
    batch_size :        float = 9
    num_steps:          int = 32
    sigma_min:          float = 0.002  
    sigma_max:          float = 80
    rho:                float = 7
    S_churn:            float = 0.0
    S_min:              float = 0.0
    S_max:              float = float('inf')
    S_noise:            float = 1

@dataclass(frozen=True)
class ConfigSimulation(Config):
    network_pkl:    str
    device:         str 
    seed:           int | None 
    input_shape:    tuple[int]
    guidance_vf:    ConfigGuidanceVF 
    diffusion:      ConfigDiffusion 

# Guidance Vector Fields
class GuidanceVF:
    def flat(self, x) : return rearrange(x, "... c h w -> ... (c h w)")
    def unflat(self, x) : return rearrange(x, "... (c h w) -> ... c h w", c=self.template.shape[-3], h=self.template.shape[-2], w=self.template.shape[-1])

    def __init__(self, template, v_0, decay_rate, latent, latent_inv, flatten_input=False):
        self.template = template
        self.v_0 = v_0
        self.decay_rate = decay_rate
        self.latent = latent
        self.latent_inv = latent_inv

        self.flatten_input = flatten_input
        self.features_template = latent(self.flat(self.template) if self.flatten_input else self.template)
        self.device = template.device
        self.dtype = template.dtype

    def __call__(self, x, t):
        raise NotImplementedError

class PixelGuidanceVF(GuidanceVF):
    def __call__(self, x, t):
        x = self.flat(x) if self.flatten_input else x 
        score = torch.sigmoid(self.decay_rate * (t - self.v_0)) * (self.template - x) / t
        score = self.unflat(score) if self.flatten_input else score
        return score

class LinearGuidanceVF(GuidanceVF):
    def __init__(self, template, v_0, decay_rate, feature_mat, flatten_input=True):
        self.template = template
        self.v_0 = v_0
        self.decay_rate = decay_rate
        self.A = feature_mat
        self.A_inv = torch.linalg.pinv(feature_mat)

        self.flatten_input = flatten_input
        if self.flatten_input:
            self.features_template = self.flat(self.template) @ self.A.T 
        else:
            self.features_template = self.template @ self.A.T 
        self.device = template.device
        self.dtype = template.dtype

    def __call__(self, x, t):
        x = self.flat(x) if self.flatten_input else x 
        diff_features = self.features_template - (x@ self.A.T)
        diff_projected = (diff_features @ self.A_inv.T)
        score = torch.sigmoid(self.decay_rate * (t - self.v_0)) * diff_projected / t
        score = self.unflat(score) if self.flatten_input else score
        return score


def create_guidance_vf(prms : ConfigGuidanceVF):
    if prms is None:
        vf = lambda x, t: torch.zeros_like(x)
        return vf
    if prms.feature_type == "linear":
        if (prms.template.shape == prms.dim_projector):
            assert prms.n_projectors == 1, "Having more then one then [] is useless"
            vf = PixelGuidanceVF(prms)
        else: 
            raise NotImplementedError

# SCHEDULER
def schedule_diffusion(cnfg : ConfigSimulation):
    # Set seed
    if cnfg.seed is not None:
        torch.manual_seed(cnfg.seed)
        print(f"Setting config seed {cnfg.seed}")

    # Load network
    print(f'Loading network from "{cnfg.network_pkl}"...')
    with dnnlib.util.open_url(cnfg.network_pkl) as f:
        net = pickle.load(f)['ema'].to(cnfg.device)

    # Load template data
    if isinstance(cnfg.guidance_vf, type(None)):
        templates=None
    elif os.path.isfile(cnfg.guidance_vf.template_path):
        img = read_image(cnfg.guidance_vf.template_path)
        templates = (img.to(device=cnfg.device, dtype=torch.float64) - 128) / 127.5 

    elif os.path.isdir(cnfg.guidance_vf.template_path):
        imgs = []
        for fname in sorted(os.listdir(cnfg.guidance_vf.template_path)): # iterate through each file in directory
            fpath = os.path.join(cnfg.guidance_vf.template_path, fname)
            if not os.path.isfile(fpath): 
                continue
            imgs.append(read_file(fpath))
        templates = torch.stack(imgs) if imgs else None

    else:
        raise ValueError(
            f"Template path must be an existing file, directory, or None; "
            f"got {cnfg.guidance_vf.template_path!r} (type {type(cnfg.guidance_vf.template_path).__name__})"
        )

    # Iterate through combinations of parameters
    raw_data = np.empty((len(cnfg.split()), cnfg.diffusion.num_steps, cnfg.diffusion.batch_size, *cnfg.input_shape)) # (N_combs, t, B, C, H, W)
    assert len(raw_data.shape) == 6, f"raw_data should have rank 6, got shape : {raw_data.shape}"
    for idx, cnfg_split in enumerate(cnfg.split()):
        vf_guide = create_guidance_vf(cnfg.guidance_vf)
        xs, ts = generate_image_grid(net, vf_guide, cnfg.seed, cnfg.device,
                                     **cnfg.diffusion.to_dict())
        raw_data[idx] = (xs * 127.5 + 128) / 255
    
    return raw_data

