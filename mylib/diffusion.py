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
from typing import List, Any, Literal
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
    def shape_combination(self) -> tuple[int]:

        def collect_dims(cfg) -> list[int]:
            dims = []
            for field in fields(cfg):
                val = getattr(cfg, field.name)
                if isinstance(val, list):
                    dims.append(len(val))
                elif isinstance(val, Config):
                    dims.extend(collect_dims(val))
            return dims

        return tuple(collect_dims(self))


@dataclass(frozen=True)
class ConfigGuidanceVF(Config):
    # Core
    vf_type:                Literal["pixel", "linear", "hf"]
    template_path:          str | None = None
    scale_template_score:   float | list[float] | None  = 1.0
    decay_rate:             float | list[float] | None = None
    v_0:                    float | list[float] | None = None
    # Linear
    n_projectors:           float | list[int] | None = None
    dim_projector:          float | list[int] | None = None
    # Hugging Face
    hf_url:                 str = None
    # Optional
    flatten_input:          bool | None = False
    threshold_weight:       float | list[float] = None
    threshold_time_min:     float | list[float] | None = None
    threshold_time_max:     float | list[float] | None = None

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

### Guidance Vector Fields

class GuidanceVF:
    def flat(self, x):
        return rearrange(x, "... c h w -> ... (c h w)")
    
    def unflat(self, x):
        return rearrange(x, "... (c h w) -> ... c h w", c=self.template.shape[-3], h=self.template.shape[-2], w=self.template.shape[-1])

    def __init__(self, template, scale, v_0, decay_rate, latent, latent_inv, *, flatten_input=False, threshold_weight=None,
                 threshold_time_min=None,
                 threshold_time_max=None):

        # Core parameters
        self.template = template
        self.scale = scale
        self.v_0 = v_0
        self.decay_rate = decay_rate
        self.latent = latent
        self.latent_inv = latent_inv
        # Optional features
        self.flatten_input = flatten_input
        self.threshold_weight = threshold_weight
        self.threshold_time_min = threshold_time_min
        self.threshold_time_max = threshold_time_max

        # Pre-process template 
        self.features_template = latent(self.flat(template)) if flatten_input else latent(template)
        # Device and type tracking
        self.device = template.device
        self.dtype = template.dtype
        # For testing
        self.history_weight = []
        self.history_apply_score = []


    def __call__(self, x, t):
        x = self.flat(x) if self.flatten_input else x
        weight = torch.sigmoid(self.decay_rate * (t - self.v_0)) * self.scale
        
        apply_score = True
        # Check weight threshold
        if self.threshold_weight is not None and weight < self.threshold_weight:
            apply_score = False
        # Check time thresholds
        if self.threshold_time_min is not None and t < self.threshold_time_min:
            apply_score = False
        if self.threshold_time_max is not None and t > self.threshold_time_max:
            apply_score = False
        
        if apply_score:
            dirac_score = self._dirac_score(x, t)
            score = weight * dirac_score
        else:
            score = torch.zeros_like(x)
        
        score = self.unflat(score) if self.flatten_input else score

        self.history_weight.append(weight)
        self.history_apply_score.append(apply_score)
        return score

    def _dirac_score(self, x, t):
        raise NotImplementedError("Subclasses must implement this method")

class PixelGuidanceVF(GuidanceVF):
    def __init__(self, *args, **kwargs):
        # Override latent mappings while passing through other params
        super().__init__(*args, **kwargs, latent=lambda x: x, latent_inv=lambda x: x)

    def _dirac_score(self, x, t):
        dirac_score =  -(self.template - x) / t
        return dirac_score

class JVPGuidanceVP(GuidanceVF):
    def _dirac_score(self, x, t):
        features = self.latent(x)
        dirac_score_latent =  -(self.features_template - features) / t
        # Jacobian vector product 
        _, dirac_score = jvp(self.latent_inv, features, dirac_score_latent, strict=True)
        return dirac_score

class NumericalGuidanceVP(GuidanceVF):
    def __init__(self, *args, epsilon=1e-3, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon  # Step size for finite differences

    def _dirac_score(self, x, t):
        features = self.latent(x)
        dirac_score_latent = -(self.features_template - features) / t
        # Numerical differentiation
        perturbed_features = features + self.epsilon * dirac_score_latent
        f_perturbed = self.latent_inv(perturbed_features)
        f_original = self.latent_inv(features)
        
        return (f_perturbed - f_original) / self.epsilon  

def create_guidance_vf(prms : ConfigGuidanceVF, templates, verbose=True):
    if verbose: 
        print(f"\n{prms}")

    # Check if vector field is defined
    if prms is None:
        vf = lambda x, t: torch.zeros_like(x)
        return vf

    # Match specicic type of vector field
    match prms.vf_type:
        case "pixel":
            kwargs_filtered = {
                k: v
                for k, v in prms.to_dict().items()
                if k not in ("vf_type", "template_path") and v is not None
            }
            kwargs_filtered['scale'] = kwargs_filtered.pop('scale_template_score')
            vf = PixelGuidanceVF(**kwargs_filtered, template=templates)

        case "hf":
            from diffusers import AutoencoderKL
            vae = AutoencoderKL.from_pretrained(prms.hf_url, subfolder="vae", use_safetensors=True)
            vae = vae.to(device=templates.device, dtype=templates.dtype)

            kwargs_filtered = {
                k: v
                for k, v in prms.to_dict().items()
                if k not in ("vf_type", "template_path") and v is not None
            }
            kwargs_filtered['scale'] = kwargs_filtered.pop('scale_template_score')

            vf = JVPGuidanceVP(
                    **kwargs_filtered,
                    template=templates,
                    latent = lambda x : vae.encode(x).latent_dist.sample(),
                    latent_inv = lambda x: vae.decode(x).sample
            )

        case _:
            raise ValueError(f"Received unexepcted vector field type: {prvs.vf_type}")

    return vf


### SCHEDULER ###

def load_templates(cnfg : ConfigSimulation):
    # Load template data
    if isinstance(cnfg.guidance_vf, type(None)):
        templates=None
    elif os.path.isfile(cnfg.guidance_vf.template_path):
        img = torch.unsqueeze(read_image(cnfg.guidance_vf.template_path), 0)
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
    return templates

def schedule_diffusion(cnfg : ConfigSimulation):
    # Set seed
    if cnfg.seed is not None:
        torch.manual_seed(cnfg.seed)
        print(f"Setting config seed {cnfg.seed}")

    # Load network
    print(f'Loading network from "{cnfg.network_pkl}"...')
    with dnnlib.util.open_url(cnfg.network_pkl) as f:
        net = pickle.load(f)['ema'].to(cnfg.device)

    # Iterate through combinations of parameters
    raw_data = np.empty((len(cnfg.split()), cnfg.diffusion.num_steps, cnfg.diffusion.batch_size, *cnfg.input_shape)) # (N_combs, t, B, C, H, W)
    assert len(raw_data.shape) == 6, f"raw_data should have rank 6, got shape : {raw_data.shape}"
    for idx, cnfg_split in enumerate(cnfg.split()):
        templates = load_templates(cnfg_split)
        vf_guide = create_guidance_vf(cnfg_split.guidance_vf, templates)
        xs, ts = generate_image_grid(net, vf_guide, cnfg.seed, cnfg.device,
                                     **cnfg.diffusion.to_dict())
        raw_data[idx] = (xs * 127.5 + 128) / 255
    
    return raw_data

