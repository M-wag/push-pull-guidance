# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/
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
from dataclasses import dataclass, asdict, replace, fields
from typing import List, Any, Literal
import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

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

        # templates score
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
    type_latent:            Literal["pixel", "linear", "hf"] = None
    type_eval:              Literal["jvp", "numdiff"] | None = None
    template_path:          str | None = None
    scale:   float | list[float] | None  = 1.0
    decay_rate:             float | list[float] | None = None
    v_0:                    float | list[float] | None = None
    # Linear
    n_features:             float | list[int] | None = None
    dim_feature:            float | list[int] | None = None
    seed_mat:               int | None = None
    T:                      int | list[int] | None = None
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
class AttentionMixture:
    def __init__(self, means, stds, weights_mixture):
        # means: (N, D), stds: (N,), weights_mixture: (N,)
        self.means = means
        self.stds = stds
        self.weights_mixture = weights_mixture
        self.D = means.size(-1)  # Dimension of the data

        # Validate that mixture weights sum to 1.0
        if not torch.isclose(torch.sum(weights_mixture), torch.tensor(1.0, dtype=weights_mixture.dtype), atol=1e-6):
            raise ValueError(f"weights_mixture must sum to 1.0, got sum={torch.sum(weights_mixture).item():.4f}")

    def __call__(self, x, T=1.0):
        """
        Args:
            x: Input tensor of shape (B, D) where B is batch size
            T: Temperature parameter (>0) controlling softmax sharpness
        Returns:
            weights_attention : Attention assocciated with gradient of log-density of mixture model, shape (B, D)
        """
        B, D = x.shape
        N, _ = self.means.shape

        # Compute squared distances between x and all means (B, N)
        diff = self.means.unsqueeze(0) - x.unsqueeze(1)  # (1, N, D) - (B, 1, D) → (B, N, D)
        mahalanobis = (diff ** 2).sum(dim=-1) / self.stds.unsqueeze(0)  # (B, N)
        energy_mahalana = -0.5 * mahalanobis  ** 2  # (B,N)

        # Compute log terms for each component 
        log_weights = torch.log(self.weights_mixture + 1e-8)                     # (N,)
        log_std_term = -self.D * torch.log(self.stds + 1e-8)                    # (N,)

        # Combine, drop the constant -(D/2)*ln(2π) 
        energy = energy_mahalana + log_weights.unsqueeze(0) + log_std_term.unsqueeze(0)  # (B,N)

        # Apply temperature and compute attention weights (B, N)
        weights_attn = F.softmax(T * energy, dim=-1)
        return weights_attn


class GuidanceVF:
    def flat(self, x):
        return rearrange(x, "... c h w -> ... (c h w)")
    
    def unflat(self, x):
        return rearrange(x, "... (c h w) -> ... c h w", c=self.templates.shape[-3], h=self.templates.shape[-2], w=self.templates.shape[-1])

    def __init__(self, templates, scale, v_0, decay_rate, latent, latent_inv, *, flatten_input=False, threshold_weight=None,
                 threshold_time_min=None,
                 threshold_time_max=None):

        # Core parameters
        self.templates = templates
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

        # Pre-process templates 
        self.features_template = latent(self.flat(self.templates)) if flatten_input else latent(self.templates)
        if self.features_template.shape[0] > 1:
            self.features_template = self.features_template.flatten(0, 1)
        # Device and type tracking
        self.device = self.templates.device
        self.dtype = self.templates.dtype
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
            if self.features_template.shape[0] == 1:
                dirac_score = self._dirac_score(x, t)
            else:
                dirac_score = self._dirac_score_attention(x, t)
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
        dirac_score =  -(self.templates - x) / t
        return dirac_score

class LinearGuidanceVF(GuidanceVF):
    def __init__(self, *args, **kwargs):
        # Override latent mappings while passing through other params
        super().__init__(*args, **kwargs)

    def _dirac_score_attention(self, x, t):
        # (B, F, L)
        features = self.latent(x)
        # (B, F * T, L)
        features_copied = torch.repeat_interleave(features, dim=1, repeats=self.templates.shape[0])
        # (B, F * T, L) = (1, F * T, L) - (B, F * T, L) 
        diff_features = self.features_template.unsqueeze(0) - features_copied
        # (B, F*T, D) 
        recons = self.latent_inv(diff_features)
        # (B, F * T)
        attention = self.attention(x)
        # (B, D) = (B, F * T) o (B, F * T, D) 
        dirac_score = torch.einsum("BN, BND -> BD", attention, recons)

class JVPGuidanceVF(GuidanceVF):
    def __init__(self, *args, **kwargs):
        # Override latent mappings while passing through other params
        super().__init__(*args, **kwargs)
    def _dirac_score(self, x, t):
        with torch.no_grad():
            features = self.latent(x)
            dirac_score_latent =  -(self.features_template - features) / t
            # Jacobian vector product 
            _, dirac_score = jvp(self.latent_inv, features, dirac_score_latent, strict=False)
        return dirac_score

class NumericalGuidanceVF(GuidanceVF):
    def __init__(self, *args, epsilon=1e-3, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon  # Step size for finite differences

    def _dirac_score(self, x, t):
        with torch.no_grad():
            features = self.latent(x)
            dirac_score_latent = -(self.features_template - features) / t
            # Numerical differentiation
            perturbed_features = features + self.epsilon * dirac_score_latent
            f_perturbed = self.latent_inv(perturbed_features)
            f_original = self.latent_inv(features)
        
        return (f_perturbed - f_original) / self.epsilon  

# VF Builders
class BuilderVFBase:
    @classmethod
    def create(cls, prms : ConfigGuidanceVF, templates):
        """ Factory method to be implemented by subclasses"""
        raise NotImplementedError
    
    @classmethod 
    def _common_setup(cls, prms, templates, extra_exclusions=None):
        """Shared initialization logic"""

        exclusions = {"type_latent", "template_path", "type_eval"}

        if extra_exclusions:
            exclusions.update(extra_exclusions)
            
        kwargs = {k: v for k, v in prms.to_dict().items() 
                if k not in exclusions and v is not None}

        return kwargs, templates

class BuilderPixelVF(BuilderVFBase):
    @classmethod
    def create(cls, prms, templates):
        kwargs, templates = cls._common_setup(prms, templates)
        return PixelGuidanceVF(**kwargs, templates=templates)

class BuilderLinearVF(BuilderVFBase):
    @classmethod
    def _create_mappings(cls, prms, templates):
        """Create the matrix representation linear mapping and its pseudoinverse"""
        g = torch.Generator(device=templates.device).manual_seed(prms.seed_mat)
        template_dim = int(torch.prod(torch.tensor(templates.shape[1:])))
        mat_latent = torch.randn(
            (prms.n_features, prms.dim_feature, template_dim),
            generator=g,
            device=templates.device,
            dtype=templates.dtype
        )
        mat_latent_inv = torch.linalg.pinv(mat_latent)

        return mat_latent, mat_latent_inv

    @classmethod
    def create(cls, prms, templates):
        kwargs, templates = cls._common_setup(prms, templates,
                                              extra_exclusions = ("n_features", "dim_feature", "seed_mat", "T"))
        # Initialize mapping
        mat_latent, mat_latent_inv = cls._create_mappings(prms, templates)

        n_templates = templates.shape[0] 
        def latent_fn(x): 
            mat_latent_inv = torch.repeat_interleave(mat_latent, dim=0, repeats=n_templates)
            return torch.einsum("NLD, BD -> BNL", mat_latent, x)
        def latent_inv_fn(x):
            mat_latent_inv_stacked = torch.repeat_interleave(mat_latent_inv, dim=0, repeats=n_templates)
            # (F * T, L, D), (B, F * T, L) -> (B, F*T, D)
            return torch.einsum("NDL ,BNL ->BND", mat_latent_inv_stacked, x)

        # If templates dim is not flattend make sure it is flattend before applying linear transformations
        if len(templates.shape[1:]) > 1:
            _orig_latent_fn = latent_fn
            latent_fn = lambda x: _orig_latent_fn(x.flatten(start_dim=1))
            _orig_latent_inv_fn = latent_inv_fn
            latent_inv_fn = lambda x : _orig_latent_inv_fn(x).reshape(-1, *templates.shape[1:])
            

        # Attention mechanism
        means_attention = latent_fn(templates).flatten(start_dim=0, end_dim=1)
        std_attention =  .....
        weights_mixture = torch.ones(n_templates * templates.shape[0]) / n_templates * templates.shape[0]
        weights_mixture = weights_mixture.to(device=prms.device)
        
        # Assign to instance not class
        vf.attention = AttentionMixture(
            means_attention,
            std_attention,
            weights_mixture
        )

        vf = LinearGuidanceVF(
            **kwargs,
            templates=templates,
            latent=latent_fn,
            latent_inv=latent_inv_fn,
            attention = attention
        )

        return vf

        if templates.shape[0] > 1:
            # Calculate attention parameters
            #TODO : DOES THIS WORK FOR DIFFERNET V_0s
            means_attention = vf.latent(templates).flatten(0, 1) #(N, D) -> (N, F, L) -> (N * F, L)
            std_attention =  vf.v_0 = 
            weights_mixture = torch.ones(templates.shape[0]) / templates.shape[0]
            weights_mixture = weights_mixture.to(device=vf.device)
            
            # Assign to instance not class
            vf.attention = AttentionMixture(
                means_attention,
                std_attention,
                weights_mixture
            )

class BuilderHuggingfaceVF(BuilderVFBase):
    @classmethod
    def create(cls, prms, templates):
        kwags, templates = cls._common_setup(prms, templates, 
                                             extra_exclusions="hf_url")
        match prms.type_eval:
            case "numdiff":
                VF = NumericalGuidanceVF
            case "jvp":
                VF = JVPGuidanceVF

        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(prms.hf_url, subfolder="vae", use_safetensors=True)
        vae = vae.to(device=templates.device, dtype=templates.dtype)

        vf = VF(
                **kwargs_filtered,
                templates=templates,
                latent = lambda x : vae.encode(x).latent_dist.sample(),
                latent_inv = lambda x: vae.decode(x).sample
        )

        return vf 

class BuilderLinearHFVF(BuilderVFBase):
    @classmethod
    def create(cls, prms, templates):
        kwags, templates = cls._common_setup(prms, templates, 
                                             extra_exclusions="hf_url")
        match prms.type_eval:
            case "numdiff":
                VF = NumericalGuidanceVF
            case "jvp":
                VF = JVPGuidanceVF

        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(prms.hf_url, subfolder="vae", use_safetensors=True)
        vae = vae.to(device=templates.device, dtype=templates.dtype)


        vf = VF(
                **kwargs_filtered,
                templates=templates,
                latent = lambda x : linear_map(vae.encode(x).latent_dist.sample()),
                latent_inv = lambda x: vae.decode(linear_inv_map(x)).sample
        )


def create_vf(prms: ConfigGuidanceVF, templates, verbose=True):
    if verbose: 
        print(f"\n{prms}")
        print(f"\ntemplates_shape \t= {tuple(templates.shape)}")

    if prms is None:
        vf = lambda x, t: torch.zeros_like(x)
        return vf
    match prms.type_latent:
        case "pixel":
            vf = BuilderPixelVF.create(prms, templates)
        case "linear":
            vf = BuilderLinearVF.create(prms, templates)
        case "hf":
            vf = BuilderHuggingfaceVF.create(prms, templates)
        case "hf-linear":
            pass

    return vf


### SCHEDULER ###

def load_templates(cnfg : ConfigSimulation, for_torch=True):
    # Load templates data
    if isinstance(cnfg.guidance_vf, type(None)):
        return np.array([0])
    elif os.path.isfile(cnfg.guidance_vf.template_path):
        templates = torch.unsqueeze(read_image(cnfg.guidance_vf.template_path), 0)

    elif os.path.isdir(cnfg.guidance_vf.template_path):
        imgs = []
        for fname in sorted(os.listdir(cnfg.guidance_vf.template_path)): # iterate through each file in directory
            fpath = os.path.join(cnfg.guidance_vf.template_path, fname)
            if not os.path.isfile(fpath): 
                continue
            imgs.append(read_image(fpath))
        templates = torch.stack(imgs) if imgs else None

    else:
        raise ValueError(
            f"Template path must be an existing file, directory, or None; "
            f"got {cnfg.guidance_vf.template_path!r} (type {type(cnfg.guidance_vf.template_path).__name__})"
    )
    if for_torch:
        templates = (templates.to(device=cnfg.device, dtype=torch.float64) - 128) / 127.5 
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
    start_time = time.time()
    for idx, cnfg_split in enumerate(cnfg.split()):
        templates = load_templates(cnfg_split)
        vf_guide = create_vf(cnfg_split.guidance_vf, templates)
        xs, ts = generate_image_grid(net, vf_guide, cnfg.seed, cnfg.device,
                                     **cnfg.diffusion.to_dict())
        raw_data[idx] = (xs * 127.5 + 128) / 255
    total_time = time.time() - start_time
    print(f"Total schedule_diffusion time: {total_time:.2f} s")
    
    return raw_data

