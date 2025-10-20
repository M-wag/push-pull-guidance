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

    def init_gradient(self, gradient_kwargs):
        self.gradient_fn = create_gradient_fn(gradient_kwargs)

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

class InjectFusionGradient:
    def __init__(
        self,
        scale_model_score,
        t_edit_start,
        gamma,
        h_exam_per_t,
        mask=None,
    ):
        self.scale_model_score = scale_model_score
        self.t_edit_start = t_edit_start
        self.gamma = gamma
        self.h_exam_by_t = h_exam_per_t
        self.mask = mask if mask else 1.0 # If no mask is passed, then apply injection to all features in bottleneck

    def __call__(self, x, t, labels, net):

        # Compute standard denoising gradient without injectios
        if t < self.t_edit_start:
            return self._gradient_denoise(x, t, labels, net)

        # Step 1 : Content Injection
        skips = net.encoder(x, t, labels)
        h_orig = skips[-1]
        h_exam = self.h_exam_by_t[t]
        
        # Normalize example features to match original's norm
        h_exam_norm = self.normalize_to_target_norm(h_exam, h_orig)
        
        # Apply mask and perform spherical interpolation
        h_masked_orig = self.mask * h_orig
        h_masked_exam = self.mask * h_exam_norm
        h_mixed = self.slerp(h_masked_orig, h_masked_exam)
        
        # Combine with original features
        h_mixed += (1 - self.mask) * h_orig
        
        # Get predictions with injected content
        skips[-1] = h_mixed
        denoised_injected = net.decode(x, skips, t, labels)

        return self.scale_model_score * (x - denoised_injected)  / t
        
        if False:
            epsilon_injected = ...
            
            # Get original prediction
            denoised_original = net.decode(x, net.encode(x, t, labels), t, labels)
            epsilon_original = ...
            # Step 2: Latent calibration
            x_calibrated = self.latent_calibration(x, t, eps_original, eps_injected)
            denoised_calibrated = net(x_calibrated, t, labels)
            grad_calibrated = self.scale_model_score * (x - denoised_calibrated) / t

    def normalize_to_target_norm(self, source, target):
        """Normalize source features to have the same norm as target features."""
        source_flat = source.view(source.shape[0], -1)
        target_flat = target.view(target.shape[0], -1)
        
        source_norm = torch.norm(source_flat, dim=-1, keepdim=True).clamp_min(1e-10)
        target_norm = torch.norm(target_flat, dim=-1, keepdim=True).clamp_min(1e-10)
        
        # Rescale source to match target's norm
        source_normalized = source_flat / source_norm * target_norm
        return source_normalized.view_as(source)

    def slerp(self, v0, v1):
        """Spherical linear interpolation between two vectors."""
        # Flatten spatial dimensions
        v0_flat = v0.view(v0.shape[0], -1)
        v1_flat = v1.view(v1.shape[0], -1)
        
        # Compute cosine of angle between vectors
        dot = (v0_flat * v1_flat).sum(dim=-1, keepdim=True)
        v0_norm = torch.norm(v0_flat, dim=-1, keepdim=True).clamp_min(1e-10)
        v1_norm = torch.norm(v1_flat, dim=-1, keepdim=True).clamp_min(1e-10)
        cos_theta = dot / (v0_norm * v1_norm)
        
        # Clamp to avoid numerical issues
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        theta = torch.acos(cos_theta)
        
        # Compute interpolation weights
        sin_theta = torch.sin(theta)
        w0 = torch.sin((1 - self.gamma) * theta) / sin_theta
        w1 = torch.sin(self.gamma * theta) / sin_theta
        
        # Handle cases where sin_theta is close to zero (use linear interpolation)
        linear_mask = sin_theta < 1e-7
        w0 = torch.where(linear_mask, 1 - self.gamma, w0)
        w1 = torch.where(linear_mask, self.gamma, w1)
        
        # Interpolate and reshape
        interpolated = w0 * v0_flat + w1 * v1_flat
        return interpolated.view_as(v0)

    def latent_calibration(self, x, epsilon_injected, epsilon_original, sqrt_alpha_t, sqrt_one_minus_alpha_t):
        """
        Implement latent calibration as described in the InjectFusion paper.
        """
        # Compute Pt for both predictions
        Pt_epsilon_injected = (x - sqrt_one_minus_alpha_t * epsilon_injected) / sqrt_alpha_t
        Pt_epsilon_original = (x - sqrt_one_minus_alpha_t * epsilon_original) / sqrt_alpha_t
        
        # Regularize Pt(epsilon_injected) to have same std as Pt(epsilon_original)
        mu_Pt_injected = torch.mean(Pt_epsilon_injected, dim=(1, 2, 3), keepdim=True)
        mu_Pt_original = torch.mean(Pt_epsilon_original, dim=(1, 2, 3), keepdim=True)
        std_Pt_injected = torch.std(Pt_epsilon_injected, dim=(1, 2, 3), keepdim=True)
        std_Pt_original = torch.std(Pt_epsilon_original, dim=(1, 2, 3), keepdim=True)
        
        # Avoid division by zero
        std_Pt_injected = std_Pt_injected.clamp_min(1e-10)
        
        # Regularize
        P_prime_t = mu_Pt_injected + (Pt_epsilon_injected - mu_Pt_injected) * (std_Pt_original / std_Pt_injected)
        
        # Compute dPt and dϵ
        dPt = P_prime_t - Pt_epsilon_original
        d_epsilon = epsilon_injected - epsilon_original
        
        # Compute dx according to Eq. 10
        dx = sqrt_alpha_t * dPt + self.omega * sqrt_one_minus_alpha_t * d_epsilon
        
        # Return calibrated latent
        return x + dx


def create_gradient_fn(grad_kwargs : dict) -> GradientMethod:
    keys = set(grad_kwargs.keys())

    if {"gvf"}.issubset(keys):
         return GVFGradient(**grad_kwargs)
    elif {"t_edit_start"}.issubset(keys):
         return InjectFusionGradient(**grad_kwargs)
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

