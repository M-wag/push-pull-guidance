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
import dnnlib
from PIL import Image
import visualization as vis
from einops import rearrange, repeat
from torchvision.io import read_image
from torch.autograd.functional import jvp


#----------------------------------------------------------------------------

def generate_image_grid(
    net, 
    vf_template,                                # Vector field induced by temlate and features      
    seed                = 0, 
    grid_w               = 2, 
    grid_h               = 2,  
    device              = torch.device('cuda'),
    num_steps           = 18, 
    sigma_min           = 0.002, 
    sigma_max           = 80, 
    rho                 = 7,
    S_churn             = 0, 
    S_min               = 0, 
    S_max               = float('inf'), 
    S_noise             = 1,
    save_all_timesteps  =True,
    scale_model_score   =1.0,
    scale_template_score=0.0,
    **kwargs,
):
    batch_size = grid_w * grid_h
    torch.manual_seed(seed)

    # Pick latents and labels.
    print(f'Generating {batch_size} images...')
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
        d_template = scale_template_score * vf_template(x_hat, t_hat)

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



import torch
from einops import rearrange

class AttentionMixture:
    def __init__(self, means, stds, mix_weights):
        # means: (N, D), stds: (N,), mix_weights: (N,)
        self.means = means
        self.stds = stds 
        self.mix_weights = mix_weights
        self.D = means.size(-1)

    def __call__(self, x, std_noise) -> torch.Tensor:
        # TODO Refactor to use softmax
        # x is now batched: (B, D)
        # Compute the covariance factor for each mixture component.
        cov_factor = self.stds**2 + std_noise**2  # shape (N,)
        cov_factor = rearrange(cov_factor, 'n -> n 1 1')  # shape (N, 1, 1)
        # Build a covariance matrix for each component: (N, D, D)
        cov_mats = cov_factor * torch.eye(self.D, device=self.means.device)
        
        # Compute the log mixture weights (to avoid log(0), add a small epsilon)
        log_mix_weights = torch.log(self.mix_weights + 1e-12)  # shape (N,)

        # Create a batch of multivariate Gaussians for the N mixture components.
        mvns = torch.distributions.MultivariateNormal(loc=self.means, covariance_matrix=cov_mats) 
        
        # Expand x to (B, 1, D) so that broadcasting computes a log probability
        # for each mixture component.
        x_expanded = x.unsqueeze(1)  # (B, 1, D)
        # Compute log-likelihoods for each mixture component for every x.
        # The resulting shape will be (B, N)
        log_gauss = mvns.log_prob(x)
        
        # Compute the weighted log densities for each mixture component.
        log_densities = log_gauss + log_mix_weights  # (B, N)
        
        # Use the log-sum-exp trick to normalize per batch element.
        max_log, _ = torch.max(log_densities, dim=1, keepdim=True)  # (B, 1)
        exp_shifted = torch.exp(log_densities - max_log)  # (B, N)
        attention = exp_shifted / exp_shifted.sum(dim=1, keepdim=True)  # (B, N)
        
        return attention

class NonLinearGradient:
    def flat(self, x) : return rearrange(x, "... c h w -> ... (c h w)")
    def unflat(self, x) : return rearrange(x, "... (c h w) -> ... c h w", c=self.template.shape[-3], h=self.template.shape[-2], w=self.template.shape[-1])

    def __init__(self, template, v_0, decay_rate, latent, latent_inv):
        self.template = template
        self.v_0 = v_0
        self.device = template.device
        self.dtype = template.dtype
        self.decay_rate = decay_rate
        self.latent = latent
        self.latent_inv = latent_inv

        self.features_template = latent(template)

    def __call__(self, x, t):
        features = self.latent(x)
        score_latent = torch.sigmoid(self.decay_rate * (t - self.v_0)) * (self.features_template - features) / t
        _, score = jvp(self.latent_inv, features, score_latent, strict=True)
        return score


class LinearLatentGradient:
    def flat(self, x) : return rearrange(x, "... c h w -> ... (c h w)")
    def unflat(self, x) : return rearrange(x, "... (c h w) -> ... c h w", c=self.template.shape[-3], h=self.template.shape[-2], w=self.template.shape[-1])

    def __init__(self, projectors, template, v_0, decay_rate):
        self.projectors = projectors
        self.template = template
        self.features_template = torch.einsum("nFD, D -> nF", self.projectors, self.flat(self.template))
        self.v_0 = v_0
        self.inv_projectors = torch.linalg.pinv(self.projectors)
        self.device = template.device
        self.dtype = template.dtype
        self.decay_rate = decay_rate
        self.n_projectors = projectors.shape[0]

        self.attention = AttentionMixture(
            means = self.features_template,
            stds = torch.ones(self.n_projectors, device=self.device) * self.v_0,
            mix_weights = torch.ones(self.n_projectors, device=self.device) / self.n_projectors, # uniform weighting of components
        )

    def __call__(self, x, t):
        features = torch.einsum("nFD, bD -> bnF", self.projectors, self.flat(x) )
        diff_features = features - self.features_template[None, :, :] # (bnd)
        diffs_projected = torch.einsum("nDF, bnF -> bnD", self.inv_projectors, diff_features) 

        # weights = torch.ones((x.shape[0], self.n_projectors), dtype=self.dtype, device=self.device)/self.n_projectors
        weights = self.attention(features,t).to(self.dtype) # -> (n)
        diff_projected = torch.einsum("bn, bnD -> bD", weights, diffs_projected)
        dxs = torch.sigmoid(self.decay_rate * (t - self.v_0)) * diff_projected/t
        dx = self.unflat(dxs[0])
        return dx


# TODO: general input of feautre parameters
def construct_vector_field_template(template, v_0, decay_rate, device, **kwargs):
    if False: 
        dim_data = template.shape[-1] * template.shape[-2] * template.shape[-3] 
        projectors = torch.randn((n_projectors, dim_projector, dim_data), device=device, dtype=template.dtype)
        vector_field_template = LinearLatentGradient(
            projectors = projectors,
            template = template[0],
            v_0 = v_0,
            decay_rate = decay_rate,
        )

        description = f"""
        vf_type         = Linear 
        n_templates     = {template.shape[0]}
        n_projectors    = {n_projectors}
        dim_data        = {dim_data} 
        dim_projector   = {dim_projector} 
        """
    else:
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae", use_safetensors=True)
        vae = vae.to(device=template.device, dtype=template.dtype)
        
        print(template.shape)
        vector_field_template = NonLinearGradient(
            template = template,
            v_0 = v_0,
            decay_rate = decay_rate,
            latent = lambda x : vae.encode(x).latent_dist.sample(),
            latent_inv = lambda x: vae.decode(x).sample
        )

        description = f"""
        vf_type         = NonLinear
        n_templates     = {template.shape[0]}
        """

    print(description)

    return vector_field_template
    


def run_diffusion_for_schedule(
    network_pkl,
    seed            = 0,
    device          = torch.device('cuda'),
    grid_h          = 3,
    grid_w          = 3,
    num_steps       = 32,
    **sched_kwargs,
):

    torch.manual_seed(seed)

    # Load network
    print(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl) as f:
        net = pickle.load(f)['ema'].to(device)

    # Convert template to torch
    sched_kwargs["template"]  = [(template.to(device).to(torch.float64) - 128) / 127.5 for template in sched_kwargs["template"]]  #IF YOU DON't DO torch.float64 you get numerical instability
    sched_keys = sched_kwargs.keys()
    sched_values = sched_kwargs.values()
    sched_shape = np.array([1 if (isinstance(vals, float) or isinstance(vals, int)) else len(vals)
                         for vals in sched_values], dtype=int)
    sched_shape_no_ones = sched_shape[np.where(sched_shape != 1)] 

    # Initialize an empty array of shape [conditons !=] x [t, B, 3, H, W]
    raw_data = np.empty(
        (*sched_shape_no_ones , num_steps, grid_h * grid_w, net.img_channels, net.img_resolution, net.img_resolution)
    )


    # Iterate over all combinations of scheduling parameters.
    for idx, idx_no_ones in tqdm.tqdm(zip(np.ndindex(*sched_shape), np.ndindex(*sched_shape_no_ones)), unit="scheduler", position=0):
        # Create dict for current schedule
        current_sched = {k: (vals if (isinstance(vals, float) or isinstance(vals, int))  else vals[i])
                         for k, vals, i in zip(sched_keys, sched_values, idx)}

        # Construct the template-derived vector field
        if current_sched['scale_template_score'] == 0: 
            vf_template = lambda x, t: torch.zeros_like(x)
        else:
            vf_template = construct_vector_field_template(device=device, **current_sched)

        # Run Diffusion Process
        xs, t_steps, = generate_image_grid(net, vf_template, 
                                           seed=seed, device=device, grid_h=grid_h, grid_w=grid_w, num_steps=num_steps, # if you dont specify these will use default
                                           **current_sched) 
        raw_data[idx_no_ones] = (xs * 127.5 + 128) / 255

    data_dict = {
        "sched_kwargs": sched_kwargs,
        "raw_data": raw_data,
    }
    return data_dict


