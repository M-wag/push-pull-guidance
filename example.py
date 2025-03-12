# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Minimal standalone example to reproduce the main results from the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import tqdm
import pickle
import numpy as np
import torch
import dnnlib
from PIL import Image
import visualization as vis
from einops import rearrange, repeat
from torchvision.io import read_image


#----------------------------------------------------------------------------

def generate_image_grid(
    net, template, score_template,
    seed=0, gridw=2, gridh=2, device=torch.device('cuda'),
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    v_0=1.0, capacity_template=1.0, decay_rate=1.0,
    scale_model_score=1.0,
    save_all_timesteps=False,
):
    batch_size = gridw * gridh
    torch.manual_seed(seed)

    # Correct format template
    template = torch.tensor(template).to(torch.float64).to(device)
    x_template = repeat(template, "c h w -> repeat c h w", repeat=batch_size)

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
        d_template = score_template(x_hat, t_hat)
        # d_template =  capacity_template * torch.sigmoid(decay_rate * (t_hat - v_0)) * (x_hat - x_template)/t_hat
        # d_template =  capacity_template * t_hat / (t_hat**2 + v_0 ** 2 ) * (x_hat - x_template)

        # model score
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_model = scale_model_score * (x_hat - denoised) / t_hat
        d_cur = (d_template + d_model) * (t_next - t_hat)
        x_next = x_hat + d_cur

        # Save intermediate timsteps
        if save_all_timesteps:
            xs[i] = x_next
            metrics[0, i] = torch.norm(x_next)

    return xs.numpy(), t_steps.cpu().numpy(), x_template.cpu().numpy()
    print('Done.')

#----------------------------------------------------------------------------





class AttentionMixture:
    def __init__(self, means, stds, mix_weights):
        self.means = means
        self.mix_weights = mix_weights
        self.stds = stds 
        self.D = means.size(-1)

    def __call__(self, x, std_noise) -> torch.Tensor :
        # make noised covariance matrix 
        cov_factor = self.stds**2 + std_noise**2  # shape (N,)
        cov_factor = rearrange(cov_factor, 'n -> n 1 1')  # shape (N, 1, 1)
        cov_mats = cov_factor * torch.eye(self.D, device=self.means.device)  # shape (N, D, D)

        # Add small epsilon to avoid log(0)
        log_mix_weights = torch.log(self.mix_weights + 1e-12)

        # Construct batch of multivariate Gaussians
        mvns = torch.distributions.MultivariateNormal(loc=self.means, covariance_matrix=cov_mats)
        # Compute log-likelihoods: shape (N,)
        log_gauss = mvns.log_prob(x)

        # Weighted log densities: log p_i(x) + log w_i
        log_densities = log_mix_weights + log_gauss

        # Log-sum-exp trick for numerical stability
        max_log = torch.max(log_densities)
        exp_shifted = torch.exp(log_densities - max_log)
        attention = exp_shifted / exp_shifted.sum()

        return attention

class LinearLatentGradient:
    def __init__(self, projectors, template, v_0, capacity, decay_rate):
        self.projectors = projectors
        self.template = template
        self.inv_projectors = torch.linalg.pinv(projectors)
        self.templates_feature = projectors @ rearrange(self.template, "c h w -> (c h w)")# (n_proj, n_template)
        self.n_projectors = projectors.shape[0]
        self.device = template.device
        self.dtype = template.dtype
        self.v_0 = torch.ones(self.n_projectors, dtype=self.dtype, device=self.device) * v_0
        self.capacity = capacity
        self.decay_rate = decay_rate

        self.attention = AttentionMixture(
            means = self.templates_feature,
            stds = self.v_0,
            mix_weights = torch.ones(self.n_projectors, device=self.device) / self.n_projectors, # uniform weighting of components
        )

    def flat(self, x) : return rearrange(x, "... c h w -> ... (c h w)")
    def unflat(self, x) : return rearrange(x, "... (c h w) -> ... c h w", c=self.template.shape[-3], h=self.template.shape[-2], w=self.template.shape[-1])

    def __call__(self, x, t):
        features = torch.einsum("nfd, bd -> bnf", self.projectors, self.flat(x))
        # calculation spatial attention and temporal attention
        # weights_spatial = self.attention(features, t) # (b, n_proj)
        weights_spatial = torch.ones((9, self.n_projectors), device=self.device, dtype=self.dtype) / self.n_projectors
        weights_temporal = torch.sigmoid(self.decay_rate * (t - self.v_0)) # (n_proj)
        # get projection of differences
        diff_projected = torch.einsum("ndf, bnf -> bnd", self.inv_projectors, features)
        # scale each delta by its temporal weight and take weighted sum using spatial weights
        dxs = self.capacity * rearrange(weights_temporal, "n -> 1 n 1")  * diff_projected
        dx = self.unflat(torch.einsum("bn,bnd->bd", weights_spatial, dxs))
        return dx



def run_diffusion_for_schedule(schedule_params):
    """
    Run the diffusion process over an arbitrary number of scheduling parameters.
    
    schedule_params: dict containing keys starting with "sched_". For example:
       {
           "sched_capacity_template": [0.125, 0.0],   # can be more than one value
           "sched_decay_rate": np.linspace(0.5, 1.5, 3),
           "sched_v0": np.linspace(1.0, 80, 5),
       }
    
    Returns a data dictionary with:
       - "scheduler_keys": the list of scheduler keys used
       - "raw_data": a NumPy array with shape (*sched_dims, num_steps, grid_h*grid_w, C, H, W)
       - additional metadata.
    """

    # Simulations parameters
    grid_h = 3
    grid_w = 3
    num_steps = 32
    seed=0

    # Load network.
    model_root = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained'
    network_pkl = f'{model_root}/edm-imagenet-64x64-cond-adm.pkl'
    # network_pkl = f'{model_root}/edm-afhqv2-64x64-uncond-vp.pkl'
    device = torch.device('cuda')
    print(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl) as f:
        net = pickle.load(f)['ema'].to(device)

    # Load template
    template = np.array(Image.open("cat.jpg").convert("RGB"))
    template = read_image("cat.jpg")
    template = (template.to(device).to(torch.float64) - 128) / 127.5 #IF YOU DON't DO torch.float64 here your template reconstruction produces large values

    # Identify all scheduling keys (all keys that start with "sched_")
    sched_keys = [k for k in schedule_params if k.startswith("sched_")]
    sched_values = [schedule_params[k] for k in sched_keys]
    sched_shape = [len(vals) for vals in sched_values]


    # (*sched_shape, num_steps, grid_h*grid_w, channels, H, W)
    raw_data = np.empty(
        (*sched_shape, num_steps, grid_h * grid_w, net.img_channels, net.img_resolution, net.img_resolution)
    )
    
    # cache for images with no control signal
    cache = {}


    # Iterate over all combinations of scheduling parameters.
    for idx in tqdm.tqdm(np.ndindex(*sched_shape), unit="scheduler", position=0):
        # create dict for current schedule
        current_sched = {k: vals[i] for k, vals, i in zip(sched_keys, sched_values, idx)}
        # extract values with default
        capacity_template = current_sched.get("sched_capacity_template", None)
        decay_rate = current_sched.get("sched_decay_rate", None)
        v_0 = current_sched.get("sched_v0", None)
        n_projectors = current_sched.get("sched_n_projectors", None)
        dim_projector = current_sched.get("sched_dim_projector", None)

        # setup additive template-derived score 
        dim_data = template.shape[-1] * template.shape[-2] * template.shape[-3] 
        
        projectors = torch.randn((n_projectors, dim_projector, dim_data), device=device, dtype=torch.float64)
        score_template = LinearLatentGradient(
            projectors = projectors,
            template = template,
            v_0 = v_0,
            capacity = capacity_template,
            decay_rate = decay_rate,
        )
        
        # when using no control signal, reuse the cached result 
        if capacity_template == 0.0 and 0.0 in cache:
            xs = cache[0.0]
        else:
            xs, t_steps, used_template = generate_image_grid(
                net, 
                template,
                score_template,
                device=device,
                seed=seed,  
                capacity_template=capacity_template,
                v_0=v_0,
                decay_rate=decay_rate,
                scale_model_score=1.0,
                num_steps=num_steps, 
                S_churn=0, S_min=0.05, S_max=50, S_noise=1.003,  # default S_churn=40, S_churn=0 turns off adding noise
                gridw=grid_w, gridh=grid_h, 
                save_all_timesteps=True,
            )
            # if no control signal is provided, cache result
            if capacity_template == 0.0:
                cache[0.0] = xs

        raw_data[idx] = (xs * 127.5 + 128) / 255

    data_dict = {
        "scheduler_keys": sched_keys,
        "schedule_params": schedule_params,
        "raw_data": raw_data,
        "grid_h": grid_h,
        "grid_w": grid_w,
        "t_steps": t_steps,
        "template": rearrange(used_template[0], "c h w -> h w c"),
    }
    return data_dict





#----------------------------------------------------------------------------

if __name__ == "__main__":

    fname = "imgs/results_feature_imgnet.pkl"
    if True:
        schedules = {
            "mod": {
                "sched_capacity_template": [0.125, 1],
                "sched_v0": np.linspace(15, 18, 1),
                "sched_decay_rate": [1.0],
                "sched_n_projectors": np.linspace(4, 64, 7).astype(int),
                "sched_dim_projector": np.linspace(32, 128, 5).astype(int),
            },
            # "og": {
            #     "sched_capacity_template": [0],
            #     "sched_v0": [0],
            #     "sched_decay_rate": [0],
            # },
        }
        
        all_results = {}
        for name, params in schedules.items():
            all_results[name] = run_diffusion_for_schedule(params)
        
        with open(fname, "wb") as f:
            pickle.dump(all_results, f)


    # with open("imgs/results_afhq_all.pkl", "rb") as f:
    with open(fname, "rb") as f:
        loaded_results = pickle.load(f)
    with open("imgs/results_imgnet_all.pkl", "rb") as f:
        two = pickle.load(f)

    
    loaded_results['og'] = two['og'] 

    data_mod = loaded_results['mod']
    data_og = loaded_results['og']

    # define the ordering of scheduler dimensions (should match generation order)
    scheduler_order = ["sched_capacity_template", "sched_v0", "sched_decay_rate", "sched_n_projectors", "sched_dim_projector"]
    raw_data_2d_mod = vis.transform_raw_data(data_mod["raw_data"], ["sched_n_projectors", "sched_dim_projector"], scheduler_order)
    raw_data_2d_og = vis.transform_raw_data(data_og["raw_data"], ["sched_capacity_template", "sched_v0"], scheduler_order[:3])
    print(raw_data_2d_mod.shape)
    print(raw_data_2d_og.shape)
    
    data_mod["raw_data"] = raw_data_2d_mod
    data_og["raw_data"] = raw_data_2d_og
    
    # visualization 
    vis.plot_condition_by_condition(data_mod, "sched_n_projectors", "sched_dim_projector", data_og)
    vis.plot(data_mod)


#----------------------------------------------------------------------------
