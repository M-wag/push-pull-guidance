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
import PIL.Image
import visualization as vis
from einops import rearrange, repeat

from torchvision.io import read_image

#----------------------------------------------------------------------------

def generate_image_grid(
    net, template,
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

        # Calculate differentials
        d_template =  capacity_template * torch.sigmoid(decay_rate * (t_hat - v_0)) * (x_hat - x_template)/t_hat
        # d_template =  capacity_template * t_hat / (t_hat**2 + v_0 ** 2 ) * (x_hat - x_template)
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

def run_diffusion_for_schedule(schedule_params):
    """
    Run the diffusion process over an arbitrary number of scheduling parameters.
    
    schedule_params: dict containing keys starting with "sched_". For example:
       {
           "sched_capacity_template": [0.125, 0.0],   # can be more than one value
           "sched_decay_rate": np.linspace(0.5, 1.5, 3),
           "sched_v0": np.linspace(1.0, 80, 5),
           # potentially more keys...
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
    # network_pkl = f'{model_root}/edm-imagenet-64x64-cond-adm.pkl'
    network_pkl = f'{model_root}/edm-afhqv2-64x64-uncond-vp.pkl'
    device = torch.device('cuda')
    print(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl) as f:
        net = pickle.load(f)['ema'].to(device)

    # Load template
    template = read_image("cat.jpg")
    template = (template.to(device).to(torch.float64) - 128) / 127.5 #IF YOU DON't DO torch.float64 here your template reconstruction produces large values

    # Identify all scheduling keys (all keys that start with "sched_")
    sched_keys = [k for k in schedule_params if k.startswith("sched_")]
    sched_values = [schedule_params[k] for k in sched_keys]
    sched_shape = [len(vals) for vals in sched_values]


    # Preallocate raw data container.
    # The resulting shape will be (*sched_shape, num_steps, grid_h*grid_w, channels, H, W)
    raw_data = np.empty(
        (*sched_shape, num_steps, grid_h * grid_w, net.img_channels, net.img_resolution, net.img_resolution)
    )
    
    # Cache for capacity_template==0.0 runs.
    cache = {}

    # Iterate over all combinations of scheduling parameters.
    for idx in tqdm.tqdm(np.ndindex(*sched_shape), unit="scheduler", position=0):
        # Create a dict for the current combination.
        current_sched = {k: vals[i] for k, vals, i in zip(sched_keys, sched_values, idx)}
        # Derive capacity_template from its related schedule.
        capacity_template = current_sched.get("sched_capacity_template", 0.125)
        # Also derive other parameters with defaults if not present.
        decay_rate = current_sched.get("sched_decay_rate", 1.0)
        v_0 = current_sched.get("sched_v0", 1.0)
        
        # If capacity_template==0.0, reuse the cached result (assumed to be independent of decay_rate and v_0).
        if capacity_template == 0.0 and 0.0 in cache:
            xs = cache[0.0]
        else:
            xs, t_steps, used_template = generate_image_grid(
                net, 
                template,
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
            # If capacity_template==0.0, store the result in cache.
            if capacity_template == 0.0:
                cache[0.0] = xs

        # Process and store the result.
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
    fname = "imgs/results_afhq_all.pkl"
    # fname = "imgs/results_imgnet_all.pkl"
    if True:
        # Define multiple schedules by name.
        schedules = {
            "mod": {
                "sched_capacity_template": [ 0.125, 0.25, 0.5],  
                "sched_decay_rate": [1.0],
                "sched_v0": np.linspace(1.0, 80, 20),
            },
            "og": {
                "sched_capacity_template": [0],
                "sched_decay_rate": [0],
                "sched_v0": [0],
            },
        }
        
        all_results = {}
        for name, params in schedules.items():
            print(f"Running schedule: {name}")
            all_results[name] = run_diffusion_for_schedule(params)
        
        # Save aggregated results.
        with open(fname, "wb") as f:
            pickle.dump(all_results, f)
    
    with open(fname, "rb") as f:
        data = pickle.load(f)["mod"]
    with open(fname, "rb") as f:
        og_data = pickle.load(f)["og"]
    
    # Define the ordering of scheduler dimensions (should match generation order)
    scheduler_order = ["sched_capacity_template", "sched_decay_rate", "sched_v0"]
    
    # Transform raw_data to only keep two dimensions, based on user input.
    # For example, if we want to plot sched_decay_rate (rows) and sched_v0 (columns),
    # we fix all others (here, sched_capacity_template) to index 0.
    raw_data_2d = vis.transform_raw_data(data["raw_data"], ["sched_capacity_template", "sched_v0"], scheduler_order)
    # Update data dictionary with the transformed raw_data.
    data["raw_data"] = raw_data_2d
    
    og_raw_data_2d = vis.transform_raw_data(og_data["raw_data"], ["sched_capacity_template", "sched_v0"], scheduler_order)
    og_data["raw_data"] = og_raw_data_2d

    # Now call the plotting functions.
    vis.plot_condition_by_condition(data, "sched_capacity_template", "sched_v0", og_data )
    vis.plot(data)

#----------------------------------------------------------------------------
