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

from einops import rearrange, repeat

import matplotlib.pyplot as plt # should be in seperate function
from mpl_toolkits.axes_grid1 import ImageGrid

from torchvision.io import read_image # should be removed given we install torchvision just for this
#----------------------------------------------------------------------------


def generate_image_grid(
    net, dest_path, template,
    seed=0, gridw=2, gridh=2, device=torch.device('cuda'),
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    v_0=1.0, guide=0,  second_order=True, 
    save_all_timesteps=False,
):
    batch_size = gridw * gridh
    torch.manual_seed(seed)

    # Correct format template
    x_template = repeat(template, "c h w -> repeat c h w", repeat=batch_size)

    # Pick latents and labels.
    print(f'Generating {batch_size} images...')
    latents = torch.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
    class_labels = None
    if net.label_dim:
        class_labels = torch.eye(net.label_dim, device=device)[batch_size * [281]]


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
    for i, (t_cur, t_next) in tqdm.tqdm(list(enumerate(zip(t_steps[:-1], t_steps[1:]))), unit='step'): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

        # Calculate differentials
        d_template = t_hat / (t_hat**2 + v_0) * (x_hat - x_template)
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_model = (x_hat - denoised) / t_hat
        d_cur = (guide * d_template + (1 - guide) * d_model) * (t_next - t_hat)
        x_next = x_hat +  d_cur

        # Save intermediate timsteps
        if save_all_timesteps:
            xs[i] = x_next
            metrics[0, i] = torch.norm(x_next)

    # Save image grid.
    # image = (xs * 127.5 + 128).to(torch.uint8)
    # image = rearrange(image, "t (b1 b2) c  h w -> (t b1 h) (b2 w) c", b1=gridh)
    return xs.numpy()
    print('Done.')

#----------------------------------------------------------------------------

def main():
    # Simulations parameters
    grid_h = 2
    grid_w = 2
    sched_guide = np.linspace(0.00, 0.24, 4)
    sched_v0 = [10e-4, 10e-3,  10e-2, 10e-2 *5]


    # Load network.
    model_root = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained'
    network_pkl = f'{model_root}/edm-imagenet-64x64-cond-adm.pkl'
    device = torch.device('cuda')
    print(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl) as f:
        net = pickle.load(f)['ema'].to(device)

    # Load template
    template = read_image("cat.jpg")
    template = (template.to(device).to(torch.float32) - 128) / 127.5

    # Run diffusion process
    images = []
    for i, guide in enumerate(sched_guide):
        for j, v_0 in enumerate(sched_v0):
            fname = f'imgs/guide={guide:.2f}_v0={v_0:.3f}.png'
            xs = generate_image_grid(
                                net, 
                                fname,
                                template,
                                device=device,
                                seed=0,  
                                guide=guide,
                                v_0=v_0,
                                num_steps=32, 
                                second_order=False,
                                S_churn=0, S_min=0.05, S_max=50, S_noise=1.003,  # default S_churn=40, S_churn=0 turns off adding noise
                                gridw=grid_w, gridh=grid_h, 
                                save_all_timesteps=True,
                            ) 
            images.append(rearrange( (xs[-1]  * 127.5 + 128)/255 , "(b1  b2) c h w -> (b1 h) (b2 w) c", b1=grid_h))

    # Plotting parameters
    num_rows = len(sched_guide)
    num_cols = len(sched_v0)
    fig = plt.figure(figsize=(num_rows * grid_h, num_cols * grid_w))

    grid = ImageGrid(fig, 121, nrows_ncols=(num_rows, num_cols), share_all=True)
    grid2 = ImageGrid(fig, 122, nrows_ncols=(1, 1), share_all=True)

    # Show template image
    grid2[0].imshow(rearrange(template.cpu(), "c h w -> h w c"))
    grid2[0].set_title("Template")

    # Disable x and y label
    for ax in grid:
        ax.set_xticks([])
        ax.set_yticks([])

    # Display y condition
    for i, guide in enumerate(sched_guide):
        grid[i * num_cols].set_ylabel(f"Guide : {guide:.3f}")

    # Display x condition
    for i, v_0 in enumerate(sched_v0):
        grid[i].set_title(f"v_0: {v_0:.3f}")

    # Plot images
    print(images)
    for i, image in enumerate(images):
        grid[i].imshow(image)

    plt.show()



#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#---------------------------------------------------------------------------

