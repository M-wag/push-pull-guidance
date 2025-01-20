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
import PIL.Image
import dnnlib

from torchvision.io import read_image # should be removed given we install torchvision just for this
#----------------------------------------------------------------------------


def generate_image_grid(
    network_pkl, dest_path,
    seed=0, gridw=2, gridh=2, device=torch.device('cuda'),
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    guide_template=0, guide_trained=1.0, second_order=True,
):
    batch_size = gridw * gridh
    torch.manual_seed(seed)

    # Load network.
    print(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl) as f:
        net = pickle.load(f)['ema'].to(device)

    # Import Template
    img = read_image("cat.jpg")
    x_template = img.to(device).to(torch.float32) 
    x_template = (x_template - torch.mean(x_template)) / torch.std(x_template) #??: Why don't we need to rescale
    x_template = x_template.view(net.img_channels, net.img_resolution, net.img_resolution)
    x_template = x_template.repeat(4, 1, 1, 1)

    # Pick latents and labels.
    print(f'Generating {batch_size} images...')
    latents = torch.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
    class_labels = None
    if net.label_dim:
        class_labels = torch.eye(net.label_dim, device=device)[[281, 281, 281, 281]]


    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in tqdm.tqdm(list(enumerate(zip(t_steps[:-1], t_steps[1:]))), unit='step'): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

        # Euler step.
        denoised = guide_trained * net(x_hat, t_hat, class_labels).to(torch.float64) if (guide_trained != 0.0) else torch.zeros_like(x_hat)
        d_cur = (x_hat - denoised - t_hat * guide_template*(x_template - x_hat)) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur


        # Apply 2nd order correction.
        if i < num_steps - 1 and second_order:
            denoised = guide_trained * net(x_next, t_next, class_labels).to(torch.float64) if (guide_trained != 0.0) else torch.zeros_like(x_hat)
            d_prime = (x_next - denoised - t_next * guide_template*(x_template - x_next)) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)



    # Save image grid.
    print(f'Saving image grid to "{dest_path}"...')
    image = (x_next * 127.5 + 128).clip(0, 255).to(torch.uint8)
    image = (x_next * 127.5 + 128).to(torch.uint8)
    image = image.reshape(gridh, gridw, *image.shape[1:]).permute(0, 3, 1, 4, 2)
    image = image.reshape(gridh * net.img_resolution, gridw * net.img_resolution, net.img_channels)
    image = image.cpu().numpy()
    PIL.Image.fromarray(image, 'RGB').save(dest_path)


    image = (x_next * 127.5 + 128).clip(0, 255).to(torch.uint8)
    image = image.reshape(gridh, gridw, *image.shape[1:]).permute(0, 3, 1, 4, 2)
    image = image.reshape(gridh * net.img_resolution, gridw * net.img_resolution, net.img_channels)
    image = image.cpu().numpy()
    PIL.Image.fromarray(image, 'RGB').save(dest_path.split(".png")[0] + "_clip.png")
    print('Done.')

#----------------------------------------------------------------------------

def main():
    model_root = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained'
    num_steps = 32
    second_order = True

    for guide_template in np.arange(0.0, 0.18, 0.02):
        fname = f'imgs/imgnet-numsteps_{num_steps}-gtmp_{guide_template:.3f}-secondorder_{second_order}.png'
        generate_image_grid(f'{model_root}/edm-imagenet-64x64-cond-adm.pkl', fname,
                            seed=0, num_steps=num_steps, guide_template=guide_template, guide_trained=1.0,
                            S_churn=40, S_min=0.05, S_max=50, S_noise=1.003, second_order=second_order,
                            gridw=2, gridh=2
                        ) 

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
