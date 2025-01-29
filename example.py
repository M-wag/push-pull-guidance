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

from einops import rearrange, repeat

import matplotlib.pyplot as plt # should be in seperate function
from matplotlib.offsetbox import AnnotationBbox, OffsetImage # should be in seperate function

from torchvision.io import read_image # should be removed given we install torchvision just for this
#----------------------------------------------------------------------------


def generate_image_grid(
    network_pkl, dest_path,
    seed=0, gridw=2, gridh=2, device=torch.device('cuda'),
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    v_0=1.0, guide_template=0, guide_trained=1.0, second_order=True, 
    save_all_timesteps=False,
):
    batch_size = gridw * gridh
    torch.manual_seed(seed)

    # Load network.
    print(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl) as f:
        net = pickle.load(f)['ema'].to(device)

    # Import Template
    img = read_image("cat.jpg")
    x_template = repeat(img, "c h w -> repeat c h w", repeat=batch_size)
    x_template = x_template.to(device).to(torch.float32) / 255

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
    if save_all_timesteps:
        # Intialize empty array to save intermediate timestaps
        mags_template = torch.empty(num_steps)
        mags_model = torch.empty(num_steps)
        xs = torch.empty((num_steps, batch_size, net.img_channels, net.img_resolution, net.img_resolution))

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    print(x_next.shape)
    for i, (t_cur, t_next) in tqdm.tqdm(list(enumerate(zip(t_steps[:-1], t_steps[1:]))), unit='step'): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)
        
        # Euler step.
        denoised = guide_trained * net(x_hat, t_hat, class_labels).to(torch.float64) if (guide_trained != 0.0) else torch.zeros_like(x_hat)
        model_score =  (x_hat - denoised ) / t_hat
        # template_score =  guide_template * (x_template - x_hat)/(t_hat**2 + v_0)
        template_score =  1.0 * (x_template - x_hat)
        d_cur = model_score + template_score
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1 and second_order:
            denoised = guide_trained * net(x_next, t_next, class_labels).to(torch.float64) if (guide_trained != 0.0) else torch.zeros_like(x_hat)
            d_prime = (x_next - denoised - t_next * guide_template*(x_template - x_next)) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        # Save intermediate timsteps
        if save_all_timesteps:
            xs[i] = x_next
            mags_template[i] = torch.norm(template_score * (t_next - t_hat)) 
            mags_model[i] = torch.norm(model_score * (t_next - t_hat)) 

    # todo: should add way to specify image output
    # Save image grid.
    print(f'Saving image grid to "{dest_path}"...')
    image = (x_next * 127.5 + 128).to(torch.uint8)
    image = rearrange(image, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=gridh)
    image = image.cpu().numpy()

    # image_over_t = (xs * 127.5 + 128).to(torch.uint8)
    image_over_t = rearrange(xs.cpu().numpy() , "t (b1 b2) c h w -> (t b1 h) (b2 w) c", b1=gridh)
    PIL.Image.fromarray(image_over_t, 'RGB').save(dest_path)

    plt.imshow(image_over_t)
    plt.show()


    print('Done.')

#----------------------------------------------------------------------------

def main():
    model_root = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained'
    second_order = False
    num_steps = 32

    # for guide_template in np.power(10, np.linspace(-2, 0, 20)):
    for guide_template in [1.0]:
        v_0 = 1.0
        
        fname = f'imgs/imgnet-numsteps_{num_steps}-v0_{v_0:.2f}-gtmp_{guide_template:.6f}-secondorder_{second_order}.png'
        generate_image_grid(f'{model_root}/edm-imagenet-64x64-cond-adm.pkl', fname,
                            seed=0,  guide_template=guide_template, guide_trained=1.0-guide_template, v_0=v_0,
                            num_steps=num_steps, second_order=second_order,
                            S_churn=0, S_min=0.05, S_max=50, S_noise=1.003,  # default S_churn=40, S_churn=0 turns off adding noise
                            gridw=2, gridh=2, save_all_timesteps=True,
                        ) 


#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------

def plot_magnitude():
    fig, ax  = plt.subplots(1, 1)
    ax.scatter(list(range(0, num_steps)), np.log10(mags_template.cpu()), label="Template")
    ax.scatter(list(range(0, num_steps)), np.log10(mags_model.cpu()), label="Model")
    ax.legend()
    ax.grid(True)


    xs = rearrange(xs, "t (b1 b2) c h w -> t (b1 h) ( b2 w) c", b1=gridh)
    min_y, max_y = ax.get_ylim()
    ax.set_ylim(min_y - 2, max_y)
    for i in range(num_steps):
        im = OffsetImage(xs[i], zoom=1/2)
        im.image.axes = ax
        ab = AnnotationBbox(im, (i, min_y - 1), frameon=False)
        ax.add_artist(ab)

