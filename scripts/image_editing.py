import numpy as np
import os 
import torch
import pickle
from einops import rearrange, repeat
from mylib.diffusion import edm_sampler, ConfigSimulation, ConfigSampler, load_templates_batch
from mylib.gvf import create_vf, ConfigGVFUnet,ConfigGVFUnetAttention, ConfigGVFHuggingFace, ConfigGVFAmbient, ConfigGVFLinear
from training.networks import EDMPrecond
from torch_utils import misc
import math 
import shutil
import dnnlib
from PIL import Image, ImageDraw, ImageFont


MODEL_ROOT = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained'

VF_AMBIENT = ConfigGVFAmbient(
    # scale = [0.0, 0.25, 0.5, 1.0],
    scale = 0.0,
    v_0 = 15,
)

VF_UNET = ConfigGVFUnet(
    type_eval = "numdiff",
    template_path = "data/data/cat_1.jpg",
    idx_skips = tuple(range(4, 16)),
    vf_latent = VF_AMBIENT,
    step_size = [0.3, 0.25, 0.2, 0.15, 0.1, 0.0],
)

VF_UNET_ATTENTION = ConfigGVFUnetAttention(
    type_eval = "numdiff",
    template_path = "data/data/cat_1.jpg",
    idxs = range(6, 9),
    vf_latent = VF_AMBIENT,
    step_size = [0.8, 0.6, 0.4, 0.2],

)


VF_HF = ConfigGVFHuggingFace(
    type_eval = "numdiff",
    template_path = "data/data/cat_1.jpg",
    hf_url = "stabilityai/sd-turbo",
    vf_latent = VF_AMBIENT,
    step_size = [1e-0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
)


def draw_labels(img: Image, p1, p2):
    # Original image dimensions
    H, W = 64, 64

    # Row and columns sizes
    n1 = len(p1)
    n2 = len(p2)

    # Font setup
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()  # Fallback to default

    # Margin sizes
    top_margin = 50  # Space for column labels
    left_margin = 80  # Space for row labels
    right_margin = 10
    bottom_margin = 10

    # Create base image
    grid_width, grid_height = img.size
    
    # Create new image with label space
    total_width = left_margin + grid_width + right_margin
    total_height = top_margin + grid_height + bottom_margin
    new_img = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    new_img.paste(img, (left_margin, top_margin))
    
    # Prepare drawing context
    draw = ImageDraw.Draw(new_img)
    
    # Draw row labels (left axis)
    for row_idx in range(n1):
        y_center = top_margin + row_idx * H + H // 2
        text = f"scale : \n {p1[row_idx]:.4f}"  # Format as needed
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        draw.text(
            (left_margin//2 - text_width//2, y_center - text_height//2),
            text, font=font, fill='black'
        )
    
    # Draw column labels (top axis)
    for col_idx in range(n2):
        x_center = left_margin + col_idx * W + W // 2
        text = f"step_size : \n {p2[col_idx]:.4f}"  # Format as needed
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        draw.text(
            (x_center - text_width//2, top_margin//2),
            text, font=font, fill='black'
        )
    
    return new_img

if __name__ == "__main__":
    cnfgs = ConfigSimulation(
        network_pkl   = f'{MODEL_ROOT}/edm-imagenet-64x64-cond-adm.pkl',
        device        = "cuda" if torch.cuda.is_available() else "cpu",
        dtype         = torch.float32,
        seed          = 1,
        input_shape   = (3, 64, 64),
        guidance_vf   = VF_UNET,
        diffusion     = ConfigSampler(
            num_steps=16, 
            class_idx=281,
            batch_size=9,
        ),
    )

    # Create network
    with dnnlib.util.open_url(cnfgs.network_pkl) as f:
        net_old = pickle.load(f)['ema'].to(cnfgs.device)
    net = EDMPrecond(*net_old.init_args, **net_old.init_kwargs).to(cnfgs.device)
    net.model.save_skips = True
    net.eval()
    misc.copy_params_and_buffers(net_old, net, require_all=True)

    results = []
    for cnfg in cnfgs.split():
        # Create guidance vectorfield
        templates = load_templates_batch([cnfg.guidance_vf.template_path] * cnfg.diffusion.batch_size, device=cnfg.device, dtype=cnfg.dtype)
        sigma = torch.tensor(1e-1).to(dtype=cnfg.dtype, device=cnfg.device)
        net(templates, sigma) # set skips
        vf = create_vf(cnfg.guidance_vf, templates, net=net, device=cnfg.device, dtype=cnfg.dtype)

        # Create latents 
        g = torch.Generator(device=cnfg.device).manual_seed(cnfg.seed)
        latents = torch.randn([cnfg.diffusion.batch_size, net.img_channels, net.img_resolution, net.img_resolution], 
                              device=cnfg.device, dtype=cnfg.dtype, generator=g)

        # Track image

        # Run sampler
        xs, (time_steps, stats) = edm_sampler( net, vf, 
                seed=cnfg.seed, device=cnfg.device, dtype=templates.dtype, **cnfg.diffusion.to_dict(), latents=latents,)
        results.append(xs[-1])


    imgs = rearrange(np.stack(results), "(n1 n2) B C H W -> B (n1 H) (n2 W) C", n2=len(cnfgs.guidance_vf.step_size))

    for i, img in enumerate(imgs):
        img = img.clip(0, 1)  
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)
        img = draw_labels(img, cnfgs.guidance_vf.vf_latent.scale, cnfgs.guidance_vf.step_size)
        img.save(os.path.join("data/parameter_evaluation/", f"{i}.png"))


