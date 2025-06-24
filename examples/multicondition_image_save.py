import numpy as np
import numpy as np
import itertools
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os 
import torch
import pickle
from einops import rearrange, repeat
from dataclasses import replace
from mylib.diffusion import edm_sampler, ConfigSimulation, ConfigSampler, ConfigGuidanceVF, load_templates_batch, create_vf, schedule_diffusion
from mylib.visual import visualize_from_path
from training.networks import EDMPrecond
from torch_utils import misc
import math 
import shutil
import dnnlib
from PIL import Image

MODEL_ROOT = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained'

VF_PIXEL= ConfigGuidanceVF(
        type_latent = "pixel",
        decay_rate = 1.0,
        v_0 = [40, 20, 10 , 5],
        scale = 0.1,
        template_path = "data/data/cat_1.jpg",
        )

VF_PIXEL_SCALE_AND_V0 = ConfigGuidanceVF(
        type_latent = "pixel",
        decay_rate = 1.0,
        v_0 = [15, 30],
        scale = [0.1, 0.5],
        template_path = "data/input/cat_1.jpg",
        )

VF_VAE_JVP = ConfigGuidanceVF(
        type_latent = "hf",
        type_eval = "jvp",
        hf_url = "stabilityai/sd-turbo",
        v_0 = [10, 20, 40],
        template_path = "data/data/cat_1.jpg",
        threshold_weight = 0.5,
        )

VF_VAE_NUMDIFF = VF_VAE_JVP(type_eval="numdiff")

VF_LINEAR = ConfigGuidanceVF(
        type_latent = "linear",
        decay_rate = 1.0,
        v_0 = [45, 30, 25, 20, 15],
        scale = 1.0,
        template_path = "data/input/",
        seed_mat = 0,
        n_features = 5,
        dim_feature = 32,
        T = 1.0,
        )

VF_LINEAR_HF = ConfigGuidanceVF(
        type_latent = "hf-linear",
        type_eval = "numdiff",
        hf_url = "stabilityai/sd-turbo",
        decay_rate = 1.0,
        v_0 = [45, 30, 15],
        scale = 1.0,
        template_path = "data/input/",
        seed_mat = 0,
        n_features = 32,
        dim_feature = 64,
        T = 1.0,
        
VF_UNET = ConfigGuidanceVF(
        type_latent = "unet",
        type_eval = "numdiff",
        template_path = "data/data/cat_1.jpg",
        idx_skips = [(i, ) for i in range(12,16)],
            vf_latent = ConfigGuidanceVF(
                type_latent = "pixel",
                scale = np.power(2, [1, 2, 3, 4, 5]).tolist(),
                v_0 = 30,
            )
        )

if __name__ == "__main__":
    cnfgs = ConfigSimulation(
        network_pkl   = f'{MODEL_ROOT}/edm-imagenet-64x64-cond-adm.pkl',
        device        = "cuda" if torch.cuda.is_available() else "cpu",
        dtype         = torch.float16,
        seed          = 0,
        input_shape   = (3, 64, 64),
        guidance_vf   = VF_UNET,
        diffusion     = ConfigSampler(
            num_steps=24, 
            class_idx=281,
            batch_size=16,
        ),
    )

    # Create network
    with dnnlib.util.open_url(cnfgs.network_pkl) as f:
        net_old = pickle.load(f)['ema'].to(cnfgs.device)
    net = EDMPrecond(*net_old.init_args, **net_old.init_kwargs).to(cnfgs.device)
    net.model.save_skips = True
    net.eval()
    misc.copy_params_and_buffers(net_old, net, require_all=True)

    for cnfg in cnfgs.split():
        # Create guidance vectorfield
        templates = load_templates_batch([cnfg.guidance_vf.template_path] * cnfg.diffusion.batch_size, device=cnfg.device, dtype=torch.float16)
        vf = create_vf(cnfg.guidance_vf, templates, net=net, cnfg_sim=cnfg)

        # Create latents 
        g = torch.Generator(device=cnfg.device).manual_seed(cnfg.seed)
        latents = torch.randn([cnfg.diffusion.batch_size, net.img_channels, net.img_resolution, net.img_resolution], 
                              device=cnfg.device,
                              dtype=cnfg.dtype,
                              generator=g
                            )

        # Run sampler
        with torch.no_grad():
            xs, _ = edm_sampler(
                    net, 
                    vf, 
                    seed=cnfg.seed, 
                    device=cnfg.device, 
                    dtype=templates.dtype, 
                    **cnfg.diffusion.to_dict(),
                    latents=latents, 
                )

            idx_skips = cnfg.guidance_vf.idx_skips[0]
            v_0 = cnfg.guidance_vf.vf_latent.v_0
            scale = cnfg.guidance_vf.vf_latent.scale

            dirname = f"data/parameter_evaluation/skips_{idx_skips}_scale_{scale}_v0_{v_0}"
            os.makedirs(dirname, exist_ok=True)
            for i, arr in enumerate(rearrange(xs[-1], "B C H W -> B H W C")):
                arr = arr.detach().cpu().numpy().clip(0, 1)  
                arr = (arr * 255).astype(np.uint8)          
                img = Image.fromarray(arr)
                img.save(dirname + f"/{i}.png")
                        
