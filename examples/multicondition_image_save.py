import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os 
import torch
import pickle
from einops import rearrange, repeat
from dataclasses import replace
from mylib.diffusion import edm_sampler, ConfigSimulation, ConfigSampler, load_templates_batch
from mylib.gvf import create_vf, ConfigGVFUnet,ConfigGVFUnetAttention, ConfigGVFHuggingFace, ConfigGVFAmbient, ConfigGVFLinear
from mylib.visual import visualize_from_path
from training.networks import EDMPrecond
from torch_utils import misc
import math 
import shutil
import dnnlib
from PIL import Image


MODEL_ROOT = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained'

VF_AMBIENT = ConfigGVFAmbient(
    scale = [0.2, 0.4, 0.6, 0.8],
    v_0 = 30.0,
    # template_path = "data/data/cat_1.jpg",
)

VF_LINEAR = ConfigGVFLinear(
    scale = 1.0,
    v_0 = 15,
    n_features= 2,
    dim_feature= 32,
)

VF_UNET = ConfigGVFUnet(
    type_eval = "numdiff",
    template_path = "data/data/cat_1.jpg",
    idx_skips = tuple(range(4, 8)),
    vf_latent = VF_AMBIENT,
)

VF_UNET_ATTENTION = ConfigGVFUnetAttention(
    type_eval = "numdiff",
    template_path = "data/data/cat_1.jpg",
    idxs = range(6, 9),
    vf_latent = VF_AMBIENT
)

VF_HF = ConfigGVFHuggingFace(
    type_eval = "numdiff",
    template_path = "data/data/cat_1.jpg",
    hf_url = "stabilityai/sd-turbo",
    vf_latent = VF_AMBIENT,
)



if __name__ == "__main__":
    cnfgs = ConfigSimulation(
        network_pkl   = f'{MODEL_ROOT}/edm-imagenet-64x64-cond-adm.pkl',
        device        = "cuda" if torch.cuda.is_available() else "cpu",
        dtype         = torch.float32,
        seed          = 1,
        input_shape   = (3, 64, 64),
        # TODO check that guidance_vf has template_path
        guidance_vf   =  [VF_AMBIENT(template_path="data/data/cat_1.jpg"), VF_HF, VF_UNET, VF_UNET_ATTENTION, VF_UNET(step_size=0.2), VF_UNET_ATTENTION(step_size=0.2)],
        diffusion     = ConfigSampler(
            num_steps=32, 
            class_idx=281,
            batch_size=1,
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
        templates = load_templates_batch([cnfg.guidance_vf.template_path] * cnfg.diffusion.batch_size, device=cnfg.device, dtype=cnfg.dtype)
        sigma = torch.tensor(1e-1).to(dtype=cnfg.dtype, device=cnfg.device)
        net(templates, sigma) # set skips
        vf = create_vf(cnfg.guidance_vf, templates, net=net, device=cnfg.device, dtype=cnfg.dtype)

        # Create latents 
        g = torch.Generator(device=cnfg.device).manual_seed(cnfg.seed)
        latents = torch.randn([cnfg.diffusion.batch_size, net.img_channels, net.img_resolution, net.img_resolution], 
                              device=cnfg.device, dtype=cnfg.dtype, generator=g)

        # Run sampler
        with torch.no_grad():
            xs, (time_steps, stats) = edm_sampler(
                    net, 
                    vf, 
                    seed=cnfg.seed, 
                    device=cnfg.device, 
                    dtype=templates.dtype, 
                    **cnfg.diffusion.to_dict(),
                    latents=latents, 
                )

            if type(cnfg.guidance_vf) is ConfigGVFAmbient:
                v_0 = cnfg.guidance_vf.v_0 
                scale = cnfg.guidance_vf.scale
            else:
                v_0 = cnfg.guidance_vf.vf_latent.v_0 
                scale = cnfg.guidance_vf.vf_latent.scale

            dirname = f"data/parameter_evaluation/scale_{scale}_v0_{v_0}"
            if False:
                if isinstance(cnfg.guidance_vf.vf_latent, ConfigGVFLinear):
                    dim_feature = cnfg.guidance_vf.vf_latent.dim_feature
                    dirname = f"data/parameter_evaluation/skips_{str(idx_skips)}_scale_{scale}_v0_{v_0}_dimfeature_{dim_feature}"
                elif isinstance(cnfg.guidance_vf.vf_latent, ConfigGVFUnet):
                    idx_skips = cnfg.guidance_vf.idx_skips
                    dirname = f"data/parameter_evaluation/skips_{str(idx_skips)}_scale_{scale}_v0_{v_0}"
                else:
                    dirname = f"data/parameter_evaluation/scale_{scale}_v0_{v_0}"

            os.makedirs(dirname, exist_ok=True)

            arr = rearrange(xs[-1], "(b1 b2) C H W -> (b1 H) (b2 W) C", b1=int(np.sqrt(cnfg.diffusion.batch_size)))
            arr = arr.detach().cpu().numpy().clip(0, 1)  
            arr = (arr * 255).astype(np.uint8)
            img = Image.fromarray(arr)

            img.save(os.path.dirname(dirname) + f"/batch.png")
            type_to_str = {
                    ConfigGVFAmbient        : "amb",
                    ConfigGVFUnet           : "unet-skip",
                    ConfigGVFUnetAttention  : "unet-attn",
                    ConfigGVFHuggingFace    : "hf",
                    }

            if getattr(cnfg.guidance_vf, "step_size", 1e-3) != 1e-3:
                type_to_str = {
                        ConfigGVFAmbient        : "amb",
                        ConfigGVFUnet           : "unet-skip-step",
                        ConfigGVFUnetAttention  : "unet-attn-step",
                        ConfigGVFHuggingFace    : "hf",
                        }


            stats.grads = None
            np.save(os.path.dirname(dirname) + f"/{scale}_{v_0}_{type_to_str[type(cnfg.guidance_vf)]}.npy", stats)

            for i, arr in enumerate(rearrange(xs[-1], "B C H W -> B H W C")):
                arr = arr.detach().cpu().numpy().clip(0, 1)  
                arr = (arr * 255).astype(np.uint8)          
                img = Image.fromarray(arr)
                # img.save(dirname + f"/{i}.png")
