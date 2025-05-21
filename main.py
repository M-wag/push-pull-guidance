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
from mylib.diffusion import schedule_diffusion, ConfigSimulation, ConfigDiffusion, ConfigGuidanceVF, load_templates, create_vf
from mylib.visual import visualize_from_path
import math 

MODEL_ROOT = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained'

VF_PIXEL= ConfigGuidanceVF(
        type_latent = "pixel",
        decay_rate = 1.0,
        v_0 = [40, 20, 10 , 5],
        scale = 1.0,
        template_path = "data/input/cat.jpg",
        )

VF_PIXEL_SCALE_AND_V0 = ConfigGuidanceVF(
        type_latent = "pixel",
        decay_rate = 1.0,
        v_0 = [15, 30],
        scale = [0.1, 0.5],
        template_path = "data/input/cat.jpg",
        )

VF_VAE_JVP = ConfigGuidanceVF(
        type_latent = "hf",
        type_eval = "jvp",
        hf_url = "stabilityai/sd-turbo",
        decay_rate = 1.0,
        v_0 = [40, 20, 10, 5],
        scale = 1.0,
        template_path = "data/input/cat.jpg",
        )

VF_LINEAR = ConfigGuidanceVF(
        type_latent = "linear",
        decay_rate = 1.0,
        v_0 = [45, 30, 15],
        scale = 1.0,
        template_path = "data/input/cat.jpg",
        seed_mat = 0,
        n_features = 3,
        dim_feature = 64,
        T = 1.0,
        )

VF_VAE_NUMDIFF = VF_VAE_JVP(type_eval="numdiff")

def run(exp_name, cnfg):
    # Set output destination
    path_exp = os.path.join(os.getcwd(), "data", "output", exp_name)
    if os.path.exists(path_exp):
        i = 1
        while True:
            path_exp = os.path.join(os.getcwd(), "data", "output", f"{exp_name}_{i}")
            if not os.path.exists(path_exp):
                break
            i += 1
    os.makedirs(path_exp)
    print(f"Making directory {path_exp}")

    # Pass to scheduler
    raw_data = schedule_diffusion(cnfg)

    # Save result
    raw_data_path = os.path.join(path_exp, "raw_data.pkl")
    cnfg_sim_path = os.path.join(path_exp, "cnfg_sim.pkl")

    with open(raw_data_path, "wb") as f:
        pickle.dump(raw_data, f)

    with open(cnfg_sim_path, "wb") as f:
        pickle.dump(cnfg, f)

    return path_exp

def run_no_guidance(cnfg, path_exp):
    # Pass to scheduler
    raw_data = schedule_diffusion(cnfg(guidance_vf=None))
    # Save result
    raw_data_path = os.path.join(path_exp, "raw_data_og.pkl")

    with open(raw_data_path, "wb") as f:
        pickle.dump(raw_data, f)

    return path_exp


def main(visualize=False):
    exp_name = "none"
    cnfg_sim = ConfigSimulation( 
                network_pkl     = f'{MODEL_ROOT}/edm-imagenet-64x64-cond-adm.pkl', 
                device          = "cuda" if torch.cuda.is_available() else "cpu",
                seed            = 0,
                input_shape     = (3, 64, 64),
                guidance_vf     = VF_PIXEL_SCALE_AND_V0,
                diffusion       = ConfigDiffusion(num_steps=24),
    )

    path_exp = run(exp_name, cnfg_sim)
    run_no_guidance(cnfg_sim, path_exp)

    if visualize:
        visualize_from_path(path_exp, exp_name )
        plt.show()

if __name__ == "__main__":
    cnfg_sim = ConfigSimulation( 
                network_pkl     = f'{MODEL_ROOT}/edm-imagenet-64x64-cond-adm.pkl', 
                device          = "cuda" if torch.cuda.is_available() else "cpu",
                seed            = 0,
                input_shape     = (3, 64, 64),
                # guidance_vf     = VF_LINEAR(template_path="data/input/cat.jpg"),
                guidance_vf     = VF_LINEAR(template_path="data/input/"),
                diffusion       = ConfigDiffusion(num_steps=24),
    )
    templates = load_templates(cnfg_sim)
    print(templates.shape)

    vf = create_vf(cnfg_sim.guidance_vf.split()[0], templates)
    x =  torch.rand(8, 3, 64, 64).to(device=vf.device, dtype=vf.dtype)
    t = torch.tensor(40, device=vf.device, dtype=vf.dtype)
    
    vf(x, t)


