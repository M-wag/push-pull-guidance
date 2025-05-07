import numpy as np
import itertools
import matplotlib.pyplot as plt
import os 
import torch
import pickle
from einops import rearrange
from dataclasses import replace
from mylib.diffusion import schedule_diffusion, ConfigSimulation, ConfigDiffusion, ConfigGuidanceVF
import math 

MODEL_ROOT = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained'

VF_PIXEL_SCALE_AND_V0 = ConfigGuidanceVF(
        type_latent = "pixel",
        decay_rate = 1.0,
        v_0 = [15, 30],
        scale_template_score = [0.0, 0.1],
        template_path = "data/input/cat.jpg",
        )

VF_VAE_JVP = ConfigGuidanceVF(
        type_latent = "hf",
        type_eval = "jvp",
        hf_url = "stabilityai/sd-turbo",
        decay_rate = 1.0,
        v_0 = [15, 30],
        scale_template_score = [0.5, 1.0],
        template_path = "data/input/cat.jpg",
        )

VF_VAE_JVP = ConfigGuidanceVF(
        type_latent = "hf",
        type_eval = "numdiff",
        hf_url = "stabilityai/sd-turbo",
        decay_rate = 1.0,
        v_0 = [15, 30],
        scale_template_score = [0.5, 1.0],
        template_path = "data/input/cat.jpg",
        )

def plot_each_condition(data, batch_size):
    n_cols = math.ceil(math.sqrt(batch_size))
    n_rows = math.ceil(batch_size / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False)
    for idx, (row, col) in enumerate(np.ndindex((n_rows, n_cols))):
        if idx == batch_size:
            break
        axes[row, col].imshow(data[idx])

    plt.show()

def run_no_guidance(exp_name, cnfg):
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

if __name__ == "__main__":
    # USER DEFINED
    RUN_FRESH = True
    cnfg_sim = ConfigSimulation( 
                network_pkl     = f'{MODEL_ROOT}/edm-imagenet-64x64-cond-adm.pkl', 
                device          = "cuda" if torch.cuda.is_available() else "cpu",
                seed            = 0,
                input_shape     = (3, 64, 64),
                guidance_vf     = VF_PIXEL_SCALE_AND_V0 (threshold_weight=0.1),
                diffusion       = ConfigDiffusion(num_steps=16),
                )

    print(cnfg_sim.diffusion)

    # USER DEFINED
    if RUN_FRESH:
        path_exp =  run_no_guidance("gamma_and_v0", cnfg_sim)
    else:
        path_exp = os.path.join(os.getcwd(), "data", "output", "gamma_and_v0_27")

    raw_data_path = os.path.join(path_exp, "raw_data.pkl")
    cnfg_sim_path = os.path.join(path_exp, "cnfg_sim.pkl")

    with open(raw_data_path, "rb") as f:
        raw_data = pickle.load(f)
    with open(cnfg_sim_path, "rb") as f:
        cnfg_sim = pickle.load(f)
    assert len(raw_data.shape) == 6, f"raw_data should have rank 6, got shape : {raw_data.shape}"

    # Reshape to be able to pick combination
    if len(cnfg_sim.shape_combination) == 0:
        shape_comb = (1, 1)
    elif len(cnfg_sim.shape_combination) == 1:
        shape_comb = cnfg_sim.shape_combination + (1,)
    else :
        shape_comb = cnfg_sim.shape_combination 

    # USER DEFINED
    # Pick combination index, time index and batch index
    idx_combinations = (slice(None), slice(None))
    idx_time = (-1, )
    idx_batch = (slice(None),)

    # Reshape data for visualize 
    data = raw_data.reshape(shape_comb + raw_data.shape[1:])
    data = data[idx_combinations + idx_time + idx_batch]
    assert len(data.shape) == 6, f"data should have rank 5, got shape: {data.shape}"
    data = rearrange(data, "p1 p2 b C H W -> b (p1 H) (p2 W) C", p1=shape_comb[0], p2=shape_comb[1])
    
    plot_each_condition(data, cnfg_sim.diffusion.batch_size)

    

