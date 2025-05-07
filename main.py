import numpy as np
import numpy as np
import itertools
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.gridspec import GridSpecFromSubplotSpec
import os 
import torch
import pickle
from einops import rearrange, repeat
from dataclasses import replace
from mylib.diffusion import schedule_diffusion, ConfigSimulation, ConfigDiffusion, ConfigGuidanceVF, load_templates
import math 

MODEL_ROOT = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained'

VF_PIXEL= ConfigGuidanceVF(
        type_latent = "pixel",
        decay_rate = 1.0,
        #v_0 = [60, 45, 30, 15, 5] ,
        v_0 = [60, 45, 30, 15] ,
        scale_template_score = 0.1,
        template_path = "data/input/cat.jpg",
        )

VF_PIXEL_SCALE_AND_V0 = ConfigGuidanceVF(
        type_latent = "pixel",
        decay_rate = 1.0,
        v_0 = [15, 30],
        scale_template_score = [0.1, 0.5],
        template_path = "data/input/cat.jpg",
        )

VF_VAE_JVP = ConfigGuidanceVF(
        type_latent = "hf",
        type_eval = "jvp",
        hf_url = "stabilityai/sd-turbo",
        decay_rate = 1.0,
        v_0 = [15, 30, 45 ],
        scale_template_score = 1.0,
        template_path = "data/input/cat.jpg",
        )

VF_VAE_NUMDIFF = VF_VAE_JVP(type_eval="numdiff")

def plot_two_conditions(data, batch_size, shape_comb):
    # USER DEFINED
    # Pick combination index, time index and batch index
    assert len(data.shape) == 6, f"data should have rank 5, got shape: {data.shape}"
    data = rearrange(data, "p1 p2 b C H W -> b (p1 H) (p2 W) C", p1=shape_comb[0], p2=shape_comb[1])
    
    plot_two_conditions(data, cnfg_sim.diffusion.batch_size)
    n_cols = math.ceil(math.sqrt(batch_size))
    n_rows = math.ceil(batch_size / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False)
    for idx, (row, col) in enumerate(np.ndindex((n_rows, n_cols))):
        if idx == batch_size:
            break
        axes[row, col].imshow(data[idx])

    plt.show()


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

def create_figure(batch_size, n_conditions, img_shape, base_tile_size=1):
    """Create figure with properly scaled subplots"""
    # Calculate dimensions
    tile_width = base_tile_size * img_shape[1] / max(img_shape)  # Normalize by image aspect ratio
    tile_height = base_tile_size * img_shape[0] / max(img_shape)
    
    # Total figure size calculation
    fig_width = ((n_conditions + 2) * tile_width) 
    fig_height = max(batch_size, 1) * tile_height  # Height determined by middle plot
    
    # Create figure with 3 subplots using GridSpec
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, n_conditions, 1],
                  wspace=0.05, hspace=0)
    
    return fig, gs

def plot_condition(ax, data, grid_shape):
    n_rows, n_cols = grid_shape
    ax.set_axis_off()
    sub_gs = ax.get_subplotspec().subgridspec(n_rows, n_cols,
                                              wspace=0, hspace=0,
                                              width_ratios = [1] * n_cols,
                                              height_ratios = [1] * n_rows)

    for i in range(n_rows):
        for j in range(n_cols):
            img = data[i, j]

            sub_ax = plt.Subplot(ax.figure, sub_gs[i, j])
            sub_ax.imshow(img)
            sub_ax.set_axis_off()
            ax.figure.add_subplot(sub_ax)
            sub_ax.margins(0, 0)

def plot_comparison(data_dict, img_shape):
    """Main plotting function for comparison visualization"""
    # Extract data dimensions
    batch_size = data_dict['middle'].shape[0]
    n_conditions = data_dict['middle'].shape[1]
    
    # Create figure and gridspec
    fig, gs = create_figure(batch_size, n_conditions, img_shape)
    
    # Create main axes
    axes = {
        'left': fig.add_subplot(gs[0]),
        'middle': fig.add_subplot(gs[1]),
        'right': fig.add_subplot(gs[2])
    }
    
    # Plot each condition
    plot_condition(axes['left'], data_dict['left'], (batch_size, 1))
    plot_condition(axes['middle'], data_dict['middle'], (batch_size, n_conditions))
    plot_condition(axes['right'], data_dict['right'], (batch_size, 1))
    
    return fig

# Usage example in main block
if __name__ == "__main__":
    # USER DEFINED
    RUN_FRESH = False
    cnfg_sim = ConfigSimulation( 
                network_pkl     = f'{MODEL_ROOT}/edm-imagenet-64x64-cond-adm.pkl', 
                device          = "cuda" if torch.cuda.is_available() else "cpu",
                seed            = 0,
                input_shape     = (3, 64, 64),
                guidance_vf     = VF_PIXEL(threshold_weight=0.1),
                diffusion       = ConfigDiffusion(num_steps=16),
    )


    # USER DEFINED
    if RUN_FRESH:
        path_exp = run("gamma_and_v0", cnfg_sim)
        run_no_guidance(cnfg_sim, path_exp)
    else:
        path_exp = os.path.join(os.getcwd(), "data", "output", "gamma_and_v0_49")


    raw_data_path = os.path.join(path_exp, "raw_data.pkl")
    raw_data_og_path = os.path.join(path_exp, "raw_data_og.pkl")
    cnfg_sim_path = os.path.join(path_exp, "cnfg_sim.pkl")

    with open(raw_data_path, "rb") as f:
        raw_data = pickle.load(f)
    with open(cnfg_sim_path, "rb") as f:
        cnfg_sim = pickle.load(f)
    with open(raw_data_og_path, "rb") as f:
        raw_data_og = pickle.load(f)
    assert len(raw_data.shape) == 6, f"raw_data should have rank 6, got shape : {raw_data.shape}"

    batch_size = cnfg_sim.diffusion.batch_size
    image_shape = cnfg_sim.input_shape

    # Reshape to be able to pick combination
    if len(cnfg_sim.shape_combination) == 0:
        shape_comb = (1, 1)
    elif len(cnfg_sim.shape_combination) == 1:
        shape_comb = cnfg_sim.shape_combination + (1,)
    else :
        shape_comb = cnfg_sim.shape_combination 

    idx_combinations = (slice(None), slice(None))
    idx_time = (-1, )
    idx_batch = (slice(None),)

    # Reshape data for visualize 
    data = raw_data.reshape(shape_comb + raw_data.shape[1:])
    data = data[idx_combinations + idx_time + idx_batch]
    data_og = raw_data_og.reshape((1,1) + raw_data.shape[1:])
    data_og = data_og[(slice(None), slice(None)) + idx_time + idx_batch]
    data_og = rearrange(data_og, "p1 p2 b h w c -> (b p1) p2 1 h w c")
    template = load_templates(cnfg_sim, for_torch=False)

    data_dict = {
        'left' : repeat(data_og, "p1 p2 b c h w -> (b p1 p2) 1 h w c"),
        'middle' : repeat(data, "p1 p2 b c h w -> b (p1 p2) h w c"),
        'right': repeat(template, "1 c h w -> b 1 h w c", b=cnfg_sim.diffusion.batch_size)
    }
    
    # Create and show plot
    fig = plot_comparison(data_dict, image_shape)
    plt.show()
