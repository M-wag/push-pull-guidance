import os
import numpy as np
import matplotlib.pyplot as plt
from mylib.diffusion import load_templates
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.gridspec import GridSpec
from einops import rearrange, repeat
import pickle


def create_figure(batch_size, n_conditions, img_shape, base_tile_size=1):
    """Create figure with properly scaled subplots"""
    # Calculate dimensions
    tile_width = base_tile_size * img_shape[1] / max(img_shape)  # Normalize by image aspect ratio
    tile_height = base_tile_size * img_shape[0] / max(img_shape)
    
    # Total figure size calculation
    fig_width = ((n_conditions + 2) * tile_width) 
    fig_height = max(batch_size, 1) * tile_height  # Height determined by middle plot
    
    # Create figure with 3 subplots using GridSpec
    # fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    fig = plt.figure()
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, n_conditions, 1],
                  wspace=0.05, hspace=0)
    
    return fig, gs


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

def plot_condition(ax, data, grid_shape, labels=None):
    n_rows, n_cols = grid_shape
    ax.set_axis_off()
    sub_gs = ax.get_subplotspec().subgridspec(
        n_rows, n_cols, wspace=0, hspace=0,
        width_ratios=[1]*n_cols, height_ratios=[1]*n_rows,
    )

    for i in range(n_rows):
        for j in range(n_cols):
            img = data[i, j]

            # create & attach the subplot
            sub_ax = plt.Subplot(ax.figure, sub_gs[i, j])
            ax.figure.add_subplot(sub_ax)

            # show image, remove axes
            sub_ax.imshow(img)
            sub_ax.set_axis_off()
            sub_ax.margins(0, 0)

            # set title only on first row
            if labels is not None and i == 0:
                sub_ax.set_title(labels[j], pad=2)

def plot_comparison(data_dict, img_shape, labels=None):
    """Main plotting function for comparison visualization"""
    batch_size = data_dict['middle'].shape[0]
    n_conditions = data_dict['middle'].shape[1]
    
    # Create figure and a 1×3 GridSpec for left/middle/right
    fig, gs = create_figure(batch_size, n_conditions, img_shape)
    
    axes = {
        name: fig.add_subplot(sub_gs)
        for name, sub_gs in zip(
            ['left','middle','right'], 
            [gs[0], gs[1], gs[2]]
        )
    }

    plot_condition(axes['left'],   data_dict['left'],   (batch_size, 1))
    plot_condition(axes['middle'], data_dict['middle'], (batch_size, n_conditions), labels)
    plot_condition(axes['right'],  data_dict['right'],  (batch_size, 1))
    
    return fig

def visualize_from_path(path_exp, title=None, labels=None):
    """
    Load data and config from a given experiment path, then call plot_comparison.

    Args:
        path_exp (str): Path to the experiment directory containing raw_data.pkl,
                        raw_data_og.pkl, and cnfg_sim.pkl.

    Returns:
        matplotlib.figure.Figure: The resulting comparison figure.
    """
    # Build file paths
    raw_data_path    = os.path.join(path_exp, "raw_data.pkl")
    raw_data_og_path = os.path.join(path_exp, "raw_data_og.pkl")
    cnfg_sim_path    = os.path.join(path_exp, "cnfg_sim.pkl")

    # Load data
    with open(raw_data_path, "rb") as f:
        raw_data = pickle.load(f)
    with open(cnfg_sim_path, "rb") as f:
        cnfg_sim = pickle.load(f)
    with open(raw_data_og_path, "rb") as f:
        raw_data_og = pickle.load(f)

    assert len(raw_data.shape) == 6, \
        f"raw_data should have rank 6, got shape: {raw_data.shape}"

    # Extract shapes
    batch_size = cnfg_sim.diffusion.batch_size
    image_shape = cnfg_sim.input_shape

    # Determine shape combination
    if not cnfg_sim.shape_combination:
        shape_comb = (1, 1)
    elif len(cnfg_sim.shape_combination) == 1:
        shape_comb = tuple(cnfg_sim.shape_combination) + (1,)
    else:
        shape_comb = tuple(cnfg_sim.shape_combination)

    # Indices for last timestep and full batch
    idx_combinations = (slice(None), slice(None))
    idx_time = (-1,)
    idx_batch = (slice(None),)

    # Reshape and select data
    data = raw_data.reshape(shape_comb + raw_data.shape[1:])
    data = data[idx_combinations + idx_time + idx_batch]

    data_og = raw_data_og.reshape((1,1) + raw_data.shape[1:])
    data_og = data_og[(slice(None), slice(None)) + idx_time + idx_batch]
    data_og = rearrange(data_og, "p1 p2 b h w c -> (b p1) p2 1 h w c")

    # Load template images
    template = rearrange(load_templates(cnfg_sim, for_torch=False), "n c h w -> 1 c h (n w)")

    # Prepare dictionary for plotting
    data_dict = {
        'left':   repeat(data_og, "p1 p2 b c h w -> (b p1 p2) 1 h w c"),
        'middle': repeat(data,    "p1 p2 b c h w -> b (p1 p2) h w c"),
        'right':  repeat(template, "1 c h w -> b 1 h w c", b=batch_size)
    }

    # Create and return figure
    fig = plot_comparison(data_dict, image_shape, labels=labels)
    if title:
        fig.suptitle(title)
    else:
        fig.suptitle(os.path.basename(path_exp))
    return fig

