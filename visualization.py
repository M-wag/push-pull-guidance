import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.gridspec import GridSpec
from einops import rearrange, repeat
import pickle

def transform_raw_data(raw_data, scheduler_keys_to_keep, scheduler_order):
    """
    Given raw_data of shape (sched_1, ..., sched_N, t, B, C, H, W) and an ordered list
    of scheduler keys (scheduler_order), return data of shape (sched_a, sched_b, t, B, C, H, W)
    based on the scheduler_keys_to_keep (a list of two keys). All other scheduler dimensions
    are fixed to index 0.
    """
    # Build a tuple of slices: if the current scheduler dimension's index (based on scheduler_order)
    # is in idx_keep, then slice(None) to keep all values; otherwise fix to index 0.
    idx_keep = [scheduler_order.index(k) for k in scheduler_keys_to_keep]
    slicer = tuple(slice(None) if i in idx_keep else 0 for i in range(len(scheduler_order)))
    transformed = raw_data[slicer]
    return transformed

def plot_condition_by_condition(data, scheduler_key_a, scheduler_key_b, og_data):
    """
    Plots a grid of images using scheduler_key_a (rows) and scheduler_key_b (columns)
    and shows a template and an unmodified (OG) image on the right.
    
    Assumes the data dictionary has:
      - "schedule_params": dict of scheduling parameters.
      - "raw_data": numpy array with shape 
            (n1, n2, ..., num_steps, grid_h*grid_w, C, H, W)
      - "grid_h", "grid_w", "t_steps", "template"
    
    For a 2-D grid, if more than two scheduling parameters exist, the extra dimensions are fixed to index 0.
    """
    sched_params = data["schedule_params"]
    cond_a = sched_params.get(scheduler_key_a, None)
    cond_b = sched_params.get(scheduler_key_b, None)
    
    # raw_data is assumed already transformed to only keep the two dimensions we want.
    raw_data_2d = data["raw_data"]
    num_rows = len(cond_a) if cond_a is not None else raw_data_2d.shape[0]
    num_cols = len(cond_b) if cond_b is not None else raw_data_2d.shape[1]
    t_steps = data["t_steps"]
    num_steps = len(t_steps)

    # Define figure dimensions.
    template_width = 1.5 
    grid_width = 3.0 * num_cols
    grid_height = 3.0 * num_rows
    total_width = grid_width + template_width
    total_height = grid_height 

    fig = plt.figure(figsize=(total_width, total_height))
    # Main gridspec: left column for image grid, right column for template and OG images.
    gs = fig.add_gridspec(2, 2, width_ratios=[grid_width, 3 * template_width], wspace=0.1)
    gs_right = gs[:, 1].subgridspec(2, 1, height_ratios=[1, 3])
    
    # Plot template image.
    ax_template = fig.add_subplot(gs_right[0])
    ax_template.imshow(data["template"], aspect='equal')
    ax_template.set_xticks([]); ax_template.set_yticks([])
    ax_template.set_title("Template Image")

    # Plot unmodified (OG) image.
    ax_og = fig.add_subplot(gs_right[1])
    data_raw_og = og_data["raw_data"]
    og_img = rearrange(data_raw_og[0, 0, -1], "(b1 b2) c h w -> (b1 h) (b2 w) c",
                        b1=data["grid_h"], b2=data["grid_w"])
    ax_og.imshow(og_img, aspect='equal')
    ax_og.set_xticks([]); ax_og.set_yticks([])
    ax_og.set_title("Unmodified Image") 

    # Create grid for final images.
    img_grid = ImageGrid(fig, gs[:, 0], nrows_ncols=(num_rows, num_cols))
    # Use the final time step.
    # raw_data_2d shape: (num_rows, num_cols, t, grid_h*grid_w, C, H, W)
    img_grid_data = rearrange(raw_data_2d[:, :, -1],
                              "... (b1 b2) C H W -> ... (b1 H) (b2 W) C",
                              b1=data["grid_h"], b2=data["grid_w"])
    for i in range(num_rows):
        for j in range(num_cols):
            idx = i * num_cols + j
            img_grid[idx].imshow(img_grid_data[i, j])
            if cond_a is not None:
                img_grid[idx].set_ylabel(f"{scheduler_key_a}: {cond_a[i]:.3f}")
            if cond_b is not None:
                img_grid[idx].set_title(f"{scheduler_key_b}: {cond_b[j]:.3f}")
    plt.show()


def plot(data):
    """
    2-D visualization using two scheduling parameters.
    If extra scheduling dimensions exist, fix the others to index 0.
    
    Expects data dictionary with "schedule_params" and "raw_data".
    For this example, we assume that the transformed raw_data was built by keeping, say,
    the last scheduling dimension from the order.
    """
    sched_params = data["schedule_params"]
    # For this example, let's assume we use "sched_capacity_template" and "sched_v0"
    cond_a = sched_params.get("sched_capacity_template", None)
    sched_v0 = sched_params.get("sched_v0", None)
    
    raw_data_2d = data["raw_data"]
    grid_h = data["grid_h"]
    grid_w = data["grid_w"]
    t_steps = data["t_steps"]
    
    def sigmoid(x): 
        return 1 / (1 + np.exp(-x))

    total_width = len(t_steps) * grid_w * 0.5 
    total_height = len(sched_v0) * grid_h * 0.5 
    fig = plt.figure(figsize=(total_width, total_height))
    img_grid = ImageGrid(fig, 111, nrows_ncols=(len(sched_v0), 1))
    
    # raw_data_2d shape: (len(cond_a), len(sched_v0), t, grid, C, H, W)
    # Use the final value of the dimension we want to plot.
    series_data = rearrange(raw_data_2d[-1],
                              "cond_B t (b1 b2) C H W -> cond_B (b1 H) (t b2 W) C",
                              b1=grid_h, b2=grid_h)
    
    coloring_width = 16
    for i, (img, v0) in enumerate(zip(series_data, sched_v0)):
        # Create a red coloring signal based on t_steps and v0.
        signal_template = sigmoid(np.array(t_steps[:-1]) - v0)[:, None] * np.array([1, 0, 0])
        coloring_signal = repeat(signal_template, "t c -> height (t repeat) c", 
                                 repeat=64 * grid_w,
                                 height=coloring_width)
        img_with_color = np.concatenate((coloring_signal, img), axis=0)
        img_grid[i].imshow(img_with_color)
        img_grid[i].set_xticks([])
        img_grid[i].set_yticks([])
    plt.show()


if __name__ == "__main__":
    # For demonstration, load some data.
    with open("imgs/results_all.pkl", "rb") as f:
        data = pickle.load(f)["mod"]
    
    # Define the ordering of scheduler dimensions (should match generation order)
    scheduler_order = ["sched_capacity_template", "sched_decay_rate", "sched_v0"]
    
    # Transform raw_data to only keep two dimensions, based on user input.
    # For example, if we want to plot sched_decay_rate (rows) and sched_v0 (columns),
    # we fix all others (here, sched_capacity_template) to index 0.
    raw_data_2d = transform_raw_data(data["raw_data"], ["sched_decay_rate", "sched_v0"], scheduler_order)
    # Update data dictionary with the transformed raw_data.
    data["raw_data"] = raw_data_2d
    
    # Now call the plotting functions.
    plot_condition_by_condition(data, "sched_decay_rate", "sched_v0")
    plot(data)
