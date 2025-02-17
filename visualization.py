import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.gridspec import GridSpec
from einops import rearrange, repeat

def plot_conditions(data):
    """
    Plots a grid of images based on two conditioning variables, with an additional template image on the right.

    Parameters:
    - raw_data: numpy array of shape (len(cond_a), len(cond_b), H, W, C)
    - cond_a: List of values for the first condition (rows)
    - cond_b: List of values for the second condition (columns)
    - template: Single image (H, W, C) to display separately
    """

    cond_a = data["sched_capacity_template"]
    cond_b = data["sched_v0"]
    raw_data = data["raw_data"] # (con_A, con_B, t, B, C, H W)
    grid_h = data['grid_h']
    grid_w = data['grid_w']
    template = data['template']
    t_steps = data['t_steps']


    num_rows = len(cond_a)
    num_cols = len(cond_b)
    num_steps = len(t_steps)


    # Define widths and heights 
    template_width = 1.5 
    grid_width = 1.5 * num_cols
    grid_height = 1.5 * num_rows
    

    total_width = grid_width + template_width
    total_height = grid_height 

    # Initalize figure Grid will be in
    fig = plt.figure(figsize=(total_width, total_height))  

    # Define grid layout and determine size
    gs = fig.add_gridspec(1, 2,
                          width_ratios=[grid_width, template_width],
    )
    
    # Create and add axis for template
    ax_template = fig.add_subplot(gs[0, 1])
    ax_template.imshow(template)

    ax_template.set_xticks([])  # remove ticks
    ax_template.set_yticks([])
    ax_template.set_title("Template Image") # add title

    # Create and add axis for final images
    img_grid = ImageGrid(fig, gs[0, 0],
                         nrows_ncols = (num_rows, num_cols))

    # Plot images in the image grid
    img_grid_data = rearrange(raw_data[:, :, -1], "... (b1 b2) C H W -> ... (b1 H) (b2 W) C", b1=grid_h, b2=grid_w)
    for i, a in enumerate(cond_a):
        for j, b in enumerate(cond_b):
            idx = i * num_cols + j
            img = np.clip(img_grid_data[i, j], 0, 1)
            img_grid[idx].imshow(img)
            img_grid[idx].set_xticks([])
            img_grid[idx].set_yticks([])

            img_grid[idx].set_ylabel(f"Guide : {a:.3f}")
            img_grid[j].set_title(f"v_0: {b:.3f}")

    plt.show()

########## 2-D ########## 

def plot(data):
    
    
    cond_a = data["sched_capacity_template"]
    sched_v0 = data["sched_v0"]
    raw_data = data["raw_data"] # (con_A, con_B, t, B, C, H W)
    grid_h = data['grid_h']
    grid_w = data['grid_w']
    template = data['template']
    t_steps = data['t_steps']

    def sigmoid(x) : return 1 / (1 + np.exp(-x))

    # Init figure
    fig = plt.figure()  
    # ImgGrid for images
    img_grid = ImageGrid(fig, 111,nrows_ncols = (len(sched_v0), 1))
    
    # Reformat data 
    series_data = rearrange(raw_data[-1], "cond_B t (b1 b2) C H W -> cond_B (b1 H) (t b2 W) C",
                           b1=grid_h, b2=grid_h)

    # Iterate through different v_0s
    coloring_width = 16
    for i, (img, v_0) in enumerate(zip(series_data, sched_v0)):
        # add color to top and bottom of images
        signal_template = sigmoid(t_steps[1::] - v_0)[:, None] * np.array([1, 0, 0])
        coloring_signal = repeat(signal_template, "t c -> height (t repeat) c", 
                                 repeat=64,
                                 height=coloring_width

        )

        # Plot image with template strength
        img=np.concatenate((coloring_signal, img))
        img_grid[i].imshow(img)
        img_grid[i].set_xticks([])
        img_grid[i].set_yticks([])
        
    plt.show()

    # TODO: fix clipping




