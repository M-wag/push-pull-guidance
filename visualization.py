import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.gridspec import GridSpec

def plot_conditions(data, cond_a, cond_b, template):
    """
    Plots a grid of images based on two conditioning variables, with an additional template image on the right.

    Parameters:
    - data: numpy array of shape (len(cond_a), len(cond_b), H, W, C)
    - cond_a: List of values for the first condition (rows)
    - cond_b: List of values for the second condition (columns)
    - template: Single image (H, W, C) to display separately
    """
    num_rows = len(cond_a)
    num_cols = len(cond_b)


    # Define widths and heights 
    template_width = 1.5 
    grid_width = 1.5 * num_cols
    total_width = grid_width + template_width
    total_height = 1.5 * num_rows


    # Initalize figure Grid will be in
    fig = plt.figure(figsize=(total_width, total_height))  

    # Define grid layout
    gs = fig.add_gridspec(1, 2,
                          width_ratios=[grid_width, template_width],
    )
    
    # Create and add axis for template
    ax_template = fig.add_subplot(gs[1])
    ax_template.imshow(template)

    ax_template.set_xticks([])  # remove ticks
    ax_template.set_yticks([])
    ax_template.set_title("Template Image") # add title

    # Create and add ImageGrid for multiple conditions
    img_grid = ImageGrid(fig, gs[0],
                         nrows_ncols = (num_rows, num_cols))

    # Plot images in the grid
    for i, a in enumerate(cond_a):
        for j, b in enumerate(cond_b):
            idx = i * num_cols + j
            img_grid[idx].imshow(data[i, j])
            img_grid[idx].set_xticks([])
            img_grid[idx].set_yticks([])

            img_grid[idx].set_ylabel(f"Guide : {a:.3f}")
            img_grid[j].set_title(f"v_0: {b:.3f}")

    plt.show()



########## 2-D ########## 

