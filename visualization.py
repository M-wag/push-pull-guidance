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

    fig = plt.figure(figsize=(num_cols * 1.5, num_rows * 1.5))  # Adjust figure size

    # Define grid layout: Left (big image grid), Right (single image)
    gs = GridSpec(2, 1, hspace=1.4)  # 3:1 width ratio

    # Left: ImageGrid for condition images
    ax_grid = fig.add_subplot(gs[1])  # This is just a placeholder
    image_grid = ImageGrid(fig, 111,  # Use standard subplot position (111)
                           nrows_ncols=(num_rows, num_cols),
                           axes_pad=0.1,
                           share_all=True,
                           aspect=True)

    # Remove axes
    fig.delaxes(ax_grid)  

    # Plot images in the grid
    for i, a in enumerate(cond_a):
        for j, b in enumerate(cond_b):
            idx = i * num_cols + j
            image_grid[idx].imshow(data[i, j])
            image_grid[idx].set_xticks([])
            image_grid[idx].set_yticks([])

            image_grid[idx].set_ylabel(f"Guide : {a:.3f}")
            image_grid[j].set_title(f"v_0: {b:.3f}")


    # Right: Single template image
    ax_template = fig.add_subplot(gs[0])
    ax_template.imshow(template)
    ax_template.set_xticks([])
    ax_template.set_yticks([])
    ax_template.set_title("Template Image")

    plt.show()

########## 2-D ########## 

