import numpy as np
import os
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Import all .npy 
    dir_data  = "data/parameter_evaluation"
    np_files = [f for f in os.listdir(dir_data) if f.endswith(".npy")]
    # Split by category
    cattoax = {
            "baseline"  :0,
            "amb"       :1,
            "hf"        :2,
            "unet-skip" :3,
            "unet-attn" :4,
            "unet-skip-step" :5,
            "unet-attn-step" :6,
    }

    fig, axes = plt.subplots(2, 7)
    for f in np_files:
        scale, v_0, cat = os.path.splitext(f)[0].split("_")
        stats = np.load(os.path.join(dir_data, f), allow_pickle=True).item()
        t, y = stats.t_steps[:-1], stats.max_mag
        # x = list(reversed(range(0, len(t))))
        x = t
        axes[0, cattoax[cat]].scatter(x, y[0], label=f"{scale}", 
                                      c="black" if cat == "baseline" else None)
        axes[0, cattoax[cat]].set_title(f"Model ({cat}) ")
        axes[1, cattoax[cat]].scatter(x, y[1], label=f"{scale}", 
                                      c="black" if cat == "baseline" else None)
        axes[1, cattoax[cat]].set_title(f"Template ({cat}) ")
        

    # For each category plot stats
    for ax in axes.flat:
        ax.set_xlabel("time")
        ax.set_ylabel("Maximum Magnitude")
        ax.legend()
        # ax.set_yscale("log")
        ax.grid()

    # Iterate through each ax and plot the
    plt.show()

