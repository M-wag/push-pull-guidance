import numpy as np
import os
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Import all data 
    dir_data = "data/parameter_evaluation/"
    np_files =  [os.path.join(dir_data, f) for f in os.listdir(dir_data) if f.endswith(".npy")] 
    time = np.load(os.path.join(dir_data, "time.npy"))
    x_axis = range(0, len(time)-1)
    fig, axes = plt.subplots(2, 2)
    for i, f in enumerate(np_files): 
        if "time" in f:
            continue
        if "baseline" in f:
            continue
        if "hf" in f:
            continue 

        scale, v_0 = os.path.splitext(os.path.basename(f))[0].split("_")
        scale, v_0 = round(float(scale), 1), round(float(v_0), 1)
        if scale == 0:
            continue
        arr = np.load(f)
        axes[0, 0].scatter(x_axis, arr[0], label=f"{scale}")
        axes[1, 0].scatter(x_axis, arr[1], label=f"{scale}")

    # Plot HF
    hf_files = [os.path.join(dir_data, f) for f in os.listdir(dir_data) if f.startswith("hf")] 
    for i, f in enumerate(hf_files): 
        print(f)
        scale, v_0 = os.path.splitext(os.path.basename(f))[0].split("_")[1::]
        scale, v_0 = round(float(scale), 1), round(float(v_0), 1)
        arr = np.load(f)
        axes[0, 1].scatter(x_axis, arr[0], label=f"{scale}")
        axes[1, 1].scatter(x_axis, arr[1], label=f"{scale}")
                   
    # Plot no guidance
    arr = np.load(os.path.join(dir_data, "baseline.npy"))
    axes[0, 1].scatter(x_axis, arr[0], c='black', label="no guidance")
    axes[1, 1].scatter(x_axis, arr[1], c='black', label="no guidance")

    for ax in axes.flat:
        ax.legend()
        ax.set_ylim(-3, 50)
        ax.set_xlabel("time step")
        ax.set_ylabel("max magnitude")

    
    
    axes[0, 0].set_title("Model (U-Net Attention Guidance)")
    axes[1, 0].set_title("Template (U-Net Attention Guidance)")
    axes[0, 1].set_title("Model (Various)")
    axes[1, 1].set_title("Template (Various)")
    plt.show()




