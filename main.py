
import os 
import pickle
# from mylib.diffusion import schedule_diffusion

if __name__ == "__main__":
    # Set output destination
    exp_name = "no_guidance"
    exp_path = os.path.join(os.getcwd(), "data", "output", exp_name)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    else:
        i = 1
        while True:
            exp_path = os.path.join(os.getcwd(), "data", "output", f"{exp_name}_{i}")
            i += 1
            if not os.path.exists(exp_path):
                os.makedirs(exp_path)
                break

    # Define your simulation parameters
    prms_sim = None

    # Pass to scheduler
    # raw_data = schedule_diffusion(prms_sim)
    raw_data = None

    # Save result
    raw_data_path = os.path.join(exp_path, "raw_data.pkl")
    prms_sim_path = os.path.join(exp_path, "prms_sim.pkl")
    with open(raw_data_path, "wb") as f:
        pickle.dump(raw_data, f)

    with open(prms_sim_path, "wb") as f:
        pickle.dump(prms_sim, f)

    # Plot result





