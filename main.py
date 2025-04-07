
from difflib import *

from dataclasses import dataclass, asdict
from typing import List, Any
import os 

@dataclass
class SimulationConfig:
    network_pkl: str

    scale_template_score: List[float] 
    decay_rate: List[float]
    v_0: List[float]
    n_projectors: List[int]
    dim_projector: List[int]
    template: Any  = None,
    scale_model_score: List[float] = 1.0

    seed: int = 0
    num_steps: int = 32
    grid_h: int = 3
    grid_w: int = 3

    sigma_min: float = 0.002  
    sigma_max: float = 80
    rho: float = 7
    S_churn: float = 0.0
    S_min: float = 0.0
    S_max: float = float('inf')
    S_noise: float = 1


    def __post_init__(self):
        # For validation
        pass

    def to_dict(self):
        return asdict(self)

    def __str__(self):
        lines = []
        max_key_len = max(len(k) for k in self.__dataclass_fields__)

        for key in self.__dataclass_fields__:
            value = getattr(self, key)
            if value is None:
                continue
            formatted_value = (
                f"[{', '.join(map(str, value))}]" if isinstance(value, list)
                else repr(value)
            )
            lines.append(f"{key:<{max_key_len}} = {formatted_value}")
        return "\n".join(lines)




if __name__ == "__main__":
    sim_name = "refactor"
    model_root = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained'
    network_pkl = f'{model_root}/edm-imagenet-64x64-cond-adm.pkl'

    params_control = SimulationConfig(
        network_pkl = network_pkl,
        template = [read_image("cat.jpg").reshape(1, 3, 64, 64)],
        scale_template_score=[0.0],
        decay_rate=[0.0],
        v_0=[0],
        n_projectors=[0],
        dim_projector=[0]
    )

    params_experiment = SimulationConfig(
        network_pkl = network_pkl,
        template = [read_image("cat.jpg").reshape(1, 3, 64, 64)],
        scale_template_score=[1.0],
        decay_rate=[0.1],
        v_0=[12, 24],
        n_projectors=[3],
        dim_projector=[192]
    )

    schedules = {"control" : params_control, "experiment": params_experiment}
    
    # Save each run to a pickle
    for name, params in schedules.items():
        result  = run_diffusion_for_schedule(**params.to_dict())

        # Check if directory already exists 
        fname = f"results/{sim_name}/{name}.pkl"
        dir_path = os.path.join("results", sim_name)
        os.makedirs(dir_path, exist_ok=True)

        fname = os.path.join(dir_path, f"{name}.pkl")
        counter = 1
        while os.path.exists(fname):
            fname = os.path.join(dir_path, f"{name}_{counter}.pkl")
            counter += 1

        with open(fname, "wb") as f:
            pickle.dump(result, f) 


def main_a():
   # VAR: File Name of Pickle
    fname = "imgs/test.pkl"
    # Run Diffusion 
    if True:
        schedules = {
            "og": {
                "sched_capacity_template": [0],
                "sched_v0": [0],
                "sched_decay_rate": [0],
            },
            "mod": {
                "sched_capacity_template": [1.0],
                "sched_v0": [12],
                "sched_decay_rate": [1.0],
                "sched_n_projectors": [8, ],
                "sched_dim_projector": np.linspace(192, 864, 3).astype(int),
            },
        }
        
        all_results = {}
        for name, params in schedules.items():
            all_results[name] = run_diffusion_for_schedule(params)
        
        with open(fname, "wb") as f:
            pickle.dump(all_results, f) # with open("imgs/results_afhq_all.pkl", "rb") as f:

    # Load file with data
    with open(fname, "rb") as f:
        loaded_results = pickle.load(f)
    # Load file with unmodifed data
    with open("imgs/results_imgnet_all.pkl", "rb") as f:
        two = pickle.load(f)

    
    # Edit file such that it has right template
    loaded_results['og'] = two['og'] 

    # Load in results
    data_mod = loaded_results['mod']
    data_og = loaded_results['og']

    # Format correctly for visualizer 
    print(data_og['scheduler_keys'])
    # define the ordering of scheduler dimensions (should match generation order)
    scheduler_order = ["sched_capacity_template", "sched_v0", "sched_decay_rate", "sched_n_projectors", "sched_dim_projector"]
    # raw_data_2d_mod = vis.transform_raw_data(data_mod["raw_data"], ["sched_n_projectors", "sched_dim_projector"], scheduler_order)
    raw_data_2d_mod = vis.transform_raw_data(data_mod["raw_data"], ["sched_capacity_template", "sched_v0"], scheduler_order)
    raw_data_2d_og = vis.transform_raw_data(data_og["raw_data"], ["sched_capacity_template", "sched_v0"], scheduler_order[:3])
    
    data_mod["raw_data"] = raw_data_2d_mod
    data_og["raw_data"] = raw_data_2d_og
    
    # visualization 
    # Return to visualizer
    vis.plot_condition_by_condition(data_mod, "sched_n_projectors", "sched_dim_projector", data_og)
