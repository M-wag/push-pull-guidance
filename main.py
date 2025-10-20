import os 
import torch
import tqdm
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import dnnlib.util as util

from typing import Dict, Tuple, Callable
from run_metrics import ExperimentRunner
from pprint import pprint
from einops import rearrange

def filter_raw_data(json_raw):
    df_raw = pl.DataFrame(json_raw) 

    df_raw = (pl.DataFrame(json_raw)).select([
        "run_id",
        "num_images",
        pl.col("sampler").struct["apply_2nd_order"],
        pl.col("sampler").struct["num_steps"],
        pl.col("sampler").struct["S_churn"],
        pl.col("sampler").struct["sigma_max"],
        pl.col("generate").struct["example_idx_range"],
        pl.col("gvf").struct["vectorfield"].struct["noise_gate"],
        pl.col("gvf").struct["latent"],
        pl.col("generate").struct["use_noisy_examples"],
        pl.col("metrics")
    ]).unnest(["noise_gate", "latent"])
    
    return df_raw

class ExperimentVisualizer:
    def __init__(self, data_raw, condition_predicates):
        many_examples = pl.col("example_idx_range").is_null()

        # transform raw data
        self.data_raw = data_raw
        self.data_all = filter_raw_data(json_raw)
        self.data = {condition : self.data_all.filter(predicates + (many_examples,) ) for condition, predicates in condition_to_predicates.items()}
        
        # add memorization

        # initialize visualiser grid
        self.grid = {}
        # figure params
        self.row_size = 5.3
        self.col_size = 5.3


        self.style = {
            #"font.family": "serif",
            #"font.size": 10,
            "axes.labelsize": 20,
            "axes.titlesize": 10,
            "legend.fontsize": 10,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.linewidth": 0.5,
            "lines.linewidth": 1.0,
            "grid.linewidth": 1.0,
            "xtick.major.width": 1.5,
            "ytick.major.width": 1.5,
        }
    def append_grid(self, pos: Tuple[int, int], fn: Callable) -> None:
        if pos in self.grid:
            self.grid[pos].append(fn)
        else:
            self.grid[pos] = [fn]

    def make_fig(self) -> None:
        # iterate through each grid
        n_rows = 1 
        n_cols = 1 

        for row, col in self.grid.keys():
            n_rows = int(max(n_rows, row + 1))
            n_cols = int(max(n_cols, col + 1))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(self.row_size * n_cols, self.col_size * n_rows), 
                                 constrained_layout=True, squeeze=False)
        return fig, axes

    def eval(self) -> Tuple[plt.Figure, plt.Axes]:
        # Make figure
        fig, axes = self.make_fig()

        with plt.rc_context(self.style):
            for pos, vizs in self.grid.items():
                for viz in vizs:
                    viz(fig, axes[pos])
            
        return fig, axes 

    def scatter_metric(self, pos, conditions, metrics, xaxis="nu") -> None:

        def _scatter_metric(fig, ax):
            # Plot (nu, metric) for each specified metric
            for condition in conditions:
                df = self.data[condition].select([xaxis, "metrics", "fd_dinov2_mem"]).unnest("metrics")
                for metric in metrics:
                    ax.scatter(df[xaxis], df[metric], label=condition)

            # Stylize axis
            ax.set_axisbelow(True)
            ax.axhline(y=0, c="black", ls="--", alpha=0.5, zorder=1)
            ax.grid(True)
            ax.legend(loc="best")
            ax.set_xscale('log')

        self.append_grid(pos, _scatter_metric)

    def imshow(self, pos, img) -> None:

        def _imshow(fig, ax):
            ax.imshow(img)

        self.append_grid(pos, _imshow)

    def show_examples(self, 
                      pos, 
                      val,
                      condition=None, 
                      path_template="data/images/examples", 
                      class_idx=0, 
                      example_idx_range=None,):

        def _show_examples(fig, ax):

            # whether to apply predicate entire data or sunset
            if condition:
                df = self.data[condition]
            else: 
                df = self.data_all
            run_id = df["run_id"][1]

            # get entry and convert to config 
            # BUG : is modifying jsonb ojects
            import copy
            entry = copy.deepcopy(util.get_entry_from_records(self.data_raw, run_id=run_id))
            config = util.convert_entry_to_config(entry)

            # get run_id by applying predicate
            # modify config to apply value
            if "sdedit" in condition:
                config["sampler"]["sigma_max"] = val
            else:
                config["gvf"]["vectorfield"]["noise_gate"]["nu"] = val
            # modify config for nice visualization
            config["generate"]["example_idx_range"] = example_idx_range
            config["generate"]["class_idx"] = class_idx
            config["sampler"]["noise_seed"] = 0
            # init runner and append configs 
            paths = {"templates" : "data/images/examples", "out" : None}
            runner = ExperimentRunner(paths, num_images=9)
            runner.config.update({f"{key}_kwargs" : val for key, val in config.items()})

            
            # generate images
            image_iter = runner.generate_images()
            for r in tqdm.tqdm(image_iter, unit='batch'):
                pass
            images = rearrange(r.images, "(b1 b2) c h w -> (b1 h) (b2 w) c ", b1=3)
            images = images.detach().cpu().numpy()
            self.imshow(pos, images)

        self.append_grid(pos, _show_examples)

    def add_border(self, pos, color):

        def _add_border(fig, ax):
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)

        self.append_grid(pos, _add_border)

    def add_line(self, pos, val):

        def _add_line(fig, ax):
            ax.axvline(val, c="red")

        self.append_grid(pos, _add_line)


    def add_memorization(self, conditions, join_axis):
        few_examples = pl.col("example_idx_range") == [0,1]
        for condition in conditions:
            for metric in ["fid", "fd_dinov2"]:
                df_few = self.data_all.filter(condition_to_predicates[condition] + (few_examples,))
                few_minus_many = self.data[condition].unnest("metrics").join(df_few.unnest("metrics"), on=join_axis, suffix="_few").select([
                    pl.col(join_axis), 
                    (pl.col(f"{metric}_few") - pl.col(metric)).alias(f"{metric}_mem")
                ])
                self.data[condition] = self.data[condition].join(few_minus_many, on=join_axis)

if __name__ == "__main__":
    # Predicates one can use to select data 
    first_order = (pl.col("apply_2nd_order") == False) & (pl.col("num_steps") == 64)
    second_order = (pl.col("apply_2nd_order") == True) & (pl.col("num_steps") == 32)

    autoencoder = (pl.col("autoencoder") == "kl") & (pl.col("id") == "stabilityai/sd-turbo")
    no_noise_gate = (pl.col("type_gate").is_null()) & (pl.col("nu").is_null())

    heaviside = pl.col("type_gate") == "heaviside"
    delayed_guidance = pl.col("noise_onset") < 80.0

    deterministic = pl.col("S_churn") == 0
    stochastic = pl.col("S_churn")  != 0
    use_noisy_examples = pl.col("use_noisy_examples") == True

    many_examples = pl.col("example_idx_range").is_null()


    # Initialize and run visualizer
    json_raw = util.read_records("data/runs_2.json")

    from mylib.diffusion import time_steps_edm
    values = time_steps_edm(num_steps=32, sigma_min=0.002, sigma_max=80, rho=7).tolist()[0:21]
    values = [2.7]
    for i, val in enumerate(values):
        # For each condition assign predicates
        condition_to_predicates = {
                "exam-2ndord-stoch"  : (
                    second_order,
                    stochastic,
                    autoencoder,
                    heaviside,
                ),
                "sdedit-2ndord-stoch"  : (
                    second_order,
                    stochastic,
                    no_noise_gate,
                    use_noisy_examples,
                    pl.col("run_id") > 265,
                )
        }

        visualizer = ExperimentVisualizer(json_raw, condition_to_predicates)
        # Make memorization data
        visualizer.add_memorization(["exam-2ndord-stoch"], join_axis="nu")
        visualizer.add_memorization(["sdedit-2ndord-stoch"], join_axis="sigma_max")

        # Add scatter
        visualizer.scatter_metric((0, 1), ["exam-2ndord-stoch"], ["fd_dinov2"])
        visualizer.scatter_metric((0, 1), ["sdedit-2ndord-stoch"], ["fd_dinov2"], xaxis="sigma_max")
        visualizer.scatter_metric((1, 1), ["exam-2ndord-stoch"], ["fd_dinov2_csmean"])
        visualizer.scatter_metric((1, 1), ["sdedit-2ndord-stoch"], ["fd_dinov2_csmean"], xaxis="sigma_max")
        visualizer.scatter_metric((2, 1), ["exam-2ndord-stoch"], ["fd_dinov2_mem"])
        visualizer.scatter_metric((2, 1), ["sdedit-2ndord-stoch"], ["fd_dinov2_mem"], xaxis="sigma_max")
        visualizer.scatter_metric((3, 1), ["exam-2ndord-stoch"], ["L2_mean"])
        visualizer.scatter_metric((3, 1), ["sdedit-2ndord-stoch"], ["L2_mean"], xaxis="sigma_max")
        # Add templates
        class_idx = 0
        path_template = "data/images/examples"
        visualizer.imshow((0, 0), mpimg.imread(os.path.join(path_template, f"{class_idx}/0.png")))
        visualizer.imshow((0, 2), mpimg.imread(os.path.join(path_template, f"{class_idx}/0.png")))
        # Generated data
        visualizer.show_examples((1,0), val, condition="exam-2ndord-stoch", class_idx=class_idx, example_idx_range=[0,1])
        visualizer.show_examples((1,2), val, condition="sdedit-2ndord-stoch", class_idx=class_idx, example_idx_range=[0,1])

        # Color in borders
        visualizer.add_border((1, 0), "blue")
        visualizer.add_border((1, 2), "orange")
        # Add line 
        visualizer.add_line((0, 1), val)
        visualizer.add_line((1, 1), val)
        visualizer.add_line((2, 1), val)
            
        fig, axes = visualizer.eval()

        # Add subtitle 
        axes[0, 1].set_title(r"FD$_\text{DINOV2}$")
        axes[1, 1].set_title(r"$<cos_\text{DINOV2}>$")
        axes[2, 1].set_title(r"Memorizaton DINO$_\text{V2}$")
        axes[3, 1].set_title(r"$<|x - \mu |_2^2>$")

        fig.suptitle(rf"$\nu_0 / t_0 = {val:.2f}$", size=40)
        fig.savefig(f"temp/temp.png")
