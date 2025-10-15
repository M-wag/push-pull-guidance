import torch
import tqdm
import polars as pl
import matplotlib.pyplot as plt
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

        # transform raw data
        self.data_raw = data_raw
        self.data_all = filter_raw_data(json_raw)
        self.data = {condition : self.data_all.filter(predicate) for condition, predicate in condition_to_predicates.items()}
        # initialize visualiser grid
        self.grid = {}
        # figure params
        self.row_size = 5.3
        self.col_size = 5.3

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

        for pos, vizs in self.grid.items():
            for viz in vizs:
                viz(fig, axes[pos])
        
        return fig, axes 

    def scatter_metric(self, pos, conditions, metrics) -> None:

        def _scatter_metric(fig, ax):
            # Plot (nu, metric) for each specified metric
            for condition in conditions:
                df = self.data[condition].select(["nu", "metrics"]).unnest("metrics")
                for metric in metrics:
                    ax.scatter(df["nu"], df[metric])

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

    def show_examples(self, pos, condition=None, predicate=None):
        def _show_examples(fig, ax):

            # whether to apply predicate entire data or sunset
            if condition:
                df = self.data[condition]
            else: 
                df = self.data_all

            # get run_id by applying predicate
            run_id = df.filter(predicate)["run_id"].item()
            # get entry and convert to config 
            entry = util.get_entry_from_records(self.data_raw, run_id=run_id)
            config = util.convert_entry_to_config(entry)

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


if __name__ == "__main__":
    # Predicates one can use to select data 
    first_order = (pl.col("apply_2nd_order") == False) & (pl.col("num_steps") == 64)
    second_order = (pl.col("apply_2nd_order") == True) & (pl.col("num_steps") == 32)
    autoencoder = (pl.col("autoencoder") == "kl") & (pl.col("id") == "stabilityai/sd-turbo")
    heaviside = pl.col("type_gate") == "heaviside"
    delayed_guidance = pl.col("noise_onset") < 80.0
    many_examples = pl.col("example_idx_range").is_null()
    deterministic = pl.col("S_churn") == 0
    stochastic = pl.col("S_churn")  != 0
    use_noisy_examples = pl.col("use_noisy_examples") == False

    # For each condition assign predicates
    condition_to_predicates = {
            "exam-2ndord-stoch"  : (
                second_order,
                stochastic,
                autoencoder,
                heaviside,
                ~delayed_guidance,
                many_examples,
            )
    }


    # Initialize and run visualizer
    json_raw = util.read_records("data/runs_2.json")
    visualizer = ExperimentVisualizer(json_raw, condition_to_predicates)
    # Add scatter
    visualizer.scatter_metric((0, 1), ["exam-2ndord-stoch"], ["fd_dinov2"])
    visualizer.scatter_metric((1, 1), ["exam-2ndord-stoch"], ["fd_dinov2_csmean"])
    # Generate image for  specific config
    visualizer.show_examples((1, 0), condition="exam-2ndord-stoch", predicate=(pl.col("nu") == 80.0))

    # Plot image
    visualizer.imshow((0, 0), torch.randn(64, 64, 3))
    # visualizer.imshow((2, 0), torch.randn(64, 64, 3))
    fig, axes = visualizer.eval()
    fig.savefig("temp.pdf")
