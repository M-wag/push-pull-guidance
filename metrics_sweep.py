"""
Metrics sweep: run a parameter grid and compute quantitative metrics per cell.

See plan.md for the full design.
"""

import argparse
import csv
import json
import os
import random
import time
import yaml

import torch
import torch.distributed

from generate import generate_images, MetadataIterable
from sweeper import extract_axes, iter_grid, unflatten, cell_label
from sweeper.schema import SDModelConfig
from calculate_metrics import calculate_metrics_from_iterable, load_stats, metric_column_names
from torch_utils import distributed as dist

from sweep import (
    SweepRunner, ProjectionCache,
    PIPELINE_SETUP, precompute_ddim,
    repeat_each, expand_seeds, load_wildti2i, load_imgnet_qualitative, load_imgnet64,
    SEED_BY_DATASET_INDEX,
)


# ---------------------------------------------------------------------------
# Datasets


# ---------------------------------------------------------------------------
# Config extensions

def _load_metrics_sweep_config(path):
    """Load YAML, split out metrics-specific fields, return (SweepConfig, extras)."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    ref_stats            = raw.pop("ref_stats")
    metrics              = raw.pop("metrics", {"fd": ["inception", "dinov2"]})
    output_csv           = raw.pop("output_csv", None)
    example_features_dir = raw.pop("example_features_dir", None)
    image_dir            = raw.pop("image_dir", None)
    solver = raw.setdefault("solver", {})
    if "type" not in solver:
        solver["type"] = raw.get("model", {}).get("type", "edm")
    from sweeper.schema import SweepConfig
    config = SweepConfig.model_validate(raw)
    if output_csv is None:
        output_csv = os.path.join(config.output_dir, "metrics.csv")
    return config, dict(ref_stats=ref_stats, metrics=metrics, output_csv=output_csv,
                        example_features_dir=example_features_dir, image_dir=image_dir)


# ---------------------------------------------------------------------------
# CSV helpers

def _read_done_indices(csv_path):
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        return {int(r["cell_idx"]) for r in reader if r.get("cell_idx")}


def _append_csv_row(csv_path, row, fieldnames):
    exists = os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------------------
# Main

def main():
    parser = argparse.ArgumentParser(description="PPG metrics sweep")
    parser.add_argument("config", help="Path to metrics sweep config YAML")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    dist.init()
    rank, world_size = dist.get_rank(), dist.get_world_size()

    config, extras = _load_metrics_sweep_config(args.config)
    ref_stats_path       = extras["ref_stats"]
    metric_names         = extras["metrics"]
    output_csv           = extras["output_csv"]
    example_features_dir = extras["example_features_dir"]
    image_dir            = extras["image_dir"]

    # --- Pipeline setup ---
    pipeline = PIPELINE_SETUP[config.model.type](config.model, config.solver)
    if isinstance(config.model, SDModelConfig):
        pipe, dynamics, solver = pipeline
    else:
        dynamics, solver, _encoder = pipeline
        pipe = None

    # --- Dataset ---
    examples_cfg = config.examples
    n_seeds      = int(examples_cfg.get("n_seeds", 1))
    if isinstance(config.model, SDModelConfig):
        paths_example, prompts, dataset_indices = load_wildti2i(
            examples_cfg.get("dataset", "data/wild-ti2i"),
            n_entries=examples_cfg.get("n_entries"),
            indices=examples_cfg.get("indices"))
        class_labels = None
        base_seeds = dataset_indices if SEED_BY_DATASET_INDEX else list(range(len(prompts)))
    else:
        ds_path   = examples_cfg.get("dataset", "data/imgnet64")
        n_entries = examples_cfg.get("n_entries")
        paths_example, class_labels, dataset_indices, prompts = load_imgnet64(
            ds_path, n_entries=n_entries, seed=examples_cfg.get("seed", 0))
        base_seeds = dataset_indices if SEED_BY_DATASET_INDEX else list(range(len(class_labels)))

    paths_example_ex = repeat_each(paths_example, n_seeds)
    prompts_ex       = repeat_each(prompts, n_seeds)
    class_labels_ex  = repeat_each(class_labels, n_seeds)
    seeds            = expand_seeds(base_seeds, n_seeds)

    # --- DDIM inversion precompute (SD only) ---
    inverted_noise = None
    if isinstance(config.model, SDModelConfig) and config.noise_source == "ddim_inversion":
        if rank == 0:
            inverted_noise = precompute_ddim(pipe, dynamics, paths_example, prompts,
                                             num_inference_steps=config.solver.num_steps)
            if n_seeds > 1:
                inverted_noise = inverted_noise.repeat_interleave(n_seeds, dim=0)
        if world_size > 1:
            if rank == 0:
                shape_tensor = torch.tensor(list(inverted_noise.shape), device="cuda")
            else:
                shape_tensor = torch.zeros(4, dtype=torch.long, device="cuda")
            torch.distributed.broadcast(shape_tensor, src=0)
            if rank != 0:
                inverted_noise = torch.zeros(*shape_tensor.tolist(), device="cuda")
            torch.distributed.broadcast(inverted_noise, src=0)
            torch.distributed.barrier()

    # --- Runner ---
    runner = SweepRunner(
        config=config,
        dynamics=dynamics, solver=solver,
        paths_example=paths_example_ex, seeds=seeds,
        projection_cache=ProjectionCache(),
        pipe=pipe, prompts=prompts_ex,
        inverted_noise=inverted_noise,
        class_labels=class_labels_ex,
    )

    # --- Load reference stats once (rank 0 fetches; broadcast implicitly via re-load on each rank) ---
    ref_stats = load_stats(ref_stats_path, verbose=True)

    # --- Resume state ---
    done = _read_done_indices(output_csv) if dist.get_rank() == 0 else set()

    # --- Sweep loop ---
    axes = extract_axes(config)
    axis_names = list(axes.keys())

    metric_columns = metric_column_names(metric_names)
    fieldnames = ["cell_idx"] + axis_names + metric_columns + ["n_images", "seed"]

    for idx, flat_cell in iter_grid(axes):
        if rank == 0 and idx in done:
            dist.print0(f"[cell {idx}] already in CSV, skipping.")
            skip = torch.tensor([1], device="cuda")
        else:
            skip = torch.tensor([0], device="cuda")
        if world_size > 1:
            torch.distributed.broadcast(skip, src=0)
        if int(skip.item()):
            continue

        cell_config = unflatten(flat_cell, config)
        runner.build(cell_config)
        inputs = runner._make_inputs()

        if 'cs' in metric_names:
            cls_ids = class_labels_ex if class_labels_ex is not None else [0] * len(seeds)
            ex_ids  = [dataset_indices[i // n_seeds] for i in range(len(seeds))]
            inputs.extensions = tuple(inputs.extensions) + (
                MetadataIterable(class_id=cls_ids, example_id=ex_ids),
            )

        cell_image_dir = os.path.join(image_dir, str(idx)) if image_dir else None
        gen_iter = generate_images(
            solver, dynamics, inputs,
            max_batch_size=runner.defaults["max_batch_size"],
            dir_out=cell_image_dir,
        )

        dist.print0(f"[cell {idx}] {flat_cell}")

        metrics = calculate_metrics_from_iterable(
            gen_iter, ref_stats,
            metrics=metric_names,
            example_features_dir=example_features_dir,
        )

        if rank == 0:
            row = {"cell_idx": idx}
            for ax in axis_names:
                row[ax] = flat_cell[ax]
            for col in metric_columns:
                row[col] = metrics.get(col)
            row["n_images"] = runner.n_images
            row["seed"] = examples_cfg.get("seed", 0)
            _append_csv_row(output_csv, row, fieldnames)
            done.add(idx)

        if world_size > 1:
            torch.distributed.barrier()

    if world_size > 1:
        torch.distributed.barrier()


if __name__ == "__main__":
    main()
