"""
Sweeper — generate images across parameter grids and build HTML viewers.

Usage:
    config  = load_sweep_config("sweeper/configs/my_sweep.yaml")
    gallery = Gallery(config)
    gallery.generate(build_fn, run_fn)
    gallery.build_html()
"""

import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from .schema import SweepConfig
from .grid import extract_axes, iter_grid, unflatten, cell_label


def _freeze(v):
    """Recursively convert dicts/lists to hashable tuples."""
    if isinstance(v, dict):
        return tuple(sorted((k, _freeze(x)) for k, x in v.items()))
    if isinstance(v, list):
        return tuple(_freeze(x) for x in v)
    return v


def _cell_key(flat_cell):
    return tuple(sorted((k, _freeze(v)) for k, v in flat_cell.items()))



class Gallery:
    """Manages a sweep directory: generation + HTML building."""

    def __init__(self, config: SweepConfig):
        self.config      = config
        self.output_dir  = config.output_dir
        self.images_dir  = os.path.join(self.output_dir, "images")
        self.manifest_path = os.path.join(self.output_dir, "manifest.json")
        self._axes       = extract_axes(config)

    def _grid(self):
        return iter_grid(self._axes)

    def _cell_dir(self, idx, flat_cell):
        parts = [f"{idx:04d}"] + [cell_label(v) for v in flat_cell.values()]
        return os.path.join(self.images_dir, "_".join(parts))

    def _save_cell_config(self, cell, cell_dir):
        os.makedirs(cell_dir, exist_ok=True)
        with open(os.path.join(cell_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(cell.model_dump(), f, sort_keys=False)

    def _log_computed_cell(self, idx, flat_cell, success, rank=0, error=None):
        os.makedirs(self.output_dir, exist_ok=True)
        record = {
            "idx":       idx,
            "flat_cell": flat_cell,
            "timestamp": time.time(),
            "success":   success,
            "rank":      rank,
        }
        if error is not None:
            record["error"] = error
        path = os.path.join(self.output_dir, "computed_cells.jsonl")
        with open(path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def _cell_complete(self, cell_dir, n_images):
        if not os.path.isdir(cell_dir):
            return False
        return all(os.path.exists(os.path.join(cell_dir, f"img_{i}.png"))
                   for i in range(n_images))

    @staticmethod
    def _arr_to_pil(img):
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[2] not in (1, 3, 4):
            img = np.transpose(img, (1, 2, 0))
        if img.ndim == 3 and img.shape[2] == 1:
            img = img.squeeze(2)
        return Image.fromarray(img)

    def _save_images(self, images, cell_dir):
        os.makedirs(cell_dir, exist_ok=True)
        for i, img in enumerate(images):
            self._arr_to_pil(img).save(os.path.join(cell_dir, f"img_{i}.png"))

    def _save_snapshots(self, snapshots, cell_dir):
        snap_dir = os.path.join(cell_dir, "snapshots")
        os.makedirs(snap_dir, exist_ok=True)
        for i, steps in enumerate(snapshots):
            for s, img in enumerate(steps):
                self._arr_to_pil(img).save(os.path.join(snap_dir, f"img_{i}_step_{s}.png"))

    def _save_logs(self, result, cell_dir):
        os.makedirs(cell_dir, exist_ok=True)
        data = {}
        if result.logs_batch is not None:
            data["logs_batch"] = result.logs_batch
        if result.logs_per_image is not None:
            data["logs_per_image"] = result.logs_per_image
        if data:
            with open(os.path.join(cell_dir, "logs.json"), "w") as f:
                json.dump(data, f)

    def _manifest_path_for_rank(self, rank):
        return os.path.join(self.output_dir, f"manifest_rank{rank}.json")

    def _load_or_create_manifest(self, path=None):
        path = path or self.manifest_path
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        snap_cfg = self.config.snapshots
        snapshot_steps = (snap_cfg.get("steps") or []
                          if snap_cfg and snap_cfg.get("enabled", True) else [])
        return {
            "axes":           self._axes,
            "snapshot_steps": snapshot_steps,
            "entries":        [],
        }

    def _save_manifest(self, manifest, path=None):
        path = path or self.manifest_path
        os.makedirs(self.output_dir, exist_ok=True)
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)

    def _validate_manifest(self, manifest):
        """Discard stale entries when axes have changed."""
        if manifest.get("axes") != self._axes:
            valid_keys = {_cell_key(fc) for _, fc in self._grid()}
            old_count = len(manifest["entries"])
            manifest["entries"] = [
                e for e in manifest["entries"]
                if _cell_key(e["flat_cell"]) in valid_keys
            ]
            kept = len(manifest["entries"])
            print(f"Config changed: kept {kept}/{old_count} entries, discarded {old_count - kept} stale.")
            manifest["axes"] = self._axes
        return manifest

    def generate(self, build_fn, run_fn, n_images=None, rank=0, world_size=1, raise_errors=False):
        """
        Run the full sweep.

        build_fn(cell: SweepConfig) — configure model for this cell.
        run_fn() -> (images, snapshots, result) or just images.
        """
        os.makedirs(self.images_dir, exist_ok=True)
        grid  = list(self._grid())
        total = len(grid)

        my_cells = grid[rank::world_size]

        manifest_path = (self._manifest_path_for_rank(rank)
                         if world_size > 1 else self.manifest_path)
        manifest = self._load_or_create_manifest(manifest_path)
        manifest = self._validate_manifest(manifest)

        existing = {_cell_key(e["flat_cell"]): e for e in manifest["entries"]}

        for idx, flat_cell in my_cells:
            cell_dir  = self._cell_dir(idx, flat_cell)
            cell_k    = _cell_key(flat_cell)

            if cell_k in existing:
                stored = existing[cell_k]
                stored_images = [os.path.join(self.output_dir, p) for p in stored["images"]]
                if n_images is None or all(os.path.exists(p) for p in stored_images):
                    print(f"[rank {rank}] [{idx+1}/{total}] skip  {flat_cell}")
                    continue

            print(f"[rank {rank}] [{idx+1}/{total}] run   {flat_cell}")
            try:
                cell = unflatten(flat_cell, self.config)
                self._save_cell_config(cell, cell_dir)
                build_fn(cell)
                output = run_fn()
            except Exception as e:
                self._log_computed_cell(idx, flat_cell, success=False, rank=rank, error=str(e))
                if raise_errors:
                    raise
                print(f"[rank {rank}] [{idx+1}/{total}] FAILED: {e}")
                continue

            if isinstance(output, tuple):
                images, snapshots, result = output
            else:
                images, snapshots, result = output, None, None

            if n_images is None:
                n_images = len(images)

            self._save_images(images, cell_dir)
            if snapshots is not None:
                self._save_snapshots(snapshots, cell_dir)
            if result is not None:
                self._save_logs(result, cell_dir)

            rel_paths = [os.path.relpath(os.path.join(cell_dir, f"img_{i}.png"), self.output_dir)
                         for i in range(len(images))]
            entry = {
                "flat_cell": flat_cell,
                "images":    rel_paths,
                "snapshots": os.path.relpath(os.path.join(cell_dir, "snapshots"), self.output_dir) if snapshots is not None else None,
                "logs":      os.path.relpath(os.path.join(cell_dir, "logs.json"), self.output_dir) if result is not None else None,
            }

            if cell_k in existing:
                manifest["entries"] = [e for e in manifest["entries"]
                                        if _cell_key(e["flat_cell"]) != cell_k]
            manifest["entries"].append(entry)
            existing[cell_k] = entry
            self._save_manifest(manifest, manifest_path)
            self._log_computed_cell(idx, flat_cell, success=True, rank=rank)

        print(f"[rank {rank}] Done. {len(my_cells)}/{total} cells.")

    def merge_manifests(self, world_size):
        """Merge per-rank manifests into the final manifest. Call from rank 0 only."""
        merged = self._load_or_create_manifest()
        seen   = set()

        for r in range(world_size):
            rank_path = self._manifest_path_for_rank(r)
            if not os.path.exists(rank_path):
                continue
            with open(rank_path) as f:
                rank_manifest = json.load(f)
            for entry in rank_manifest["entries"]:
                key = _cell_key(entry["flat_cell"])
                if key not in seen:
                    merged["entries"].append(entry)
                    seen.add(key)

        grid_order = {_cell_key(fc): idx for idx, fc in self._grid()}
        merged["entries"].sort(key=lambda e: grid_order.get(
            _cell_key(e["flat_cell"]), 0))

        self._save_manifest(merged)
        for r in range(world_size):
            rank_path = self._manifest_path_for_rank(r)
            if os.path.exists(rank_path):
                os.remove(rank_path)
        print(f"Merged {len(merged['entries'])} entries from {world_size} ranks.")

    def _generate_plots(self, manifest, plot_fn):
        for entry in manifest["entries"]:
            logs_rel = entry.get("logs")
            if not logs_rel:
                continue
            logs_path = os.path.join(self.output_dir, logs_rel)
            if not os.path.exists(logs_path):
                continue
            with open(logs_path) as f:
                logs_data = json.load(f)
            plot_fn(logs_data, os.path.dirname(logs_path))

    def build_html(self, output_path=None, example_paths=None, prompts=None,
                   plot_fn=None, baseline_paths=None, n_seeds=1):
        if output_path is None:
            output_path = os.path.join(self.output_dir, "viewer.html")

        manifest = self.load_manifest(self.manifest_path)
        if plot_fn is not None:
            self._generate_plots(manifest, plot_fn)

        # Persist viewer metadata so standalone build_html() calls recover it.
        meta_changed = False
        if n_seeds != manifest.get("n_seeds", 1):
            manifest["n_seeds"] = n_seeds
            meta_changed = True
        if prompts is not None and prompts != manifest.get("prompts"):
            manifest["prompts"] = prompts
            meta_changed = True
        if meta_changed:
            self._save_manifest(manifest)

        # Fall back to stored values when called without arguments.
        if n_seeds == 1:
            n_seeds = manifest.get("n_seeds", 1)
        if prompts is None:
            prompts = manifest.get("prompts")

        # Reconstruct example paths from the already-copied examples/ directory.
        if example_paths is None:
            examples_dir = os.path.join(self.output_dir, "examples")
            if os.path.isdir(examples_dir):
                files = sorted(
                    (f for f in os.listdir(examples_dir) if f.startswith("example_")),
                    key=lambda f: int(f.split("_")[1].split(".")[0]),
                )
                example_paths = [os.path.join(examples_dir, f) for f in files] or None

        # Reconstruct baseline paths from baseline_dir when not passed explicitly.
        baseline_rel = None
        if baseline_paths:
            baseline_rel = [os.path.relpath(p, self.output_dir) for p in baseline_paths]
        elif self.config.baseline_dir and os.path.isdir(self.config.baseline_dir):
            n_images = len(manifest["entries"][0]["images"]) if manifest["entries"] else 0
            candidates = [
                os.path.join(self.config.baseline_dir, f"img_{i}.png")
                for i in range(n_images)
            ]
            if n_images and all(os.path.exists(p) for p in candidates):
                baseline_rel = [os.path.relpath(p, self.output_dir) for p in candidates]

        from .viewer import build_viewer_html
        html = build_viewer_html(
            manifest,
            base_dir=self.output_dir,
            example_paths=example_paths,
            prompts=prompts,
            baseline_paths=baseline_rel,
            title=self.config.title,
            n_seeds=n_seeds,
        )
        Path(output_path).write_text(html)
        print(f"Viewer saved to {output_path}")

    @staticmethod
    def load_manifest(manifest_path):
        with open(manifest_path) as f:
            return json.load(f)
