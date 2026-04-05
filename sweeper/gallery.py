"""
Sweeper — generate images across parameter grids and build HTML viewers.

Usage:
    gallery = Gallery("sweeper/configs/my_sweep.yaml")
    gallery.generate(build_fn, run_fn)
    gallery.build_html()
"""

import itertools
import json
import os
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from typing import Optional

from .viewer import build_viewer_html


class Gallery:
    """Manages a sweep directory: generation + HTML building."""

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.output_dir = self.config["output_dir"]
        self.images_dir = os.path.join(self.output_dir, "images")
        self.manifest_path = os.path.join(self.output_dir, "manifest.json")
        self._axes = self._parse_axes()

    def _parse_axes(self):
        """Parse axis definitions into {name: [values]} dict."""
        axes = {}
        for name, spec in self.config.get("axes", {}).items():
            if "values" in spec:
                axes[name] = list(spec["values"])
            elif "linspace" in spec:
                start, stop, num = spec["linspace"]
                vals = np.linspace(start, stop, int(num))
                if spec.get("round_int"):
                    vals = np.round(vals).astype(int)
                axes[name] = vals.tolist()
            else:
                raise ValueError(f"Axis '{name}' must have 'values' or 'linspace'")
        return axes

    def _validate_manifest(self, manifest):
        """Check that manifest axes match current config. Discard stale entries."""
        manifest_axes = manifest.get("axes", {})
        if manifest_axes != self._axes:
            # Build set of valid param combos from current config
            valid_keys = {tuple(sorted(p.items())) for _, p in self._grid()}
            old_count = len(manifest["entries"])
            manifest["entries"] = [
                e for e in manifest["entries"]
                if tuple(sorted(e["params"].items())) in valid_keys
            ]
            kept = len(manifest["entries"])
            print(f"Config changed: kept {kept}/{old_count} entries, discarded {old_count - kept} stale.")
            manifest["axes"] = {name: vals for name, vals in self._axes.items()}
        return manifest

    def _grid(self):
        """Cartesian product of all axes. Yields (index, params_dict)."""
        names = list(self._axes.keys())
        value_lists = [self._axes[n] for n in names]
        for idx, combo in enumerate(itertools.product(*value_lists)):
            yield idx, dict(zip(names, combo))

    def _cell_dir(self, idx, params):
        """Directory name for a single grid cell."""
        parts = [f"{idx:04d}"]
        for v in params.values():
            parts.append(str(v))
        return os.path.join(self.images_dir, "_".join(parts))

    def _cell_complete(self, cell_dir, n_images):
        """Check if a cell directory has all expected images."""
        if not os.path.isdir(cell_dir):
            return False
        for i in range(n_images):
            if not os.path.exists(os.path.join(cell_dir, f"img_{i}.png")):
                return False
        return True

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
        """Save final images as img_{i}.png."""
        os.makedirs(cell_dir, exist_ok=True)
        for i, img in enumerate(images):
            self._arr_to_pil(img).save(os.path.join(cell_dir, f"img_{i}.png"))

    def _save_snapshots(self, snapshots, cell_dir):
        """Save snapshots[i][s] as snapshots/img_{i}_step_{s}.png."""
        snap_dir = os.path.join(cell_dir, "snapshots")
        os.makedirs(snap_dir, exist_ok=True)
        for i, steps in enumerate(snapshots):
            for s, img in enumerate(steps):
                self._arr_to_pil(img).save(os.path.join(snap_dir, f"img_{i}_step_{s}.png"))

    def _save_logs(self, result, cell_dir):
        """Save RunResult logs as logs.json."""
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
        """Per-rank manifest path for multi-GPU runs."""
        return os.path.join(self.output_dir, f"manifest_rank{rank}.json")

    def generate(self, build_fn, run_fn, n_images=None, rank=0, world_size=1):
        """
        Run the full sweep.

        build_fn(params: dict) — configure model for these params.
        run_fn() -> list[np.ndarray] — run inference, return list of images.
        n_images: expected number of images per cell (for crash recovery check).
                  If None, determined from first run.
        rank: current process rank (0 for single-GPU).
        world_size: total number of processes (1 for single-GPU).
        """
        os.makedirs(self.images_dir, exist_ok=True)

        grid = list(self._grid())
        total = len(grid)

        # Each rank handles a slice of the grid
        my_cells = grid[rank::world_size]

        # Load existing manifest for crash recovery (per-rank if multi-GPU)
        if world_size > 1:
            manifest_path = self._manifest_path_for_rank(rank)
        else:
            manifest_path = self.manifest_path
        manifest = self._load_or_create_manifest(manifest_path)

        # Validate manifest axes match current config — discard stale entries
        manifest = self._validate_manifest(manifest)

        existing = {tuple(sorted(e["params"].items())) for e in manifest["entries"]}

        for idx, params in my_cells:
            cell_dir = self._cell_dir(idx, params)
            params_key = tuple(sorted(params.items()))

            # Skip if already completed
            if params_key in existing and (n_images is None or self._cell_complete(cell_dir, n_images)):
                print(f"[rank {rank}] [{idx+1}/{total}] skip  {params}")
                continue

            print(f"[rank {rank}] [{idx+1}/{total}] run   {params}")
            build_fn(params)
            output = run_fn()

            # run_fn may return (images, snapshots, RunResult) or just images
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

            # Record in manifest
            rel_paths = [os.path.relpath(os.path.join(cell_dir, f"img_{i}.png"), self.output_dir)
                         for i in range(len(images))]
            entry = {
                "params":    params,
                "images":    rel_paths,
                "snapshots": os.path.relpath(os.path.join(cell_dir, "snapshots"), self.output_dir) if snapshots is not None else None,
                "logs":      os.path.relpath(os.path.join(cell_dir, "logs.json"), self.output_dir) if result is not None else None,
            }

            if params_key in existing:
                manifest["entries"] = [e for e in manifest["entries"]
                                       if tuple(sorted(e["params"].items())) != params_key]
            manifest["entries"].append(entry)
            existing.add(params_key)

            # Write manifest after each cell for crash recovery
            self._save_manifest(manifest, manifest_path)

        print(f"[rank {rank}] Done. {len(my_cells)}/{total} cells.")

    def merge_manifests(self, world_size):
        """Merge per-rank manifests into the final manifest. Call from rank 0 only."""
        merged = self._load_or_create_manifest()
        seen = set()

        for r in range(world_size):
            rank_path = self._manifest_path_for_rank(r)
            if not os.path.exists(rank_path):
                continue
            with open(rank_path) as f:
                rank_manifest = json.load(f)
            for entry in rank_manifest["entries"]:
                key = tuple(sorted(entry["params"].items()))
                if key not in seen:
                    merged["entries"].append(entry)
                    seen.add(key)

        # Sort entries by grid index for consistent ordering
        grid_order = {tuple(sorted(p.items())): idx for idx, p in self._grid()}
        merged["entries"].sort(key=lambda e: grid_order.get(tuple(sorted(e["params"].items())), 0))

        self._save_manifest(merged)

        # Clean up per-rank manifests
        for r in range(world_size):
            rank_path = self._manifest_path_for_rank(r)
            if os.path.exists(rank_path):
                os.remove(rank_path)

        print(f"Merged {len(merged['entries'])} entries from {world_size} ranks.")

    def _load_or_create_manifest(self, path=None):
        """Load existing manifest or create a new one."""
        path = path or self.manifest_path
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)

        snap_cfg = self.config.get("snapshots", {})
        snapshot_steps = (snap_cfg.get("steps") or []
                          if snap_cfg and snap_cfg.get("enabled", True) else [])

        examples_cfg = self.config.get("examples", {})
        return {
            "examples":       examples_cfg,
            "axes":           {name: vals for name, vals in self._axes.items()},
            "fixed":          self.config.get("fixed", {}),
            "snapshot_steps": snapshot_steps,
            "entries":        [],
        }

    def _save_manifest(self, manifest, path=None):
        """Write manifest to disk."""
        path = path or self.manifest_path
        os.makedirs(self.output_dir, exist_ok=True)
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)

    def _generate_plots(self, manifest, plot_fn):
        """Call plot_fn(logs_data, cell_dir) for each entry that has a logs.json."""
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

    def build_html(self, output_path=None, example_paths=None, prompts=None, plot_fn=None):
        """
        Build HTML viewer from manifest.

        example_paths: list of paths to example images (shown alongside outputs).
        prompts: list of prompt strings (shown as captions).
        plot_fn: optional callable(logs_data, cell_dir) that generates and saves plot images.
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, "viewer.html")

        manifest = self.load_manifest(self.manifest_path)
        if plot_fn is not None:
            self._generate_plots(manifest, plot_fn)
        html = build_viewer_html(
            manifest,
            base_dir=self.output_dir,
            example_paths=example_paths,
            prompts=prompts,
            title=self.config.get("title", "Sweep Viewer"),
        )
        Path(output_path).write_text(html)
        print(f"Viewer saved to {output_path}")

    @staticmethod
    def load_manifest(manifest_path):
        """Load an existing manifest."""
        with open(manifest_path) as f:
            return json.load(f)
