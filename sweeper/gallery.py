"""
Sweeper — generate images across parameter grids and build HTML viewers.

Usage:
    config  = load_sweep_config("sweeper/configs/my_sweep.yaml")
    gallery = Gallery(config)
    gallery.generate(build_fn, run_fn)
    gallery.build_html()
"""

import copy
import itertools
import json
import os
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from pydantic import BaseModel

from .schema import SweepConfig, ListAxis, LinspaceAxis


def _resolve_axis(axis) -> list:
    if isinstance(axis, ListAxis):
        return list(axis.values)
    start, stop, n = axis.linspace
    vals = np.linspace(start, stop, int(n))
    if axis.round_int:
        vals = np.round(vals).astype(int)
    return vals.tolist()


def _extract_axes(node, prefix="") -> dict:
    """Recursively find all Axis fields in a Pydantic model tree."""
    axes = {}
    if isinstance(node, BaseModel):
        for name in node.model_fields:
            value = getattr(node, name)
            path  = f"{prefix}.{name}" if prefix else name
            if isinstance(value, (ListAxis, LinspaceAxis)):
                axes[path] = _resolve_axis(value)
            else:
                axes.update(_extract_axes(value, path))
    elif isinstance(node, list):
        for i, item in enumerate(node):
            axes.update(_extract_axes(item, f"{prefix}[{i}]"))
    return axes


def _set_path(obj, path: str, value):
    """Set a dotted/indexed path on a nested structure of dicts and lists."""
    parts = _split_path(path)
    for part in parts[:-1]:
        if isinstance(part, int):
            obj = obj[part]
        else:
            obj = obj[part]
    last = parts[-1]
    if isinstance(last, int):
        obj[last] = value
    else:
        obj[last] = value


def _split_path(path: str) -> list:
    """Split 'ppg.gate.n' or 'maps[1].dim_out' into a list of str/int keys."""
    parts = []
    for segment in path.replace("]", "").split("."):
        if "[" in segment:
            name, idx = segment.split("[")
            if name:
                parts.append(name)
            parts.append(int(idx))
        else:
            parts.append(segment)
    return parts


def _unflatten(flat_cell: dict, base: SweepConfig) -> SweepConfig:
    """Return a new SweepConfig with all axis paths replaced by their cell values."""
    raw = base.model_dump()
    for path, value in flat_cell.items():
        # Axis values that are Pydantic models (e.g. MapConfig) need model_dump too
        if isinstance(value, BaseModel):
            value = value.model_dump()
        _set_path(raw, path, value)
    return SweepConfig.model_validate(raw)


class Gallery:
    """Manages a sweep directory: generation + HTML building."""

    def __init__(self, config: SweepConfig):
        self.config      = config
        self.output_dir  = config.output_dir
        self.images_dir  = os.path.join(self.output_dir, "images")
        self.manifest_path = os.path.join(self.output_dir, "manifest.json")
        self._axes       = _extract_axes(config)

    def _grid(self):
        """Cartesian product of all axes. Yields (index, flat_cell dict)."""
        names = list(self._axes.keys())
        value_lists = [self._axes[n] for n in names]
        for idx, combo in enumerate(itertools.product(*value_lists)):
            yield idx, dict(zip(names, combo))

    def _cell_dir(self, idx, flat_cell):
        parts = [f"{idx:04d}"] + [str(v) for v in flat_cell.values()]
        return os.path.join(self.images_dir, "_".join(parts))

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
            valid_keys = {tuple(sorted(fc.items())) for _, fc in self._grid()}
            old_count = len(manifest["entries"])
            manifest["entries"] = [
                e for e in manifest["entries"]
                if tuple(sorted(e["flat_cell"].items())) in valid_keys
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

        existing = {tuple(sorted(e["flat_cell"].items())) for e in manifest["entries"]}

        for idx, flat_cell in my_cells:
            cell_dir  = self._cell_dir(idx, flat_cell)
            cell_key  = tuple(sorted(flat_cell.items()))

            if cell_key in existing and (n_images is None or self._cell_complete(cell_dir, n_images)):
                print(f"[rank {rank}] [{idx+1}/{total}] skip  {flat_cell}")
                continue

            print(f"[rank {rank}] [{idx+1}/{total}] run   {flat_cell}")
            try:
                cell = _unflatten(flat_cell, self.config)
                build_fn(cell)
                output = run_fn()
            except Exception as e:
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

            if cell_key in existing:
                manifest["entries"] = [e for e in manifest["entries"]
                                        if tuple(sorted(e["flat_cell"].items())) != cell_key]
            manifest["entries"].append(entry)
            existing.add(cell_key)
            self._save_manifest(manifest, manifest_path)

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
                key = tuple(sorted(entry["flat_cell"].items()))
                if key not in seen:
                    merged["entries"].append(entry)
                    seen.add(key)

        grid_order = {tuple(sorted(fc.items())): idx for idx, fc in self._grid()}
        merged["entries"].sort(key=lambda e: grid_order.get(
            tuple(sorted(e["flat_cell"].items())), 0))

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
                   plot_fn=None, baseline_paths=None):
        if output_path is None:
            output_path = os.path.join(self.output_dir, "viewer.html")

        manifest = self.load_manifest(self.manifest_path)
        if plot_fn is not None:
            self._generate_plots(manifest, plot_fn)

        baseline_rel = None
        if baseline_paths:
            baseline_rel = [os.path.relpath(p, self.output_dir) for p in baseline_paths]

        from .viewer import build_viewer_html
        html = build_viewer_html(
            manifest,
            base_dir=self.output_dir,
            example_paths=example_paths,
            prompts=prompts,
            baseline_paths=baseline_rel,
            title=self.config.title,
        )
        Path(output_path).write_text(html)
        print(f"Viewer saved to {output_path}")

    @staticmethod
    def load_manifest(manifest_path):
        with open(manifest_path) as f:
            return json.load(f)
