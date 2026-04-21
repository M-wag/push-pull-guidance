"""Axis discovery, grid iteration, and cell materialization for parameter sweeps."""

import itertools

import numpy as np
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


def extract_axes(node, prefix="") -> dict:
    """Recursively find all Axis fields in a Pydantic model tree."""
    axes = {}
    if isinstance(node, BaseModel):
        for name in node.model_fields:
            value = getattr(node, name)
            path  = f"{prefix}.{name}" if prefix else name
            if isinstance(value, (ListAxis, LinspaceAxis)):
                axes[path] = _resolve_axis(value)
            else:
                axes.update(extract_axes(value, path))
    elif isinstance(node, list):
        for i, item in enumerate(node):
            axes.update(extract_axes(item, f"{prefix}[{i}]"))
    return axes


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


def _set_path(obj, path: str, value):
    """Set a dotted/indexed path on a nested structure of dicts and lists."""
    parts = _split_path(path)
    for part in parts[:-1]:
        obj = obj[part]
    obj[parts[-1]] = value


def unflatten(flat_cell: dict, base: SweepConfig) -> SweepConfig:
    """Return a new SweepConfig with all axis paths replaced by their cell values."""
    raw = base.model_dump()
    for path, value in flat_cell.items():
        if isinstance(value, BaseModel):
            value = value.model_dump()
        _set_path(raw, path, value)
    return SweepConfig.model_validate(raw)


def iter_grid(axes: dict):
    """Cartesian product of axes. Yields (index, flat_cell dict)."""
    names = list(axes.keys())
    value_lists = [axes[n] for n in names]
    for idx, combo in enumerate(itertools.product(*value_lists)):
        yield idx, dict(zip(names, combo))
