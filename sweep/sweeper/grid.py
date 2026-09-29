"""Axis discovery, grid iteration, and cell materialization for parameter sweeps."""

import itertools
import math

import numpy as np
from pydantic import BaseModel

from .schema import SweepConfig, ListAxis, LinspaceAxis


# JSON has no literal for non-finite floats. Python's json writes the bare
# token `Infinity`, which is valid inline JS but breaks JSON.parse and renders
# as `null` under JSON.stringify. Round-trip non-finite floats as strings so the
# on-disk JSON is valid everywhere and the value stays self-describing.
def inf_to_json(obj):
    """Recursively replace non-finite floats with string sentinels for JSON."""
    if isinstance(obj, float) and not math.isfinite(obj):
        if math.isnan(obj):
            return "nan"
        return "inf" if obj > 0 else "-inf"
    if isinstance(obj, dict):
        return {k: inf_to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [inf_to_json(v) for v in obj]
    return obj


def json_to_inf(obj):
    """Inverse of inf_to_json: restore non-finite floats from string sentinels."""
    if isinstance(obj, str):
        return {"inf": math.inf, "-inf": -math.inf, "nan": math.nan}.get(obj, obj)
    if isinstance(obj, dict):
        return {k: json_to_inf(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_to_inf(v) for v in obj]
    return obj


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


def cell_label(v) -> str:
    """Readable, path-safe string for a cell value (scalar or variant dict)."""
    if isinstance(v, dict):
        parts = [str(v["type"])] if "type" in v else []
        for k, vv in v.items():
            if k == "type":
                continue
            parts.append(f"{k}={cell_label(vv)}")
        return "-".join(parts) if parts else "variant"
    return str(v)


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


def _variant_key_unions(obj, path=()):
    """Find every variants axis and the union of keys across its option dicts.

    Returns a list of (path_to_parent, union_of_keys). Used to strip keys that
    _prefill_variants copied from the first variant onto the parent but that the
    chosen variant does not set (e.g. a numdiff variant's `kwargs` leaking into a
    jvp cell).
    """
    results = []
    if isinstance(obj, dict):
        variants = obj.get("variants")
        if isinstance(variants, dict):
            keys = set()
            for val in (variants.get("values") or []):
                if isinstance(val, dict):
                    keys |= set(val.keys())
            if keys:
                results.append((path, keys))
        for k, v in obj.items():
            results.extend(_variant_key_unions(v, path + (k,)))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            results.extend(_variant_key_unions(item, path + (i,)))
    return results


def _merge_variants(obj):
    """Recursively merge any 'variants' dict into its parent and drop the key.

    A variant axis (ListAxis of dicts) lets you sweep a bundle like
    {type: hill, n: 3} as a single cell. After _set_path, the chosen dict
    sits at parent.variants; this pass hoists its keys onto the parent so
    pydantic can validate without a stray 'variants' field.
    """
    if isinstance(obj, dict):
        variant = obj.pop("variants", None)
        if isinstance(variant, dict):
            obj.update(variant)
        for v in obj.values():
            _merge_variants(v)
    elif isinstance(obj, list):
        for item in obj:
            _merge_variants(item)


def unflatten(flat_cell: dict, base: SweepConfig) -> SweepConfig:
    """Return a new SweepConfig with all axis paths replaced by their cell values."""
    raw = base.model_dump()
    # Capture variant key sets before _set_path overwrites the variants axis.
    variant_unions = _variant_key_unions(raw)
    for path, value in flat_cell.items():
        if isinstance(value, BaseModel):
            value = value.model_dump()
        _set_path(raw, path, value)
    # Drop keys prefilled from the first variant that the chosen variant omits,
    # so they don't leak across variants (e.g. numdiff `kwargs` into jvp).
    for path, keys in variant_unions:
        parent = raw
        for part in path:
            parent = parent[part]
        chosen = parent.get("variants")
        chosen_keys = set(chosen) if isinstance(chosen, dict) else set()
        for k in keys - chosen_keys:
            parent.pop(k, None)
    _merge_variants(raw)
    return SweepConfig.model_validate(raw)


def iter_grid(axes: dict):
    """Cartesian product of axes. Yields (index, flat_cell dict)."""
    names = list(axes.keys())
    value_lists = [axes[n] for n in names]
    for idx, combo in enumerate(itertools.product(*value_lists)):
        yield idx, dict(zip(names, combo))
