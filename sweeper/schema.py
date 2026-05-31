from __future__ import annotations

import yaml
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from typing_extensions import Annotated
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Axis specs

class ListAxis(BaseModel):
    values: List[Union[float, int, str, bool, Dict[str, Any]]]

class LinspaceAxis(BaseModel):
    linspace: Tuple[float, float, int]
    round_int: bool = False

Axis = Union[ListAxis, LinspaceAxis]


# ---------------------------------------------------------------------------
# Gate / PPG

class GateConfig(BaseModel):
    type: Literal["quadratic", "heaviside", "hill"] = "quadratic"
    nu:   Union[float, Axis]
    n:    Union[int, float, Axis] = 3  # n = .inf turns the hill gate into a heaviside gate
    variants: Optional[ListAxis] = None


class PPGConfig(BaseModel):
    mean_scale:         Literal["vp", "ve"] = "vp"
    normalize_variance: Union[Literal["none", "split", "decomposed", "global"], Axis] = "split"
    gate:               GateConfig
    use_net_below:      Optional[float] = None


# ---------------------------------------------------------------------------
# Model

class SDModelConfig(BaseModel):
    type:       Literal["sd"]
    checkpoint: str = "runwayml/stable-diffusion-v1-5"


class EDMModelConfig(BaseModel):
    type:    Literal["edm"]
    net_pkl: str = "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl"


class EDM2ModelConfig(BaseModel):
    type:    Literal["edm2"]
    net_pkl: str = "https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-xs-2147483-0.200.pkl"


ModelConfig = Annotated[
    Union[SDModelConfig, EDMModelConfig, EDM2ModelConfig],
    Field(discriminator="type")
]


# ---------------------------------------------------------------------------
# Solver

class DDIMSolverConfig(BaseModel):
    type:           Literal["ddim"] = "ddim"
    num_steps:      int   = 50
    guidance_scale: float = 7.5
    ddim_eta:       Union[float, Axis] = 0.0


class EDMSolverConfig(BaseModel):
    type:         Literal["edm"] = "edm"
    num_steps:    int = 50
    sigma_max:    Union[float, Axis] = 80.0
    stochastic:   Union[bool, Axis]  = False
    second_order: Union[bool, Axis]  = True
    solver_seed:  Optional[int]      = None


SolverConfig = Annotated[
    Union[DDIMSolverConfig, EDMSolverConfig],
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Maps

class BaseMapConfig(BaseModel):
    scale:    float = 1.0
    variants: Optional[ListAxis] = None


class ProjectedMapConfig(BaseMapConfig):
    dim_out: Optional[Union[float, Axis]] = None


class LinearMapConfig(ProjectedMapConfig):
    type:        Literal["linear"]
    projection:  Union[Literal["orthonormal", "coordinate", "lowpass", "matrix"], Axis] = "orthonormal"
    n_features:  Union[int, Axis] = 1
    seed:        int = 2
    matrix_path: Optional[str] = None  # path to .pt file when projection == "matrix"


class SpgMapConfig(ProjectedMapConfig):
    type:         Literal["spg"]
    basis:        Union[Literal["ambient", "orthonormal", "frequency"], Axis] = "ambient"
    basis_kwargs: Optional[Dict[str, Any]] = None
    k_min:        Union[int, Axis] = 0


class IdentityMapConfig(BaseMapConfig):
    type: Literal["identity"]


class PullbackConfig(BaseModel):
    type:     Literal["jvp", "numdiff", "linear"] = "jvp"
    kwargs:   Optional[Dict[str, Any]] = None
    variants: Optional[ListAxis] = None


class NonlinearMapConfig(BaseMapConfig):
    type:       Literal["nonlinear"]
    map_type:   str
    map_kwargs: Optional[Dict[str, Any]] = None
    pullback:   PullbackConfig = Field(default_factory=PullbackConfig)


class InterpolationMapConfig(BaseMapConfig):
    type:          Literal["interpolation"]
    shape_out:     Tuple[int, int]
    shape_in:      Optional[Tuple[int, int]] = None
    mode:          str = "bilinear"
    align_corners: Union[bool, Axis] = False
    antialias:     Union[bool, Axis] = False


MapConfig = Annotated[
    Union[LinearMapConfig, SpgMapConfig, IdentityMapConfig, NonlinearMapConfig, InterpolationMapConfig],
    Field(discriminator="type")
]


# ---------------------------------------------------------------------------
# Top-level sweep config

class SweepConfig(BaseModel):
    title:        str = "Sweep"
    output_dir:   str
    baseline_dir: Optional[str] = None
    model:        ModelConfig
    examples:     Dict[str, Any] = {}
    logging:      Dict[str, Any] = {}
    snapshots:    Dict[str, Any] = {}
    noise_source: Literal["random", "ddim_inversion", "sdedit"] = "random"
    solver:       SolverConfig
    ppg:          Optional[PPGConfig] = None
    maps:         List[MapConfig] = []
    max_batch_size: Optional[int] = None


# ---------------------------------------------------------------------------

def _prefill_variants(node) -> None:
    """Recursively fill missing fields from the first variant entry.

    Allows required fields to be omitted from the base config and supplied
    only via variants, while still satisfying pydantic validation at load
    time. The actual per-cell values are written by unflatten() after
    variant expansion.
    """
    if isinstance(node, dict):
        variants = node.get("variants")
        if isinstance(variants, dict):
            first = (variants.get("values") or [None])[0]
            if isinstance(first, dict):
                for k, v in first.items():
                    if k not in node:
                        node[k] = v
        for v in node.values():
            _prefill_variants(v)
    elif isinstance(node, list):
        for item in node:
            _prefill_variants(item)


def load_sweep_config(path: str) -> SweepConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    # Default solver.type to match model.type when omitted (keeps old YAMLs working).
    solver = raw.setdefault("solver", {})
    if "type" not in solver:
        model_type = raw.get("model", {}).get("type", "edm")
        solver["type"] = "edm" if model_type == "edm2" else model_type
    _prefill_variants(raw)
    return SweepConfig.model_validate(raw)
