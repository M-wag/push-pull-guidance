from __future__ import annotations

import yaml
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from typing_extensions import Annotated
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Axis specs

class ListAxis(BaseModel):
    values: List[Union[float, int, str, bool]]

class LinspaceAxis(BaseModel):
    linspace: Tuple[float, float, int]
    round_int: bool = False

Axis = Union[ListAxis, LinspaceAxis]


# ---------------------------------------------------------------------------
# Gate / PPG

class GateConfig(BaseModel):
    type: Literal["quadratic", "heaviside", "hill"] = "quadratic"
    nu:   Union[float, Axis]
    n:    Union[int, Axis] = 3


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


ModelConfig = Annotated[
    Union[SDModelConfig, EDMModelConfig],
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


SolverConfig = Annotated[
    Union[DDIMSolverConfig, EDMSolverConfig],
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Maps

class BaseMapConfig(BaseModel):
    scale: float = 1.0


class ProjectedMapConfig(BaseMapConfig):
    dim_out: Optional[Union[float, Axis]] = None


class LinearMapConfig(ProjectedMapConfig):
    type:       Literal["linear"]
    projection: Union[Literal["orthonormal", "coordinate", "lowpass"], Axis] = "orthonormal"
    n_features: Union[int, Axis] = 1
    seed:       int = 2


class SpgMapConfig(ProjectedMapConfig):
    type:         Literal["spg"]
    basis:        Union[Literal["ambient", "orthonormal", "frequency"], Axis] = "ambient"
    basis_kwargs: Optional[Dict[str, Any]] = None
    k_min:        Union[int, Axis] = 0


class IdentityMapConfig(BaseMapConfig):
    type: Literal["identity"]


class NonlinearMapConfig(BaseMapConfig):
    type:            Literal["nonlinear"]
    map_type:        str
    map_kwargs:      Optional[Dict[str, Any]] = None
    pullback:        Literal["jvp", "numdiff", "linear"] = "jvp"
    pullback_kwargs: Optional[Dict[str, Any]] = None


MapConfig = Annotated[
    Union[LinearMapConfig, SpgMapConfig, IdentityMapConfig, NonlinearMapConfig],
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


# ---------------------------------------------------------------------------

def load_sweep_config(path: str) -> SweepConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    # Default solver.type to match model.type when omitted (keeps old YAMLs working).
    solver = raw.setdefault("solver", {})
    if "type" not in solver:
        solver["type"] = raw.get("model", {}).get("type", "edm")
    return SweepConfig.model_validate(raw)
