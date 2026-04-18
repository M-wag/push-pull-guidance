from __future__ import annotations

import yaml
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from typing_extensions import Annotated
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Axis specs

class ListAxis(BaseModel):
    values: List[Union[float, int, str]]

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

class SolverConfig(BaseModel):
    num_steps:      int   = 50
    guidance_scale: float = 7.5   # SD only
    ddim_eta:       float = 0.0   # SD only


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


MapConfig = Annotated[
    Union[LinearMapConfig, SpgMapConfig, IdentityMapConfig],
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
    noise_source: Literal["random", "ddim_inversion"] = "random"
    solver:       SolverConfig = SolverConfig()
    ppg:          Optional[PPGConfig] = None
    maps:         List[MapConfig] = []


# ---------------------------------------------------------------------------

def load_sweep_config(path: str) -> SweepConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return SweepConfig.model_validate(raw)
