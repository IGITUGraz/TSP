from dataclasses import dataclass
from typing import Any, Dict, Optional

from omegaconf import MISSING


@dataclass
class LayerConfig:
    func: str = MISSING
    hyper_params: Optional[dict[str, Any]] = None
    params_name: Optional[str] = None
    states_name: Optional[str] = None
    fb_name: Optional[str] = None
    init_params: Optional[dict[str, Any]] = None
    init_states: Optional[dict[str, Any]] = None


@dataclass
class ModelConfig:
    layers_definitions: Dict[str, LayerConfig] = MISSING
    graph_ff_structure: str = MISSING
    graph_fb_structure: Optional[str] = None
    checkpoint_path: Optional[str] = None
