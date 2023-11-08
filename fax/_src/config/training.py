from dataclasses import dataclass
from lib2to3.pgen2.token import OP
from typing import Any, Callable, Optional

from omegaconf import MISSING


@dataclass
class Scheduler:
    _target_: str = MISSING


@dataclass
class ConstantSchedule(Scheduler):
    _target_ = "optax.constant_schedule"
    value: float = MISSING


@dataclass
class ExponentialDecay(Scheduler):
    _target_: str = "optax.exponential_decay"
    init_value: float = MISSING
    transition_steps: int = MISSING
    decay_rate: float = MISSING
    transition_begin: int = 0
    staircase: bool = False
    end_value: Optional[float] = None


@dataclass
class ScaleBySchedule:
    _target_: str = "optax.scale_by_schedule"
    step_size_fn: Scheduler = MISSING


@dataclass
class ReplaceBySchedule:
    _target_: str = "fax.nn.optimizers.replace_by_schedule"
    step_size_fn: Scheduler = MISSING

@dataclass
class GradientStop:
    _target_: str = "fax.nn.utils.tree_stop_gradient_wrapper"
    labels_struct: Optional[dict[str, Any]] = None

@dataclass
class StateTransform:
    _target_: str = "fax.nn.utils.tree_to_transformed_states_wrapper"
    labels_struct: Optional[dict[str, Any]] = None
@dataclass
class Identity:
    _target_: str = "optax.identity"
    
@dataclass
class MultiTranform:
    _target_: str = "fax.nn.optimizers.multi_tranform_wrapper"
    transforms: Optional[dict[str, Any]] = None
    labels_struct: Optional[dict[str, Any]] = None
    
@dataclass
class Optimizer:
    _target_: str = MISSING
    learning_rate: Scheduler = MISSING

@dataclass
class SGD(Optimizer):
    _target_: str = "optax.sgd"
    momentum: Optional[float] = None
    nesterov: bool = False


@dataclass
class ADAMW(Optimizer):
    _target_: str = "optax.adamw"
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8
    eps_root: float = 0.0
    weight_decay: float = 1e-4

@dataclass
class AdaBelief(Optimizer):
    _target_: str = "optax.adabelief"
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-16
    eps_root: float = 1e-16

@dataclass
class NoisySgd(Optimizer):
    _target_: str = "optax.noisy_sgd"
    eta: float = 0.01
    gamma: float = 0.55
    seed: int = 0

@dataclass
class ClipByGlobalNorm:
    _target_: str = "optax.clip_by_global_norm"
    max_norm: float = MISSING
