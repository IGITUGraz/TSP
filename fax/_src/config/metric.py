from dataclasses import dataclass
from typing import Any, Optional
from omegaconf import MISSING


@dataclass
class MetricConfig:
    _target_: str = MISSING

@dataclass
class Store(MetricConfig):
    _target_: str = "fax.nn.metrics.store"
    fields: list[str] = MISSING
    
@dataclass
class Accuracy(MetricConfig):
    _target_: str = "fax.nn.metrics.accuracy"
    from_logits: bool = MISSING
    
@dataclass
class LogStepFunction(MetricConfig):
    _target_: str = "fax.nn.metrics.log_step_function"
    key: str = MISSING
    func: Any = MISSING

@dataclass
class ContextAccuracy(MetricConfig):
    _target_: str = "fax.nn.metrics.context_accuracy"
    from_logits: bool = MISSING
    context_id: Any = None


@dataclass
class Average(MetricConfig):
    _target_: str = "fax.nn.metrics.average"
    key: str = MISSING


@dataclass
class ConfusionMatrix(MetricConfig):
    _target_ = "fax.nn.metrics.confusion_matrix"
    from_logits: bool = MISSING
    nb_classes: int = MISSING


@dataclass
class ContextConfusionMatrix(MetricConfig):
    _target_ = "fax.nn.metrics.context_confusion_matrix"
    from_logits: bool = True
    nb_classes: int = MISSING
    nb_context: int = MISSING
