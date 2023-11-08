from platform import node
from hydra.core.config_store import ConfigStore

from .model import LayerConfig, ModelConfig
from .task import BabiTask, AssociatedPairs, BabiTaskRL, FearConditioning, MatchingPairs, RadialMaze
from .training import ExponentialDecay, ConstantSchedule, SGD, ADAMW, AdaBelief, ClipByGlobalNorm, GradientStop, Identity, NoisySgd, MultiTranform, ReplaceBySchedule, ScaleBySchedule, StateTransform
from .metric import Average, Accuracy, ContextAccuracy, ConfusionMatrix,\
    ContextConfusionMatrix, LogStepFunction, Store, MetricConfig

def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(group="layer", name="base_layer", node=LayerConfig)
    cs.store(group="model", name="base_model", node=ModelConfig)
    cs.store(group="task", name="babi", node=BabiTask)
    cs.store(group="task", name="fear_cond", node=FearConditioning)
    cs.store(group="training/scheduler", name="exp_scheduler", node=ExponentialDecay)
    cs.store(group="training/scheduler", name="const_scheduler", node=ConstantSchedule)
    cs.store(group="training/optimizer", name="adamw", node=ADAMW)
    cs.store(group="training/optimizer", name="sgd", node=SGD)
    cs.store(group="training/optimizer", name="multi_transform", node=MultiTranform)
    cs.store(group="training/optimizer", name="adabelief", node=AdaBelief)
    cs.store(group="training/optimizer", name="noisy_sgd", node=NoisySgd)
    cs.store(group="training/optimizer",
             name="global_norm_clip", node=ClipByGlobalNorm)
    cs.store(group="training/optimizer",
            name="identity", node=Identity)
    cs.store(group="training/optimizer", name="scale_by_schedule", node=ScaleBySchedule)
    cs.store(group="training/optimizer", name="replace_by_schedule", node=ReplaceBySchedule)
    cs.store(group="training/transform", name="gradient_stop", 
             node=GradientStop)
    cs.store(group="training/transform", name="state_transform",
             node=StateTransform)
    cs.store(group="task", name="rl_ap", node=AssociatedPairs)
    cs.store(group="task", name="rl_mp", node=MatchingPairs)
    cs.store(group="task", name="rl_radial", node=RadialMaze)
    cs.store(group="task", name="rl_babitask", node=BabiTaskRL)
    cs.store(group="metric", name="average", node=Average)
    cs.store(group="metric", name="accuracy", node=Accuracy)
    cs.store(group="metric", name="context_accuracy", node=ContextAccuracy)
    cs.store(group="metric", name="confusion_matrix", node=ConfusionMatrix)
    cs.store(group="metric", name="context_confusion_matrix", node=ContextConfusionMatrix)
    cs.store(group="metric", name="context_confusion_matrix", node=ContextConfusionMatrix)
    cs.store(group="metric", name="store", node=Store)
    cs.store(group="metric", name="log_step_func", node=LogStepFunction)
    
