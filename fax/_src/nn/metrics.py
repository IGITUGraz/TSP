from pathlib import Path
import shutil
import pickle
import jax
from matplotlib.pyplot import axis
# import mlflow
import numpy as np
from sklearn import metrics as skmetrics
from typing import Any, Callable, NamedTuple, Optional, Sequence, Union
import chex
from fax.config import flatten_dict
PyTree = Any
Shape = Sequence[int]
# Transformation states are (possibly empty) namedtuples.
MetricState = NamedTuple
# Parameters are arbitrary nests of `jnp.ndarrays`.
Params = chex.ArrayTree
Array = chex.Array
# Gradient updates are of the same type as parameters.
Updates = Params

TransformInitFn = Callable[
    [Params],
    Union[MetricState, Sequence[MetricState]]]
TransformUpdateFn = Callable[
    [Updates, MetricState], Union[MetricState, Sequence[MetricState]]]
TransformAggregateFn = Callable[[MetricState], dict]

NO_PARAMS_MSG = (
    'You are using a transformation that requires the current value of '
    'parameters, but you are not passing `params` when calling `update`.')


try:
    # aim was used for monitoring but for the publishing code we 
    # simplify monitoring to be stdout    
    from aim import Run, Text
    def aim_logging(
        aim_run: Run, data: dict, 
        step:int, prefix: str = "",
        file_path:str = "./data/aim_track_file/"):
        
        flat_dict = flatten_dict(data, separator="/", exclude_list=[])
        
        non_scalar_dict = {}
        for k, v in flat_dict.items():
            if np.ndim(v) == 0 and not isinstance(v, str):
                aim_run.track(float(v), name=k, step=step, context={"subset":prefix})
            else:
                non_scalar_dict[k] = v
        if non_scalar_dict:
            run_id = aim_run.hash
            exp_name = aim_run.experiment
            file_path = Path(file_path)
            file_path = (file_path / exp_name) / f"{run_id}_{step}.pkl"
            file_path.mkdir(exist_ok=True, parents=True)
            with open(file_path , "wb") as f_w:
                pickle.dump(non_scalar_dict, f_w)
            file_path_aim = Text(file_path.as_uri())
            aim_run.track(file_path_aim, name="non_scalar_path", step=step,
                        context={"subset":prefix})
except ImportError:
    pass

def reduce_temporal_dimension(data: np.ndarray):
    data_shape = data.shape
    reduced_data = data.reshape(
        (data_shape[0] * data_shape[1],) + data_shape[2:])
    return reduced_data


class MetricTransformation(NamedTuple):
    """
    Metric transformations consists of a function triple:
    (initialise, update, aggregate).
    """
    init: TransformInitFn
    update: TransformUpdateFn
    aggregate: TransformAggregateFn


class EmptyState(MetricState):
    """
    An empty state for the simplest stateless transformations.
    """


def identity() -> MetricTransformation:
    """
    Stateless identity transformation that leaves input gradients untouched.

    Returns:
        An (init_fn, update_fn) tuple.
    """

    def init_fn(_):
        return EmptyState()

    def update_fn(updates, state):
        return state

    def aggregate_fn(state):
        return {}

    return MetricTransformation(init_fn, update_fn, aggregate_fn)

class LogStepFunction(MetricState):
    func: Callable
    res: float

def log_step_function(key:str, func:Callable):
    def init_fn(_):
        return LogStepFunction(func=func, res=func(0))
    def update_fn(updates, states: LogStepFunction):
        step = updates["step"]
        res = states.func(step)
        return LogStepFunction(func=states.func, res=res)
    def aggregate_fn(states: LogStepFunction):
        return {key: states.res}
    return MetricTransformation(init_fn, update_fn, aggregate_fn)
    
            
class StoreState(MetricState):
    buffer: dict[str, list[Any]]
def store(fields: list[str]):
    def init_fn(_):
        return StoreState(buffer={f: [] for f in fields})
    def update_fn(updates, states: StoreState):
        buffer = states.buffer
        for k in buffer.keys():
            buffer[k].append(updates[k])
        return StoreState(buffer=buffer)
    def aggregate_fn(states: StoreState):
        buffer = states.buffer
        stacked_buffer = {
            k: jax.tree_map(lambda *x: np.stack(x, axis=0), *v) 
                          for k,v in buffer.items()} 
        return stacked_buffer
    return MetricTransformation(init_fn, update_fn, aggregate_fn)
        
class AveragingState(MetricState):
    buffer: list[Any]


def average(key: str) -> MetricTransformation:
    def init_fn(_):
        return AveragingState(buffer=[])

    def update_fn(updates, states):
        data = updates[key]
        prefix_dim = updates["prefix_dim"]
        if prefix_dim == 2:
            temporal_mask = updates.get("temporal_mask")
            mask = temporal_mask.astype(float) \
                if temporal_mask is not None else np.ones(len(data))
            data = data * mask
            data = data.sum(axis=1)
            batch_mean = data.sum() / mask.sum()
        else:
            batch_mean = data.mean(axis=0)
        average_buffer: list = states.buffer
        average_buffer.append(batch_mean)
        return AveragingState(buffer=average_buffer)

    def aggregate_fn(states):
        buffer_array = np.array(states.buffer)
        buffer_mean = buffer_array.mean(axis=0)
        return {f"mean_{key}": buffer_mean}

    # typing errors does not really make sense here
    return MetricTransformation(init_fn, update_fn, aggregate_fn)


class AccuracyState(MetricState):
    frequencies_count: int
    total_count: int


def accuracy(from_logits: bool) -> MetricTransformation:
    def init_fn(_):
        return AccuracyState(frequencies_count=0,
                             total_count=0)

    def update_fn(updates, states):
        predicted_class: np.ndarray = updates["pred"]
        targets: np.ndarray = updates["target"]

        prefix_dim = updates["prefix_dim"]
        if prefix_dim == 2:
            mask = updates.get("temporal_mask")
            predicted_class = reduce_temporal_dimension(predicted_class)
            targets = reduce_temporal_dimension(targets)
            if mask is not None:
                mask = reduce_temporal_dimension(mask)
                mask_idx = np.nonzero(mask)
                predicted_class = predicted_class[mask_idx]
                targets = targets[mask_idx]

        if from_logits:
            predicted_class = np.argmax(predicted_class, axis=-1)

        new_freq = int(skmetrics.accuracy_score(targets,
                                                predicted_class,
                                                normalize=False))
        new_count = predicted_class.shape[0]
        frequencies_count = states.frequencies_count + new_freq
        total_count = states.total_count + new_count
        return AccuracyState(frequencies_count, total_count)

    def aggregate_fn(states):
        return {"accuracy": states.frequencies_count / states.total_count}

    # typing errors does not really make sense here
    return MetricTransformation(init_fn, update_fn, aggregate_fn)


class ContextAccuracy(MetricState):
    context_freq: np.ndarray
    context_total: np.ndarray


def context_accuracy(from_logits: bool, context_id: str):
    context_id = context_id.split(",")
    context_map = {k:v for v,k in enumerate(context_id)}
    nb_context = len(context_id)
    def init_fn(_):
        return ContextAccuracy(
            context_freq=np.zeros((nb_context,), dtype=int),
            context_total=np.zeros((nb_context,), dtype=int)
            )

    def update_fn(updates, states):
        predicted_class: np.ndarray = updates["pred"]
        targets: np.ndarray = updates["target"]
        contexts: np.ndarray = updates["context"]
        prefix_dim = updates["prefix_dim"]
        if prefix_dim == 2:
            mask = updates.get("temporal_mask")
            predicted_class = reduce_temporal_dimension(predicted_class)
            targets = reduce_temporal_dimension(targets)
            contexts = reduce_temporal_dimension(contexts)
            if mask is not None:
                mask = reduce_temporal_dimension(mask)
                mask_idx = np.nonzero(mask)
                predicted_class = predicted_class[mask_idx]
                targets = targets[mask_idx]
                contexts = contexts[mask_idx]
        if from_logits:
            predicted_class = np.argmax(predicted_class, axis=-1)
        new_context_freq = states.context_freq
        new_context_total = states.context_total
        contexts_collections: list[tuple[list, list]] = [
            ([], []) for _ in range(nb_context)]
        for i in range(len(predicted_class)):            
            context_id = context_map[str(contexts[i])]
            contexts_collections[context_id][0].append(targets[i])
            contexts_collections[context_id][1].append(predicted_class[i])
        for i, (x, y) in enumerate(contexts_collections):
            x = np.array(x)
            y = np.array(y)
            # if x and y are empty accuracy_score return 0
            new_context_freq[i] += int(
                skmetrics.accuracy_score(x, y, normalize=False))
            new_context_total[i] += x.shape[0]
        return ContextAccuracy(new_context_freq, new_context_total)

    def aggregate_fn(states):
        context_res = {}
        for k, v in context_map.items():
            if states.context_total[v] != 0:
                context_res[str(k)] = \
                    states.context_freq[v] / states.context_total[v]
        return {"context_accuracy": context_res}

        # typing errors does not really make sense here

    return MetricTransformation(init_fn, update_fn, aggregate_fn)


class ConfusionMatrixState(MetricState):
    conf_m: np.ndarray


def confusion_matrix(from_logits: bool, nb_classes: int) -> \
        MetricTransformation:
    def init_fn(_):
        conf_m = np.zeros((nb_classes, nb_classes), dtype=int)
        return ConfusionMatrixState(conf_m)

    def update_fn(updates, states: ConfusionMatrixState):
        predicted_class: np.ndarray = updates["pred"]
        targets: np.ndarray = updates["target"]
        prefix_dim = updates["prefix_dim"]
        if prefix_dim == 2:
            mask = updates.get("temporal_mask")
            predicted_class = reduce_temporal_dimension(predicted_class)
            targets = reduce_temporal_dimension(targets)
            if mask is not None:
                mask = reduce_temporal_dimension(mask)
                mask_idx = np.nonzero(mask)
                predicted_class = predicted_class[mask_idx]
                targets = targets[mask_idx]
        if from_logits:
            predicted_class = np.argmax(predicted_class, axis=-1)
        new_conf_m = skmetrics.confusion_matrix(targets,
                                                predicted_class,
                                                labels=np.arange(0, nb_classes)
                                                )
        return ConfusionMatrixState(states.conf_m + new_conf_m)

    def aggregate_fn(states):
        return {"confusion_matrix": states.conf_m}

    return MetricTransformation(init_fn, update_fn, aggregate_fn)


class MultiContextConfusionMatrixState(MetricState):
    context_conf_m: np.ndarray


def context_confusion_matrix(from_logits: bool,
                             nb_context: int,
                             nb_classes: int
                             ) -> MetricTransformation:
    def init_fn(_):
        context_conf_m = np.zeros(
            (nb_context, nb_classes, nb_classes), dtype=int)
        return MultiContextConfusionMatrixState(context_conf_m)

    def update_fn(updates, states: MultiContextConfusionMatrixState):
        predicted_class: np.ndarray = updates["pred"]
        targets: np.ndarray = updates["target"]
        contexts: np.ndarray = updates["context"]
        prefix_dim = updates["prefix_dim"]
        if prefix_dim == 2:
            mask = updates.get("temporal_mask")
            predicted_class = reduce_temporal_dimension(predicted_class)
            targets = reduce_temporal_dimension(targets)
            contexts = reduce_temporal_dimension(contexts)
            if mask is not None:
                mask = reduce_temporal_dimension(mask)
                mask_idx = np.nonzero(mask)
                predicted_class = predicted_class[mask_idx]
                targets = targets[mask_idx]
                contexts = contexts[mask_idx]
        if from_logits:
            predicted_class = np.argmax(predicted_class, axis=-1)
        new_context_cm = states.context_conf_m
        labels = np.arange(0, nb_classes)
        for i in range(len(predicted_class)):
            # if x and y are empty confusion_matrix return 0-filled matrix
            new_context_cm[contexts[i]] += skmetrics.confusion_matrix(
                targets[i],
                predicted_class[i],
                labels=labels)
        return MultiContextConfusionMatrixState(new_context_cm)

    def aggregate_fn(states):
        context_cm = states.context_conf_m
        contexts_results = {}
        for i in range(nb_context):
            contexts_results[f"c_{i}"] = context_cm[i]
        return {"context_confusion_matrix": contexts_results}

    return MetricTransformation(init_fn, update_fn, aggregate_fn)


def chain(
        *args: MetricTransformation
) -> MetricTransformation:
    init_fns, update_fns, aggregate_fns = zip(*args)

    def init_fn(params):
        return [fn(params) for fn in init_fns]

    def update_fn(updates, states):
        if len(update_fns) != len(states):
            raise ValueError("The number of updates "
                             "and states has to be the same in "
                             "chain! Make sure you have called init first!")

        new_state = []
        prefix_dim = updates.get("prefix_dim")
        updates["prefix_dim"] = prefix_dim if prefix_dim else 1
        for s, fn in zip(states, update_fns):
            new_s = fn(updates, s)
            new_state.append(new_s)
        return new_state

    def aggregate_fn(states):
        if len(update_fns) != len(states):
            raise ValueError("The number of updates and states"
                             " has to be the same in "
                             "chain! Make sure you have called init first!")

        metric_outs = {}
        for s, fn in zip(states, aggregate_fns):
            metric_out = fn(s)
            metric_outs |= metric_out
        return metric_outs

    return MetricTransformation(init_fn, update_fn, aggregate_fn)


# def mlflow_logging(data: dict, step: int, prefix: str = "", 
#                    tmp_path: str = "./data/tmp/"):
#     """
#     mlflow logger helper, split data to log between scalar a
#     nd non-scalar metrics while adding a possible prefix,
#     scalar will be use to call  mlflow.log_metrics and non-scalar
#     mlflow.log_dict with path
#     $run$/non_scalar_metrics/{step}.json
#     Args:
#         step:
#         prefix:
#         data:

#     Returns:

#     """
#     flat_dict = flatten_dict(data, separator="/", exclude_list=[])
#     scalar_dict = {}
#     non_scalar_dict = {}
#     for k, v in flat_dict.items():
#         if np.ndim(v) == 0 and not isinstance(v, str):
#             scalar_dict[f"{prefix}/{k}"] = float(v)
#         else:
#             non_scalar_dict[f"{prefix}/{k}"] = v
#     mlflow.log_metrics(scalar_dict, step)
#     if non_scalar_dict:
#         run = mlflow.active_run()
#         run_id = run.info.run_id
#         tmp_path = Path(tmp_path)
#         run_path = tmp_path / f"{run_id}/{step}"
#         print(run_path)
#         run_path.mkdir(exist_ok=True, parents=True)
#         with open(run_path / "data.pkl", "wb") as f_w:
#             pickle.dump(non_scalar_dict, f_w)
        
#         mlflow.log_artifact(run_path, artifact_path="non_scalar_metrics")
def artifact_cleanup(path: str = "./data/tmp/"):
    path: Path = Path(path)
    if path.exists():
        print(f"clean file tree at {path} ...")
        shutil.rmtree(path)
        print("cleaning completed")
    

    