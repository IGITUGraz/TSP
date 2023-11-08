from typing import Hashable
import jax
from typing import Mapping
from optax._src import base, numerics
from optax import ScaleByScheduleState, multi_transform
from fax.data.utils import label_struct_to_label_function
from jax import numpy as jnp

def replace_by_schedule(
    step_size_fn: base.Schedule
) -> base.GradientTransformation:
  """Scale updates using a custom schedule for the `step_size`.

  Args:
    step_size_fn: a function that takes an update count as input and proposes
      the step_size to multiply the updates by.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(_):
    return ScaleByScheduleState(count=jnp.zeros([], jnp.int32))

  def update_fn(updates, state, params=None):
    del params
    step_size = step_size_fn(state.count)
    updates = jax.tree_map(
        lambda g: jnp.ones_like(g, dtype=g.dtype) * jnp.array(step_size, dtype=g.dtype), updates)
    return updates, ScaleByScheduleState(
        count=numerics.safe_int32_increment(state.count))

  return base.GradientTransformation(init_fn, update_fn)

def multi_tranform_wrapper(
    transforms: Mapping[Hashable, base.GradientTransformation],
    labels_struct):
    
    if transforms.get("fallback") is None:
        raise ValueError(
            f"Multi-transform optimizer must contain"
            f" a valid fallback optimization method")
    label_func = label_struct_to_label_function(labels_struct)
    return multi_transform(transforms, label_func)

        