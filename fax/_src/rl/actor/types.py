from typing import NamedTuple

from jax import numpy as jnp


class Qaction(NamedTuple):
    action: int
    pi: jnp.ndarray
    q_values: jnp.ndarray


