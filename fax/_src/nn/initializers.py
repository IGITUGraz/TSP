from typing import Union, Callable

import jax.numpy as jnp
from jax.nn.initializers import *


def one_hot(key, shape, dtype=jnp.float32):
    row_l, *col_l = shape
    if not col_l:
        col_l = None
    else:
        col_l = col_l[0]
    return jnp.eye(N=row_l, M=col_l, dtype=dtype)


_initializers_dispatch: dict[str, Callable] = {
    "delta_orthogonal": delta_orthogonal,
    "glorot_normal": glorot_normal,
    "glorot_uniform": glorot_uniform,
    "he_normal": he_normal,
    "he_uniform": he_uniform,
    "kaiming_normal": kaiming_normal,
    "kaiming_uniform": kaiming_uniform,
    "lecun_normal": lecun_normal,
    "lecun_uniform": lecun_uniform,
    "normal": normal,
    "ones": ones,
    "orthogonal": orthogonal,
    "relu_orthogonal": orthogonal,
    "uniform": uniform,
    "variance_scaling": variance_scaling,
    "xavier_normal": xavier_normal,
    "xavier_uniform": xavier_uniform,
    "zeros": zeros,
    "one_hot": one_hot,
}


def get(identifier: Union[str, Callable]) -> Callable:
    if isinstance(identifier, str):
        try:
            return _initializers_dispatch[identifier]
        except KeyError:
            valid_ids_msg = "\n".join(_initializers_dispatch.keys())
            print(f"{identifier} does not exist in the lookup table \n"
                  f"valid identifier are:\n {valid_ids_msg}")
    elif isinstance(identifier, Callable):
        return identifier
