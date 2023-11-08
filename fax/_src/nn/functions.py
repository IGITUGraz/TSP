from collections.abc import Callable
from typing import Union

from chex import Array
from jax.nn import *


def linear(x: Array) -> Array:
    return x


_functions_dispatch: dict[str, Callable] = {
    "relu": relu,
    "softplus": softplus,
    "soft_sign": soft_sign,
    "sigmoid": sigmoid,
    "silu": silu,
    "log_sigmoid": log_sigmoid,
    "elu": elu,
    "leaky_relu": leaky_relu,
    "hard_tanh": hard_tanh,
    "celu": celu,
    "selu": selu,
    "gelu": gelu,
    "glu": glu,
    "logsumexp": logsumexp,
    "log_softmax": log_softmax,
    "normalize": normalize,
    "relu6": relu6,
    "hard_sigmoid": hard_sigmoid,
    "hard_silu": hard_silu,
    "hard_swish": hard_swish,
    "linear": linear,
    "one_hot": one_hot,
    "tanh": tanh,
}


def get(identifier: Union[str, Callable]) -> Callable:
    if isinstance(identifier, str):
        try:
            return _functions_dispatch[identifier]
        except KeyError:
            valid_ids_msg = "\n".join(_functions_dispatch.keys())
            print(f"{identifier} does not exist in the lookup table \n"
                  f"valid identifier are:\n {valid_ids_msg}")
    elif isinstance(identifier, Callable):
        return identifier
