from typing import Union, Callable

import chex
import jax
from jax import numpy as jnp
from optax import (l2_loss, huber_loss, smooth_labels, sigmoid_binary_cross_entropy,
                   cosine_similarity, cosine_distance, log_cosh,
                   )


def cross_entropy(
        preds: chex.Array,
        labels: chex.Array,
        from_logits: bool = True
) -> chex.Array:
    chex.assert_type([preds], float)
    if from_logits:
        preds = jax.nn.log_softmax(preds, axis=-1)
    else:
        preds = jnp.log(preds, axis=-1)

    return -jnp.sum(labels * preds, axis=-1)


def sparse_cross_entropy(
        preds: chex.Array,
        labels: chex.Array,
        from_logits: bool = True
) -> chex.Array:
    # assuming here that preds are logits
    chex.assert_type([preds], float)
    probs = jax.nn.log_softmax(preds, axis=-1)
    return -jnp.take_along_axis(probs, jnp.expand_dims(labels, 1), axis=-1)


_losses_dispatch = {
    "l2": l2_loss,
    "huber": huber_loss,
    "smooth_labels": smooth_labels,
    "sigmoid_binary_cross_entropy": sigmoid_binary_cross_entropy,
    "cosine_similarity": cosine_similarity,
    "cosine_distance": cosine_distance,
    "log_cosh": log_cosh,
    "cross_entropy": cross_entropy,
    "sparse_cross_entropy": sparse_cross_entropy,
}


def get(identifier: Union[str, Callable]):
    if isinstance(identifier, str):
        try:
            return _losses_dispatch[identifier]
        except KeyError:
            valid_ids_msg = "\n".join(_losses_dispatch.keys())
            print(f"{identifier} does not exist in the lookup table \n"
                  f"valid identifier are:\n {valid_ids_msg}")
    elif isinstance(identifier, Callable):
        return identifier
