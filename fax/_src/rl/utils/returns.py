# https://github.com/deepmind/rlax/blob/3566d85af6d41abba82c20828e1e50f05bc2a559/rlax/_src/multistep.py
from typing import Optional
import chex
import jax
from jax import numpy as jnp
from jax import lax
from numpy import array

Numeric = chex.Numeric
Array = chex.Array
Scalar = chex.Scalar


def truncated_generalized_advantage_estimation(
        r: Array, gamma: Array, lambda_: Numeric, v: Array) -> Array:
    lambda_arr = jnp.ones_like(gamma) * lambda_
    delta_td = r + gamma * v[1:] - v[:-1]

    def step_adv(adv_tp1: Scalar, data: tuple[Scalar, Scalar, Scalar]):
        gamma_t, lambda_t, delta_t = data
        adv_t = delta_t + gamma_t * lambda_t * adv_tp1
        return adv_t, adv_t
    
    advantages = lax.scan(step_adv, 0.,
                          (gamma, lambda_arr, delta_td), reverse=True)[1]
    return advantages[::]

def temporal_difference(r: Array, gamma: Array, v: Array):
    delta_td = r + gamma * v[1:] - v[:-1]
    return delta_td

def lambda_return(r: Array, gamma: Array, lambda_: Numeric, v: Array) -> Array:
    lambda_arr = jnp.ones_like(gamma) * lambda_

    
    def step_lr(u_tp1: Scalar, data: tuple[Scalar, Scalar, Scalar, Scalar]):
        gamma_t, lambda_t, r_t, v_t = data
        u_t = r_t + gamma_t * (lambda_t * u_tp1 + (1 - lambda_t) * v_t)
        return u_t, u_t
    u = lax.scan(step_lr, v[-1], (gamma, lambda_arr, r, v), reverse=True)[1]
    return u[::]


def n_step_bootstrapped_returns(
        r_t: Array,
        discount_t: Array,
        v_t: Array,
        n: int,
        lambda_t: Numeric = 1.) -> Array:
    seq_len = r_t.shape[0]

    # Maybe change scalar lambda to an array.
    lambda_t = jnp.ones_like(discount_t) * lambda_t

    # Shift bootstrap values by n and pad end of sequence with last value v_t[-1].
    pad_size = min(n - 1, seq_len)
    targets = jnp.concatenate([v_t[n - 1:], jnp.array([v_t[-1]] * pad_size)])

    # Pad sequences. Shape is now (T + n - 1,).
    r_t = jnp.concatenate([r_t, jnp.zeros(n - 1)])
    discount_t = jnp.concatenate([discount_t, jnp.ones(n - 1)])
    lambda_t = jnp.concatenate([lambda_t, jnp.ones(n - 1)])
    v_t = jnp.concatenate([v_t, jnp.array([v_t[-1]] * (n - 1))])

    # Work backwards to compute n-step returns.
    for i in reversed(range(n)):
        r_ = r_t[i:i + seq_len]
        discount_ = discount_t[i:i + seq_len]
        lambda_ = lambda_t[i:i + seq_len]
        v_ = v_t[i:i + seq_len]
        targets = r_ + discount_ * ((1. - lambda_) * v_ + lambda_ * targets)

    return targets
