from functools import partial
from typing import Callable

import chex
import fax._src.nn.losses
import jax.nn
from jax import numpy as jnp

from fax.rl.utils import returns

Array = chex.Array
PRNGKey = chex.PRNGKey


def batched_index(
        values: Array, indices: Array, keepdims: bool = False
) -> Array:
    """Index into the last dimension of a tensor, preserving all others dims.
  Args:
    values: a tensor of shape [..., D],
    indices: indices of shape [...].
    keepdims: whether to keep the final dimension.
  Returns:
    a tensor of shape [...] or [..., 1].
  """
    indexed = jnp.take_along_axis(values, indices[..., None], axis=-1)
    if not keepdims:
        indexed = jnp.squeeze(indexed, axis=-1)
    return indexed


def create_policy(policy_params: dict, model: Callable):
    def actor_policy(rng: PRNGKey, params: dict,
                     policy_state: dict, obs_tm1, evaluation: bool):
        k1, k2, k3 = jax.random.split(rng, 3)
        policy_state, q_values = model(k1, params, policy_state, obs_tm1)
        q_values = q_values[0]
        probs = jax.nn.softmax(q_values / policy_params["temperature"])
        greedy_action = jnp.argmax(q_values)
        soft_action = jax.random.choice(
          k2, q_values.shape[0], (1,), p=probs)[0]
        action = jax.lax.select(evaluation, greedy_action, soft_action)
        # state, action, action_logp, v, action_predictor
        return policy_state, action, \
            jnp.log(probs[action]), q_values[action], q_values

    def train_policy(rng: PRNGKey,
                     params: dict, policy_state: dict, data: dict):
        partial_q_act = partial(model, rng, params)
        masks = data["info"]["mask"]
        actions = data["action"]
        rewards = data["reward"]
        done = data["done"]

        policy_state, out = jax.lax.scan(
          partial_q_act, policy_state, data["obs"])
        q_values, = out

        a_tm1 = actions[:-1]    
        q_tm1 = q_values[:-1]
        q_t = q_values[1:]
        r_t = rewards[1:]
        # 0 iff obs_t is t is terminal
        not_terminals = (1 - done)
        not_termninal_tm1 = not_terminals[:-1]
        gamma_t = policy_params["gamma"] * not_terminals[1:]
        lambda_t = policy_params["lambda"]

        qa_tm1 = batched_index(q_tm1, a_tm1)
        # On-policy
        # v_t = jax.lax.stop_gradient(jnp.concatenate(
        #      (batched_index(q_t[:-1], actions[2:]),
        #       jnp.expand_dims(jnp.max(q_t[-1], axis=-1), 0))))
        # off-policy
        v_t = jax.lax.stop_gradient(jnp.max(q_t, axis=-1))
        target_tm1 = returns.lambda_return(
          r=r_t, gamma=gamma_t, lambda_=lambda_t, v=v_t)
        td_errors_tm1 = jax.lax.stop_gradient(target_tm1) - qa_tm1
        global_mask = not_termninal_tm1 * masks[:-1]
        td_errors_tm1 *= jax.lax.stop_gradient(global_mask)
        loss = jnp.sum(
          fax._src.nn.losses.l2_loss(td_errors_tm1)) / jax.lax.stop_gradient(
            jnp.sum(global_mask))
        return policy_state, loss

    return actor_policy, train_policy