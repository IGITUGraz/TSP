from typing import Callable

import chex
import jax.nn
from jax import numpy as jnp

from fax.rl.types import ObsType

Array = chex.Array
PRNGKey = chex.PRNGKey


def create_policy(policy_params: dict, model: Callable):
    def actor_policy(rng: PRNGKey, params: dict,
                     policy_state: dict, obs_tm1, evaluation: bool):
        k1, k2, k3 = jax.random.split(rng, 3)
        policy_state, q_values = model(k1, params, policy_state, obs_tm1)
        probs = jax.nn.softmax(q_values / policy_params["temperature"])
        greedy_action = jnp.argmax(q_values)
        soft_action = jax.random.choice(
            k2, q_values.shape[0], (1,), p=probs)[0]
        action = jax.lax.select(evaluation, greedy_action, soft_action)
        return policy_state, action, \
            jnp.log(probs[action]), q_values[action], q_values

    def train_policy(rng: PRNGKey, params: dict,
                     policy_state: dict, data: dict) -> tuple[float, dict]:
        k1, k2 = jax.random.split(rng, 2)
        obs_tm1, act_tm1, obs_t = \
            data["obs_tm1"], data["action"], data["obs_t"]
        r_t = obs_t.reward
        discount_t = policy_params["discount"] * (
            not obs_t.type == ObsType.terminal.value)
        policy_state, q_tm1 = model(k1, params, policy_state, obs_tm1)
        policy_state, q_t = model(k2, params, policy_state, obs_t)
        target_tm1 = r_t + discount_t * jnp.max(q_t)
        td_error = 0.5 * (
            jax.lax.stop_gradient(target_tm1) - q_tm1[act_tm1]) ** 2
        td_error *= obs_tm1.type == ObsType.recall.value
        return policy_state, td_error

    return actor_policy, train_policy
