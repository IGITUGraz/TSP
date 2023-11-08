from functools import partial
from typing import Callable

import chex
import jax.nn
from jax import numpy as jnp

from fax._src.rl.actor.q_lambda import batched_index
from fax._src.rl.utils import returns
from fax._src.nn import losses
from jax.scipy.special import entr
import optax

Array = chex.Array
PRNGKey = chex.PRNGKey


def create_policy(policy_params: dict, model: Callable):
    gamma = policy_params["gamma"]
    _lambda = policy_params["lambda"]
    temp = policy_params["temperature"]
    clip_param = policy_params["clip_param"]
    entropy_coef = policy_params["entropy_coef"]["init_value"]
    # entropy_schedule = optax.exponential_decay(**entropy_coef)
    importance_sampling = policy_params["importance_sampling"]
        
    def actor_policy(rng: PRNGKey, params: dict,
                     policy_state: dict,
                     obs_tm1, evaluation: bool):
        k1, k2, k3 = jax.random.split(rng, 3)
        policy_state, out = model(k1, params, policy_state, obs_tm1)
        v, logits = out
        v = jnp.squeeze(v)
        probs = jax.nn.softmax(logits)
        greedy_action = jnp.argmax(logits)
        soft_action = jax.random.choice(k2, logits.shape[0], (1,), p=probs)[0]
        action = jax.lax.select(evaluation, greedy_action, soft_action)
        return policy_state, (action, jnp.log(probs[action]), v, logits)

    def train_policy(rng: PRNGKey, params: dict, 
                     policy_state: dict, data: dict):
        # entropy_coef = entropy_schedule(step)
        partial_q_act = partial(model, rng, params)
        masks: jnp.ndarray = data["info"]["mask"]
        actions: jnp.ndarray = data["action"]
        action_logp: jnp.ndarray = data["action_logp"]
        rewards: jnp.ndarray = data["reward"]
        done: jnp.ndarray = data["done"]

        policy_state, out = jax.lax.scan(partial_q_act, policy_state, data["obs"])
        v, logits = out
        # v is (T, 1) transform to (T,)
        v = jnp.squeeze(v)
        a_tm1 = actions
        
        probs = jax.nn.softmax(logits)
        log_probs = jax.nn.log_softmax(logits)
        # control_log_probs_tm1 = action_logp[:-1]
        entropy = -jnp.sum(entr(probs), axis=-1)[:-1]
        
        log_pi_a = log_probs[jnp.arange(len(log_probs)), a_tm1]
        # pi_atm1 -= jax.lax.stop_gradient(control_log_probs_tm1)
        log_pi_atm1 = log_pi_a[:-1]
        r_t = rewards[1:]
        # 0 iff obs_t is t is terminal
        not_terminals = (1 - done)
        not_termninal_tm1 = not_terminals[:-1]
        gamma_t = gamma * not_terminals[1:]
        
        # qa_tm1 = batched_index(q_tm1, a_tm1)
        # On-policy
        # v_t = jax.lax.stop_gradient(jnp.concatenate(
        #      (batched_index(q_t[:-1], actions[2:]),
        #       jnp.expand_dims(jnp.max(q_t[-1], axis=-1), 0))))
        # off-policy
        # v_t = jax.lax.stop_gradient(jnp.max(q_t, axis=-1))

        adv_tm1 = returns.temporal_difference(
            r=r_t, gamma=gamma_t,
            v=jax.lax.stop_gradient(v))
        target_tm1 = returns.lambda_return(r=r_t, gamma=gamma_t,
                                           lambda_=_lambda,
                                           v=jax.lax.stop_gradient(v[1:]))
        adv_tm1 = jax.lax.stop_gradient(adv_tm1)
        td_errors_tm1 = jax.lax.stop_gradient(target_tm1) - v[:-1]
        global_mask = not_termninal_tm1 * masks[:-1]
        td_errors_tm1 *= jax.lax.stop_gradient(not_termninal_tm1)
        if importance_sampling:
            ratio = jnp.exp(log_pi_atm1 - jax.lax.stop_gradient(log_pi_atm1))
            clip_ratio = jnp.clip(ratio, 1 - clip_param, 1 + clip_param)
            loss_policy = -jnp.minimum(ratio*adv_tm1, clip_ratio*adv_tm1)
        # normal sampling
        else:
            loss_policy = -adv_tm1 * log_pi_atm1
        return_loss = jnp.sum(
            losses.l2_loss(td_errors_tm1)) / jax.lax.stop_gradient(jnp.sum(not_termninal_tm1))
        policy_loss = jnp.sum(
            (loss_policy + entropy_coef*entropy) * jax.lax.stop_gradient(global_mask)) / jax.lax.stop_gradient(jnp.sum(global_mask))
        loss = return_loss + policy_loss
        return policy_state, loss
    return actor_policy, train_policy
