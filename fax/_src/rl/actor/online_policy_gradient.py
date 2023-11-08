import copy
import imp
from typing import Callable
from chex import PRNGKey
import jax
from jax import numpy as jnp
from fax.nn.models import vectorized_model
from fax.rl.types import VectorWrapper, ObsType
from fax.nn import losses
from jax.scipy.special import entr
import optax
def create_policy(policy_params: dict, model: Callable):
    l2_loss = losses.get("l2")
    gamma = policy_params["gamma"]
    _lambda = policy_params["lambda"]
    temp = policy_params["temperature"]
    clip_param = policy_params["clip_param"]
    entropy_coef = policy_params["entropy_coef"]
    supervised = policy_params.get("supervised", False)
    
    entropy_schedule = optax.exponential_decay(**entropy_coef)
    importance_sampling = policy_params["importance_sampling"]
    vectorize_model = policy_params["vectorize_model"]
    if vectorize_model:
        model = vectorized_model(
            model, {"in_axes": (0, 0, 0), "out_axes": (0, 0)})
        
    def actor_policy_batch(rng: PRNGKey, params: dict,
                     policy_state: dict, obs_tm1):
        policy_state, out = model(rng, params, policy_state, obs_tm1)
        v, logits = out
        v = jnp.squeeze(v)
        probs = jax.nn.softmax(logits, axis=-1)
        action = jnp.argmax(logits, axis=-1)
        return policy_state, (action, probs[jnp.arange(len(logits)), action], v, logits)
    
    def actor_policy(rng: PRNGKey, params: dict,
                     policy_state: dict, obs_tm1):
        policy_state, out = model(rng, params, policy_state, obs_tm1)
        v, logits = out
        v = jnp.squeeze(v)
        probs = jax.nn.softmax(logits)
        action = jnp.argmax(logits)
        return policy_state, (action, probs[action], v, logits)
    def local_derivation(rng, params, policy_state, obs_tm1):
        k1, k2, k3 = jax.random.split(rng, 3)
        states_tm1, out = model(k1, params, policy_state, obs_tm1)
        v_tm1, logits_tm1 = out
        v_tm1 = jnp.squeeze(v_tm1)
        # act_tm1 = jax.random.choice(k2, pi_tm1.shape[0], (1,), p=pi_tm1)[0]
        return (v_tm1, logits_tm1), states_tm1
    @jax.custom_vjp
    def before_loss(rng, params, policy_state, obs_tm1):
        res, vjpfun, aux = jax.vjp(
            local_derivation, rng, params, policy_state, obs_tm1, has_aux=True)
        v_tm1, logtis_tm1 = res
        states_tm1 = aux
        eligibity_v_t = vjpfun((1.0, jnp.zeros_like(logtis_tm1, dtype=logtis_tm1.dtype)))[1]
        eligibity_p_t = vjpfun((0.0, jnp.ones_like(logtis_tm1, dtype=logtis_tm1.dtype)))[1]
        states_tm1["delta_value"] = jax.tree_map(
            lambda x, y: _lambda*x + y, 
            policy_state["delta_value"], eligibity_v_t)
        states_tm1["delta_policy"] = jax.tree_map(
            lambda x, y: _lambda*x + y, 
            policy_state["delta_policy"], eligibity_p_t)

        return (v_tm1, logtis_tm1), states_tm1
    def fwd_before_loss(rng, params, policy_state, obs_tm1):
        res, aux = before_loss(rng, params, policy_state, obs_tm1)
        v_tm1, log_pa_tm1 = res
        states_tm1 = aux
        return (v_tm1, log_pa_tm1), states_tm1,\
            (states_tm1["delta_value"], states_tm1["delta_policy"])
    def bwd_before_loss(res, g):
        delta_value, delta_policy = res
        (l_v, l_p), (_, _) = g
        print(l_p)
        grad_params = jax.tree_map(lambda x, y: l_v*x + l_p *y,
                                   delta_value, delta_policy)
        return None, grad_params, None, None
    before_loss.defvjp(fwd_before_loss, bwd_before_loss)
    
    def train_policy_batch(env_vect_fun, rng, params, env_states, policy_state, pi_tm1_prime, obs_tm1, step):
        entropy_coef = entropy_schedule(step)
        first_key = rng[0]
        k1, k2, *batch_keys = jax.random.split(first_key, len(pi_tm1_prime) + 2)
        batch_keys = jnp.array(batch_keys)
        states_tm1, out = model(batch_keys, params, policy_state, obs_tm1)
        v_tm1, logits_tm1 = out
        v_tm1 = jnp.squeeze(v_tm1)
        # states_tm1["delta_value"] = policy_state["delta_value"]
        # states_tm1["delta_policy"] = policy_state["delta_policy"]
        pi_tm1 = jax.nn.softmax(logits_tm1, axis=-1)
        act_tm1 = jax.random.categorical(k2, logits_tm1, axis=-1)
        # act_tm1 = jax.random.choice(k2, pi_tm1.shape[0], (1,), p=pi_tm1)[0]
        new_env_state, new_actor_state = env_vect_fun(env_states, act_tm1)
        target = new_actor_state.info["target"]
        mask = obs_tm1[1] == ObsType.recall
        done = new_actor_state.done
        new_obs = new_actor_state.obs
        r_t = new_actor_state.reward
        # copy_states_tm1 = copy.deepcopy(states_tm1)
        k1, *batch_keys = jax.random.split(k1, len(pi_tm1_prime) + 1)
        batch_keys = jnp.array(batch_keys)
        _, out = model(
            batch_keys, 
            jax.lax.stop_gradient(params), 
            jax.lax.stop_gradient(states_tm1), 
            jax.lax.stop_gradient(new_obs)
            )
        v_t, logits_t = out
        entropy = -jnp.sum(entr(pi_tm1), axis=-1)
        pi_t = jax.nn.log_softmax(logits_t, axis=-1)
        
        v_t = jnp.squeeze(v_t)
        loss_return = l2_loss(
            jax.lax.stop_gradient(
                r_t + gamma*(1-done)*v_t) - v_tm1)
        td_t = jax.lax.stop_gradient(
            r_t + gamma*(1-done)*v_t - v_tm1)
        # importance sampling
        if importance_sampling:
            log_p_tm1 = jax.nn.log_softmax(logits_tm1, axis=-1)
            log_pa_tm1 = log_p_tm1[jnp.arange(len(pi_tm1)), act_tm1]
            ratio = jnp.exp(log_pa_tm1 - jax.lax.stop_gradient(
                pi_tm1_prime[jnp.arange(len(pi_tm1)), act_tm1]))
            clip_ratio = jnp.clip(ratio, 1 - clip_param, 1 + clip_param)
            loss_policy = -jnp.minimum(ratio*td_t, clip_ratio*td_t)
        # normal sampling
        else:
            loss_policy = -td_t * pi_tm1[jnp.arange(len(pi_tm1)), act_tm1]
        total_loss = loss_return + mask*(loss_policy + entropy_coef*entropy)
        if supervised:
            total_loss = -1.0 * mask*pi_t[jnp.arange(len(pi_tm1)), target]
        return total_loss, (states_tm1, new_env_state, new_obs,done, pi_t)
    
    def train_policy(env_step_fun, rng, params, env_state, policy_state, pi_tm1_prime, obs_tm1, step):
        k1, k2 = jax.random.split(rng, 2)
        entropy_coef = entropy_schedule(step)
        # (v_tm1, log_pa_tm1), (act_tm1, states_tm1) =\
        #     before_loss(k1, params, policy_state, obs_tm1)
        (v_tm1, logits_tm1), states_tm1 =\
            local_derivation(k1, params, policy_state, obs_tm1)    
        # states_tm1["delta_value"] = policy_state["delta_value"]
        # states_tm1["delta_policy"] = policy_state["delta_policy"]
        
        pi_tm1 = jax.nn.softmax(logits_tm1)
        act_tm1 = jax.random.choice(k2, pi_tm1.shape[0], (1,), p=pi_tm1)[0]
        new_env_state, new_actor_state = env_step_fun(env_state, act_tm1)
        target = new_actor_state.info["target"]
        mask = obs_tm1[1] == ObsType.recall
        done = new_actor_state.done
        new_obs = new_actor_state.obs
        r_t = new_actor_state.reward
        # copy_states_tm1 = copy.deepcopy(states_tm1)
        _, out = model(
            k2, 
            jax.lax.stop_gradient(params), 
            jax.lax.stop_gradient(states_tm1), 
            jax.lax.stop_gradient(new_obs)
            )
        v_t, logits_t = out
        entropy = -jnp.sum(entr(pi_tm1))
        pi_t = jax.nn.log_softmax(logits_t)
        
        v_t = jnp.squeeze(v_t)
        loss_return = l2_loss(
            jax.lax.stop_gradient(
                r_t + gamma*(1-done)*v_t) - v_tm1)
        td_t = jax.lax.stop_gradient(
            r_t + gamma*(1-done)*v_t - v_tm1)
        # importance sampling
        if importance_sampling:
            log_p_tm1 = jax.nn.log_softmax(logits_tm1)
            log_pa_tm1 = log_p_tm1[act_tm1]
            ratio = jnp.exp(log_pa_tm1 - jax.lax.stop_gradient(pi_tm1_prime[act_tm1]))
            clip_ratio = jnp.clip(ratio, 1 - clip_param, 1 + clip_param)
            loss_policy = -jnp.minimum(ratio*td_t, clip_ratio*td_t)
        # normal sampling
        else:
            loss_policy = -td_t * pi_tm1[act_tm1]
        total_loss = loss_return + mask*(loss_policy + entropy_coef*entropy)
        if supervised:
            total_loss = -1.0 * mask*pi_t[target]
        return total_loss, (states_tm1, new_env_state, new_obs,done, pi_t)
    if vectorize_model:
        return actor_policy_batch, train_policy_batch
    else:
        return actor_policy, train_policy