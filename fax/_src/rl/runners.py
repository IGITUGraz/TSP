import datetime
import pickle
from typing import Optional
import numpy as np
import jax
from jax import numpy as jnp
import optax
from chex import PRNGKey
from fax.config import flatten_dict
from fax.nn import metrics
from fax.rl.types import Env


def train_step_online(train_policy, tree_stop_gradient, rng: PRNGKey,
                      params: dict, policy_state, env_state, pi_t, obs, step):
    def online_loss(params):
        params = tree_stop_gradient(params)
        total_loss, aux = train_policy(
            rng, params, env_state, policy_state, pi_t, obs, step)
        total_loss = jnp.mean(total_loss)
        return total_loss, aux
    grad_fn = jax.value_and_grad(online_loss, has_aux=True)
    (new_loss, (new_policy_states, new_env_state,\
        new_obs, done, pi_t)), grads = grad_fn(params)
    return grads, new_policy_states,\
        new_env_state, new_obs, done, new_loss, pi_t

def online_training(key, step, initial_states, params, env:Env,
                    train_func, train_batch_size,
                    optimizer, params_transform,
                    opt_state, pt_state, state_reduce_func,
                    loss_log_frequency = 0, checkpoint_frequency = 0,
                    checkpoint_path = None, aim_run = None, env_aux = None):
    key, train_env_episode_key, train_episode_key = jax.random.split(key, 3)
    env_state, actor_state, env_aux = env.reset(train_env_episode_key, env_aux)
    grads = jax.tree_util.tree_map(
        lambda x: jnp.zeros_like(x, dtype=jnp.float32),
                                   params)
    obs = actor_state.obs
    actor_state_dict = actor_state._asdict()
    data = dict(action=None, action_logp=None, pred=None, value=None)
    data |= actor_state_dict
    done = actor_state.done
    pi_t = jnp.tile(jnp.zeros((2,), dtype=jnp.float32),(train_batch_size,1))
    act_policy_state = jax.tree_map(
        lambda x: jnp.tile(x, (train_batch_size,) + (1,) * jnp.ndim(x)),
        initial_states)
    while not jnp.all(done):
        train_episode_key, *batch_keys = jax.random.split(
            train_episode_key, 1+train_batch_size)
        new_grads, act_policy_state, env_state, obs,\
            done, loss_res, pi_t = train_func(jnp.array(batch_keys), params,
                act_policy_state, env_state, pi_t, obs, step)
        grads = jax.tree_util.tree_map(lambda x, y: x + y, grads, new_grads)
    grads = jax.tree_util.tree_map(lambda x: x/train_batch_size, grads)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    params, pt_state = params_transform.update(params, pt_state)

    step += 1
    if loss_log_frequency and step % loss_log_frequency == 0 and aim_run is not None:
        aim_run.track(float(loss_res), name="loss", step=step, 
                      context={"subset":"train"})
    
    new_states = state_reduce_func(act_policy_state)
    return key, params, grads, opt_state, pt_state, new_states, step, env_aux

def training(key: PRNGKey, step, initial_states, params, env,
              train_policy, train_func, env_batch_size, train_batch_size,
              opt_state, pt_state, accumulator, state_reduce_func, loss_log_frequency, 
              checkpoint_frequency = None, checkpoint_path = None, aim_run = None, env_aux = None):
    key, train_env_episode_key, train_episode_key = jax.random.split(
    key, 3)
    env_state, actor_state, env_aux = env.reset(train_env_episode_key, env_aux)
    actor_state_dict = actor_state._asdict()
    data = dict(action=None, action_logp=None, pred=None, value=None)
    data |= actor_state_dict
    accumulator.push(**data)
    done = actor_state.done
    
    act_policy_state = jax.tree_util.tree_map(
        lambda x: jnp.tile(x, (env_batch_size,) + (1,) * jnp.ndim(x)),
        initial_states)
    while not jnp.all(done):
        train_episode_key, *batch_keys = jax.random.split(
            train_episode_key, 1+env_batch_size)
        results = train_policy(jnp.array(batch_keys), params, act_policy_state,
                               actor_state.obs)
        act_policy_state, (action, action_logp, v, pred) = results
        env_state, actor_state = env.step(env_state, action)
        done = actor_state.done

        data = {"action": action,
                "action_logp": action_logp,
                "pred": pred,
                "value": v}
        data |= actor_state._asdict()
        accumulator.push(**data)
    train_policy_state = state_reduce_func(act_policy_state)
    if accumulator.is_ready(batch_size=train_batch_size):
        train_episode_key, sample_key, *batch_keys = jax.random.split(
            train_episode_key, 2+train_batch_size)
        data = accumulator.sample(train_batch_size, sample_key)
        act_policy_state = train_policy_state
        params, opt_state, pt_state, grads, _, loss_res = train_func(
            jnp.array(batch_keys), params, opt_state, pt_state, act_policy_state, data)

        step += 1
        if aim_run is not None:
            if loss_log_frequency and step % loss_log_frequency == 0:
                aim_run.track(float(loss_res),
                            name="loss", step=step, context={"subset": "train"})
                
    return key, params, grads, opt_state, pt_state, train_policy_state, step, env_aux


def evaluation(key: PRNGKey, nb_episodes_per_eval: int, initial_states, params,
               env: Env, eval_policy, batch_size: int, metric_chain, step:int, 
               aim_run = None, env_aux = None):
    metric_states = metric_chain.init({})
    for i in range(nb_episodes_per_eval):
        act_policy_state = jax.tree_util.tree_map(
            lambda x: jnp.tile(x, (batch_size,) + (1,) * jnp.ndim(x)),
            initial_states)
        
        key, test_env_episode_key, test_episode_key =\
            jax.random.split(key, 3)
        env_state, actor_state, env_aux = env.reset(test_env_episode_key, env_aux)
        
        actor_state_dict = actor_state._asdict()
        data = dict(action=None, action_logp=None,
                    value=None, pred=None)
        data |= actor_state_dict
        done = actor_state.done
        
        episode = []
        while not jnp.all(done):
            test_episode_key, *batch_keys = jax.random.split(
                test_episode_key, 1+batch_size)
            results = eval_policy(jnp.array(batch_keys),
                                  params,
                                  act_policy_state,
                                  actor_state.obs)
            act_policy_state, (action, action_prob, v, pred) = results
            env_state, actor_state = env.step(env_state, action)
            done = actor_state.done
            data = {"action": action,
                    "action_prob": action_prob,
                    "value": v,
                    "pred": pred}
            data |= actor_state._asdict()
            data |= data["info"]
            data |= flatten_dict(act_policy_state, separator="/",
                                 exclude_list=[])
            del data["info"]
            episode.append(data)
        batched_data = jax.tree_map(
            lambda *x: np.stack(x, axis=1), *episode)
        # eval_dic = evaluation_dispatch(
        #     test_episode_key, test_env, network_apply,
        #     act_policy_state, params, env_state, eval_params)
        batched_data["prefix_dim"] = 2
        batched_data["step"] = step*batch_size
        # batched_data |= eval_dic
        metric_states = metric_chain.update(batched_data,
                                            metric_states)
        
    metric_results = metric_chain.aggregate(metric_states)
    if aim_run is not None:
        metrics.aim_logging(aim_run, 
                            data=metric_results, 
                            step=step*batch_size,
                            prefix="test")
    return key, metric_results, step, env_aux

def train_step_base(train_policy, tree_stop_gradient, optimizer, params_transform,
                    rng: PRNGKey, params: dict, opt_state: optax.OptState, pt_state,
                    policy_state: dict, data: dict):
    """Train one step."""
    
    def loss(params):
        params = tree_stop_gradient(params)
        new_states, batch_loss = train_policy(rng, params, policy_state, data)
        loss_mean = jnp.mean(batch_loss)
        return loss_mean, (new_states,)

    grad_fn = jax.value_and_grad(loss, has_aux=True)
    (new_loss, (new_policy_state,)), grads = grad_fn(params)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    new_params, new_pt_state = params_transform.update(new_params, pt_state)
    return new_params, new_opt_state, new_pt_state, grads, new_policy_state, new_loss