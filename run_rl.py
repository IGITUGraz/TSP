from datetime import datetime
import functools
from hashlib import new
from pathlib import Path
from fax._src.config import register_configs
from fax._src.nn import losses

import jax
import jax.numpy as jnp
import numpy as np
from hydra.utils import instantiate
from jax import jit
import optax
from functools import partial
import pickle
from omegaconf import DictConfig, OmegaConf
from optax import chain
import hydra

from fax._src.data.utils import hash_dictionary
from fax.nn import metrics
from fax.rl.types import Env
from fax.nn.utils import maybe_set_random_seed
from fax.nn.models import model_factory, vectorized_model
from fax.rl import actor
from fax.rl.utils import data_structures, resolve_task
from fax.rl.runners import train_step_base, train_step_online,\
    training, online_training, evaluation
import chex

register_configs()

PRNGKey = chex.PRNGKey
Shape = chex.Shape
@hydra.main(config_path="config", config_name="rl_config", version_base="1.2")
def main(cfg: DictConfig):
    rl_experiment(cfg)  
    
def rl_experiment(cfg: DictConfig):
    global experiment_cfg_str
    task_params = OmegaConf.to_object(cfg.task)
    cfg.training.seed = maybe_set_random_seed(cfg.training.seed)
    key = jax.random.PRNGKey(cfg.training.seed)
    train_env, test_env, task_params = resolve_task(task_params,
                                       cfg.training.indep_test_env,
                                       key)
    cfg.task = task_params
    experiment_cfg_str = OmegaConf.to_container(cfg, resolve=True)
    experiment_cfg = instantiate(experiment_cfg_str)
    experiment_cfg = OmegaConf.to_container(experiment_cfg)
    run(experiment_cfg, train_env, test_env)

def run(global_config: dict, train_env: Env, test_env: Env):
    model_config = global_config["model"]
    training_config = global_config["training"]
    metrics_config = global_config["metric"]
    task_config = global_config["task"]
    env_batch = task_config["batch"]
    seed = training_config["seed"]
    
    eval_params = training_config["eval_params"]
    
    gradient_transform = training_config["gradient_transform"]
    state_transform = training_config["state_transform"]
    params_transform = training_config["params_transform"]
    optimizer_transform = training_config["optimizer"]
    nb_policy_iteration = training_config["nb_policy_iteration"]
    train_batch = training_config["batch"]
    nb_episodes_per_eval = training_config["nb_episodes_per_eval"]
    loss_log_frequency = training_config["loss_log_frequency"]
    eval_frequency = training_config["eval_frequency"]
    
    checkpoint_frequency = training_config["checkpoint_frequency"]
    checkpoint_path = Path(training_config["checkpoint_path"])
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    
    online_learning = training_config["online_learning"]
    
    task_model_dict = task_config | model_config
    task_model_id = hash_dictionary(task_model_dict)
    
    metric_chain = metrics.metric_chain(*tuple(metrics_config.values()))
    
    dummy_obs = train_env.observation_space.sample()
    input_shape = jax.tree_map(lambda x: jnp.shape(x), dummy_obs)
    accumulator = data_structures.TransitionStore(
        **training_config["accumulator"])

    params, base_states, network_apply, output_dim = model_factory(
        model_config,
        training_config["temporal_type"],
        training_config["batch_config"],
        input_shape, seed=seed)
    network_apply = jit(network_apply)
    loss_fn = training_config["loss"]
    if not online_learning:
        actor_policy, train_policy = actor.get(loss_fn["func"])(
            loss_fn["params"], network_apply)
        actor_policy_train = jit(vectorized_model(
             partial(actor_policy, evaluation=False),
        {"in_axes": (0, 0, 0), "out_axes": (0, 0)}))
        actor_policy_eval = jit(vectorized_model(
            partial(actor_policy, evaluation=True),
        {"in_axes": (0, 0, 0), "out_axes": (0, 0)}))
        train_policy = jit(vectorized_model(train_policy,
                                        {"in_axes": (0, None, 0),
                                         "out_axes": (0, 0)}))
    else:
        actor_policy, train_policy = actor.get(
            "online_policy_gradient")(loss_fn["params"], network_apply)
        if loss_fn["params"]["vectorize_model"]:
            actor_policy_eval = actor_policy
        else:
            actor_policy_eval = jit(
                vectorized_model(actor_policy,
                                {"in_axes": (0, 0, 0), "out_axes": (0, 0)}))
    
    optimizer_transform = tuple(optimizer_transform.values())
    optimizer_opt = chain(*optimizer_transform)
    opt_state = optimizer_opt.init(params)
    pt_state = params_transform.init(params)
    
    states_reduce = partial(state_transform, func=lambda x: jnp.mean(x, axis=0),
                                   base_states=base_states)
    if not online_learning:
        train_step = jit(partial(train_step_base,
                train_policy, gradient_transform, optimizer_opt, params_transform))
    else:
        if loss_fn["params"]["vectorize_model"]:
            train_step = jit(
                partial(
                    train_step_online, partial(train_policy, train_env.step),
                    gradient_transform)
                )
        else:
            train_step = jit(partial(train_step_online,
                            jax.vmap(partial(train_policy, jit(train_env.env.step)),
                                    in_axes=(0, None, 0, 0, 0, 0, None)),
                            gradient_transform))
    

    train_policy_state = {**base_states}
    key = jax.random.PRNGKey(seed)
    key, train_key, test_key = jax.random.split(key, 3)
    policy_iteration = 0
    test_env_aux = test_env.create_auxilatory_data()
    test_key, metric_results, step, test_env_aux = evaluation(
                test_key, nb_episodes_per_eval, train_policy_state, params,
                test_env, actor_policy_eval, env_batch, metric_chain,
                policy_iteration, None, test_env_aux)
    test_metrics_str = [f"{k}:{v}" for k,v in metric_results.items()]
    test_metrics_str = " ".join(test_metrics_str)
    print(f"Episode n-{policy_iteration*train_batch}: {test_metrics_str}")
    
    train_env_aux = train_env.create_auxilatory_data()
    while policy_iteration < nb_policy_iteration:
        if online_learning:
            results = online_training(
                train_key, policy_iteration, train_policy_state,
                params, train_env, train_step, train_batch,
                optimizer_opt, params_transform,
                opt_state, pt_state, states_reduce,
                loss_log_frequency, checkpoint_frequency,
                checkpoint_path, None, train_env_aux)
            
            train_key, params, grads, opt_state, pt_state, \
                train_policy_state, policy_iteration, train_env_aux = results
        else: 
            result = training(
                train_key, policy_iteration, train_policy_state,
                params,train_env, actor_policy_train, 
                train_step, env_batch, train_batch, opt_state, pt_state,
                accumulator, states_reduce, loss_log_frequency, 
                checkpoint_frequency, checkpoint_path, None, train_env_aux)
            train_key, params, grads, opt_state, pt_state,\
                train_policy_state, policy_iteration, train_env_aux = result
        
            
        if policy_iteration % eval_frequency == 0:
            test_key, metric_results, step, test_env_aux = evaluation(
                test_key, nb_episodes_per_eval, train_policy_state, params,
                test_env, actor_policy_eval, env_batch, metric_chain,
                policy_iteration, None, test_env_aux)
            test_metrics_str = [f"{k}:{v}" for k,v in metric_results.items()]
            test_metrics_str = " ".join(test_metrics_str)
            print(f"Episode n-{policy_iteration*train_batch}: {test_metrics_str}")
        
        #if checkpoint_frequency and policy_iteration % checkpoint_frequency == 0:
    if checkpoint_path is not None:
        date = datetime.now().strftime("%Y_%m_%d-%H:%M:%S")
        path: Path = checkpoint_path  /  (date + ".pkl")
        with open(path, "wb") as f_w:
            params_states = {"params": params, "states": train_policy_state, 
                                "exp_config": experiment_cfg_str}
            pickle.dump(params_states, f_w)
if __name__ == "__main__":
    main()