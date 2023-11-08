import functools
import pathlib
import pickle
import time
from pathlib import Path
from datetime import datetime

import optax
from optax import chain

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate

from jax import jit
from jax import numpy as jnp
import jax

from fax.data import fear_conditioning_dataset
from fax.data.utils import hash_dictionary
from fax.data import read_numpy_data
from fax.nn.utils import maybe_set_random_seed
from fax.nn.utils import get_data_shape
from fax.nn.models import model_factory
from fax.nn import losses, metrics
from fax.config import BabiTask, FearConditioning, register_configs


register_configs()

@hydra.main(config_path="config", config_name="babi_config",
            version_base="1.2")
def main(cfg: DictConfig):
    supervised_experiment(cfg)    
    
@functools.singledispatch
def resolve_task(task_params: object):
    raise NotImplementedError(f"Task of type {type(task_params)} is not implemented")

@resolve_task.register
def _(task_params: FearConditioning):
    train_set, test_set, sample_size = fear_conditioning_dataset(**task_params.__dict__)
    task_metadata = {
        "train_size": sample_size,
        "test_size": sample_size,
        "nb_classes": 2,
        # "task_labels_train":  train_set[1].tolist(),
        # "task_labels_test": test_set[1].tolist(),
        "name": "fear_cond"
    }
    return train_set, train_set, test_set, task_metadata

def supervised_experiment(cfg: DictConfig):
    global experiment_cfg_str
    task_params = OmegaConf.to_object(cfg.task)
    train_set, valid_set, test_set, task_metadata = read_numpy_data(
        task_params.__dict__, "./datasets/tasks/babi")
    batch_size = cfg.training.batch_size
    task_metadata["train_n_batch"] = task_metadata["train_size"] // batch_size
    task_metadata["test_n_batch"] = task_metadata["test_size"] // batch_size
    task_params = OmegaConf.merge(task_params.__dict__, task_metadata)
    cfg.task = task_params
    experiment_cfg_str = OmegaConf.to_container(cfg, resolve=True)
    experiment_cfg_str["training"]["seed"] = maybe_set_random_seed(
        experiment_cfg_str["training"]["seed"])
        
    experiment_cfg = instantiate(experiment_cfg_str)
    experiment_cfg = OmegaConf.to_container(experiment_cfg)
    

    run(experiment_cfg, train_set, valid_set, test_set)

def shuffle_dataset(dataset, data_size, rng):
    shuffle_idx = jax.random.permutation(rng, jnp.arange(0, data_size, dtype=jnp.int32))
    new_dataset = jax.tree_map(lambda arr: arr[shuffle_idx], dataset)
    return new_dataset

shuffle_dataset = jit(shuffle_dataset, static_argnums=(1))
def shuffle_idx(data_size, rng):
    shuffle_idx = jax.random.permutation(rng, jnp.arange(0, data_size, dtype=jnp.int32))
    return shuffle_idx



def slice_batch(dataset, batch_size, batch_number):
    i = batch_size * batch_number
    batch = jax.tree_map(lambda arr: arr[
        i:i + batch_size], dataset)
    return batch


def run(global_config: dict, train_set, valid_set, test_set):
    model_config = global_config["model"]
    training_config = global_config["training"]
    task_config = global_config["task"]
    metrics_config = global_config["metric"]
    checkpoint_path = training_config["checkpoint_path"]
    
    epochs = training_config["epochs"]
    batch_size = training_config["batch_size"]
    
    optimizer_transform = training_config["optimizer"]
    gradient_transform = training_config["gradient_transform"]
    state_transform = training_config["state_transform"]
    params_transform = training_config["params_transform"]
    
    loss_fn = losses.get(training_config["loss"])
    
    seed = training_config["seed"]
    train_size = task_config["train_size"]
    train_n_batch = task_config["train_n_batch"]
    test_size = task_config["test_size"]
    test_n_batch = task_config["test_n_batch"]

    metric_chain = metrics.metric_chain(*tuple(metrics_config.values()))

    key = jax.random.PRNGKey(seed)
    train_data = train_set
    dimension_to_abstract = (training_config["batch_config"] is not None) \
                                + (training_config["temporal_type"] is not None)
    input_shape = get_data_shape(train_data[0], dimension_to_abstract)

    # TODO: make dataset could be a abitrary function
    # batch and temporal dimension can be abstracted by high order function,
    # we only need the shape without batch and/or temporal dimension

    task_model_dict = task_config | model_config
    task_model_id = hash_dictionary(task_model_dict)
    if checkpoint_path is not None:
        checkpoint_path = (Path(checkpoint_path) / training_config["experiment_name"]) / task_model_id
        checkpoint_path.mkdir(parents=True, exist_ok=True)
    params, base_states, network_apply, output_dim = model_factory(model_config, training_config["temporal_type"],
                                                        training_config["batch_config"], input_shape, seed=seed)

    optimizer_opt = chain(*optimizer_transform.values())

    opt_state = optimizer_opt.init(params)
    pt_state = params_transform.init(params)
    states_reduce = functools.partial(state_transform, func=lambda x: jnp.mean(x, axis=0),
                                   base_states=base_states)

    @jax.jit
    def test_step(rng, params, states, batch):
        x_batch, y_batch = batch

        def loss_and_pred(params):
            
            new_states, outputs = network_apply(rng, params, states, x_batch)
            pred = outputs[0][:,-1, :]
            loss = loss_fn(pred, y_batch).mean()
            return new_states, loss, pred

        new_states, loss, pred = loss_and_pred(params)
        # assignment here can only be made inside a tuple [-1] is perform in 
        # order to return the last element of the tuple (the reshaped vector)
        return new_states, pred, loss

    @jax.jit
    def train_step(rng, params, opt_state, pt_state, states, batch):
        """Train one step."""
        x_batch, y_batch = batch

        def loss_and_metrics(params):
            params = gradient_transform(params)
            new_states, outputs = network_apply(rng, params, states, x_batch)
            outputs = outputs[0]
            pred = outputs[:, -1, :]
            loss = loss_fn(pred, y_batch).mean()
            return loss, (new_states, pred)

        grad_fn = jax.value_and_grad(loss_and_metrics, has_aux=True)
        (loss, (new_states, pred)), grads = grad_fn(params)
        updates, new_opt_state = optimizer_opt.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        new_params, new_pt_state = params_transform.update(new_params, pt_state)
        return new_params, new_opt_state, new_pt_state, grads, new_states, pred, loss

    train_states = {**base_states}

    if checkpoint_path is not None:
        date = datetime.now().strftime("%Y_%m_%d-%H:%M:%S")
        path: pathlib = checkpoint_path  /  (date + ".pkl")
        with open(path, "wb") as f_w:
            params_states = {"params": params, "states": train_states, 
                                "exp_config": experiment_cfg_str}
            pickle.dump(params_states, f_w)
    train_context = jnp.array(task_config["task_labels_train"])
    test_context = jnp.array(task_config["task_labels_test"])
    train_data = train_set
    test_data = test_set
    time_start = time.time()
    for epoch in range(epochs):

        train_metric_states = metric_chain.init({})
        test_metrics_states = metric_chain.init({})

        key, key2, key3 = jax.random.split(key, 3)
        shuffled_train_context = shuffle_dataset(
            train_context, train_size, key2)
        shuffled_train_set = shuffle_dataset(train_data, train_size, key2)
        for i in range(train_n_batch):
            context = slice_batch(shuffled_train_context, batch_size, i)
            batch_data = slice_batch(shuffled_train_set, batch_size, i)
            key2, *keys = jax.random.split(key2, batch_size + 1)
            keys = jnp.array(keys)
            params, opt_state, pt_state, new_grads,\
                new_train_states, pred, loss = train_step(
                    keys, params, opt_state, pt_state, train_states, batch_data)
            new_train_metrics = {"pred": pred, 
                                 "target": batch_data[1],
                                 "loss": jnp.expand_dims(loss, 0),
                                 "context": context}

            train_metric_states = metric_chain.update(
                new_train_metrics,train_metric_states)

            train_states = states_reduce(new_train_states)
        # testing
        key3, *keys = jax.random.split(key3, test_size + 1)
        keys = jnp.array(keys)
        new_test_states, pred, loss = test_step(keys, params, train_states, test_data)
        
        new_test_metrics = {"loss":  jnp.expand_dims(loss, 0), "pred": pred,
                            "target": test_data[1],
                            "context": test_context}
        test_metrics_states = metric_chain.update(
            new_test_metrics, test_metrics_states)
        
        train_metric_results = metric_chain.aggregate(train_metric_states)
        test_metric_results = metric_chain.aggregate(test_metrics_states)
        
        train_metrics_str = [f"{k}:{v}" for k,v in train_metric_results.items()]
        train_metrics_str = " ".join(train_metrics_str)
        print(f"Train: {train_metrics_str}")
        
        test_metrics_str = [f"{k}:{v}" for k,v in test_metric_results.items()]
        test_metrics_str = " ".join(test_metrics_str)
        print(f"Test: {test_metrics_str}")
        


        if checkpoint_path is not None:
            date = datetime.now().strftime("%Y_%m_%d-%H:%M:%S")
            path: Path = checkpoint_path / (date + ".pkl")
            with open(path, "wb") as f_w:
                params_states = {"params": params, "states": train_states, 
                                    "exp_config": experiment_cfg_str}
                pickle.dump(params_states, f_w)
    time_stop = time.time()
    elapsed_time = time_stop - time_start
    m, s = divmod(elapsed_time, 60)
    h, m = divmod(m, 60)
    print(f"elapse_time: {h}:{m}:{s}" )

if __name__ == "__main__":
    main()