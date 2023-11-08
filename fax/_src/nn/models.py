import pickle
from functools import partial
from pathlib import Path
from typing import Union, Optional, Callable

import jax
from chex import Shape
from jax import lax

from fax.nn.graph import neural_net_graphs_factory, initial_states_and_params


def create_model(layers_definitions,
                 graph_ff_structure: Union[str, Path],
                 graph_fb_structure: Union[str, Path, None],
                 input_shape: Shape, seed,
                 checkpoint_path: Optional[str] = None,):
    init_fun_states, init_fun_params, \
        initialization_graph, applicative_graph = neural_net_graphs_factory(
            graph_ff_structure, graph_fb_structure, layers_definitions)
    init_fun_states, params, states, output_dim = initial_states_and_params(
        initialization_graph, init_fun_params,
        init_fun_states, input_shape, seed)
    if checkpoint_path is not None:
        checkpoint_file_path = Path(checkpoint_path)
        with open(checkpoint_file_path, "rb") as f_r:
            check_pt = pickle.load(f_r)
            states_chp = check_pt["states"]
            params_chp = check_pt["params"]
            params["w_emb"] = params_chp["w_emb"]
            params["p_emb"] = params_chp["p_emb"]
    return init_fun_states, params, states, output_dim,\
        initialization_graph, applicative_graph


def vectorized_model(model: Callable, batch_config):
    def apply_model(rng, params, states, x_batch):
        def model_fn(rng, states, example):
            new_states, output = model(rng, params, states, example)
            return new_states, output

        return jax.vmap(model_fn,
                        in_axes=batch_config["in_axes"],
                        out_axes=batch_config["out_axes"])(rng, states, x_batch)
    return apply_model


def temporize_model(model: Callable, temporal_type: str):
    if temporal_type == "seq2seq":
        def fold_wrapper(rng, p, s, inputs):
            param_cloned = partial(model, rng, p)
            return lax.scan(param_cloned, s, inputs)
    elif temporal_type == "rnn":
        def fold_wrapper(rng, p, s, inputs):
            param_cloned = partial(model, rng, p)
            output_states, outputs = lax.scan(param_cloned, s, inputs)
            outputs = jax.tree_map(lambda x: x[-1], outputs)
            return output_states, outputs
    else:
        raise NotImplementedError(
            f"Temporal type {temporal_type} is not implemented")
    return fold_wrapper


def model_factory(model_parameters: dict, temporal_type: Optional[str],
                  batch_config: Optional[dict], input_shape: Shape, seed):
    init_fun_states, params, states, output_dim, init_g, apply_g = \
        create_model(**model_parameters, input_shape=input_shape, seed=seed)
    base_model = apply_g
    if temporal_type is not None:
        base_model = temporize_model(base_model, temporal_type)
    if batch_config is not None:
        base_model = vectorized_model(base_model, batch_config)
    return params, states, base_model, output_dim
