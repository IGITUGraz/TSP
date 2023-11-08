from collections.abc import Callable
from pathlib import Path
from typing import Any, Union, Optional

import jax
from chex import PRNGKey
from jax import numpy as jnp

from fax.data import parsing_graph_from_path, parsing_graph_from_text
from fax.nn.layers import get as fax_funcs
from fax.nn.types import Graph, Overlay, Connect, Vertex, Empty


def _init_vertex(vertex: Vertex, *args):
    return vertex.v(*args)


def _init_connect(connect: Connect, *args):
    rng, init_params, init_states, *new_args = args
    k1, k2 = jax.random.split(rng, 2)
    rhs_init_states, rhs_params, rhs_states, rhs_output = init_graph(connect.rhs, k1,
                                                                     init_params, init_states, *new_args)

    init_states |= rhs_init_states
    lhs_init_states, lhs_params, lhs_states, lhs_output = \
        init_graph(connect.lhs, k2, init_params, init_states, *rhs_output)
    init_states |= lhs_init_states
    params = rhs_params | lhs_params
    states = rhs_states | lhs_states

    return init_states, params, states, lhs_output


def _init_overlay(overlay: Overlay, *args):
    rng, init_params, init_states, *new_args = args
    k1, k2 = jax.random.split(rng, 2)

    rhs_init_states, rhs_params, rhs_states, rhs_output = init_graph(
        overlay.rhs, k1, init_params, init_states, *new_args)
    lhs_init_states, lhs_params, lhs_states, lhs_output = init_graph(
        overlay.lhs, k2, init_params, init_states, *new_args)
    init_states |= rhs_init_states | lhs_init_states
    params = rhs_params | lhs_params
    states = rhs_states | lhs_states

    return init_states, params, states, lhs_output + rhs_output


def _apply_vertex(vertex: Vertex, *args):
    return vertex.v(*args)


def _apply_connect(connect: Connect, *args):
    rng, params, states, *new_args = args
    k1, k2 = jax.random.split(rng, 2)
    rhs_states, rhs_output = apply_graph(connect.rhs, k1, params, states, *new_args)
    states |= rhs_states
    lhs_states, lhs_output = apply_graph(connect.lhs, k2, params, states, *rhs_output)
    states |= lhs_states
    return states, lhs_output


def _apply_overlay(overlay: Overlay, *args):
    rng, params, states, *new_args = args
    k1, k2 = jax.random.split(rng, 2)
    (rhs_states, rhs_output) = apply_graph(overlay.rhs, k1, params, states, *new_args)
    (lhs_states, lhs_output) = apply_graph(overlay.lhs, k2, params, states, *new_args)
    states |= rhs_states | lhs_states
    return states, lhs_output + rhs_output


def init_graph(g: Graph, *args):
    if isinstance(g, Vertex):
        return _init_vertex(g, *args)
    if isinstance(g, Connect):
        return _init_connect(g, *args)
    if isinstance(g, Overlay):
        return _init_overlay(g, *args)


def apply_graph(g: Graph, *args):
    if isinstance(g, Vertex):
        return _apply_vertex(g, *args)
    if isinstance(g, Connect):
        return _apply_connect(g, *args)
    if isinstance(g, Overlay):
        return _apply_overlay(g, *args)


def _vertex_to_function(vertex: Vertex):
    func = vertex.v

    def inner_vertex_func(*args):
        return func(*args)

    return inner_vertex_func


def _connect_to_function(connect: Connect):
    rhs_func = graph_to_function(connect.rhs)
    lhs_func = graph_to_function(connect.lhs)

    def inner_functional_connect(*args):
        rng, params, states, *new_args = args
        k1, k2 = jax.random.split(rng, 2)
        rhs_states, rhs_output = rhs_func(k1, params, states, *new_args)
        states |= rhs_states
        lhs_states, lhs_output = lhs_func(k2, params, states, *rhs_output)
        states |= lhs_states
        return states, lhs_output

    return inner_functional_connect


def _overlay_to_function(overlay: Overlay):
    rhs_func = graph_to_function(overlay.rhs)
    lhs_func = graph_to_function(overlay.lhs)

    def inner_functional_overlay(*args):
        rng, params, states, *new_args = args
        k1, k2 = jax.random.split(rng, 2)
        rhs_states, rhs_output = rhs_func(k1, params, states, *new_args)
        lhs_states, lhs_output = lhs_func(k2, params, states, *new_args)
        states |= rhs_states | lhs_states
        return states, lhs_output + rhs_output

    return inner_functional_overlay


def graph_to_function(g: Graph) -> Callable:
    if isinstance(g, Vertex):
        return _vertex_to_function(g)
    if isinstance(g, Connect):
        return _connect_to_function(g)
    if isinstance(g, Overlay):
        return _overlay_to_function(g)


def _resolve_definition(node_definition: dict, node_name: str, do_feedback: bool):
    func = node_definition.get("func")
    func = fax_funcs(func)
    hyper_params = node_definition.get("hyper_params", {})
    init, apply = func(**hyper_params)

    global_name = node_definition.get("name", node_name)
    params_namespace = node_definition.get("params_name", global_name)
    states_namespace = node_definition.get("states_name", global_name)
    fb_namespace = node_definition.get("fb_name", "output" if do_feedback else None)
    if fb_namespace == "":
        fb_namespace = None
    init_params = node_definition.get("init_params", {})
    init_states = node_definition.get("init_states", {} if fb_namespace is None else {f"{fb_namespace}_shape": (1,)})

    apply_ff = _create_apply_ff_wrapper(apply, params_namespace, states_namespace, fb_namespace)
    init_ff = _create_init_ff_wrapper(init, params_namespace, states_namespace, fb_namespace)
    init_params_env = {params_namespace: init_params}
    init_states_env = {states_namespace: init_states}

    if do_feedback:
        apply_fb = _create_apply_fb_wrapper(states_namespace, fb_namespace)
        init_fb = _create_init_fb_wrapper(states_namespace, fb_namespace)
        return apply_ff, apply_fb, init_ff, init_fb, init_params_env, init_states_env
    else:
        return apply_ff, init_ff, init_params_env, init_states_env


def _create_apply_ff_wrapper(func: Callable, params_namespace: str, states_namespace: str,
                             fb_namespace: Optional[str] = None):
    store_output = fb_namespace == "output"

    def apply_ff(rng: PRNGKey, params: dict, states: dict, *args):
        new_states, output = func(rng, params[params_namespace], states[states_namespace], *args)
        if store_output:
            new_states["output"] = output
        # if fb_namespace is define it must refers to a tag inside new_states dict
        return {states_namespace: new_states}, (output,)

    return apply_ff


def _create_apply_fb_wrapper(states_namespace: str, fb_namespace: str):
    def apply_fb(rng: PRNGKey, params: dict, states: dict, *args):
        return {}, (states[states_namespace][fb_namespace],)

    return apply_fb


def _create_init_ff_wrapper(func: Callable, params_namespace: str, states_namespace: str,
                            fb_namespace: Optional[str] = None):
    store_output = fb_namespace == "output"

    def init_ff(rng: PRNGKey, init_params: dict, init_states: dict, *args):
        new_init_states, params, states, output_shape \
            = func(rng, init_params[params_namespace], init_states[states_namespace], *args)
        if store_output:
            states["output"] = jnp.zeros(output_shape, dtype=jnp.float32)
        if fb_namespace is not None:
            new_init_states[f"{fb_namespace}_shape"] = output_shape if store_output else jnp.shape(states[fb_namespace])
        return {states_namespace: new_init_states}, \
               {params_namespace: params}, {states_namespace: states}, (output_shape,)

    return init_ff


def _create_init_fb_wrapper(states_namespace: str, fb_namespace: str):
    def init_fb(rng: PRNGKey, init_params: dict, init_states: dict, *args):
        return {}, {}, {}, (init_states[states_namespace][f"{fb_namespace}_shape"],)

    return init_fb


def neural_net_graphs_factory(graph_ff_structure: Union[Path, str], graph_fb_structure: Union[Path, str, None],
                              nodes_definitions: dict[str, dict],
                              jit_node: bool = False):
    if isinstance(graph_ff_structure, Path):
        ff_out_edges, ff_in_edges = parsing_graph_from_path(graph_ff_structure)

    elif isinstance(graph_ff_structure, str):
        ff_out_edges, ff_in_edges = parsing_graph_from_text(graph_ff_structure)
    else:
        raise TypeError
    if isinstance(graph_fb_structure, Path):
        fb_out_edges, fb_in_edges = parsing_graph_from_path(graph_fb_structure)

    elif isinstance(graph_fb_structure, str):
        fb_out_edges, fb_in_edges = parsing_graph_from_text(graph_fb_structure)
    elif graph_fb_structure is None:
        fb_out_edges, fb_in_edges = dict.fromkeys(ff_out_edges.keys(), []), dict.fromkeys(ff_in_edges.keys(), [])
    else:
        raise TypeError

    leaves_nodes = [k for k, v in ff_out_edges.items() if v == []]
    fb_nodes = [k for k, v in fb_out_edges.items() if v != []]
    tag_to_init_ff_value = {}
    tag_to_init_fb_value = {}
    tag_to_apply_ff_value = {}
    tag_to_apply_fb_value = {}

    graph_init_states = {}
    graph_init_params = {}

    for k in ff_out_edges.keys():
        if k in fb_nodes:
            apply_ff, apply_fb, init_ff, init_fb, init_params, init_states = _resolve_definition(
                nodes_definitions[k], k, do_feedback=True)
            tag_to_init_fb_value[k] = init_fb
            tag_to_apply_fb_value[k] = jax.jit(apply_fb) if jit_node else apply_ff
        else:
            apply_ff, init_ff, init_params, init_states = _resolve_definition(
                nodes_definitions[k], k, do_feedback=False)
        tag_to_init_ff_value[k] = init_ff
        tag_to_apply_ff_value[k] = jax.jit(apply_ff) if jit_node else apply_ff
        graph_init_states |= init_states
        graph_init_params |= init_params

    initialization_graph = create_graph(ff_in_edges, fb_in_edges,
                                        tag_to_init_ff_value, tag_to_init_fb_value,
                                        leaves_nodes)
    applicative_graph = create_graph(ff_in_edges, fb_in_edges, tag_to_apply_ff_value,
                                     tag_to_apply_fb_value, leaves_nodes)
    return graph_init_params, graph_init_states, initialization_graph, applicative_graph


def initial_states_and_params(initial_graph: Union[Graph, Callable], init_params: dict,
                              init_states, input_shape, seed, k: int = 2):

    key = jax.random.PRNGKey(seed)
    if isinstance(initial_graph, Graph):
        graph_func = lambda s, x: init_graph(initial_graph, key, init_params, s, x)
    else:
        raise TypeError
    params = {}
    states = {}
    output_dim = None
    for i in range(k):
        init_states, params, states, output_dim = graph_func(init_states, input_shape)
    return init_states, params, states, output_dim


def create_graph(tag_to_incoming_feedforward_nodes: dict[str, list[str]],
                 tag_to_incoming_feedback_nodes: dict[str, list[str]],
                 tag_to_feedforward_value: dict[str, Any],
                 tag_to_feedback_value: dict[str, Any],
                 leaf_nodes: list[str],
                 reduce: bool = True) -> Graph[Any]:
    def _create_sub_graph(tag: str):
        v_node = Vertex(tag_to_feedforward_value[tag])
        feedforward_list = [_create_sub_graph(t) for t in tag_to_incoming_feedforward_nodes.get(tag, [])]
        feedback_list = [Vertex(tag_to_feedback_value[t]) for t in tag_to_incoming_feedback_nodes.get(tag, [])]
        incoming_edges = feedforward_list + feedback_list
        sub_graph = v_node * sum(incoming_edges, start=Empty())
        return sub_graph

    leaf_overlay = [_create_sub_graph(t) for t in leaf_nodes]
    graph = sum(leaf_overlay, start=Empty())

    return reduce_graph(graph) if reduce else graph


def reduce_graph(g: Graph[Any]) -> Graph[Any]:
    new_g = _reduce_graph(g)
    while new_g != g:
        g = new_g
        new_g = _reduce_graph(g)
    return new_g


def _reduce_graph(g: Graph[Any]) -> Graph[Any]:
    if isinstance(g, Overlay):
        # x + x = x
        if g.lhs == g.rhs:
            return _reduce_graph(g.lhs)
        # empty + x = x
        if isinstance(g.lhs, Empty):
            return _reduce_graph(g.rhs)
        # x + empty = x
        if isinstance(g.rhs, Empty):
            return _reduce_graph(g.lhs)
        # factorizations
        if isinstance(g.lhs, Connect) and isinstance(g.rhs, Connect):
            # x * y + x * z -> x * (x + y)
            if g.lhs.lhs == g.rhs.lhs:
                return _reduce_graph(Connect(g.lhs.lhs, Overlay(g.lhs.rhs, g.rhs.rhs)))
            # x * z + y * z -> (x + y) * z
            elif g.lhs.rhs == g.rhs.rhs:
                return _reduce_graph(Connect(Overlay(g.lhs.lhs, g.rhs.lhs), g.lhs.rhs))
        return Overlay(_reduce_graph(g.lhs), _reduce_graph(g.rhs))
    if isinstance(g, Connect):
        # empty * x = x
        if isinstance(g.lhs, Empty):
            return _reduce_graph(g.rhs)
        # x * empty = x
        if isinstance(g.rhs, Empty):
            return _reduce_graph(g.lhs)
        return Connect(_reduce_graph(g.lhs), _reduce_graph(g.rhs))
    if isinstance(g, Vertex) or isinstance(g, Empty):
        return g
