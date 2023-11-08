from collections import deque
from collections.abc import Callable
from functools import partial, singledispatch
from pathlib import Path
from typing import Any, Union, Optional
import copy

import jax
from chex import PRNGKey
from jax import numpy as jnp

from fax.data import parsing_graph_from_path, parsing_graph_from_text, parsing_graph_from_list
from fax.nn.layers import get as fax_funcs
from fax.nn.utils import copy_func

"""
Programmable and differientiable functional graph:
    Main idea:
Deep learning algorithm can be generally understood as directed functional graph,
where vertices are atomic differentiable functions and edges define the input-output relation between such functions.
As human, it is generally convenient to conceptualize such systems by directly drawing the computational graph, 
while defining the semantics of each node separately.
this work tends to bridge the gap between conceptualisation and implementation.

To this end, we propose that a deep learning model can be built and evaluated in a programmed way 
only from a simple definition of the graph structure (vertices and edges) and the semantics of the nodes.

    Graph structure:
In order to represent and parse the structure we use a lightweight version of the DOT language:
The grammar of this simple language is the following:

    graph       :  '{' stmt_list '}'
    stmt_list   : (node_id | edge_stmt) '\n' stmt_list
    edge_stmt 	:  node_id  edgeRHS
    edgeRHS 	: 	'->' node_id  [ edgeRHS ]
    node_id 	: 	ID
    ID          : [a-zA-Z0-9_]+
    
    ID are case un-sensitives
For instance, if we want a three nodes (v1, v2, v3) graph where v1 project to v2 and v3, we can write:
    { v1 -> v2
      v1 -> v3 }

        
        Feedforward and feedback graph:
Our system does not limit it self to directed acyclic graph DAG, in deep-learning system its now common to define 
flow of information going backward from the standard feedforward hierarchy of the graph.
This flow was coined as feedback edges.

The most common way to implement these feedback edges is to send results 
from the previous computation to the current one.
For instance, suppose that our feedforward computation is define as v1 -> v2, and we have a feedback edge v2 -> v1:
At computation time t, we express the graph as v2^{t-1} -> v1^{t} -> v2^{t}.

In our system, because, feedforward and feedback flows are generally Indistinguishable from each other, 
user must define the two structure structure separately:



example of node definition

    """


@singledispatch
def resolve_graph_structure(graph: object):
    raise NotImplementedError(f"graph structure of type {type(graph)} was not understood")


@resolve_graph_structure.register
def _(graph: Path):
    return parsing_graph_from_path(graph)


@resolve_graph_structure.register
def _(graph: str):
    return parsing_graph_from_text(graph)


@resolve_graph_structure.register
def _(graph: list):
    return parsing_graph_from_list(graph)


def initial_states_and_params(initial_graph: Callable, init_params: dict,
                              init_states, input_shape, seed, k: int = 2):
    params = {}
    states = {}
    output_dim = None
    key = jax.random.PRNGKey(seed)

    for i in range(k):
        init_states, params, states, output_dim = initial_graph(key, init_params, init_states, input_shape)
        key, subkey = jax.random.split(key, 2)
    return init_states, params, states, output_dim


def neural_net_graphs_factory(graph_ff_structure: Union[Path, str, list],
                              graph_fb_structure: Union[Path, str, list, None],
                              nodes_definitions: dict[str, dict]):
    ff_out_edges, ff_in_edges = resolve_graph_structure(graph_ff_structure)
    if graph_fb_structure is not None:
        fb_out_edges, fb_in_edges = resolve_graph_structure(graph_fb_structure)
    else:
        fb_out_edges, fb_in_edges = dict.fromkeys(ff_out_edges.keys(), []), dict.fromkeys(ff_in_edges.keys(), [])
    root_nodes = [k for k, v in ff_in_edges.items() if v == []]
    # TODO: check if root nodes is already a valid unique input node
    leaves_nodes = [k for k, v in ff_out_edges.items() if v == []]
    # TODO: check if leave nodes is already a valid unique output node
    fb_nodes = [k for k, v in fb_out_edges.items() if v != []]
    # create input or output nodes if needed
    if ff_in_edges.get("input") is None:
        ff_in_edges["input"] = []
        ff_out_edges["input"] = []
        for root in root_nodes:
            ff_in_edges[root].append("input")
            ff_out_edges["input"].append(root)
    elif root_nodes != ["input"]:
        raise ValueError("If input node is present, it must connect all roots nodes")
    if ff_in_edges.get("output") is None:
        ff_in_edges["output"] = []
        ff_out_edges["output"] = []
        for leaf in leaves_nodes:
            ff_out_edges[leaf].append("output")
            ff_in_edges["output"].append(leaf)
    elif leaves_nodes != ["output"]:
        raise ValueError("If output node is present, all leaves must connect to it")
    ranked_nodes = topological_sort_by_rank(ingoing_edges=ff_in_edges, outgoing_edges=ff_out_edges)
    fb_nodes_definitions: dict[str, list[tuple[str, str]]] = {k: [] for k in ff_out_edges}
    # for each node that receive feedback for another node, we resolve the feedback namespace of the sending node
    for sending_fb_node in fb_nodes:
        sending_node_def = nodes_definitions[sending_fb_node]
        global_name = sending_node_def.get("name", sending_fb_node)
        states_namespace = sending_node_def.get("states_name", global_name)
        fb_namespace = sending_node_def.get("fb_name", "output")
        for receiving_fb_node in fb_out_edges[sending_fb_node]:
            fb_nodes_definitions[receiving_fb_node].append((states_namespace, fb_namespace))

    tag_to_init_wrapper = {}
    tag_to_apply_wrapper = {}

    graph_init_states = {}
    graph_init_params = {}

    for k in set(ff_out_edges) - {"input", "output"}:
        init_wrapper, apply_wrapper, init_states, init_params, = _resolve_definition(k,
                                                                                     nodes_definitions[k],
                                                                                     ff_in_edges[k],
                                                                                     fb_nodes_definitions[k],
                                                                                     do_feedback=k in fb_nodes)
        tag_to_init_wrapper[k] = init_wrapper
        tag_to_apply_wrapper[k] = apply_wrapper
        graph_init_states |= init_states
        graph_init_params |= init_params

    graph_init_states |= {"input": {}} | {"output": {}}
    graph_init_params |= {"input": {}} | {"output": {}}

    init_graph = _create_init_graph(ranked_nodes, tag_to_init_wrapper, ff_in_edges["output"])
    apply_graph = _create_apply_graph(ranked_nodes, tag_to_apply_wrapper, ff_in_edges["output"])

    return graph_init_states, graph_init_params, init_graph, apply_graph


def _create_rank_apply_computation(funcs_on_rank: list[Callable]):
    _set_len = len(funcs_on_rank)

    def rank_apply_computation(rng, params, states, input_store: dict):
        key, *keys = jax.random.split(rng, _set_len + 1)
        results = [func(sub_key, params, states, input_store) for func, sub_key in
                   zip(funcs_on_rank, keys)]
        new_states_list, outputs_list = zip(*results)

        new_states = {}
        new_states |= new_states_list
        input_store |= outputs_list
        return new_states, input_store

    return rank_apply_computation


def _create_rank_init_computation(funcs_on_rank: list[Callable]):
    _set_len = len(funcs_on_rank)

    def rank_init_computation(rng, init_params, init_states, input_store: dict):
        key, *keys = jax.random.split(rng, _set_len + 1)
        results = [func(sub_key, init_params, init_states, input_store) for func, sub_key in
                   zip(funcs_on_rank, keys)]
        new_init_states_list, apply_params_list, apply_states_list, outputs_shape_list = zip(*results)
        apply_states = {}
        apply_params = {}
        new_init_states = {}
        apply_states |= apply_states_list
        apply_params |= apply_params_list
        new_init_states |= new_init_states_list
        input_store |= outputs_shape_list
        return new_init_states, apply_params, apply_states, input_store

    return rank_init_computation


def _create_apply_graph(nodes_by_ranks: list[list[str]], tag_to_apply: dict[str, Callable],
                        leaf_keys):
    rank_funcs = []

    for rank in range(1, len(nodes_by_ranks) - 1):
        funcs = [tag_to_apply[k] for k in nodes_by_ranks[rank]]
        rank_funcs.append(_create_rank_apply_computation(funcs))

    def apply_graph_func(rng, params, states, inputs):
        inputs_store = {"input": inputs}
        key = rng
        global_new_states = {}
        for rank_func in rank_funcs:
            key, subkey = jax.random.split(key, 2)
            new_states, outputs_store = rank_func(subkey, params, states, inputs_store)
            inputs_store = outputs_store
            global_new_states |= new_states
        outputs_values = tuple([inputs_store[k] for k in leaf_keys])
        return global_new_states, outputs_values

    return apply_graph_func


def _create_init_graph(nodes_by_ranks: list[list[str]], tag_to_init: dict[str, Callable],
                       leaf_keys):
    # number of rank can be determined nodes by rank is available
    rank_funcs = []

    for rank in range(1, len(nodes_by_ranks) - 1):
        funcs = [tag_to_init[k] for k in nodes_by_ranks[rank]]
        rank_funcs.append(_create_rank_init_computation(funcs))

    def init_graph_func(rng, init_params, init_states, inputs):
        inputs_store = {"input": inputs}
        key = rng
        global_states, global_params, global_new_init_states = {}, {}, {}
        for rank_func in rank_funcs:
            key, subkey = jax.random.split(key, 2)
            new_init_states, apply_params, apply_states, outputs_store = rank_func(subkey, init_params,
                                                                                   init_states, inputs_store)
            inputs_store = outputs_store
            global_states |= apply_states
            global_params |= apply_params
            global_new_init_states |= new_init_states
        outputs_values = tuple([inputs_store[k] for k in leaf_keys])
        return global_new_init_states, global_params, global_states, outputs_values

    return init_graph_func


def _resolve_definition(node_name: str, node_definition: dict, parents_nodes: list[str],
                        fb_nodes: list[tuple[str, str]], do_feedback: bool):
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
    init_states = node_definition.get("init_states", {} if fb_namespace is None else {f"{fb_namespace}_shape": (1,)})
    init_params = node_definition.get("init_params", {})

    apply_wrapper = _create_apply_wrapper(apply, node_name,
                                          params_namespace, states_namespace, parents_nodes, fb_nodes, fb_namespace)
    init_wrapper = _create_init_wrapper(init, node_name,
                                        params_namespace, states_namespace, parents_nodes, fb_nodes, fb_namespace)
    init_states_env = {states_namespace: init_states}
    init_params_env = {params_namespace: init_params}
    return init_wrapper, apply_wrapper, init_states_env, init_params_env,

#TODO: handle a better way the fact that one node can have not ff link but only fb
def _create_init_wrapper(func: Callable, node_name: str, params_namespace: str, states_namespace: str,
                         parents_nodes: list[str], fb_nodes: list[tuple[str, str]],
                         fb_namespace: Optional[str] = None):
    store_output = fb_namespace == "output"

    def init_ff(rng: PRNGKey, init_params: dict, init_states: dict, input_store: dict):
        ff_input = tuple([input_store[p_node] for p_node in parents_nodes if p_node != "empty"] )
        fb_input = tuple([init_states[state_name][f"{fb_name}_shape"] for state_name, fb_name in fb_nodes if state_name != "empty"])
        inputs = ff_input + fb_input
        new_init_states, params, states, output_shape \
            = func(rng, init_params[params_namespace], init_states[states_namespace], *inputs)
        if store_output:
            states[fb_namespace] = jnp.zeros(output_shape, dtype=jnp.float32)
        if fb_namespace is not None:
            new_init_states[f"{fb_namespace}_shape"] = output_shape if store_output else jnp.shape(states[fb_namespace])
        return (states_namespace, new_init_states), (params_namespace, params), \
               (states_namespace, states), (node_name, output_shape)

    return init_ff


def _create_apply_wrapper(func: Callable, node_name: str, params_namespace: str, states_namespace: str,
                          parents_nodes: list[str], fb_nodes: list[tuple[str, str]],
                          fb_namespace: Optional[str] = None):
    store_output = fb_namespace == "output"

    def apply_ff(rng: PRNGKey, params: dict, states: dict, input_store: dict):
        ff_input = tuple([input_store[p_node] for p_node in parents_nodes if p_node != "empty"])
        fb_input = tuple([states[state_name][fb_name] for state_name, fb_name in fb_nodes if state_name != "empty"])
        inputs = ff_input + fb_input

        new_states, output = func(rng, params[params_namespace], states[states_namespace], *inputs)
        if store_output:
            new_states["output"] = output
        # if fb_namespace is define it must refers to a tag inside new_states dict
        return (states_namespace, new_states), (node_name, output)

    return apply_ff


def topological_sort_by_rank(ingoing_edges: dict[str, list[str]],
                             outgoing_edges: dict[str, list[str]]) -> list[list[str]]:
    no_ingoing_q: deque[tuple[str, int]] = deque()
    topo_sorted_set: list[list[str]] = [[]]
    temp_ingoing: dict[str, list[str]] = copy.deepcopy(ingoing_edges)
    for key in ingoing_edges.keys():
        if not ingoing_edges[key]:
            no_ingoing_q.append((key, 0))
            topo_sorted_set[0].append(key)
    if len(no_ingoing_q) == 0:
        raise ValueError("Graph must contain as least one node with no incoming edges!")
    while no_ingoing_q:
        parent_key, parent_rank = no_ingoing_q.popleft()
        for child_key in outgoing_edges[parent_key]:
            temp_ingoing[child_key].remove(parent_key)
            if not temp_ingoing[child_key]:
                child_pair = (child_key, parent_rank + 1)
                no_ingoing_q.append(child_pair)
                if len(topo_sorted_set) == parent_rank + 1:
                    topo_sorted_set.append([])
                topo_sorted_set[parent_rank + 1].append(child_key)

    for i in range(len(topo_sorted_set)):
        # remove duplicates will preserving order
        topo_sorted_set[i] = list(dict.fromkeys(topo_sorted_set[i]))

    return topo_sorted_set
