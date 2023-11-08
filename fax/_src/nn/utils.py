import binascii
import os
import types
from typing import Any, Optional
import optax

from sklearn.model_selection import KFold
from fax.data.utils import label_struct_to_label_function
import jax

def get_data_shape(data: Any, number_of_dim_to_abstract: int):
    return jax.tree_map(lambda x: jax.numpy.shape(x[(0,) * number_of_dim_to_abstract]), data)


def maybe_set_random_seed(seed: Optional[int] = None):
    if seed is None:
        seed = int(binascii.hexlify(os.urandom(4)), 16)
    return seed

def hyper_opt_labels(transforms, label_struct):
    label_func = label_struct_to_label_function(label_struct)
    def label_data_func(data, label):
        from ray import tune
        if label ==  "fallback":
            return data
        else:
            return eval(transforms[label])
    def config_to_hyper_config(config):
        labels = label_func(config)
        config_tree = jax.tree_map(label_data_func, config, labels)
        return config_tree
    return config_to_hyper_config

def tree_stop_gradient_wrapper(labels_struct):
    stop_key = list(labels_struct.keys())[0]
    if stop_key == "fallback":
        raise ValueError("stop label cannot be name fallback")
    label_func = label_struct_to_label_function(labels_struct)

    _func_map = {stop_key: lambda x: jax.lax.stop_gradient(x),
                 "fallback": lambda x: x}
    def stop_gradient(params):
        labels = label_func(params)
        return jax.tree_map(
            lambda data, label: _func_map[label](data), params, labels)
    return stop_gradient
def tree_to_transformed_states_wrapper(labels_struct):
    """ Allow to determine states post-traitement procedure for states 
    the differents wanted cases are define as keep, reduce and reset (fallback case)
    
    the keep case do not touch states and is just identity function
    the reduce case apply a function tranforming the states
    the reset case, express as the fallback label, 
    will reset to a provided intitial state
    Args:
        labels_struct (dict): a dictonary de label containing prefix and postfix
        structure in order to determine which leafs need to be labeled 

    Returns:
        _type_: a f(states, base_states, optional(func)) function that produce
        operations on the tree determined by the 
    """
    #TODO: it could be interesting to allow any (stateful) tranformation
    # in the config but this will need to create very complexe setup 
    # example being if we want to chain tranformation over a sub-tree
    # this would need to create function that a full definition "name" + hyperparameters
    # for each function 
    label_func = label_struct_to_label_function(labels_struct)
    #TODO: function that do not have states define an empty state dictonary,
    # this is semantically sound but for jax this will mean that the empty dictonary
    # have as a leaf the atomic None element
    _func_map = {"fallback": lambda _, b_s: b_s,
                 "keep": lambda s, _: s}
    def states_transform(states, base_states, func = None):
        if func is not None:
            _func_map["reduce"] = lambda s, _: func(s)
        states_labels = label_func(states)
        return jax.tree_map(lambda s, b_s, l: _func_map[l](s, b_s),
                     states, base_states, states_labels)
    return states_transform



def copy_func(f, name):
    WRAPPER_ASSIGNMENTS = ('__kwdefaults__', '__module__', '__doc__',
                           '__annotations__', '__dict__')
    print(f.__closure__)
    """Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)"""
    g = types.FunctionType(f.__code__, f.__globals__, name=name,
                           argdefs=f.__defaults__,
                           closure=f.__closure__)
    g.__qualname__ = name
    for attr in WRAPPER_ASSIGNMENTS:
        try:
            value = getattr(f, attr)
        except AttributeError:
            pass
        else:
            setattr(g, attr, value)
    return g
