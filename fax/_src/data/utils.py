from functools import update_wrapper
import imp
from typing import Hashable, Mapping, NamedTuple, Optional, Union
# from mlflow.tracking.client import MlflowClient
# from mlflow.entities import ViewType
import jax
import copy
from optax import multi_transform
from optax._src import base
from optax._src import wrappers

def convert_dict_to_readable_string(dictionary: dict):
    def _rec_convert(sub_dict: dict):
        global_dict_str = ""
        for k, v in sub_dict.items():
            if isinstance(v, dict):
                sub_dict_str = f"{str(k)}=\u007b{_rec_convert(v)}\u007d"
            else:
                sub_dict_str = f"{str(k)}={str(v).replace(' ', '')}"
            global_dict_str += sub_dict_str + "_"
        return global_dict_str[:-1]

    return _rec_convert(dictionary)


def hash_dictionary(dictionary: dict):
    import hashlib
    import json
    return hashlib.sha256(json.dumps(dictionary, default=str, sort_keys=True).encode("ascii"),
                          usedforsecurity=False).hexdigest()

def simple_dict_cache(user_function):
    cache = {}
    sentinel = object()
    hits = misses = 0
    def simple_dict_cache_wrapper(dict_object):
        if not isinstance(dict_object, dict):
            raise TypeError(
                f"function wait a dict object but was provided with {type(dict_object)}")
        nonlocal hits, misses
        key = hash_dictionary(dict_object)
        result = cache.get(key, sentinel)
        if result is not sentinel:
            hits += 1
            return result
        misses += 1
        result = user_function(dict_object)
        cache[key] = result
        return result
    def cache_info():
        return {"hits": hits, "misses": misses, "cache_size": len(cache)}
    def clear_cache():
        nonlocal cache, hits, misses
        cache = {}
        hits = misses = 0
    
    simple_dict_cache_wrapper.cache_info = cache_info
    simple_dict_cache_wrapper.clear_cache = clear_cache
    return update_wrapper(simple_dict_cache_wrapper, user_function)

# def get_mlflow_runs(experiment_id: Union[str, int], 
#                     filter_string: str = "", max_results: int = 1,
#                     order_by: Optional[str] = None):
#     client = MlflowClient(tracking_uri="./data/mlflow")
#     client.list_experiments
#     print(client)
#     if isinstance(experiment_id, str):
        
#         exp = client.get_experiment_by_name(experiment_id)
#         print(exp)
#         experiment_id = exp.experiment_id
#     runs = client.search_runs(experiment_ids=experiment_id, 
#                                       filter_string=filter_string,
#                                       run_view_type=ViewType.ACTIVE_ONLY,
#                                       max_results=max_results,
#                                       order_by=order_by
#                                       )
#     return runs


def in_set_or_empty_set(element, _set):
    
    return not _set or element in _set

def is_in_singleton_or_list_empty(element, _list):
    
    return (not _list) or (len(_list) == 1 and _list[0] == element)
def tagged_tree_labeling(label, prefix_list: list[str],
                         postfix_list: list[str], base_tree):
    """Label each leaf of the tagged tree with "label" value, 
    that correspond of any paths respecting the following structure:
    (/\w+)*/prefix_list(/\w+)*/postfix_list(/\w+)*
    where \w correspond to any alpha-numeric value
    and prefix_list and postfix_list must be understood as regex unions.
    empty list means that no restrictions are made on prefix or postfix 
    
    example:
    tag_tree = {a: {b: 1, c :{b: 2, d: 3}}}
    possible paths can be written as parent/child:
    a/b
    a/b/1
    a/c
    a/c/b
    a/c/d
    a/c/b/2
    a/c/d/3
    where 1,2,3 are leafs
    
    prefix: [a], postfix: empty
    change leafs,1,2,3 from a/b, a/c/b, a/c/d to label value
    prefix: [a], postfix: [b]
    change leaf 1 from a/b to label value
    prefix: [a], postfix: [c]
    change leaf 2,3 from a/c/b and a/c/d to label value
    prefix: empty, postfix: [b]
    change leaf 1,2 from a/b and a/c/b to label value
    Args:
        label (str): a string label
        prefix_list (list[str]): 
        postfix_list (list[str]): 
        base_tree (dict): base label tree were each leaf contain 
        a fallback label

    Returns:
        _type_: tagged_tree where leaf are label value or fallback value
        depending on the prefix and postfix constraints
    """
    
    postfix_set = set(postfix_list)
    if prefix_list == [] or prefix_list is None:
        prefix_list = [""]
        
    def _rec_label(_prefix_list, sub_tree):
        new_sub_tree = {}
        for k in sub_tree.keys():
            if not _prefix_list and in_set_or_empty_set(k, postfix_set):
                new_sub_tree[k] = jax.tree_map(lambda _: label, sub_tree[k])
            elif isinstance(sub_tree[k], dict):
                if _prefix_list and _prefix_list[0] == k:
                    new_prefix_list = _prefix_list[1:]
                else:
                    new_prefix_list = _prefix_list
                new_sub_tree[k] = _rec_label(new_prefix_list, sub_tree[k])
            else:
                new_sub_tree[k] = sub_tree[k]
        return new_sub_tree

    for prefix in prefix_list:
        prefix = prefix.split("/")
        prefix = [] if prefix == [""] else prefix
        base_tree = _rec_label(prefix, base_tree)
    return base_tree

def label_struct_to_label_function(labels_struct):
    """multi_tranform_wrapper rewrite the multi_tranform procedure
    with tagged tree as base structure.
    rewrite need to be done because even if we can fit the procedure on tagged tree
    inside of function returning a pytree from a set of parameters 
    this function will be costly and label_tree is better to store inside

    Args:
        transforms (Mapping[Hashable, base.GradientTransformation]): _description_
        labels_struct (_type_): _description_

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    # multi_tranform need to be total function of the leafs,
    # the best way to do it without adding all possible path in the tree is
    # to add a fallback label
    @simple_dict_cache
    def base_labels_to_labels(base_labels:dict):
        labelled_tag_tree = base_labels
        for label in labels_struct.keys():
            labelled_tag_tree = tagged_tree_labeling(
                label, labels_struct[label]["prefix"],
                labels_struct[label]["postfix"], labelled_tag_tree)
        return labelled_tag_tree
    
    def label_func(params:dict):
        base_labels = jax.tree_map(lambda _: "fallback", params)
        base_labels = base_labels_to_labels(base_labels)
        return base_labels
    return label_func

    
    