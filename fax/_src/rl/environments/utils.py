import functools
from pathlib import Path
from typing import Optional
from chex import PRNGKey
import numpy as np
import jax
from jax import numpy as jnp
from omegaconf import OmegaConf
from fax._src.config import AssociatedPairs, MatchingPairs, RadialMaze, BabiTaskRL
from fax._src.rl.environments.types import VectorBabiWrapper
from fax.rl.environments import associated_pairs, matching_pairs, radial_maze, babi_task
from fax.rl.types import VectorWrapper, ObsType, Env


@functools.singledispatch
def resolve_task(task_params: object,
                 independent_test_env: bool, key: PRNGKey):
    raise NotImplementedError(
        f"Task of type {type(task_params)} is not implemented")


@resolve_task.register
def _(task_params: AssociatedPairs, independent_test_env: bool, key: PRNGKey):
    dict_task_params = task_params.__dict__
    batch_size = dict_task_params["batch"]
    del dict_task_params["batch"]
    if independent_test_env:
        k1, k2 = jax.random.split(key)
    else:
        k1 = k2 = key
    dict_task_params["generator_params"]["test_env"] = False
    train_env = associated_pairs.AssociatedPairs(**dict_task_params, space_key=k1)
    dict_task_params["generator_params"]["test_env"] = True
    test_env = associated_pairs.AssociatedPairs(**dict_task_params, space_key=k2)
    task_params.__dict__["batch"] = batch_size
    return VectorWrapper(train_env, batch_size), VectorWrapper(test_env, batch_size), task_params

@resolve_task.register
def _(task_params: BabiTaskRL, independent_test_env: bool, key: PRNGKey):
    """
        BabiTask metadata fields
    metadata = {
        "nb_sentences": max_num_sentences,
        "nb_words": max_words,
        "vocab_size": vocab_size,
        "nb_classes": answer_vocab_size if class_space == "answer_corpus" else vocab_size,
        "full_corpus_dict": full_corpus_idx,
        "answer_corpus_dict": answer_corpus_idx,
        "train_size": len(trainS),
        "valid_size": len(validS),
        "test_size": len(testS)
    }
    """
    from fax.data import read_numpy_data
    dict_task_params = task_params.__dict__
    train_set, valid_set, test_set, task_metadata = read_numpy_data(
        dict_task_params["babi_parameters"], "./datasets/tasks/babi")
    state_dim = task_metadata["nb_words"]
    nb_classes = task_metadata["nb_classes"]
    nb_story_train = task_metadata["train_size"]
    nb_story_test = task_metadata["test_size"]

    success_reward = dict_task_params["success_reward"]
    failure_reward = dict_task_params["failure_reward"]
    base_reward = dict_task_params["base_reward"]

    batch_size = dict_task_params["batch"]
    del dict_task_params["batch"]
    if independent_test_env:
        k1, k2 = jax.random.split(key)
    else:
        k1 = k2 = key
    dict_task_params["generator_params"]["test_env"] = False
    train_env = babi_task.BabiTask(
        x_data=train_set[0],
        y_data=train_set[1],
        nb_story=nb_story_train, nb_classes=nb_classes,
        state_dim=state_dim,
        success_reward=success_reward,failure_reward=failure_reward,
        base_reward=base_reward, 
        generator_params=dict_task_params["generator_params"], 
        space_key=k1)
    dict_task_params["generator_params"]["test_env"] = True
    test_env = babi_task.BabiTask(
        x_data=test_set[0],
        y_data=test_set[1],
        nb_story=nb_story_test, nb_classes=nb_classes,
        state_dim=state_dim,
        success_reward=success_reward,failure_reward=failure_reward,
        base_reward=base_reward, 
        generator_params=dict_task_params["generator_params"], 
        space_key=k1)
    task_params.__dict__["batch"] = batch_size
    task_params = OmegaConf.merge(task_params.__dict__, task_metadata)
    return VectorBabiWrapper(train_env, batch_size), VectorBabiWrapper(test_env, batch_size), task_params
    
@resolve_task.register
def _(task_params: MatchingPairs, independent_test_env: bool, key: PRNGKey):
    dict_task_params = task_params.__dict__
    batch_size = dict_task_params["batch"]
    del dict_task_params["batch"]
    if independent_test_env:
        k1, k2 = jax.random.split(key)
    else:
        k1 = k2 = key
    dict_task_params["generator_params"]["test_env"] = False
    train_env = matching_pairs.MatchingPairs(**dict_task_params, space_key=k1)
    dict_task_params["generator_params"]["test_env"] = True
    test_env = matching_pairs.MatchingPairs(**dict_task_params, space_key=k2)
    task_params.__dict__["batch"] = batch_size
    return VectorWrapper(train_env, batch_size), VectorWrapper(test_env, batch_size), task_params

@resolve_task.register
def _(task_params: RadialMaze, independent_test_env: bool, key: PRNGKey):
    dict_task_params = task_params.__dict__
    batch_size = dict_task_params["batch"]
    del dict_task_params["batch"]
    if independent_test_env:
        k1, k2 = jax.random.split(key)
    else:
        k1 = k2 = key
    dict_task_params["generator_params"]["test_env"] = False
    train_env = radial_maze.RadiaMaze(**dict_task_params, space_key=k1)
    dict_task_params["generator_params"]["test_env"] = True
    test_env = radial_maze.RadiaMaze(**dict_task_params, space_key=k2)
    task_params.__dict__["batch"] = batch_size
    return VectorWrapper(train_env, batch_size), VectorWrapper(test_env, batch_size), task_params


@functools.singledispatch
def evaluation_dispatch(key: PRNGKey, env: Env, model, initial_state: dict,
                        params: dict, env_state, eval_params: dict):
    # here after end of an episode we can sample obs from test_env 
    # in order to get latter evaluate differents representation
    # states and parameters must be fixed but sampling can be task 
    # dep so... maybe create function that dispatch
    # over environment
    # what we want:
    # the representations of obs
    #   obs can be in two context store and recall, 
    #   store can introduce context to obs (radial maze)
    #   
    # the representation of states:
    #   if we use store the representation of the states will change
    #   and we may want to see how this representation change
    #   but if we use recall obs
    #   the temporal normalizer change but not the states of hmem
    #   we probably don't want to return states of recall obs
    #   
    # introduce the following structure:
    # recall = {"rpz": [states], "context" : context_dic}
    # store = {"rpz": [states], "context": context_dic}
    # for radial maze (recall)
    # context_dic = {"context_id": int}
    # for radial maze (store)
    # context_dic = {"contex_id": [int], "action": [int], "reward": [int]}
    # for associated pairs:
    # context_dic = {"pair_a": [int], "pair_b": [int]}
    # for matching pair:
    # context_dic = {"a_id": int, "b_id": int, left_on_match: bool}
    pass

@evaluation_dispatch.register
def _(key: PRNGKey, env: radial_maze.RadiaMaze, model, initial_state: dict,
      params: dict, env_state: radial_maze.RadialState, eval_params: dict):
    # eval_params: 
    #   nb_sample: int
    #   obs_type: str (recall, store, both)
    data = []
    
    for context_id in range(env.nb_pairs):
        key, k1, k2 = jax.random.split(key, 3)
        obs_without_context = env.space_generator(context_id, k1)
        new_obs = jnp.concatenate((obs_without_context,
                                   jnp.array([0.0, 0.0, 0.0, 0.0])),
                                  axis=-1)
        new_obs = (new_obs, ObsType.recall)
        before_store_states, out = model(k2, params, initial_state, new_obs)
        reward_location = env_state.reward_locations[context_id]
        if reward_location == 0:
            # correct prediction simulation
            action_rpz_correct = env.action_rpz[0]
            reward_rpz_correct = env.reward_rpz[0]
            
            action_rpz_wrong = env.action_rpz[1]
            reward_rpz_wrong = env.reward_rpz[1]
        else:
            action_rpz_correct = env.action_rpz[1]
            reward_rpz_correct = env.reward_rpz[0]
            
            action_rpz_wrong = env.action_rpz[0]
            reward_rpz_wrong = env.reward_rpz[1]
            
        new_obs_correct = jnp.concatenate(
            (obs_without_context, action_rpz_correct, reward_rpz_correct),
            axis=-1)
        new_obs_correct = (new_obs_correct, ObsType.store)
        
        new_obs_wrong = jnp.concatenate(
            (obs_without_context, action_rpz_wrong, reward_rpz_wrong),
            axis=-1)
        new_obs_wrong = (new_obs_wrong, ObsType.store)
        key, sub_key_1, sub_key_2 = jax.random.split(key, 3)
        
        states_store_correct, out = model(
            sub_key_1, params, before_store_states, new_obs_correct)
        states_store_wrong, out = model(
            sub_key_2, params, before_store_states, new_obs_wrong)
        
        key, sub_key_1, sub_key_2 = jax.random.split(key, 3)
        new_obs = jnp.concatenate((obs_without_context,
                                   jnp.array([0.0, 0.0, 0.0, 0.0])),
                                  axis=-1)
        
        new_obs = (new_obs, ObsType.recall)
        states_recall_after_store_correct, out = model(
            sub_key_1, params, states_store_correct, new_obs)
        states_recall_after_store_wrong, out = model(
            sub_key_2, params, states_store_wrong, new_obs)
        
        new_store = {
            "states_before_recall": before_store_states,
            "states_store_correct": states_store_correct,
            "states_store_wrong": states_store_wrong,
            "states_recall_after_store_correct":
                states_recall_after_store_correct,
            "states_recall_after_store_wrong":
                states_recall_after_store_wrong,
            "context": {
                "context_id": context_id,
                "reward_location": reward_location,
                }
            }
        data.append(new_store)
    data = \
        jax.tree_util.tree_map(lambda *x: np.stack(x, axis=0),
                               *data)
    eval_dic = {"eval": {"data": data,
                         "init_state": initial_state,
                         "reward_locations": env_state.reward_locations,
                         }}
    return eval_dic
