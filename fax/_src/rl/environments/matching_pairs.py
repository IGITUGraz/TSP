from typing import Any, NamedTuple, Optional

import jax
import jax.numpy as jnp
from chex import PRNGKey, Array

from gym import spaces
from fax.rl.datasets import omniglot_embedding_space
from fax.rl.types import EnvState, ActorState, Env, ObsType

class MPState(NamedTuple):
    step_count: int
    states: Optional[tuple[jnp.array, int]]
    context: int
    target: int
    key: PRNGKey
    info: dict[str, Any]
    
class MatchingPairs(Env):
    def render(self, mode="human"):
        raise NotImplementedError

    def __init__(self,
                 nb_states: int,
                 episode_length: int,
                 state_dim: int,
                 state_type: str,
                 base_reward: float,
                 success_reward: float,
                 failure_reward: float,
                 reward_on_match_proba: float,
                 context_type: str,
                 generator_params: dict,
                 space_key: PRNGKey):
        space_key, *context_keys = jax.random.split(space_key, 3)
        self.space_generator = self.space_creation(
            context_type, state_type, nb_states, state_dim, space_key, generator_params)
        self.failure_reward: float = failure_reward
        self.success_reward: float = success_reward
        self.base_reward: float = base_reward
        self.state_type: str = state_type
        self.state_dim: int = state_dim
        self.episode_length: int = episode_length
        self.nb_states: int = nb_states
        self.reward_on_match_proba = reward_on_match_proba
        self.context_type = context_type
        if context_type == "before_obs":
            matching_rpz = jnp.eye(2, state_dim, dtype=jnp.float32)
            self.matching_context = matching_rpz[0]
            self.not_matching_context = matching_rpz[1]
            output_dim = (self.state_dim,)
        elif context_type == "concat_with_obs":
            self.matching_context = jnp.array([1.0, 0.0])
            self.not_matching_context = jnp.array([0.0, 1.0])
            output_dim = (self.state_dim + 2,)
        else:
            output_dim = (self.state_dim + 2,)
        self.action_space: spaces.Space = spaces.Discrete(2)
        self.observation_space: spaces.Space = spaces.Tuple(
            (spaces.Box(-jnp.inf, jnp.inf, output_dim), spaces.Discrete(3)))
        self.states: Optional[tuple[jnp.array, int]] = None
        self.target: int

    def step(self, env_state: MPState, action):
        new_obs = jax.tree_map(lambda x: x[env_state.step_count],
                               env_state.states)
        is_terminal = new_obs[1] == ObsType.terminal.value
        new_reward = self.base_reward + is_terminal * jax.lax.select(
            action == env_state.target,
            self.success_reward, 
            self.failure_reward)
        
        env_state = MPState(step_count=env_state.step_count + 1,
                            states=env_state.states,
                            target=env_state.target,
                            context=env_state.context,
                            key=env_state.key,
                            info={})
        actor_state = ActorState(
            obs=new_obs,
            reward=new_reward,
            done=is_terminal,
            info={"mask": new_obs[1] == ObsType.recall,
                  "context": env_state.context,
                  "target": env_state.target,
                  "temporal_mask": new_obs[1] == ObsType.terminal,
                  })
        return env_state, actor_state

    def reset(self, key: PRNGKey):
        new_key, *choices_keys = jax.random.split(key, 3)
        # must push button if match (True case), 
        # or must push_button if not match (False case)
        reward_on_match = jax.random.bernoulli(choices_keys[0],
                                          self.reward_on_match_proba, ())
        
        does_match, states, states_type = self.space_generator(
            self.episode_length, choices_keys[1])
        # does_match: True pairs are matching, False pair are not matching
        target = (reward_on_match == does_match).astype(int)
        does_match = does_match.astype(int)
        # target = [reward_on_match] * (self.noise_length+1) + [target]
        # target = jnp.array(target)
        # target 0 if agent need to press button if reward is on match and 
        # pairs are matching, or reward is on not match and the two pairs are not
        # matching
        # target is 1 if agent do not need to press anything
        

        if self.context_type == "before_obs":
            context_list = reward_on_match * self.matching_context \
                + (1 - reward_on_match) * self.not_matching_context
            context_array = jnp.array(context_list, dtype=jnp.float32)
            context_array = jnp.expand_dims(context_array, 0)
            states = jnp.concatenate((context_array, states), axis=0)
        elif self.context_type == "concat_with_obs":
            context_list = reward_on_match * self.matching_context \
                + (1 - reward_on_match) * self.not_matching_context
            context_list = jnp.array(context_list, dtype=jnp.float32)
            # context_array = jnp.expand_dims(context_array, 0)
            context_array = jnp.zeros((len(states), 2))
            context_array = context_array.at[0].set(context_list)
            
            # context_array = jnp.tile(
            #     context_array, (len(states), 1))
            states = jnp.concatenate((states, context_array,), axis=1)

        new_state = states[0]
        new_type = states_type[0]
        new_obs = (new_state, new_type)
        states = (states, states_type)
        env_state = MPState(step_count=1,
                            states=states,
                            target=target,
                            context=reward_on_match.astype(int),
                            key=new_key,
                            info={})
        actor_state = ActorState(
            obs=new_obs,
            reward=self.base_reward,
            done=False,
            info={"mask": new_obs[1] == ObsType.recall,
                  "context": env_state.context,
                  "target": target,
                  "temporal_mask": new_obs[1] == ObsType.terminal
                  })
        return env_state, actor_state

    @staticmethod
    def space_creation(context_type: str, state_type_str: str, nb_states: int, state_dim: int,
                       space_key: PRNGKey, generator_params: dict):
        if state_type_str == "uniform":
            space_key, subkeys = jax.random.split(space_key, 2)
            objects_rpz = jax.random.uniform(subkeys, (nb_states, state_dim),
                                             minval=-1.0, maxval=1.0)
            real_dim = state_dim
        elif state_type_str == "one_hot":
            objects_rpz = jnp.eye(nb_states, state_dim, dtype=jnp.float32)
            real_dim = state_dim
        elif state_type_str == "omniglot":
            omniglot_path = generator_params.get(
                "omniglot_path", "./datasets/omniglot_proto_emb/train_set")
            space_key, k1 = jax.random.split(space_key)
            x1_seed = int(k1[0])
            x1_datax, x1_datay, x1_selected_file = omniglot_embedding_space(
                omniglot_path, nb_classes=nb_states, seed=x1_seed)
            objects_rpz = x1_datax
            real_dim = state_dim
        else:
            raise ValueError(f"Type {state_type_str} was not understood")
            

        def gen(episode_length: int, key: PRNGKey):
            new_key, *subkeys = jax.random.split(key, 6)
            need_to_match = jax.random.bernoulli(subkeys[0], p=0.5)
            data = jax.random.normal(
                subkeys[1], (episode_length + 1, real_dim))
            # data = jnp.ones((episode_length + 1, real_dim))
            # in_btw_noise = jnp.ones((noise_length, real_dim), dtype=jnp.float32)
            # prevent to construct a while loop in order to have
            # p1_object_idx != p2_object_idx when need_to_match is False
            idx = jax.random.choice(subkeys[2],
                                    nb_states, shape=(2,), replace=False)
            p1_object_idx = idx[0]
            p2_object_idx = jax.lax.select(need_to_match,
                                           p1_object_idx, idx[1])
            p1_object = objects_rpz[p1_object_idx]
            p2_object = objects_rpz[p2_object_idx]
            if state_type_str == "omniglot":
                if generator_params.get("fixed_sample", True):
                    p1_object = p1_object[0]
                    p2_object = p2_object[0]
                else:
                    if generator_params["test_env"]:
                        k1, k2 = jax.random.split(new_key, 2)
                        p1_data_idx, p2_data_idx = jax.random.choice(
                            k1, jnp.array([19, 20], dtype=int),
                            shape=(2,), replace=False)
                        # p2_data_idx = jax.random.randint(
                        #     k2, shape=(), minval=18, maxval=20)
                        p1_object = p1_object[p1_data_idx]
                        p2_object = p2_object[p2_data_idx]
                    else:
                        k1, k2 = jax.random.split(new_key, 2)
                        p1_data_idx = jax.random.randint(
                            k1, shape=(), minval=0, maxval=18)
                        p2_data_idx = jax.random.randint(
                            k2, shape=(), minval=0, maxval=18)
                        p1_object = p1_object[p1_data_idx]
                        p2_object = p2_object[p2_data_idx]
            if not generator_params["test_env"] \
                and generator_params["dynamic_length"]:
                p2_idx = jax.random.choice(subkeys[3], 
                                            jnp.arange(1, episode_length), shape=())
            else:
                p2_idx = episode_length - 1
                        
            data = data.at[0].set(p1_object)
            data = data.at[p2_idx].set(p2_object)
            if context_type == "before_obs":
                states_type = jnp.array(
                    [ObsType.store] * (episode_length + 2))
                
                states_type = states_type.at[p2_idx + 1].set(ObsType.recall)
            else:
                states_type = jnp.array(
                    [ObsType.store] * (episode_length + 1))
            idx_pos = jnp.arange(episode_length + 1)
            terminal_states = jnp.array([ObsType.terminal] * (episode_length + 1))
            terminal_data = jnp.zeros_like(data, dtype=data.dtype)
            flags = idx_pos > p2_idx
            states_type = jnp.where(flags, terminal_states, states_type)
            flags = jnp.expand_dims(flags, 1)
            data = jnp.where(flags, terminal_data, data)
            states_type = states_type.at[p2_idx].set(ObsType.recall)
            return need_to_match, data, states_type

        return gen
