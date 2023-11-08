from cgi import test
from typing import Any, NamedTuple, Optional

import jax
import jax.numpy as jnp
from chex import PRNGKey, Array

from gym import spaces
from fax.rl.types import Env, ObsType, ActorState, EnvState
from fax.rl.datasets import omniglot_embedding_space
from itertools import permutations
import numpy as np
class APState(NamedTuple):
    step_count: int
    states: Optional[tuple[jnp.array, int]]
    targets: Optional[jnp.array]
    key: PRNGKey
    info: dict[str, Any]

class AssociatedPairs(Env):
    def __init__(self,
                 nb_states: int,
                 nb_store_state: int,
                 hops: int,
                 state_dim: int,
                 state_type: str,
                 base_reward: float,
                 success_reward: float,
                 failure_reward: float,
                 unique_pair: bool,
                 generator_params: dict,
                 space_key: PRNGKey):
        self.space_generator = self.space_creation(
            state_type, nb_states, state_dim, space_key, generator_params)

        self.unique_pair = unique_pair
        self.failure_reward: float = failure_reward
        self.success_reward: float = success_reward
        self.base_reward: float = base_reward
        self.state_type: str = state_type
        self.state_dim: int = state_dim
        self.nb_store_state: int = nb_store_state
        self.hops: int = hops
        self.nb_states: int = nb_states
        choices_list = jnp.array(sorted(
            list(set(permutations(np.arange(self.nb_states))))), 
            dtype=jnp.int32)
        if generator_params["test_env"]:
            self.choices = choices_list
        else:
            self.choices = choices_list
        self.action_space: spaces.Space = spaces.Discrete(nb_states)
        self.observation_space: spaces.Space = spaces.Tuple(
            (spaces.Box(-jnp.inf, jnp.inf, (2 * self.state_dim,)),
             spaces.Discrete(3)))

    def step(self, env_state: APState, action):
        prev_obs = jax.tree_map(
            lambda x: x[env_state.step_count - 1], env_state.states)
        new_obs = jax.tree_map(
            lambda x: x[env_state.step_count], env_state.states)
        new_obs_type: int = new_obs[1]
        prev_obs_type: int = prev_obs[1]
        flag = jnp.logical_or(new_obs_type == ObsType.terminal,
                              jnp.logical_and(new_obs_type == prev_obs_type, 
                                              new_obs_type == ObsType.recall))
        
        new_reward = jax.lax.select(flag,
                                    jax.lax.select(
                                        action == env_state.targets[-1],
                                        self.success_reward/self.hops,
                                        self.failure_reward/self.hops),
                                    self.base_reward)
                                
        new_env_state = APState(
            step_count=env_state.step_count + 1,
            key=env_state.key,
            states=env_state.states,
            targets=env_state.targets,
            info=env_state.info
            )
        actor_state = ActorState(
            obs=new_obs,
            reward = new_reward,
            done = new_obs[1] == ObsType.terminal,
            info={
                "mask": new_obs[1] == ObsType.recall,
                "temporal_mask": flag,
                "target": env_state.targets[-1],
                "context": env_state.targets[-1]
                })
        return new_env_state, actor_state

    def reset(self, key: PRNGKey):
        number_idx = jnp.arange(self.nb_states, dtype=jnp.int32)
        new_key, *choices_keys = jax.random.split(key, 7)
        # give which A class are associated with B class
        # where B class labels are fixed
        p1_perm = jax.random.permutation(choices_keys[1], self.nb_states)
        # give order over the class of B
        # choices = jax.random.choice(
        #     choices_keys[2], self.nb_states, (self.nb_store_state,),
        #     replace=not self.unique_pair)
        choice_idx = jax.random.randint(choices_keys[2], (), 
                                        minval=0, maxval=len(self.choices))
        choices = self.choices[choice_idx]
        p1_idx = p1_perm[choices]
        p2_idx = number_idx[choices]
        states = self.space_generator(p1_idx, p2_idx, choices_keys[3])
        # choose a pair association id for query
        queried_id = jax.random.choice(choices_keys[4], choices, (1,))
        p1_query_idx = p1_perm[queried_id]
        target_idx = number_idx[queried_id]
        query = self.space_generator(p1_query_idx, target_idx, choices_keys[5])
        query = query.at[0, self.state_dim:].set(0.0)
        query = jnp.tile(query, (self.hops, 1))
        terminal_state = jnp.expand_dims(jnp.zeros_like(states[0]), 0)
        states = jnp.concatenate((states, query, terminal_state), axis=0)
        states_type = [ObsType.store] * self.nb_store_state \
            + [ObsType.recall] * self.hops + [ObsType.terminal]
        states_type = jnp.array(states_type)
        p2_target = jnp.concatenate((p2_idx, target_idx))
        new_state = states[0]
        new_type = states_type[0]
        new_obs = (new_state, new_type)
        states = (states, states_type)
        env_state = APState(step_count=1,
                            states=states,
                            targets=p2_target,
                            key=new_key,
                            info={})
        actor_state = ActorState(
            obs=new_obs,
            reward=self.base_reward,
            done=new_obs[1] == ObsType.terminal,
            info={"mask": new_obs[1] == ObsType.recall,
                  "temporal_mask": new_obs[1] == ObsType.terminal,
                  "target": p2_target[-1],
                  "context": p2_target[-1]
                  })
        return env_state, actor_state

    def render(self, mode="human"):
        pass

    @staticmethod
    def space_creation(state_type: str, nb_states: int, state_dim,
                       env_key: PRNGKey, generator_params: dict):
        if state_type == "uniform":
            key, *subkeys = jax.random.split(env_key, 3)
            p1_space = jax.random.uniform(
                subkeys[0], (nb_states, state_dim), minval=-1.0, maxval=1.0)
            p2_space = jax.random.uniform(
                subkeys[1], (nb_states, state_dim), minval=-1.0, maxval=1.0)
            
            def gen(a: Array, b: Array, key: PRNGKey):
                p1_data = p1_space[a]
                p2_data = p2_space[b]
                res = jnp.concatenate((p1_data, p2_data), axis=-1)
                return res
            return gen
        
        elif state_type == "one_hot":
            p1_space = jnp.eye(nb_states, state_dim, dtype=jnp.float32)
            p2_space = jnp.array(p1_space)
            
            def gen(a: Array, b: Array, key: PRNGKey):
                p1_data = p1_space[a]
                p2_data = p2_space[b]
                res = jnp.concatenate((p1_data, p2_data), axis=-1)
                # noise = 0.1 * jax.random.normal(key, res.shape)
                #
                # res = res + noise
                return res
            return gen
                    
        elif state_type == "cluster":
            from sklearn.datasets import make_classification
            n_samples = generator_params.get("n_samples", 2 * 1_00)
            n_clusters_per_class = generator_params.get(
                "n_clusters_per_class", 1)
            flip_y = generator_params.get("flip_y", 0.0)
            class_sep = generator_params.get("class_sep", 16)
            k1, k2 = jax.random.split(env_key)

            x1_seed = int(k1[0])
            x2_seed = int(k2[1])
            x1, y1 = make_classification(
                n_samples=n_samples, n_features=state_dim,
                n_informative=state_dim, n_redundant=0, n_repeated=0,
                n_classes=nb_states, scale=0.01,
                n_clusters_per_class=n_clusters_per_class, flip_y=flip_y,
                class_sep=class_sep, shuffle=True, random_state=x1_seed)
            x2, y2 = make_classification(
                n_samples=n_samples, n_features=state_dim, 
                n_informative=state_dim, n_redundant=0, n_repeated=0,
                n_classes=nb_states, scale=0.01,
                n_clusters_per_class=n_clusters_per_class, flip_y=flip_y,
                class_sep=class_sep, shuffle=True, random_state=x2_seed)
            p1_space = [[] for _ in range(nb_states)]
            p2_space = [[] for _ in range(nb_states)]
            for i in range(n_samples):
                p1_space[y1[i]].append(x1[i])
                p2_space[y2[i]].append(x2[i])
            p1_space = jnp.array(p1_space)
            p2_space = jnp.array(p2_space)
            
            def gen(a: Array, b: Array, key: PRNGKey):
                p1_data = p1_space[a]
                p2_data = p2_space[b]
                k1, k2 = jax.random.split(key, 2)
                p1_data_idx = jax.random.choice(
                    k1, len(n_samples), (len(a),))
                p2_data_idx = jax.random.choice(
                    k2, len(n_samples), (len(b),))
                p1_data = p1_data[jnp.arange(len(a)), p1_data_idx]
                p2_data = p2_data[jnp.arange(len(b)), p2_data_idx]
                res = jnp.concatenate((p1_data, p2_data), axis=-1)
                # noise = 0.1 * jax.random.normal(key, res.shape)
                #
                # res = res + noise
                return res
            return gen
        elif state_type == "omniglot":
            test_env = generator_params.get("test_env", False)
            omniglot_path = generator_params.get(
                "omniglot_path", 
                "./datasets/omniglot_proto_emb/train_set")


            k1, k2 = jax.random.split(env_key)
            x1_seed = int(k1[0])
            x1_datax, x1_datay, x1_selected_classes = omniglot_embedding_space(
                omniglot_path, nb_classes=nb_states, seed=x1_seed)
            
            x2_seed = int(k2[1])
            x2_datax, x2_datay, x2_selected_file = omniglot_embedding_space(
                omniglot_path, nb_classes=nb_states,
                excluded_classes=x1_selected_classes, seed=x2_seed)
            fixed_sample = generator_params.get("fixed_sample", False)
            p1_space = x1_datax
            p2_space = x2_datax
            
            def gen(a: Array, b: Array, key: PRNGKey):
                p1_data = p1_space[a]
                p2_data = p2_space[b]
                if fixed_sample:
                    p1_data = p1_data[jnp.arange(len(a)), 0]
                    p2_data = p2_data[jnp.arange(len(b)), 0]
                    
                else:
                    k1, k2 = jax.random.split(key, 2)
                    if test_env:
                        p1_data_idx = 19
                        p2_data_idx = 19
                    else:
                        p1_data_idx = jax.random.choice(k1, 19, (len(a),))
                        p2_data_idx = jax.random.choice(k2, 19, (len(b),))
                    p1_data = p1_data[jnp.arange(len(a)), p1_data_idx]
                    p2_data = p2_data[jnp.arange(len(b)), p2_data_idx]
                res = jnp.concatenate((p1_data, p2_data), axis=-1)
                return res
            return gen

        else:
            raise ValueError(f"Type {state_type} was not understood")