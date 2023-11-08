from typing import Dict, Any
from typing import NamedTuple
import jax
import jax.numpy as jnp
from chex import PRNGKey, Array
from gym import spaces
from fax.rl.datasets import omniglot_embedding_space
from fax.rl.types import Env, ActorState, ObsType


class RadialState(NamedTuple):
    create_store_obs: bool
    step_count: int
    selected_pair: Array
    last_obs: Array
    reward_locations: Array
    key: PRNGKey
    info: Dict[str, Any]


class RadiaMaze(Env):
    def render(self, mode="human"):
        raise NotImplementedError

    def __init__(self,
                 nb_pairs: int,
                 switch_reward_rule: str,
                 switch_pair_rule: str,
                 state_type: str,
                 state_dim: int,
                 base_reward: float,
                 success_reward: float,
                 failure_reward: float,
                 horizon: int,
                 generator_params: dict,
                 space_key: PRNGKey):
        self.nb_pairs = nb_pairs
        self.switch_reward_rule = switch_reward_rule
        self.switch_pair_rule = switch_pair_rule
        self.state_dim = state_dim
        self.base_reward = jnp.array(base_reward, dtype=jnp.float32)
        self.success_reward = jnp.array(success_reward, dtype=jnp.float32)
        self.failure_reward = jnp.array(failure_reward, dtype=jnp.float32)
        self.state_type = state_type
        self.space_generator = self.space_creation(
            state_type, nb_pairs, state_dim, space_key, generator_params)
        self.horizon = horizon

        def uniform_pair_selection(_key: PRNGKey, selected_pair: int):
            return jax.random.randint(_key, (), minval=0, maxval=self.nb_pairs)

        def circular_pair_selection(_key: PRNGKey, selected_pair: int):
            return (selected_pair + 1) % self.nb_pairs
        if self.switch_pair_rule == "uniform":
            self.pair_selection = uniform_pair_selection
        elif self.switch_pair_rule == "circular":
            self.pair_selection = circular_pair_selection
        else:
            raise NotImplementedError(
                f"Pair selection method {self.switch_pair_rule} "
                "is not implemented")
        # for each pairs of branches, one branch is baited, the agent cannot
        # see the bait, we use 0 as
        # "left branch is baited and 1 for "right branch is baited"
        self.action_space: spaces.Space = spaces.Discrete(2)
        self.observation_space: spaces.Space = spaces.Tuple(
                (
                    spaces.Box(-jnp.inf, jnp.inf, (self.state_dim + 4,)),
                    spaces.Discrete(3)
                )
            )
        self.switch_baited = self.switch_reward_rule in ["everytime", 
                                                         "on_reward"]
        k1, k2 = jax.random.split(space_key)
        self.reward_rpz = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        self.action_rpz = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        self.rewards = jnp.array(
            [
                self.success_reward / self.horizon,
                self.failure_reward / self.horizon
            ],
            dtype=jnp.float32)

        def store_obs(args):
            _, action, last_obs, previous_pair_idx, previous_reward_location = args
            baited_branch = previous_reward_location[previous_pair_idx]
            reward_idx = jax.lax.select(baited_branch == action, 0, 1)

            reward_rpz = self.reward_rpz[reward_idx]
            action_rpz = self.action_rpz[action]
            reward = self.rewards[reward_idx]
            new_obs = jnp.concatenate(
                (last_obs, reward_rpz, action_rpz),
                axis=-1)
            new_obs = (new_obs, ObsType.store)
            results = (reward, last_obs, new_obs, previous_pair_idx,
                       previous_reward_location, baited_branch)
            return results

        def recall_obs(args):
            key, _, _, previous_pair_idx, previous_reward_location = args
            idx_key, sample_key = jax.random.split(key, 2)
            previous_baited_branch = previous_reward_location[
                previous_pair_idx]
            new_baited_branch = (previous_baited_branch + 1) % 2
            reward_locations = jax.lax.select(self.switch_baited,
                                              previous_reward_location
                                              .at[previous_pair_idx]
                                              .set(new_baited_branch),
                                              previous_reward_location)
            selected_pair_idx = self.pair_selection(idx_key, previous_pair_idx)
            new_obs_without_context = self.space_generator(selected_pair_idx,
                                                           sample_key)
            baited_branch = reward_locations[selected_pair_idx]
            reward = self.base_reward
            new_obs = jnp.concatenate((new_obs_without_context,
                                       jnp.array([0.0, 0.0, 0.0, 0.0])
                                       ),
                                      axis=-1)
            new_obs = (new_obs, ObsType.recall)
            results = (reward, new_obs_without_context, new_obs,
                       selected_pair_idx, reward_locations, baited_branch)
            return results
        self._store_obs = store_obs
        self._recall_obs = recall_obs

    def step(self, env_state: RadialState, action):
        step_count = env_state.step_count + 1
        is_terminal = (step_count >= self.horizon*2)
        key = env_state.key
        new_key, sub_key = jax.random.split(key)
        results = jax.lax.cond(
            env_state.create_store_obs, self._store_obs, self._recall_obs,
            operand=(sub_key, action, env_state.last_obs,
                     env_state.selected_pair, env_state.reward_locations)
            )
        reward, new_obs_without_context, new_obs, selected_pair_idx, \
            reward_locations, baited_branch = results

        new_env_state = RadialState(
            create_store_obs=(1-env_state.create_store_obs),
            step_count=step_count,
            selected_pair=selected_pair_idx,
            last_obs=new_obs_without_context,
            reward_locations=reward_locations,
            key=sub_key,
            info=env_state.info
            )
        actor_state = ActorState(obs=new_obs, reward=reward,
                                 done=is_terminal,
                                 info={
                                     "mask": new_obs[1] == ObsType.recall,
                                     "target": baited_branch,
                                     "temporal_mask":
                                         new_obs[1] == ObsType.store,
                                     "context": selected_pair_idx
                                 })

        return new_env_state, actor_state
    
    def reset(self, key: PRNGKey):
        new_key, *subkeys = jax.random.split(key, 4)
        new_pair_idx = jax.random.choice(subkeys[0], self.nb_pairs, ())
        # for each pairs of branches, one branch is baited,
        # the agent cannot see the bait, we use 0 as
        # "left branch is baited and 1 for "right branch is baited"
        reward_locations = jax.random.choice(subkeys[1],
                                             2,
                                             (self.nb_pairs,))
        new_obs_without_context = self.space_generator(
            new_pair_idx, subkeys[2])
        baited_branch = reward_locations[new_pair_idx]
        new_obs = jnp.concatenate((new_obs_without_context,
                                   jnp.array([0.0, 0.0, 0.0, 0.0])),
                                  axis=-1)
        new_obs = (new_obs, ObsType.recall.value)
        env_state = RadialState(create_store_obs=True,
                                step_count=1, selected_pair=new_pair_idx,
                                last_obs=new_obs_without_context,
                                reward_locations=reward_locations,
                                key=new_key,
                                info={})
        actor_state = ActorState(
            obs=new_obs,
            reward=self.base_reward,
            done=False,
            info={"mask": True,
                  "target": baited_branch,
                  "temporal_mask": False,
                  "context": new_pair_idx}
        )
        return env_state, actor_state
    
    @staticmethod
    def space_creation(state_type: str, nb_pairs: int, state_dim: int,
                       space_key: PRNGKey, generator_params: dict):
        test_env = generator_params.get("test_env", False)
        if state_type == "uniform":
            space_key, *subkeys = jax.random.split(space_key, 3)
            pairs_contexts = jax.random.uniform(
                subkeys[0],
                (nb_pairs, state_dim),
                minval=-1.0,
                maxval=1.0)
        elif state_type == "one_hot":
            pairs_contexts = jnp.eye(nb_pairs, state_dim)
        elif state_type == "omniglot":
            omniglot_path = generator_params.get(
                "omniglot_path", "./datasets/omniglot_proto_emb/train_set")
            space_key, k1 = jax.random.split(space_key)
            x1_seed = int(k1[0])
            pairs_contexts, _, _ = omniglot_embedding_space(
                omniglot_path, nb_classes=nb_pairs, seed=x1_seed)
        else:
            raise ValueError(f"Type {state_type} was not understood")

        def gen(selected_pair: int, key: PRNGKey):
            new_key, subkey = jax.random.split(key, 2)
            selected_pair_context = pairs_contexts[selected_pair]
            if state_type == "omniglot":
                fixed_sample = generator_params.get("fixed_sample", True)
                if fixed_sample:
                    selected_pair_context = selected_pair_context[0]
                else:
                    if test_env:
                        sample_idx = 19
                    else:
                        subkey, k2 = jax.random.split(subkey, 2)
                        sample_idx = jax.random.choice(k2, 19, ())
                    selected_pair_context = selected_pair_context[sample_idx]
                               
            return selected_pair_context

        return gen
