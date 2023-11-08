from typing import Any, NamedTuple, Optional

import jax
import jax.numpy as jnp
from chex import PRNGKey, Array

from gym import spaces
from fax.rl.types import ActorState, Env, ObsType



class BabiTaskState(NamedTuple):
    step: int
    story_id: int
    x: Array
    y: Array
    key: PRNGKey
    info: dict[str, Any]

class BabiTask(Env):
    """
    generator space function creation:
    get x, y pair of a given task,
    depending on eval_env flag
    get train set or test set
    divide in batch from batch_size
    dim(x) = (nb_batch, batch_size, story_len, max_word)
    dim(y) = (nb_batch, batch_size,)
    
    signature:
        gen(batch_id)

    

    """
    def __init__(self,
                 x_data,
                 y_data,
                 nb_story,
                 nb_classes,
                 state_dim,
                 success_reward,
                 failure_reward,
                 base_reward,
                 generator_params,
                 space_key,
                 ):
        self.state_dim = state_dim
        self.nb_story = nb_story
        self.nb_classes = nb_classes
        self.space_generation  =\
            self.space_creation(x_data, y_data, generator_params)
        self.success_reward = success_reward
        self.failure_reward = failure_reward
        self.base_reward = base_reward
        self.observation_space: spaces.Space = spaces.Tuple(
            (spaces.Box(-jnp.inf, jnp.inf, (self.state_dim,)),
             spaces.Discrete(3)))
        self.action_space: spaces.Space = spaces.Discrete(self.nb_classes)

    def step(self, env_state: BabiTaskState, action):
        new_obs = jax.tree_map(
            lambda x: x[env_state.step], env_state.x)
        new_obs_type: int = new_obs[1]
        terminal_flag = new_obs_type == ObsType.terminal
        reward_flag = action == env_state.y
        terminal_reward = jax.lax.select(
            reward_flag, self.success_reward, self.failure_reward)
        rewards = jax.lax.select(
            terminal_flag, terminal_reward, self.base_reward)
        new_env_state = BabiTaskState(
                step=env_state.step + 1,
                story_id=env_state.story_id,
                x=env_state.x,
                y=env_state.y,
                key=env_state.key,
                info=env_state.info
            )
        actor_state = ActorState(
            obs=new_obs,
            reward = rewards,
            done = terminal_flag,
            info={
                "mask": new_obs[1] == ObsType.recall,
                "temporal_mask": terminal_flag,
                "target": env_state.y,
                })
        return new_env_state, actor_state
    def create_auxilatory_data(self):
        return {"story_id": 0}
    def reset(self, key: PRNGKey, story_id):
        x, y = self.space_generation(story_id, key)
        # dim(x[0]) = (story_size, nb_words)
        # dim(x[1]) = (story_size,)
        # dim(y) = ()
        # add a new story at the end as terminale story
        
        (states, states_type) = x
        states_shape = jnp.shape(states)
        terminal_state = jnp.zeros((1, states_shape[-1]), dtype=jnp.float32)
        states = jnp.concatenate((states, terminal_state), axis=0)
        
        terminal_type = jnp.array([ObsType.terminal])
        states_type = jnp.concatenate((states_type, terminal_type), axis=0)
        x = (states, states_type)
        y = y
        actor_state = ActorState(
            obs=jax.tree_map(lambda u: u[0], x),
            reward=self.success_reward,
            done=False,
            info={"mask": False,
                  "temporal_mask": False,
                  "target": y,
                  })
        env_state = BabiTaskState(
            step=1,
            story_id=story_id,
            x=x, y=y, key=key,
            info={}
        )
        return env_state, actor_state
    def render(self, mode="human"):
        pass

    @staticmethod
    def space_creation(x, y, generator_params: dict):
        x = jax.tree_util.tree_map(lambda u: jnp.asarray(u), x)
        y = jnp.asarray(y)
        
        def gen(n:int, key):
            new_story_x = jax.tree_util.tree_map(lambda u: u[n], x)
            new_story_y = y[n]
            
            
            return new_story_x, new_story_y

        return gen
