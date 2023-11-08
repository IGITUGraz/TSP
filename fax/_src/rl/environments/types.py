# fork based on
# Copyright 2021 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import binascii
import enum
import os
from typing import ClassVar, Optional, Dict, Any, NamedTuple

import gym
from gym.vector import utils
import abc
from dataclasses import dataclass, replace, field

import jax
from jax import numpy as jnp
from chex import PRNGKey, ArrayTree, Array, Scalar


class ObsType(enum.IntEnum):
    store = 0
    recall = 1
    terminal = 2


class EnvState(NamedTuple):
    # here we define the information that define
    # the current configuration and observation of the environment
    pass


class ActorState(NamedTuple):
    obs: object
    reward: Scalar
    done: Scalar
    info: Dict[str, Any]


class Env(abc.ABC):
    # here we define everything that are read only
    # TODO: make gym.Space with jax, pytree transformation would be very helpful here
    action_space: gym.Space
    observation_space: gym.Space

    @abc.abstractmethod
    def reset(self, key: PRNGKey) -> tuple[EnvState, ActorState]:
        """Resets the environment to an initial state."""

    @abc.abstractmethod
    def step(self, state: EnvState, action: object) -> tuple[EnvState,
                                                             ActorState]:
        """Run one time-step of the environment's dynamics."""
    def create_auxilatory_data(self):
        return None

    @property
    def observation_shape(self) -> object:
        """The size of the observation vector returned in step and reset."""
        sample_obs = self.observation_space.sample()
        obs_shape = jax.tree_map(lambda x: jnp.shape(x), sample_obs)
        return obs_shape

    @property
    def action_shape(self) -> object:
        """The size of the action vector expected by step."""
        sample_obs = self.action_space.sample()
        obs_shape = jax.tree_map(lambda x: jnp.shape(x), sample_obs)
        return obs_shape

    @property
    def unwrapped(self) -> 'Env':
        return self


class Wrapper(Env):
    """Wraps the environment to allow modular transformations."""

    def __init__(self, env: Env):
        self.env: Env = env

    def reset(self, key: PRNGKey) -> tuple[EnvState, ActorState]:
        return self.env.reset(key)

    def step(self, state: EnvState, action: object) -> tuple[EnvState, ActorState]:
        return self.env.step(state, action)

    @property
    def observation_shape(self) -> object:
        return self.env.observation_shape

    @property
    def action_shape(self) -> object:
        return self.env.action_shape

    @property
    def unwrapped(self) -> Env:
        return self.env.unwrapped

    def __getattr__(self, name):
        if name == '__setstate__':
            raise AttributeError(name)
        return getattr(self.env, name)



class VectorWrapper(Wrapper):
    """Vectorizes Brax env."""

    def __init__(self, env: Env, batch_size: int):
        super().__init__(env)
        self.batch_size = batch_size
        self._step = jax.jit(jax.vmap(self.env.step))
        self._reset = jax.jit(jax.vmap(self.env.reset))
    def reset(self, key: PRNGKey, aux=None) -> tuple[EnvState, ActorState]:
        new_key, *batched_key = jax.random.split(key, 1 + self.batch_size)
        batched_key = jnp.array(batched_key)
        env_state, act_state = self._reset(batched_key)
        return env_state, act_state, aux
    def step(self, state: EnvState, action: object) -> tuple[EnvState, ActorState]:
        return self._step(state, action)

class VectorBabiWrapper(VectorWrapper):
    def create_auxilatory_data(self):
        return self.env.create_auxilatory_data()
    def reset(self, key: PRNGKey, aux=None) -> tuple[EnvState, ActorState]:
        batched_key: PRNGKey = jax.random.split(key, self.batch_size)
        story_id = aux["story_id"]
        env = self.env
        nb_story = env.nb_story
        idx = jnp.arange(story_id, story_id + self.batch_size,dtype=jnp.int32)
        idx = idx % nb_story
        env_state, act_state = jax.vmap(self.env.reset)(batched_key, idx)
        aux["story_id"] = (idx[-1] + 1) % nb_story
        return env_state, act_state, aux
    
class EpisodeWrapper(Wrapper):
    """Maintains episode step count and sets done at episode end."""

    def __init__(self, env: Env, episode_length: int,
                 action_repeat: int):
        super().__init__(env)
        self.episode_length = episode_length
        self.action_repeat = action_repeat

    def reset(self, rng: PRNGKey) -> tuple[EnvState, ActorState]:
        env_state, actor_state = self.env.reset(rng)
        env_state.info['steps'] = jnp.zeros(())
        env_state.info['truncation'] = jnp.zeros(())
        return env_state, actor_state

    def step(self, state: EnvState, action: object) -> tuple[EnvState, ActorState]:
        env_state, actor_state = self.env.step(state, action)
        steps = env_state.info['steps'] + self.action_repeat
        one = jnp.ones_like(actor_state.done)
        zero = jnp.zeros_like(actor_state.done)
        done = jnp.where(steps >= self.episode_length, one, actor_state.done)
        env_state.info['truncation'] = jnp.where(steps >= self.episode_length,
                                                 1 - actor_state.done, zero)
        env_state.info['steps'] = steps
        return env_state, ActorState(actor_state.obs, actor_state.reward, done, actor_state.info)


class AutoResetWrapper(Wrapper):
    """Automatically resets Brax envs that are done."""

    def reset(self, rng: PRNGKey) -> tuple[EnvState, ActorState]:
        env_state, actor_state = self.env.reset(rng)
        env_state.info['first_obs'] = actor_state.obs
        return env_state, actor_state

    def step(self, state: EnvState, action: object) -> tuple[EnvState, ActorState]:
        env_state, actor_state = self.env.step(state, action)

        def where_done(x, y):
            done = actor_state.done
            if done.shape:
                done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
            return jnp.where(done, x, y)

        obs = jax.tree_multimap(where_done, env_state.info['first_obs'], actor_state.obs)
        return env_state, ActorState(obs, actor_state.reward, actor_state.done, actor_state.info)


class GymWrapper(gym.Env):
    """A wrapper that converts Brax Env to one that follows Gym API."""

    # Flag that prevents `gym.register` from misinterpreting the `_step` and
    # `_reset` as signs of a deprecated gym Env API.
    _gym_disable_underscore_compat: ClassVar[bool] = True

    def __init__(self,
                 env: Env,
                 seed: int = 0,
                 backend: Optional[str] = None):
        self._env = env
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': 1
        }
        self.seed(seed)
        self.backend = backend
        self._env_state: Optional[EnvState] = None
        self._key: Optional[PRNGKey] = None

        self.observation_space: gym.Space = env.observation_space

        self.action_space: gym.Space = env.action_space

        def reset(key):
            env_state, actor_state = self._env.reset(key)
            return env_state, actor_state.obs

        self._reset = jax.jit(reset, backend=self.backend)

        def step(env_state, action):
            env_state, actor_state = self._env.step(env_state, action)
            return env_state, actor_state.obs, actor_state.reward, actor_state.done, actor_state.info

        self._step = jax.jit(step, backend=self.backend)

    def reset(self):
        self._env_state, obs = self._reset(self._key)
        return obs

    def step(self, action):
        self._env_state, obs, reward, done, info = self._step(self._env_state, action)
        return obs, reward, done, info

    def seed(self, seed: Optional[int] = None):
        if seed is None:
            seed = int(binascii.hexlify(os.urandom(4)), 16)
        self._key = jax.random.PRNGKey(seed)

    def render(self, mode='human'):
        # pylint:disable=g-import-not-at-top
        return super().render(mode=mode)  # just raise an exception


class VectorGymWrapper(gym.vector.VectorEnv):
    """A wrapper that converts batched Brax Env to one that follows Gym VectorEnv API."""

    # Flag that prevents `gym.register` from misinterpreting the `_step` and
    # `_reset` as signs of a deprecated gym Env API.
    _gym_disable_underscore_compat: ClassVar[bool] = True

    def __init__(self,
                 env: Env,
                 seed: int = 0,
                 backend: Optional[str] = None):
        self._env = env
        self._key: Optional[PRNGKey] = None
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': 1
        }
        if not hasattr(self._env, 'batch_size'):
            raise ValueError('underlying env must be batched')

        self.num_envs = self._env.__getattribute__("batch_size")
        self.seed(seed)
        self.backend = backend
        self._env_state: Optional[EnvState] = None
        self.observation_space = utils.batch_space(self.observation_space,
                                                   self.num_envs)

        self.action_space = utils.batch_space(self.action_space,
                                              self.num_envs)

        def reset(key):
            env_state, actor_state = self._env.reset(key)
            return env_state, actor_state.obs

        self._reset = jax.jit(reset, backend=self.backend)

        def step(env_state, action):
            env_state, actor_state = self._env.step(env_state, action)
            return env_state, actor_state.obs, actor_state.reward, actor_state.done, actor_state.info

        self._step = jax.jit(step, backend=self.backend)

    def reset(self):
        self._env_state, obs = self._reset(self._key)
        return obs

    def step(self, action):
        self._env_state, obs, reward, done, info = self._step(self._env_state, action)
        return obs, reward, done, info

    def seed(self, seed: Optional[int] = None):
        if seed is None:
            seed = int(binascii.hexlify(os.urandom(4)), 16)
        self._key = jax.random.PRNGKey(seed)

    def render(self, mode='human'):
        return super().render(mode=mode)  # just raise an exception
