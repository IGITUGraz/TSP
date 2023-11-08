from functools import partial
import jax
from jax import numpy as jnp
import numpy as np
from chex import Array, PRNGKey


def stack_and_pad(max_episode_len, constant_values: int = 0, *arr):
    l_arr = list(arr)
    
    stacked_arr = np.stack(l_arr, axis=1)
    # here we right-pad only the temporal dimension 
    stacked_pad_arr = np.pad(
        stacked_arr,
        ((0, 0),) + ((0, max(0, max_episode_len - stacked_arr.shape[1])),) 
        + ((0, 0),) * (np.ndim(stacked_arr) - 2),
        mode="constant", constant_values=constant_values)
    return stacked_pad_arr


class TransitionStore:
    """
    TransitionStore
    """
    _BATCH_STORE_ = {
        "action": None,
        "action_logp": None,
        "pred": None,
        "value": None,
        "obs": None,
        "done": None,
        "reward": None,
        "info": None,
    }
    # actions and obs can be complex structured containers
    # we can use pytree definition to accumulate over leaves transparently
    _LIVE_QUEUE_ = {
        "action": [],
        "action_logp": [],
        "pred": [],
        "value": [],
        "obs": [],
        "done": [],
        "reward": [],
        "info": []
    }
    queue_size = 0
    nb_batch = 0
    atomic_leaves = None

    @staticmethod
    def is_leaf_atomic(x):
        return isinstance(x, (Array, float, int, bool))

    def __init__(self, max_episode_len: int, multiple_episodes_in_batch: bool):
        self.multiple_episodes_in_batch = multiple_episodes_in_batch
        self.max_episode_len = max_episode_len

    def reset(self):
        self._LIVE_QUEUE_ = {k: [] for k in self._LIVE_QUEUE_.keys()}
        self._BATCH_STORE_ = {k: None for k in self._BATCH_STORE_.keys()}
        self.queue_size = 0
        self.nb_batch = 0
        self.atomic_leaves = None

    def is_ready(self, batch_size):
        return self.nb_batch >= batch_size

    def sample(self, batch_size: int, rng: PRNGKey):
        batch_idx = jax.random.choice(rng, self.nb_batch,
                                      (batch_size,), replace=False)
        samples = jax.tree_map(
            lambda x: jnp.take(x, batch_idx, axis=0), self._BATCH_STORE_)
        self._BATCH_STORE_ = jax.tree_map(
            lambda x: jnp.delete(x, batch_idx, axis=0),
            self._BATCH_STORE_)
        self.nb_batch -= batch_size
        return samples

    def push(self, **data):
        # we don't need to differentiate btw first obs and
        # the rest if we can sample a dummy action from action space.
        # the action will not be used anyway
        #TODO: batched data
        # in batched data
        # data is a dict with (b,) + (n1, .., ni) shape
        # when one data["done"][i] is True, this means 
        # that at least one environment have reach a terminal states
        # sol 1:
        #   terminate all environments an store each of them as a batch
        # sol 2: 
        #   keep track off env that have terminated (not really possible)
        # sol 3:
        #   continue until max_episode_len is reach or
        # all envs have terminated 
        all_done = jnp.all(data["done"])
        for k in data:
            self._LIVE_QUEUE_[k].append(data[k])
        self.queue_size += 1
        if (all_done and not self.multiple_episodes_in_batch) or \
                self.queue_size >= self.max_episode_len:
            for k in self._LIVE_QUEUE_.keys():
                if k == "done":
                    constant_value = 1
                else:
                    constant_value = 0
                padding_func = partial(
                    stack_and_pad, self.max_episode_len, constant_value)
                padded_results = jax.tree_map(
                    padding_func, *[x for x in self._LIVE_QUEUE_[k] if x is not None])

                if self._BATCH_STORE_[k] is None:
                    self._BATCH_STORE_[k] = padded_results
                else:
                    # _BATCH_STORE_[k] may be a jaxtree 
                    # so we concatenate at leafs
                    
                    self._BATCH_STORE_[k] = jax.tree_map(
                        lambda x, y: jnp.concatenate((x, y), axis=0),
                        self._BATCH_STORE_[k], padded_results)
            self.nb_batch += data["done"].shape[0]
            self.queue_size = 0
            self._LIVE_QUEUE_ = {k: [] for k in self._LIVE_QUEUE_.keys()}
