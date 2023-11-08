from functools import partial
from math import comb
from jax import numpy as jnp
import jax
import numpy as np
def sample_binary(number_set:set, nb_ones:int, dim_size:int):
    already_present = True
    power_list = np.arange(0, dim_size,dtype=int)
    while already_present:
        powers = np.random.choice(power_list, nb_ones, replace=False)
        k = np.sum([2**p for p in powers])
        already_present = k in number_set
        if not already_present:
            zeros_vector = np.zeros((dim_size), dtype=float)
            zeros_vector[powers] = 1.0
    return k, zeros_vector
def generate_datasets(dim_size:int, nb_ones:int, max_pattern: int = None):
    ncr = comb(dim_size, nb_ones)
    if max_pattern is not None and ncr > 2*max_pattern:
        ncr = 2*max_pattern 
    numbers_set = set()
    
    train_set = []
    test_set = []

    for _ in range(ncr//2):
        k, train_binary_sample = sample_binary(numbers_set, nb_ones, dim_size)
        numbers_set.add(k)
        k, test_binary_sample = sample_binary(numbers_set, nb_ones, dim_size)
        numbers_set.add(k)
        train_set.append(train_binary_sample)        
        test_set.append(test_binary_sample)
    train_set = np.array(train_set)
    test_set = np.array(test_set)
    hash_train_set = set([hash(x.tostring()) for x in train_set])
    hash_test_set = set([hash(x.tostring()) for x in test_set])
    inter_set = hash_train_set.intersection(hash_test_set)
    assert len(inter_set) == 0,\
        "the test set and train set have non empty intersection"
    return train_set, test_set
def create_episode(dynamic_size, fear_context, nb_non_fear_pattern, fear_set, not_fear_set):
        f_p = fear_set[np.random.choice(len(fear_set), ())]
        c_f_p = jnp.concatenate((f_p, fear_context[0]), axis=-1)
        q_f_p = jnp.concatenate((f_p, fear_context[2]), axis=-1)
        non_fear_pattern_list = []
        if dynamic_size:
            max_size = nb_non_fear_pattern
            nb_non_fear_pattern = np.random.randint(1, nb_non_fear_pattern)
        for i in range(nb_non_fear_pattern):
            not_f_p = f_p
            while np.array_equal(f_p, not_f_p):
                not_f_p = not_fear_set[np.random.choice(len(not_fear_set),())]
            c_n_f_p = jnp.concatenate((not_f_p, fear_context[1]), axis=-1)
            non_fear_pattern_list.append(c_n_f_p)
        non_fear_pattern_list = jnp.array(non_fear_pattern_list)
        facts = jnp.concatenate(
            (non_fear_pattern_list, jnp.expand_dims(c_f_p, 0)), axis=0)
        non_fear_query_pattern = non_fear_pattern_list[np.random.choice(
            len(non_fear_pattern_list), ())]
        q_n_f_p = non_fear_query_pattern.at[-2:].set(0.0)
        query = jnp.array([q_f_p, q_n_f_p])
        facts_idx = np.random.permutation(len(facts))
        query_idx = np.random.permutation(len(query))

        shuffled_facts = facts[facts_idx]
        if dynamic_size:
            shuffled_facts = jnp.pad(shuffled_facts, (
                (0, max(0, 1 + max_size - len(shuffled_facts) )),
                (0,0)))
        
        shuffled_query = query[query_idx]
        
        
        x = (jnp.concatenate((shuffled_facts, shuffled_query), axis=0),
                        jnp.array([0]*len(shuffled_facts) + [1]*len(query), dtype=int))
        y = jnp.array([0, 1])[query_idx]
        return x, y
def sampling_data_set(dynamic_size, 
                      sample_size, nb_non_fear_pattern, fear_set, not_fear_set):
    fear_context = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    x = []
    y = []
    for _ in range(sample_size):
        x_sample, y_sample = create_episode(dynamic_size, fear_context, nb_non_fear_pattern,
            fear_set, not_fear_set)
        x.append(x_sample)
        y.append(y_sample)
    x = jax.tree_util.tree_map(
        lambda *x: jnp.stack(list(x), axis=0), *x)
    y = np.array(y)
    return (x, y)
    
def fear_conditioning_dataset(dim_size:int, nb_ones:int,
                              sample_size:int, 
                              nb_non_fear_pattern:int,
                              disjoint_pattern_set:bool,
                              dynamic_size:bool):
    assert dim_size >= nb_ones,\
        "the number of ones in pattern must be less than the space dimension"
    train_set, test_set = generate_datasets(dim_size, nb_ones)
    if disjoint_pattern_set:
        train_f_set, train_not_f_set  = np.array_split(train_set, 2, axis=0)
        test_f_set, test_not_f_set = np.array_split(test_set, 2, axis=0)
    else:
        train_f_set = train_not_f_set = train_set
        test_f_set = test_not_f_set = test_set

    train_set = partial(sampling_data_set, dynamic_size, sample_size,
        nb_non_fear_pattern, train_f_set, train_not_f_set)
    test_set = partial(sampling_data_set, dynamic_size, sample_size,
                       nb_non_fear_pattern, test_f_set, test_not_f_set)
    return train_set, test_set, sample_size