from dataclasses import dataclass
from typing import Any, Optional

from omegaconf import MISSING


@dataclass
class Task:
    pass


@dataclass
class BabiTask(Task):
    task_id: Any = 1
    training_set_type: str = "10k"
    mode: str = "default"
    class_space: str = "full_corpus"
    max_num_sentences: int = -1
    hops: int = 1
    
@dataclass
class FearConditioning(Task):
    dynamic_size: bool = True
    dim_size:int = 20
    nb_ones:int = 3
    nb_non_fear_pattern: int = 5
    sample_size:int = 1000
    disjoint_pattern_set:bool = False
    
@dataclass
class BabiTaskRL(Task):
    babi_parameters: Optional[dict[str, Any]] = None
    batch: int = 1
    base_reward: float = 0.0
    success_reward: float = 1.0
    failure_reward: float = -1.0
    generator_params: Optional[dict[str, Any]] = None
    

@dataclass
class AssociatedPairs(Task):
    nb_states: int = 5
    nb_store_state: int = 5
    hops: int = 1
    state_dim: int = 20
    state_type: str = "uniform"
    base_reward: float = 0.0
    success_reward: float = 1.0
    failure_reward: float = -1.0
    unique_pair: bool = True
    batch: int = 1
    generator_params: Optional[dict[str, Any]] = None
    


@dataclass
class MatchingPairs(Task):
    nb_states: int = 5
    episode_length: int = 4
    state_dim: int = 20
    state_type: str = "uniform"
    base_reward: float = 0.0
    success_reward: float = 1.0
    failure_reward: float = -1.0
    reward_on_match_proba: float = 0.5
    context_type: str = "concat_with_obs"
    batch: int = 1
    generator_params: Optional[dict[str, Any]] = None



@dataclass
class RadialMaze(Task):
    nb_pairs: int = 1
    switch_reward_rule: str = "everytime"
    switch_pair_rule: str = "uniform"
    state_type: str = "one_hot"
    state_dim: int = 20
    base_reward: float = 0.0
    success_reward: float = 1.0
    failure_reward: float = -1.0
    horizon: int = 10
    batch: int = 1
    generator_params: Optional[dict[str, Any]] = None
