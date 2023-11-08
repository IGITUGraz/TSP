from typing import Union, Callable
from . import policy_gradient, q_lambda, q_learning, online_policy_gradient
_actor_method = {
    "q_learning": q_learning.create_policy,
    "q_lambda": q_lambda.create_policy,
    "vanilla_policy_gradient": policy_gradient.create_policy,
    "online_policy_gradient": online_policy_gradient.create_policy
}


def get(identifier: Union[str, Callable]):
    if isinstance(identifier, str):
        try:
            return _actor_method[identifier]
        except KeyError:
            valid_ids_msg = "\n".join(_actor_method.keys())
            print(f"{identifier} does not exist in the lookup table \n"
                  f"valid identifier are:\n {valid_ids_msg}")
    elif isinstance(identifier, Callable):
        return identifier
