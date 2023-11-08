from pathlib import Path
from typing import Optional
from chex import PRNGKey
import numpy as np
import jax
from jax import numpy as jnp

def omniglot_embedding_space(embedding_path: str, nb_classes: int,
                             excluded_classes: Optional[list[Path]] = None,
                             seed: Optional[int] = None):
    """
    Select nb_classes classes that are not in excluded_classes
    and read samples from omniglot embedded datasets.
    Because iterdir order is OS dependent we impose a lexicographic order
    """
    grnd = np.random.default_rng(seed)
    embedding_path: Path = Path(embedding_path)
    list_of_classes = [x.name for x in embedding_path.iterdir() if x.is_file()]
    if excluded_classes is not None:
        list_of_classes_set = set(list_of_classes)
        list_of_classes = list(list_of_classes_set - set(excluded_classes))
    list_of_classes = sorted(list_of_classes)
    chosen_classes = grnd.choice(list_of_classes, nb_classes,
                                 replace=False)
    datax = []
    datay = []
    chosen_files = embedding_path / chosen_classes
    for file in chosen_files:
        with open(file, "rb") as f_r:
            x = np.load(f_r)
        datax.append(jnp.array(x, dtype=jnp.float32))
        # remove extension
        y = file.name[:-4]
        datay.append(y)
    datax = jnp.array(datax, dtype=jnp.float32)
    return datax, datay, chosen_classes