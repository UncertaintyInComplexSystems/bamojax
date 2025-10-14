import jax
import jax.numpy as jnp
from typing import Callable
from jax.flatten_util import ravel_pytree

def iid_likelihood(L: Callable):
    r"""
    
    We typically have multiple observations and assume the likelihood factorizes 
    as: 

    $$    
        \log p\left(Y \mid \theta\right) = \sum_{i=1}^N \log p\left(y_i \mid \theta\right) \enspace.
    $$

    """
    return lambda x: jnp.sum(L()(x))

#
def flatten_dict_to_array(samples: dict):
    """ Bamojax states are dictionaries, with entries per model variable. Here we flatten them so the 
    proposal distribution can be one single multivariate distribution.

    """
    if len(samples.keys()) == 1:
        per_sample_template = jax.tree.map(lambda x: jnp.zeros(x.shape[1:], x.dtype), samples)
        _, unravel_one_sample = ravel_pytree(per_sample_template)
        samples_flattened, _ = ravel_pytree(samples)
        return samples_flattened, unravel_one_sample
    else:
        leaves, treedef = jax.tree_util.tree_flatten(samples)
        flattened_leaves = [x.reshape(x.shape[0], -1) for x in leaves]
        sizes = [f.shape[1] for f in flattened_leaves]
        cumulative_sizes = jnp.cumsum(jnp.array(sizes))
        samples_flattened = jnp.concatenate(flattened_leaves, axis=-1) 
        leaf_shapes = [x.shape[1:] for x in leaves]

        def unravel_one_sample(vec):
            splits = jnp.split(vec, cumulative_sizes[:-1])
            reshaped = [v.reshape(s) for v, s in zip(splits, leaf_shapes)]
            return jax.tree_util.tree_unflatten(treedef, reshaped)

    return samples_flattened, unravel_one_sample

#