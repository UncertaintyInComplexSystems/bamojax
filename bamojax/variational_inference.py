from typing import NamedTuple, Callable
from jaxtyping import Array
import jax
import jax.random as jrnd
import jax.numpy as jnp
from blackjax.types import ArrayTree, PRNGKey

from blackjax import meanfield_vi


from .base import Model

def get_bijectors(model: Model, model_pytree):
    bijectors = {}
    for k in model_pytree.keys():
        if hasattr(model.nodes[k].distribution, '_bijector'):                        
            transform = lambda x: model.nodes[k].distribution._bijector.forward(x=x)
        else:
            transform = lambda x: x
        bijectors[k] = transform
    return bijectors

#

def meanfield_variational_inference(key, model, num_steps, num_samples: int = 10, optimizer: Callable = None):
    # get bijectors
    bijectors = get_bijectors(model, model.sample_prior(jrnd.PRNGKey(0)))

    def logdensity_fn(z):
        z = jax.tree.map(lambda f, v: f(v), bijectors, z)
        return model.loglikelihood_fn()(z) + model.logprior_fn()(z)

    #
    mfvi = meanfield_vi(logdensity_fn, optimizer, num_samples)
    initial_position = {'beta': jnp.array([1.0, 1.0]), 'sigma': -3.0}  # note: these values are overridden! by Blackjax!
    initial_state = mfvi.init(initial_position)

    @jax.jit
    def one_step(state, rng_key):
        state, info = mfvi.step(rng_key, state)
        return state, (state, info)

    #
    keys = jrnd.split(key, num_steps)
    _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)
    return states, infos

#
