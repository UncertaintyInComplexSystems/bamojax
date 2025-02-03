import jax.flatten_util
from blackjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey
from blackjax.base import SamplingAlgorithm
from typing import Callable
import jax.random as jrnd
import jax.numpy as jnp
import blackjax

##### BINARY LATENT VARIABLES #####

def generate_bernoulli_noise(rng_key: PRNGKey, position, theta):
    p, unravel_fn = jax.flatten_util.ravel_pytree(position)
    sample = jrnd.bernoulli(rng_key, shape=p.shape, p=theta)
    return unravel_fn(sample)

#
def bernoulli(theta: Array) -> Callable:
    def propose(rng_key: PRNGKey, position) -> ArrayTree:
        return generate_bernoulli_noise(rng_key, position, theta=theta)
    
    #
    return propose

#
def build_xor_step():
    def kernel(
        rng_key: PRNGKey, state, logdensity_fn: Callable, random_step: Callable
    ):
        def proposal_generator(key_proposal, position):
            move_proposal = jax.tree_util.tree_map(lambda x: x.astype(int), random_step(key_proposal, position)) 
            new_position = jax.tree_util.tree_map(jnp.bitwise_xor, position, move_proposal)
            return new_position

        inner_kernel = blackjax.mcmc.random_walk.build_rmh()
        return inner_kernel(rng_key, state, logdensity_fn, proposal_generator)

    return kernel

#  
def xor_step_random_walk(logdensity_fn: Callable, random_step: Callable) -> SamplingAlgorithm:

    kernel = build_xor_step()
    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        return blackjax.mcmc.random_walk.init(position, logdensity_fn)

    def step_fn(rng_key: PRNGKey, state):
        return kernel(rng_key, state, logdensity_fn, random_step)

    return SamplingAlgorithm(init_fn, step_fn)

#
def bernoulli_random_walk(logdensity_fn: Callable, theta):
    return xor_step_random_walk(logdensity_fn, bernoulli(theta))

#