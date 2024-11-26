from typing import NamedTuple, Callable
from jaxtyping import Array
import jax
import jax.random as jrnd
import jax.numpy as jnp
from blackjax.types import ArrayTree, PRNGKey

from blackjax import generate_top_level_api_from, normal_random_walk



from .base import Model

# blackjax.vi implements an init (similar to MCMC), and a step fn
# install optax

# How does this deal with distrax transformations? Can we optimize in the untransformed space? 
# In the case of full covariance, we might need the reparametrization trick

# generic:
#  - define variational distribution using dx.Distribution (default: dx.MultivariateNormalFullCovariance / dx.MultivariateNormalDiag)
#  - THE STEP FUNCTION
#  - define ELBO(theta) := E_q(z|theta) [log p(y, z) - log q(z))] = E_q(z) [log p(y|z) + log p(z) - log q(z|theta)], implemented as:
#     - elbo_fn(theta):
#       - z ~ q(z|theta), M samples 
#       - log p(z|theta) via dx.Dist(params=**theta).log_prob(value=z)  # in vmap over M
#       - log p(y|z) + log p(z) via model.loglikelihood_fn()(z) and model.logprior_fn()(z)
#       - return (logq - logp).mean()
#    - elbo, elbo_grad = jax.value_and_grad(elbo_fn)(theta)
#    - updates, new_opt_state = optimizer.update(elbo_grad, state.opt_state, theta)  # optimizer from optax
#    - new_parameters = jax.tree.map(lambda p, u: p + u, parameters, updates)
#    - new_state = VIState(new_parameters, new_opt_state)
#    - return new_state, elbo
#  - Gradient-based optimization using jax.grad / jax.value_and_grad

# tricks: see blackjax' implementation and use 'stick-the-landing' to end the gradient computation in the ELBO function; simply add theta' = jax.lax.stop_gradient(theta') for theta' \in theta
