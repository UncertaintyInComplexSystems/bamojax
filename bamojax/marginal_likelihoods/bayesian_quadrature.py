
import jax
from jax import Array
import gpjax as gpx
import optax
import jax.numpy as jnp
import jax.random as jrnd

from jax.flatten_util import ravel_pytree
from bamojax.base import Model




def select_initial_points(key, model: Model, param_bounds: dict, num_eval: int = 100, temperature: float = 0.5) -> Array:

    init_params = model.sample_prior(key)
    # _, unravel_fn = ravel_pytree(init_params)

    # use slice sampling and a tempered logdensity to sample some initial points
    loglikelihood_fn = model.loglikelihood_fn()
    logprior_fn = model.logprior_fn()

    def tempered_logdensity(state):
        return temperature * loglikelihood_fn(state) + logprior_fn(state)
    
    #
    _, sample_chain_fn = make_slice_sampler(
        tempered_logdensity=tempered_logdensity,
        initial_params=init_params,
        bracket_width=1.0,
        max_step_out=50,
        max_shrink=200,
    )

    samples = sample_chain_fn(key, init_params=init_params, num_samples=num_eval, num_burnin=0,)   
    return samples

#
def fit_GP(model: Model, X: Array, param_bounds: dict = None, num_eval: int = 100):

    pass

#
def bayesian_quadrature_wsabi(X: Array, y: Array, lb: float = 1e-3, ub: float = 1e3) -> Array:
    
    pass

#
def bayesian_quadrature(model: Model, param_bounds: dict, num_eval: int = 100) -> Array:
    """Bayesian quadrature for estimating the marginal likelihood (model evidence)"""
    
    pass

#
