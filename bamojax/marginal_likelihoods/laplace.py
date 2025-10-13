import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_map

import jaxopt
from typing import Callable


from bamojax.base import Model


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


def laplace_approximation(key,
                          model: Model,
                          iid_obs: bool= True,
                          **opt_args):

    r"""Compute the Laplace approximation of the log marginal likelihood of model

    The Laplace approximation approximates the posterior density of the model 
    with a Gaussian, centered at the mode of the density and with its curvature
    determined by the Hessian matrix of the negative log posterior density.

    The marginal likelihood of this proxy distribution is known in closed-form,
    and is used to approximate the actual marginal likelihood.

    See https://en.wikipedia.org/wiki/Laplace%27s_approximation

    """

    # The objective function is the unnormalized posterior
    @jax.jit
    def fun(x):
        return -1.0 * (loglikelihood_fn(x) + logprior_fn(x))

    #
    if iid_obs:
        loglikelihood_fn = iid_likelihood(model.loglikelihood_fn)
    else:
        loglikelihood_fn = model.loglikelihood_fn
    logprior_fn = model.logprior_fn()

    # Get initial values in the same PyTree structure as the model expects
    init_params = tree_map(jnp.asarray, 
                           model.sample_prior(key))

    # For some models, the parameters are bounded
    if 'bounds' in opt_args:
        print(opt_args['bounds'])
        solver = jaxopt.ScipyBoundedMinimize(fun=fun)
    else:
        solver = jaxopt.ScipyMinimize(fun=fun)
        
    
    # Derive the number of parameters
    D = 0
    vars_flattened, _ = tree_flatten(init_params)
    for varval in vars_flattened:
        D += varval.shape[0] if varval.shape else 1

    # Compute MAP
    sol = solver.run(init_params, **opt_args)   

    # We fit a Gaussian(\hat{\theta}, \Sigma) with 
    # \hat{\theta} = \argmax_\theta p(\theta \mid y)
    # \Sigma^-1 is the Hessian of -\log p(\theta \mid y) at \theta=\hat{\theta}

    mode = sol.params
    H = jax.hessian(fun)(mode)
    h, _ = tree_flatten(H)
    if D > 1:
        S = jnp.squeeze(jnp.linalg.inv(jnp.reshape(jnp.asarray(h), 
                                                   newshape=(D, D))))
        _, logdet = jnp.linalg.slogdet(S)
    else: 
        S = 1.0 / jnp.squeeze(jnp.asarray(h))
        logdet = jnp.log(S)

    log_posterior = -1.0 * sol.state.fun_val
    lml = log_posterior + 1/2*logdet + D/2 * jnp.log(2*jnp.pi)
    return lml

#

