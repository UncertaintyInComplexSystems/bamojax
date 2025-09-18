
import jax
import jax.numpy as jnp
import jax.random as jrnd

import jaxopt
from tqdm import tqdm

from jax.scipy.special import logsumexp
from jax.tree_util import tree_flatten, tree_unflatten, tree_map
from distrax._src.distributions.distribution import Distribution
from distrax._src.bijectors.bijector import Bijector

from jaxtyping import Float
from typing import Callable, Dict

from bamojax.base import Model

# TODO: implement testing for these functions

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
def naive_monte_carlo(key, 
                      model: Model, 
                      num_prior_draws: int = 1_000, 
                      num_chunks: int = 5,
                      iid_obs: bool = True,
                      pb = True) -> Float:   
    r"""The Naive Monte Carlo (NMC) estimator

    The marginal likelihood is defined as 
    $$
        p(D) = \int_\Theta p\left(D \mid \theta\right) p(\theta) \,\text{d}\theta \enspace .
    $$

    In NMC we draw samples from the prior and approximate the ML as
    $$
        p(D) \approx 1/N \sum_{i=1}^N p\left(D \mid \theta_i\right), with \theta_i ~ p(\theta) \enspace.
    $$
    In nontrivial models, we need a *large* $N$ for this approximation to be 
    reasonable.

    """

    if iid_obs:
        loglikelihood_fn = iid_likelihood(model.loglikelihood_fn)
    else:
        loglikelihood_fn = model.loglikelihood_fn

    loglikelihoods = jnp.zeros((num_prior_draws, 
                                num_chunks))
    
    # We don't want to vmap this loop, as the reason for the loop is to avoid
    # running out of memory!
    for i in tqdm(range(num_chunks), disable=not pb):
        key, subkey = jrnd.split(key)
        keys = jrnd.split(subkey, num_prior_draws)
        prior_draws = jax.vmap(model.sample_prior)(keys)
        loglikelihoods = loglikelihoods.at[:, i].set(jax.vmap(loglikelihood_fn)(prior_draws))
    return logsumexp(loglikelihoods.flatten()) - jnp.log(num_prior_draws*num_chunks)

#
def harmonic_mean(key, model: Model) -> Float:
    raise NotImplementedError

def generalized_harmonic_mean(key, model: Model) -> Float:
    raise NotImplementedError

#
def importance_sampling(key, 
                        model: Model, 
                        g_IS: Distribution,
                        num_samples: int = 1_000,
                        iid_obs: bool = True) -> Float:
    
    r"""Importance sampling routine for a given BayesianModel.

    Importance sampling is based around the following approximation to the log
    marginal likelihood [Gronau et al., 2017]:

    $$
    p(D) \approx 1/N \sum_{i=1}^N p\left(D \mid \theta_i\right) \frac{p(\theta_i)}{g_IS(\theta_i)}\enspace,
    $$
    with $\theta_i ~ g_IS(\theta)$

    Here, g_IS is the importance density, which should meet these criteria:

    1. It is easy to evaluate.
    2. It has the same domain as the posterior p(\theta \mid D).
    3. It matches the posterior as closely as possible.
    4. It has fatter tails than the posterior.

    There is no one-size-fits-all importance density; this needs to be crafted
    carefully for each specific problem.

    Note that the importance density can also be a mixture distribution, which 
    can make it easier to introduce heavy tails.

    References:

    - Gronau, Q. F., Sarafoglou, A., Matzke, D., Ly, A., Boehm, U., Marsman, M., Leslie, D. S., Forster, J. J., Wagenmakers, E.-J., & Steingroever, H. (2017). A tutorial on bridge sampling. Journal of Mathematical Psychology, 81, 80-97. https://doi.org/10.1016/j.jmp.2017.09.005


    """

    def g_eval(state):
        logprob = 0        
        values_flat, _ = tree_flatten(state)
        for value, dist in zip(values_flat, g_flat):
            logprob += jnp.sum(dist.log_prob(value))
        return logprob
    
    # 
    def adjusted_likelihood(state):
        return loglikelihood_fn(state) + logprior_fn(state) - g_eval(state)

    #

    if iid_obs:
        loglikelihood_fn = iid_likelihood(model.loglikelihood_fn)
    else:
        loglikelihood_fn = model.loglikelihood_fn

    logprior_fn = model.logprior_fn()

    g_flat, g_treedef = tree_flatten(g_IS, 
                                     lambda l: isinstance(l, (Distribution, Bijector)))
       
    samples = list()
    for g in g_flat:
        key, subkey = jrnd.split(key)
        samples.append(g.sample(seed=subkey, sample_shape=(num_samples, )))

    importance_samples = tree_unflatten(g_treedef, samples)
    adjusted_likelihoods = jax.vmap(adjusted_likelihood)(importance_samples)
    return logsumexp(adjusted_likelihoods) - jnp.log(num_samples)

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

