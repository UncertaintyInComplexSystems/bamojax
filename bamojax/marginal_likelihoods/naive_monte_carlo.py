import jax
import jax.numpy as jnp
import jax.random as jrnd
from jax.scipy.special import logsumexp

from typing import Callable
from jaxtyping import Float
from tqdm import tqdm


from bamojax.base import Model
from bamojax.marginal_likelihoods.utility import iid_likelihood


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
        p(D) \approx \frac{1}{N} \sum_{i=1}^N p\left(D \mid \theta_i\right) \enspace,
    $$  with $\theta_i \sim p(\theta)$.

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