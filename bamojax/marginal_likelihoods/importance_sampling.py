import jax
import jax.numpy as jnp
import jax.random as jrnd
from jax.tree_util import tree_flatten, tree_unflatten
from jax.scipy.special import logsumexp

from numpyro.distributions import Distribution, TransformedDistribution
from typing import Callable, Float


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
def importance_sampling(key, 
                        model: Model, 
                        g_IS: Distribution,
                        num_samples: int = 1_000,
                        iid_obs: bool = True) -> Float:
    
    r"""Importance sampling routine for a given BayesianModel.

    Importance sampling is based around the following approximation to the log
    marginal likelihood [Gronau et al., 2017]:

    $$
    p(D) \approx \frac{1}{N} \sum_{i=1}^N p\left(D \mid \theta_i\right) \frac{p(\theta_i)}{g_IS(\theta_i)}\enspace,
    $$
    with $\theta_i \sim g_IS(\theta)$

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
                                     lambda l: isinstance(l, (Distribution, TransformedDistribution)))
       
    samples = list()
    for g in g_flat:
        key, subkey = jrnd.split(key)
        samples.append(g.sample(key=subkey, sample_shape=(num_samples, )))

    importance_samples = tree_unflatten(g_treedef, samples)
    adjusted_likelihoods = jax.vmap(adjusted_likelihood)(importance_samples)
    return logsumexp(adjusted_likelihoods) - jnp.log(num_samples)

#