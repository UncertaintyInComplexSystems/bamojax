import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

import numpyro.distributions as dist
import numpyro.distributions.transforms as nprb

from bamojax.base import Model
from bamojax.marginal_likelihoods.utility import flatten_dict_to_array
from jax.flatten_util import ravel_pytree


def apply_bijectors(samples, bijectors):
    """Transform variables from the real line.

    """
    return {k: b(samples[k]) for k, b in bijectors.items()}

#
def apply_inverse_bijectors(samples, bijectors):
    """Transform variables to the real line.

    """
    return {k: b._inverse(samples[k]) for k, b in bijectors.items()}
#
def get_jacobians(samples, bijectors):
    """Get the Jacobian for the change-of-variables due to the bijector

    """
    jac = 0.0
    for k, b in bijectors.items():
        jac += b.log_abs_det_jacobian(samples[k], None)
    return jac

#
def get_proposal_distribution(transformed_samples):
    """Create a proposal distribution on the real line.

    Also returns an unravel function to transform proposals back to the dicts Bamojax uses for sampling.

    """

    samples_flattened, unravel_one_sample = flatten_dict_to_array(transformed_samples)
    mu, cov = jnp.mean(samples_flattened, axis=0), jnp.cov(samples_flattened, rowvar=False)

    if len(transformed_samples.keys()) == 1:
        proposal_distribution = dist.Normal(loc=jnp.atleast_1d(mu), scale=jnp.atleast_1d(jnp.sqrt(cov)))
    else:        
        proposal_distribution = dist.MultivariateNormal(loc=mu, covariance_matrix=cov)

    return proposal_distribution, jax.vmap(unravel_one_sample)

#
def sample_from_proposal_distribution(key, prop_dist, unravel_fn, N):
    """Sample from the proposal distribution and unravel back into a dictionary.

    """
    proposal_samples_flattened = prop_dist.sample(key, sample_shape=(N, ))
    proposal_samples = unravel_fn(proposal_samples_flattened)
    squeeze_fn = lambda x: jnp.squeeze(x)
    return jax.tree.map(squeeze_fn, proposal_samples)

#
def proposal_distribution_logprob(prop_dist, samples):
    """Flatten a sample dictionary and compute the log probability according to the proposal distribution.

    """

    samples_flattened, _ = flatten_dict_to_array(samples)
    return prop_dist.log_prob(samples_flattened)

#
def logposterior_proposals(model, bijectors: dict, samples):
    r""" Compute the log joint probability for a sample, and adjusting for the Jacobian
    
    p(D \mid \theta)p(\theta)|J(f(\theta)|
    
    """

    loglikelihood_fn = model.loglikelihood_fn()
    logprior_fn = model.logprior_fn()
    logjoint = lambda x: loglikelihood_fn(x) + logprior_fn(x)

    jacobian = get_jacobians(samples, bijectors)
    log_prob = jax.vmap(logjoint)(apply_bijectors(samples, bijectors))   
    return log_prob + jacobian

#
def get_importance_weights(model, bijectors: dict, prop_dist, samples):
    """ Compute the importance weight reflecting the relative probability under the posterior vs the proposal distribution.

    """
    return logposterior_proposals(model, bijectors, samples) - proposal_distribution_logprob(prop_dist, samples)

#
def bridge_sampling(key, model: Model, posterior_samples, bijectors: dict, proposal_type: str = 'gaussian', N2: int = 1000, max_iter: int = 20, tol: float = 1e-6):
    """ Run the warp-II bridge sampling algorithm, using the optimal bridge function by Meng & Wong (1996).

    Args:
        - model: The bamojax model to get the marginal likelihood for.
        - posterior_samples: Posterior samples of the model, obtained via any suitable way.
        - bijectors: A dictionary of transformations to more closely align the proposal distribution with
        the posterior. When no bijector is provided for a variable, an identity transformation is used for
        convenience.
        - proposal_type: Indicates the kind of proposals to use. For now, only Gaussians are supported, 
        ideally we make this generic. Student-T distributions are also common, due to their wide tails.
        - N2: The number of draws from the proposal distribution.
        - max_iter: The maximum number of bridge sampling iterations. Not often used.
        - tol: The convergence criterion.

    Returns:

        - The log marginal likelihood of `model`
        - The number of iterations of the bridge sampler (typically small)

    References:

        - Meng & Wong, 1996, Simulating ratios of normalizing constants via a simple identity: a theoretical 
        exploration. Statistica Sinica, 831--860.
        - Gronau et al., 2017, A tutorial on bridge sampling, Journal of Mathematical Psychology 81, 80--97.

    """

    for k in model.get_latent_nodes():
        if k not in bijectors:
            bijectors[k] = nprb.IdentityTransform()

    posterior_samples_batch_1 = jax.tree.map(lambda x: x[1::2, ...], posterior_samples)
    posterior_samples_batch_2 = jax.tree.map(lambda x: x[::2, ...], posterior_samples)

    # N1 = ravel_pytree(jax.tree.map(lambda x: x.shape[0], posterior_samples_batch_1))[0][0]
    N1 = list(posterior_samples_batch_1.values())[0].shape[0]

    if proposal_type == 'gaussian':
        # apply bijector to posterior samples batch 1
        transformed_samples_1 = apply_inverse_bijectors(posterior_samples_batch_1, bijectors)

        # get posterior distribution and unravel function
        proposal_distribution, unravel_fn = get_proposal_distribution(transformed_samples_1)
    else:
        raise NotImplementedError(f'Proposal type "{proposal_type}" is not implemented')
    
    proposal_samples = sample_from_proposal_distribution(key, proposal_distribution, unravel_fn, N2)

    L2 = get_importance_weights(model, bijectors, proposal_distribution, proposal_samples)
    transformed_samples_2 = apply_inverse_bijectors(posterior_samples_batch_2, bijectors)
    L1 = get_importance_weights(model, bijectors, proposal_distribution, transformed_samples_2)

    s1 = N1 / (N1 + N2)
    s2 = N2 / (N1 + N2)

    # For numerical stability, Gronau et al. suggest the following approach:
    l_star = jnp.median(L1)
    L1_c = L1 - l_star
    L2_c = L2 - l_star

    def one_iter(r_prev):
        num_terms = L2_c - jnp.log(s1 * jnp.exp(L2_c) + s2 * r_prev)
        log_num = logsumexp(num_terms) - jnp.log(N2)
        denom_terms = -jnp.log(s1 * jnp.exp(L1_c) + s2 * r_prev)
        log_denom = logsumexp(denom_terms) - jnp.log(N1)
        return jnp.exp(log_num - log_denom)

    #
    def cond_fn(carry):
        t, r_prev, r_new = carry
        return jnp.logical_and(t < max_iter, jnp.abs(r_new - r_prev) > tol)

    #
    def body_fn(carry):
        t, _, r = carry
        r_new = one_iter(r)
        return (t + 1, r, r_new)

    #
    init_state = (0, 1.0, 0.0)
    n_iter, _, r_new = jax.lax.while_loop(cond_fn, body_fn, init_state)

    log_Z = jnp.log(r_new) + l_star  # see Gronau et al., page 95

    return log_Z, n_iter

#