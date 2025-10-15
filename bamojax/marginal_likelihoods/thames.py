
import jax

import jax.numpy as jnp
import jax.random as jrnd
from jax.scipy.special import gammaln

from bamojax.base import Model
from bamojax.marginal_likelihoods.utility import flatten_dict_to_array

def sample_unit_sphere(key, d, n):
    """ Randomly determine angles and radii on the unit sphere.
    
    """
    key_angle, key_radius = jrnd.split(key)
    angles_unnorm = jrnd.normal(key_angle, (n, d))
    angles_norm = angles_unnorm / jnp.linalg.norm(angles_unnorm, axis=1, keepdims=True)

    rad = jrnd.uniform(key_radius, (n, 1))
    rad_scaled = rad**(1 / d)
    return rad_scaled * angles_norm

#
def sample_ellipsoid(key, Sigma, n, c=None, mu=None, jitter=1e-12):
    """ Affine transformation of points on the unit sphere to the ellipsoid.
    
    """
    d = Sigma.shape[0]
    if c is None:
        c = jnp.sqrt(d+1)
    if mu is None:
        mu = jnp.zeros((d, ))

    L = jnp.linalg.cholesky(Sigma + jitter * jnp.eye(d))
    u = sample_unit_sphere(key, d, n)
    return mu + (u @ L.T)*c

#
def log_truncated_mvn_volume(d, cov, c=None):
    if c is None:
        c = jnp.sqrt(d + 1)
    _, logdet = jnp.linalg.slogdet(cov)
    return d/2*jnp.log(d+1) + d/2*jnp.log(jnp.pi) + 1/2*logdet - gammaln(d/2 + 1)

#
def thames(key, model: Model, posterior_samples, M=100, adjust_volume=True):
    """ Implements the Truncated Harmonic Mean Estimator (THAMES), by Metodiev et al. (2025)
    
    Args:
        - key: The random seed for the volume adjustment.
        - model: The bamojax model to get the marginal likelihood for.
        - posterior_samples: Samples from the posterior of the model.
        - M: The number of samples to draw from the ellipsoid to correct for the different
        volume when using constrained parameters in the model.

    Returns:
        - The log marginal likelihood of `model`

    References:

        -  Martin Metodiev, Marie Perrot-Dockès, Sarah Ouadah, Nicholas J. Irons, Pierre 
        Latouche, Adrian E. Raftery. "Easily Computed Marginal Likelihoods from Posterior 
        Simulation Using the THAMES Estimator." Bayesian Anal. 20 (3) 1003 - 1030, 2025. 
        https://doi.org/10.1214/24-BA1422 



    """

    d = model.get_model_size()

    posterior_samples_batch_1 = jax.tree.map(lambda x: x[1::2, ...], posterior_samples)
    posterior_samples_batch_2 = jax.tree.map(lambda x: x[::2, ...], posterior_samples)

    N = list(posterior_samples_batch_1.values())[0].shape[0]

    samples_batch_1_flat, unravel_one = flatten_dict_to_array(posterior_samples_batch_1)
    samples_batch_2_flat, unravel_one = flatten_dict_to_array(posterior_samples_batch_2)

    mu = jnp.mean(samples_batch_1_flat, axis=0)
    cov = jnp.cov(samples_batch_1_flat, rowvar=False)

    L = jnp.linalg.cholesky(cov)
    samples_std = jnp.linalg.solve(L, (samples_batch_2_flat - mu).T).T

    # Determine the relative heaviness of the posterior tails
    truncation_subset = jnp.linalg.norm(samples_std, axis=1)**2 < (d + 1)

    flat_loglikelihood_fn = lambda x: model.loglikelihood_fn()(unravel_one(x))
    flat_logprior_fn = lambda x: model.logprior_fn()(unravel_one(x))

    logliks = jax.vmap(flat_loglikelihood_fn)(samples_batch_2_flat)
    logpriors = jax.vmap(flat_logprior_fn)(samples_batch_2_flat)

    logV = log_truncated_mvn_volume(d, cov)  

    log_terms = -logliks - logpriors
    log_sum = jax.scipy.special.logsumexp(jnp.where(truncation_subset, log_terms, -jnp.inf))

    log_Z = jnp.log(N) + logV - log_sum

    if adjust_volume:
        M = 100
        samples_ellipse = sample_ellipsoid(key, n=M, Sigma=cov, c=jnp.sqrt(d+1), mu=mu)

        logliks_ellipse = jax.vmap(flat_loglikelihood_fn)(samples_ellipse)
        logprior_ellipse = jax.vmap(flat_logprior_fn)(samples_ellipse)

        support = ~jnp.isnan(logliks_ellipse + logprior_ellipse)

        # R is the ratio of volumes of the constrained space vs the assumed (truncated) Gaussian 'importance' density
        R = jnp.sum(support) / M

        log_Z += jnp.log(R)

    return log_Z
#