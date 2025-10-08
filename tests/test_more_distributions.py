import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrnd

import jaxkern as jk

import numpyro as npr
import numpyro.distributions as dist
import numpyro.distributions.transforms as nprb

from bamojax.more_distributions import AscendingDistribution, GaussianProcessFactory, Zero

def test_ascending_distribution():

    min_val, max_val, num_el = 0.0, 5.0, 100
    asc_dist = AscendingDistribution(min_val, max_val, num_el)
    num_draws = 1000
    draws = asc_dist.sample(key=jrnd.PRNGKey(0), sample_shape=(num_draws, ))
    is_sorted = lambda a: jnp.all(a[:-1] <= a[1:])

    assert jnp.isclose(min_val, jnp.mean(draws[:, 0]), atol=0.1, rtol=1e-3)
    assert jnp.isclose(jnp.mean(draws[:, -1]), max_val, rtol=1e-3)
    assert jnp.all(jax.vmap(is_sorted)(draws))

#
def test_gaussian_processes():

    cov_fn = jk.RBF().cross_covariance
    n = 100
    x = jnp.linspace(0, 1, num=n)

    params = dict(lengthscale=0.5, variance=10.0)
    gp_dist = GaussianProcessFactory(cov_fn=cov_fn)(input=x, **params)
    num_draws = 100
    draws = gp_dist.sample(key=jrnd.PRNGKey(0), sample_shape=(num_draws, ))

    assert gp_dist.event_shape == (n, )   
    assert draws.shape == (num_draws, n)

    # verify MLE parameters by grid search
    likelihood = lambda ls, var: jnp.sum(GaussianProcessFactory(cov_fn=cov_fn)(input=x, lengthscale=ls, variance=var).log_prob(value=draws))
    scales = jnp.logspace(-4, 0, num=100)
    variances = jnp.logspace(-3, 2, num=50)

    scores = jax.vmap(jax.vmap(likelihood, in_axes=(None, 0)), in_axes=(0, None))(scales, variances)
    flat_index = jnp.argmax(scores)
    i, _ = jnp.unravel_index(flat_index, scores.shape)
    
    assert jnp.isclose(scales[i], params['lengthscale'], atol=1e-1)
    
#


