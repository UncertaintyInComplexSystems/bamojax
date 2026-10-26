import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrnd

from bamojax.more_distributions import GaussianProcessFactory

def rbf(params, x, y):
    """Minimal RBF kernel with the (params, x, y) signature GaussianProcessFactory expects."""
    d = x[:, None] - y[None, :]
    return params["variance"] * jnp.exp(-0.5 * d**2 / params["lengthscale"]**2)



def test_gaussian_processes():

    cov_fn = rbf  # can be drawn from e.g. GPJax; we now use the old jaxkern syntax
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


