import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrnd

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

from bamojax.more_distributions import AscendingDistribution, Wishart

def test_ascending_distribution():

    min_val, max_val, num_el = 0.0, 5.0, 100
    asc_dist = AscendingDistribution(min_val, max_val, num_el)
    num_draws = 1000
    draws = asc_dist.sample(seed=jrnd.PRNGKey(0), sample_shape=(num_draws, ))
    is_sorted = lambda a: jnp.all(a[:-1] <= a[1:])

    assert jnp.isclose(min_val, jnp.mean(draws[:, 0]), atol=0.1, rtol=1e-3)
    assert jnp.isclose(jnp.mean(draws[:, -1]), max_val, rtol=1e-3)
    assert jnp.all(jax.vmap(is_sorted)(draws))

#
def test_wishart_distribution():

    p = 3
    dof = p + 3
    V = jnp.eye(p)

    num_draws = 10000
    wishart_dist = Wishart(dof=dof, scale=V)
    draws = wishart_dist.sample(seed=jrnd.PRNGKey(0), sample_shape=(num_draws, ))

    monte_carlo_expectation = jnp.mean(draws, axis=0)
    exact_expectation = wishart_dist.mean()
    exact_mode = wishart_dist.mode()

    assert wishart_dist.event_shape == (p, p)
    assert jnp.allclose(monte_carlo_expectation, dof*V, atol=1e-1, rtol=1e-3)
    assert jnp.allclose(exact_expectation, dof*V, atol=1e-5, rtol=1e-5)

    prob_exact_expectation = wishart_dist.log_prob(exact_expectation)
    prob_exact_mode = wishart_dist.log_prob(exact_mode)

    assert prob_exact_mode > prob_exact_expectation


#


