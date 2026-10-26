""" Shared code for the test suite. 

Many tests compare approximate inference / evidence estimates against conjugate
models whose posteriors and marginal likelihoods are known analytically. Those
reference computations live here so every test module uses the same ground truth.

Run the fast suite with ``pytest -m "not slow"``; the full suite with ``pytest``.

"""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jrnd
import numpyro.distributions as dist
import numpyro.distributions.transforms as nprb
import pytest
from jax.scipy.special import betaln, gammaln

from bamojax.base import Model


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long-running statistical test (deselect with -m 'not slow')")


# ---------------------------------------------------------------------------
# Closed-form references
# ---------------------------------------------------------------------------

def normal_known_sd_posterior(y, sd, mu0, sd0):
    """Posterior N(mu_n, sd_n^2) of the mean of N(mu, sd^2) data under a N(mu0, sd0^2) prior."""
    n = y.shape[0]
    prec = 1.0 / sd0**2 + n / sd**2
    mean = (mu0 / sd0**2 + jnp.sum(y) / sd**2) / prec
    return mean, jnp.sqrt(1.0 / prec)


def normal_known_sd_lml(y, sd, mu0, sd0):
    """log p(y) for y_i ~ N(mu, sd^2), mu ~ N(mu0, sd0^2): y ~ N(mu0 1, sd^2 I + sd0^2 11')."""
    n = y.shape[0]
    cov = sd**2 * jnp.eye(n) + sd0**2 * jnp.ones((n, n))
    return dist.MultivariateNormal(loc=mu0 * jnp.ones(n), covariance_matrix=cov).log_prob(y)


def beta_binomial_lml(k, N, a, b):
    """log p(k) for k ~ Binomial(N, theta), theta ~ Beta(a, b)."""
    log_binom = gammaln(N + 1) - gammaln(k + 1) - gammaln(N - k + 1)
    return log_binom + betaln(k + a, N - k + b) - betaln(a, b)


def gamma_poisson_lml(y, alpha, beta):
    """log p(y) for y_i ~ Poisson(lam), lam ~ Gamma(alpha, rate=beta)."""
    n, s = y.shape[0], jnp.sum(y)
    return (alpha * jnp.log(beta) - gammaln(alpha)
            + gammaln(alpha + s) - (alpha + s) * jnp.log(beta + n)
            - jnp.sum(gammaln(y + 1)))


def bayesian_linreg_lml(X, y, sd, prior_sd):
    """log p(y | X) for y = X w + eps, w ~ N(0, prior_sd^2 I), eps ~ N(0, sd^2 I)."""
    n = X.shape[0]
    cov = sd**2 * jnp.eye(n) + prior_sd**2 * X @ X.T
    return dist.MultivariateNormal(loc=jnp.zeros(n), covariance_matrix=cov).log_prob(y)


# ---------------------------------------------------------------------------
# Data + model fixtures
# ---------------------------------------------------------------------------

GUK = dict(true_mean=1.5, sd=2.0, mu0=0.0, sd0=3.0, n=50)


@pytest.fixture(scope="session")
def guk_data():
    return GUK["true_mean"] + GUK["sd"] * jrnd.normal(jrnd.PRNGKey(123), shape=(GUK["n"],))


def make_guk_model(y, sd=GUK["sd"], mu0=GUK["mu0"], sd0=GUK["sd0"], name="Gaussian, unknown mean"):
    model = Model(name)
    mu = model.add_node("mu", distribution=dist.Normal(loc=mu0, scale=sd0))
    model.add_node("y", distribution=dist.Normal, parents=dict(loc=mu, scale=sd), observations=y)
    return model


@pytest.fixture
def guk_model(guk_data):
    return make_guk_model(guk_data)


BB = dict(k=7, N=20, a=2.0, b=3.0)


def make_beta_binomial_model(k=BB["k"], N=BB["N"], a=BB["a"], b=BB["b"]):
    model = Model("beta-binomial")
    theta = model.add_node("theta", distribution=dist.Beta(concentration1=a, concentration0=b))
    model.add_node("k", distribution=dist.Binomial, parents=dict(probs=theta, total_count=N), observations=k)
    return model


@pytest.fixture
def beta_binomial_model():
    return make_beta_binomial_model()


def make_eight_schools():
    means = jnp.array([28., 8., -3., 7., -1., 1., 18., 12.])
    stddevs = jnp.array([15., 10., 16., 11., 9., 11., 10., 18.])
    J = means.shape[0]
    model = Model("eight schools")
    mu = model.add_node("mu", distribution=dist.Normal(loc=0., scale=10.))
    tau = model.add_node("tau", distribution=dist.TransformedDistribution(dist.Normal(loc=1., scale=1.), nprb.ExpTransform()))
    theta = model.add_node("theta", distribution=dist.Normal, parents=dict(loc=mu, scale=tau), shape=(J,))
    model.add_node("y", distribution=dist.Normal, parents=dict(loc=theta, scale=stddevs), observations=means)
    return model, means, stddevs
