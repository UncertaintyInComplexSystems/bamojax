import jax.numpy as jnp
import jax.random as jrnd
from bamojax.utility.StanParser import stan_to_bamojax

import numpyro.distributions as dist

def logdensity(model, state):
    return model.logprior_fn()(state) + model.loglikelihood_fn()(state)
 
#

def test_eight_schools():
    code = """
    data { int<lower=0> J; vector[J] y; vector<lower=0>[J] sigma; }
    parameters { real mu; real<lower=0> tau; vector[J] eta; }
    transformed parameters { vector[J] theta = mu + tau * eta; }
    model {
        mu ~ normal(0, 5);
        tau ~ cauchy(0, 5);
        eta ~ std_normal();
        y ~ normal(theta, sigma);   // likelihood
    }"""
    y = jnp.array([28., 8, -3, 7, -1, 1, 18, 12])
    sigma = jnp.array([15., 10, 16, 11, 9, 11, 10, 18])
    model = stan_to_bamojax(code, dict(J=8, y=y, sigma=sigma))
    state = model.sample_prior(jrnd.PRNGKey(0))
    assert state['eta'].shape == (8, ) and state['tau'] > 0
    expected = (dist.Normal(0, 5).log_prob(state['mu']) + dist.HalfCauchy(5).log_prob(state['tau'])
                + dist.Normal().log_prob(state['eta']).sum()
                + dist.Normal(state['mu'] + state['tau'] * state['eta'], sigma).log_prob(y).sum())
    assert jnp.allclose(logdensity(model, state), expected)

#
def test_vectorized_or_loop():
    data = dict(N=20, K=3, J=4, x=jrnd.normal(jrnd.PRNGKey(1), (20, 3)), g=jnp.arange(20) % 4 + 1,
                    y=jrnd.poisson(jrnd.PRNGKey(2), 2.0, (20,)))
    head = """
    data { int N; int K; int J; matrix[N, K] x; array[N] int<lower=1, upper=J> g; array[N] int y; }
    parameters { vector[K] beta; vector[J] alpha; real<lower=0> sigma; }
    """
    looped = head + """
    transformed parameters {
        vector[N] eta;
        for (n in 1:N) eta[n] = alpha[g[n]] + x[n] * beta;
    }
    model {
        beta ~ normal(0, 1);
        sigma ~ exponential(1);
        for (j in 1:J) { alpha[j] ~ normal(0, sigma); }
        for (n in 1:N) y[n] ~ poisson_log(eta[n]);
    }"""
    vectorized = head + """
    model {
        beta ~ normal(0, 1);
        sigma ~ exponential(1);
        alpha ~ normal(0, sigma);
        y ~ poisson_log(alpha[g] + x * beta);
    }"""
    a, b = stan_to_bamojax(looped, data), stan_to_bamojax(vectorized, data)
    state = b.sample_prior(jrnd.PRNGKey(3))
    assert a.sample_prior(jrnd.PRNGKey(3))['alpha'].shape == (4,)
    expected = (dist.Normal(0, 1).log_prob(state['beta']).sum() + dist.Exponential(1).log_prob(state['sigma'])
                + dist.Normal(0, state['sigma']).log_prob(state['alpha']).sum()
                + dist.Poisson(jnp.exp(state['alpha'][data['g'] - 1] + data['x'] @ state['beta'])).log_prob(data['y']).sum())
    assert jnp.allclose(logdensity(a, state), expected)
    assert jnp.allclose(lambda s: logdensity(b, s)(state), expected)

#
def test_multivariate():
    code = """
    data { int N; int K; array[N] vector[K] y; }
    transformed data { vector[K] zero = rep_vector(0, K); }
    parameters { cholesky_factor_corr[K] L; vector<lower=0>[K] tau; simplex[K] w; }
    model {
        L ~ lkj_corr_cholesky(2.0);
        tau ~ cauchy(0, 2.5);
        w ~ dirichlet(rep_vector(1.0, K));
        y ~ multi_normal_cholesky(zero, diag_pre_multiply(tau .* w, L));
    }"""
    y = jrnd.normal(jrnd.PRNGKey(0), (10, 3))
    model = stan_to_bamojax(code, dict(N=10, K=3, y=y))
    state = model.sample_prior(jrnd.PRNGKey(1))
    assert state['L'].shape == (3, 3) and state['tau'].shape == (3,) and jnp.all(state['tau'] > 0)
    scale_tril = (state['tau'] * state['w'])[:, None] * state['L']
    expected = (dist.LKJCholesky(3, 2.0).log_prob(state['L']) + dist.Dirichlet(jnp.ones(3)).log_prob(state['w'])
                + dist.HalfCauchy(2.5).log_prob(state['tau']).sum()
                + dist.MultivariateNormal(jnp.zeros(3), scale_tril=scale_tril).log_prob(y).sum())
    assert jnp.allclose(logdensity(model, state), expected)

#