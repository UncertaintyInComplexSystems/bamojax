import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrnd

import distrax as dx
from bamojax.base import Model
from bamojax.inference import gibbs_sampler, mcmc_sampler, SMCInference, MCMCInference
import blackjax

 
def guk_lml(y, sd, mu0, tau):
    # See https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
    y_bar = jnp.mean(y)
    n = len(y)
    return jnp.log(sd) - n/2*jnp.log(2*jnp.pi*sd**2) - 1/2*jnp.log(n*tau**2 + sd**2) - jnp.sum(y**2) / (2*sd**2) - mu0 / (2*tau**2) + (tau**2 * n**2 * y_bar**2 / sd**2 + sd**2*mu0**2 / tau**2 + 2*n*y_bar*mu0) / (2*(n*tau**2 + sd**2))
    
#
def exact_posterior(y, sd, mu0, sd0) -> dx.Distribution:
    """For this simple model, the exact posterior is available (given a Gaussian prior on the mean), which allows us to compare the approximate SMC result with a ground truth.
    
    """
    n = len(y)
    sd_post = 1.0 / (1/sd0**2 + n  / sd**2)
    mu_post = sd_post*(mu0 / sd0**2 + jnp.sum(y) / sd**2)
    return dx.Normal(loc=mu_post, scale=jnp.sqrt(sd_post))

#
def test_gibbs_inference():
    means = jnp.array([28, 8, -3, 7, -1, 1, 18, 12])
    stddevs = jnp.array([15, 10, 16, 11, 9, 11, 10, 18])

    J = len(means)
    ES = Model('eight schools')
    mu = ES.add_node('mu', distribution=dx.Normal(loc=0, scale=10))
    tau = ES.add_node('tau', distribution=dx.Transformed(dx.Normal(loc=5, scale=1), tfb.Exp()))
    theta = ES.add_node('theta', distribution=dx.Normal, parents=dict(loc=mu, scale=tau), shape=(J, ))
    _ = ES.add_node('y', distribution=dx.Normal, parents=dict(loc=theta, scale=stddevs), observations=means)

    step_fns = dict(mu=blackjax.normal_random_walk,
                    tau=blackjax.normal_random_walk,
                    theta=blackjax.normal_random_walk)
    step_fn_params = dict(mu=dict(sigma=10.0),
                            tau=dict(sigma=5.0),
                            theta=dict(sigma=10.0*jnp.eye(J)))
    gibbs_kernel = gibbs_sampler(model=ES, step_fns=step_fns, step_fn_params=step_fn_params)

    num_samples = 50000
    num_burn = 10000
    num_chains = 4
    engine = MCMCInference(model=ES, mcmc_kernel=gibbs_kernel, num_chains=num_chains, num_samples=num_samples, num_burn=num_burn)
    result = engine.run(jrnd.PRNGKey(0))
    assert jnp.isclose(result['states']['theta'].mean(), 7.1, atol=0.05)

#
def test_smc_inference():

    num_particles = 10_000
    num_mcmc_steps = 500
    num_chains = 1
    stepsize = 0.01

    mcmc_params = dict(sigma=stepsize*jnp.eye(gukmodel.get_model_size()))
    rmh = mcmc_sampler(gukmodel, mcmc_kernel=blackjax.normal_random_walk, mcmc_parameters=mcmc_params)

    engine = SMCInference(model=gukmodel, num_chains=num_chains, mcmc_kernel=rmh, num_mutations=num_mcmc_steps, num_particles=num_particles)
    result = engine.run(jrnd.PRNGKey(1))
    final_state = result['final_state']
    lml = result['lml']

    exact_lml = guk_lml(mu0=mu0, tau=sd0, sd=true_sd, y=y)
    exact_posterior_dist = exact_posterior(y, true_sd, mu0, sd0)

    assert jnp.isclose(exact_lml, lml, rtol=1e-3)
    assert jnp.isclose(jnp.mean(final_state.particles['mu']), exact_posterior_dist.mean(), atol=0.05)
    assert jnp.isclose(jnp.var(final_state.particles['mu']), exact_posterior_dist.variance(), atol=0.05)


#
def test_nuts_inference():

    num_warmup = 500
    num_samples = 1_000
    num_chains = 4

    cold_nuts_parameters = dict(step_size=0.5, inverse_mass_matrix=0.0001*jnp.eye(gukmodel.get_model_size()))  # these will be overriden by the window adaptation
    nuts_kernel = mcmc_sampler(model=gukmodel, mcmc_kernel=blackjax.nuts, mcmc_parameters=cold_nuts_parameters)

    engine = MCMCInference(model=gukmodel, num_chains=num_chains, num_samples=num_samples, num_warmup=num_warmup, num_burn=500, mcmc_kernel=nuts_kernel)
    result = engine.run(jrnd.PRNGKey(0))

    states = result['states']

    exact_posterior_dist = exact_posterior(y, true_sd, mu0, sd0)

    assert jnp.isclose(jnp.mean(states['mu']), exact_posterior_dist.mean(), atol=0.05)
    assert jnp.isclose(jnp.var(states['mu']), exact_posterior_dist.variance(), atol=0.05)

#

true_mean = 5.0
true_sd = 3.0
n = 100
y = dx.Normal(loc=true_mean, scale=true_sd).sample(seed=jrnd.PRNGKey(0), sample_shape=(n, ))
mu0 = 0.0
sd0 = 2.0

gukmodel = Model('Gaussian with unknown mean')
unknown_mean = gukmodel.add_node('mu', distribution=dx.Normal(loc=mu0, scale=sd0))
_ = gukmodel.add_node('y', distribution=dx.Normal, observations=y, parents=dict(loc=unknown_mean, scale=true_sd))

