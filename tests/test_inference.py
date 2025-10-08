import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrnd
import jaxkern as jk

import numpyro as npr
import numpyro.distributions as dist
import numpyro.distributions.transforms as nprb

from bamojax.base import Model
from bamojax.inference import SMCInference, MCMCInference, VIInference, SGMCMCInference
from bamojax.samplers import gibbs_sampler, mcmc_sampler
from bamojax.more_distributions import GaussianProcessFactory
import blackjax

 
def guk_lml(y, sd, mu0, tau):
    # See https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
    y_bar = jnp.mean(y)
    n = len(y)
    return jnp.log(sd) - n/2*jnp.log(2*jnp.pi*sd**2) - 1/2*jnp.log(n*tau**2 + sd**2) - jnp.sum(y**2) / (2*sd**2) - mu0 / (2*tau**2) + (tau**2 * n**2 * y_bar**2 / sd**2 + sd**2*mu0**2 / tau**2 + 2*n*y_bar*mu0) / (2*(n*tau**2 + sd**2))
    
#
def exact_posterior(y, sd, mu0, sd0) -> dist.Distribution:
    """For this simple model, the exact posterior is available (given a Gaussian prior on the mean), which allows us to compare the approximate SMC result with a ground truth.
    
    """
    n = len(y)
    sd_post = 1.0 / (1/sd0**2 + n  / sd**2)
    mu_post = sd_post*(mu0 / sd0**2 + jnp.sum(y) / sd**2)
    return dist.Normal(loc=mu_post, scale=jnp.sqrt(sd_post))

#
def test_gibbs_inference():
    """ Replication of tensorflow probability's example at https://www.tensorflow.org/probability/examples/Eight_Schools

    """

    num_chains = 4
    num_samples = 100000
    num_burn = 100000

    step_fns = dict(mu=blackjax.normal_random_walk,
                    tau=blackjax.normal_random_walk,
                    theta=blackjax.normal_random_walk)
    step_fn_params = dict(mu=dict(sigma=10.0),
                            tau=dict(sigma=10.0),
                            theta=dict(sigma=5.0*jnp.eye(J)))
    gibbs_kernel = gibbs_sampler(model=ES, step_fns=step_fns, step_fn_params=step_fn_params)

    engine = MCMCInference(model=ES, mcmc_kernel=gibbs_kernel, num_chains=num_chains, num_samples=num_samples, num_burn=num_burn)
    result = engine.run(jrnd.PRNGKey(0))

    assert jnp.allclose(jnp.mean(result['states']['mu'], axis=1), 5.8, atol=0.2)

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
    assert jnp.isclose(jnp.mean(final_state.particles['mu']), exact_posterior_dist.mean, atol=0.05)
    assert jnp.isclose(jnp.var(final_state.particles['mu']), exact_posterior_dist.variance, atol=0.05)


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

    assert jnp.isclose(jnp.mean(states['mu']), exact_posterior_dist.mean, atol=0.05)
    assert jnp.isclose(jnp.var(states['mu']), exact_posterior_dist.variance, atol=0.05)

#
def test_vi():
    import optax

    num_chains = 4
    num_steps = 100_000
    num_gradient_samples = 10
    num_draws = 1_000

    engine = VIInference(ES, 
                        num_chains=num_chains, 
                        num_steps=num_steps, 
                        num_gradient_samples=num_gradient_samples, 
                        optimizer=optax.sgd(learning_rate=1e-1), 
                        optimizer_chain_args=optax.clip_by_global_norm(1.0))

    result = engine.run(key=jrnd.PRNGKey(0))
    assert jnp.allclose(result['states'].mu['mu'][:,-1], 6.1, atol=0.1)

    vi_samples = engine.sample_from_variational(jrnd.PRNGKey(1), vi_result=result, num_draws=num_draws)
    assert jnp.min(vi_samples['tau'].flatten()) > 0.0
    assert vi_samples['theta'].shape == (num_chains, num_draws, J)

    num_chains = 4
    num_steps = 100_000
    num_gradient_samples = 10
    num_draws = 1_000

    engine = VIInference(ES, 
                        num_chains=num_chains, 
                        num_steps=num_steps, 
                        num_gradient_samples=num_gradient_samples, 
                        optimizer=optax.sgd(learning_rate=1e-1), 
                        optimizer_chain_args=optax.clip_by_global_norm(1.0))

    result = engine.run(key=jrnd.PRNGKey(0))
    assert jnp.allclose(result['states'].mu['mu'][:,-1], 6.1, atol=0.1)

    vi_samples = engine.sample_from_variational(jrnd.PRNGKey(1), vi_result=result, num_draws=num_draws)
    assert jnp.min(vi_samples['tau'].flatten()) > 0.0
    assert vi_samples['theta'].shape == (num_chains, num_draws, J)

#
def test_sginference():
    n = 500
    x = jnp.linspace(-1, 1, n)
    true_lengthscale = 0.15
    true_variance = 3.0
    true_noise = 0.1

    cov_fn = jk.RBF().cross_covariance

    f = GaussianProcessFactory(cov_fn=cov_fn)(input=x, lengthscale=true_lengthscale, variance=true_variance).sample(key=jrnd.PRNGKey(0))
    y = f + 0.1 * jax.random.normal(jrnd.PRNGKey(0), shape=(n, ))

    def marginal_gp_fn(x, lengthscale, variance, obs_noise):
        n = x.shape[0]
        params = dict(lengthscale=lengthscale, variance=variance)
        K = cov_fn(params, x, x)
        Sigma = K + obs_noise**2 * jnp.eye(n)
        return dict(loc=jnp.zeros(n), covariance_matrix=Sigma)

    #

    cov_fn = jk.RBF().cross_covariance

    mgpmodel = Model('Marginal GP')
    lengthscale = mgpmodel.add_node('lengthscale', distribution=dist.TransformedDistribution(dist.Normal(loc=0., scale=1.), nprb.ExpTransform()))
    variance = mgpmodel.add_node('variance', distribution=dist.TransformedDistribution(dist.Normal(loc=0., scale=1.), nprb.ExpTransform()))
    obs_noise = mgpmodel.add_node('obs_noise', distribution=dist.TransformedDistribution(dist.Normal(loc=0., scale=1.), nprb.ExpTransform()))
    x_node = mgpmodel.add_node(name='x', observations=x)
    y_node = mgpmodel.add_node(name='y', 
                            distribution=dist.MultivariateNormal, 
                            observations=y, 
                            parents=dict(x=x_node, 
                                            lengthscale=lengthscale, 
                                            variance=variance, 
                                            obs_noise=obs_noise), 
                            link_fn=marginal_gp_fn)
    
    batch_size = 25
    num_samples = 200_000
    num_burn = 200_000
    num_thin = 20
    num_chains = 1
    stepsize = 1e-5  # with larger learning rates we encounter nans

    engine = SGMCMCInference(model=mgpmodel, 
                            num_chains=num_chains, 
                            sgmcmc_kernel=blackjax.sgld, 
                            data_size=n, 
                            batch_size=batch_size, 
                            batch_nodes=[x_node, y_node], 
                            stepsize=stepsize, 
                            num_samples=num_samples, 
                            num_burn=num_burn, 
                            num_thin=num_thin)


    result = engine.run(jrnd.PRNGKey(0))

    assert jnp.isclose(jnp.mean(result['states']['lengthscale']), true_lengthscale, atol=0.01)
    assert jnp.isclose(jnp.mean(result['states']['obs_noise']), true_noise, atol=0.01)  
    # Note that we do not check the variance, as it is not identifiable in the model.
#

true_mean = 5.0
true_sd = 3.0
n = 100
y = dist.Normal(loc=true_mean, scale=true_sd).sample(key=jrnd.PRNGKey(0), sample_shape=(n, ))
mu0 = 0.0
sd0 = 2.0

gukmodel = Model('Gaussian with unknown mean')
unknown_mean = gukmodel.add_node('mu', distribution=dist.Normal(loc=mu0, scale=sd0))
_ = gukmodel.add_node('y', distribution=dist.Normal, observations=y, parents=dict(loc=unknown_mean, scale=true_sd))

means = jnp.array([28, 8, -3, 7, -1, 1, 18, 12])
stddevs = jnp.array([15, 10, 16, 11, 9, 11, 10, 18])

J = len(means)
ES = Model('eight schools')
mu = ES.add_node('mu', distribution=dist.Normal(loc=0, scale=10))
tau = ES.add_node('tau', distribution=dist.TransformedDistribution(dist.Normal(loc=5., scale=1.), nprb.ExpTransform()))
theta = ES.add_node('theta', distribution=dist.Normal, parents=dict(loc=mu, scale=tau), shape=(J, ))
_ = ES.add_node('y', distribution=dist.Normal, parents=dict(loc=theta, scale=stddevs), observations=means)

