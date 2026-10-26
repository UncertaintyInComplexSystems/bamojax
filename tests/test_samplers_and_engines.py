"""Tests for samplers (``bamojax.samplers``) and inference engines (``bamojax.inference``).

Wherever possible the approximate posterior is compared against an exact answer:
conjugate Gaussian models, a GP regression with fixed hyperparameters (whose latent
posterior is Gaussian), and a binary model small enough to enumerate.
"""
import itertools

import blackjax
import jax
import jax.numpy as jnp
import jax.random as jrnd
import numpyro.distributions as dist
import numpyro.distributions.transforms as nprb
import optax
import pytest
from jax.scipy.special import logsumexp

from bamojax.base import MetaModel, Model
from bamojax.inference import LaplaceInference, MCMCInference, SMCInference, VIInference, get_model_bijectors
from bamojax.modified_blackjax.more_proposals import bernoulli_random_walk
from bamojax.more_distributions import GaussianProcessFactory
from bamojax.samplers import gibbs_sampler, mcmc_sampler, reversible_jump_mcmc

from conftest import GUK, gamma_poisson_lml, make_eight_schools, normal_known_sd_lml, normal_known_sd_posterior


@pytest.fixture
def guk_posterior(guk_data):
    return normal_known_sd_posterior(guk_data, GUK["sd"], GUK["mu0"], GUK["sd0"])


def rbf(params, x, y):
    """Minimal RBF kernel with the (params, x, y) signature GaussianProcessFactory expects."""
    d = x[:, None] - y[None, :]
    return params["variance"] * jnp.exp(-0.5 * d**2 / params["lengthscale"]**2)


# ---------------------------------------------------------------------------
# mcmc_sampler
# ---------------------------------------------------------------------------

def test_mcmc_sampler_requires_kernel_and_parameters_together(guk_model):
    with pytest.raises(ValueError):
        mcmc_sampler(guk_model, mcmc_kernel=blackjax.normal_random_walk)
    with pytest.raises(ValueError):
        mcmc_sampler(guk_model, mcmc_parameters=dict(sigma=1.0))


def test_mcmc_sampler_defaults_to_nuts(guk_model, guk_posterior):
    kernel = mcmc_sampler(guk_model)
    result = MCMCInference(guk_model, mcmc_kernel=kernel, num_samples=1_000, num_burn=0, num_warmup=300).run(jrnd.PRNGKey(0))
    mean, sd = guk_posterior
    assert jnp.isclose(jnp.mean(result["states"]["mu"]), mean, atol=0.05)
    assert jnp.isclose(jnp.std(result["states"]["mu"]), sd, atol=0.05)


def test_mcmc_sampler_temperature_zero_targets_the_prior(guk_model):
    """With temperature=0 the tempered density is the prior: a key ingredient of tempered SMC."""
    kernel = mcmc_sampler(guk_model, mcmc_kernel=blackjax.normal_random_walk, mcmc_parameters=dict(sigma=3.0 * jnp.eye(1)))
    state = kernel.init({"mu": jnp.asarray(0.0)})

    def body(state, key):
        state, _ = kernel.step(key, state, temperature=0.0)
        return state, state.position["mu"]

    _, mus = jax.lax.scan(body, state, jrnd.split(jrnd.PRNGKey(1), 40_000))
    assert jnp.isclose(jnp.mean(mus), GUK["mu0"], atol=0.15)
    assert jnp.isclose(jnp.std(mus), GUK["sd0"], rtol=0.05)


# ---------------------------------------------------------------------------
# gibbs_sampler
# ---------------------------------------------------------------------------

def test_gibbs_default_step_functions(guk_model, guk_posterior, capsys):
    kernel = gibbs_sampler(guk_model)   # no step functions: Gaussian random walk defaults
    assert "setting defaults" in capsys.readouterr().out
    result = MCMCInference(guk_model, mcmc_kernel=kernel, num_samples=40_000, num_burn=2_000).run(jrnd.PRNGKey(0))
    mean, _ = guk_posterior
    assert jnp.isclose(jnp.mean(result["states"]["mu"]), mean, atol=0.1)


def test_gibbs_elliptical_slice_recovers_exact_gp_posterior():
    """GP regression with fixed hyperparameters: p(f | y) is Gaussian and known in closed form."""
    n, noise = 15, 0.3
    x = jnp.linspace(0, 1, n)
    y = jnp.sin(6 * x) + noise * jrnd.normal(jrnd.PRNGKey(1), (n,))
    hyper = dict(lengthscale=0.2, variance=1.0)

    model = Model("GP regression")
    x_node = model.add_node("x", observations=x)
    f = model.add_node("f", distribution=GaussianProcessFactory(cov_fn=rbf), parents=dict(input=x_node, **hyper))
    model.add_node("y", distribution=dist.Normal, parents=dict(loc=f, scale=noise), observations=y)

    K = rbf(hyper, x, x) + 1e-6 * jnp.eye(n)
    A = K @ jnp.linalg.inv(K + noise**2 * jnp.eye(n))
    post_mean, post_sd = A @ y, jnp.sqrt(jnp.diag(K - A @ K))

    kernel = gibbs_sampler(model,
                           step_fns=dict(f=blackjax.elliptical_slice),
                           step_fn_params=dict(f=dict(implied_mvn_prior=True, name="elliptical_slice")))
    result = MCMCInference(model, mcmc_kernel=kernel, num_samples=4_000, num_burn=500).run(jrnd.PRNGKey(0))
    fs = result["states"]["f"]
    assert fs.shape == (4_000, n)
    assert jnp.max(jnp.abs(jnp.mean(fs, axis=0) - post_mean)) < 0.06
    assert jnp.max(jnp.abs(jnp.std(fs, axis=0) - post_sd)) < 0.05


def test_gibbs_binary_random_walk_matches_enumeration():
    """Binary variable selection: compare inclusion probabilities against brute-force enumeration."""
    c = jnp.array([1.0, -0.5, 2.0])

    def link_fn(z):
        return dict(loc=jnp.sum(z * c), scale=0.5)

    model = Model("binary selection")
    z = model.add_node("z", distribution=dist.Bernoulli(probs=0.3 * jnp.ones(3)))
    model.add_node("y", distribution=dist.Normal, parents=dict(z=z), link_fn=link_fn, observations=jnp.array(2.3))

    all_states = jnp.array(list(itertools.product([0, 1], repeat=3)))
    logp = jax.vmap(lambda s: model.loglikelihood_fn()({"z": s}) + model.logprior_fn()({"z": s}))(all_states)
    exact_inclusion = jnp.exp(logp - logsumexp(logp)) @ all_states

    kernel = gibbs_sampler(model, step_fns=dict(z=bernoulli_random_walk), step_fn_params=dict(z=dict(theta=0.3)))
    result = MCMCInference(model, mcmc_kernel=kernel, num_samples=20_000, num_burn=1_000).run(jrnd.PRNGKey(0))
    samples = result["states"]["z"]
    assert jnp.all((samples == 0) | (samples == 1))
    assert jnp.allclose(jnp.mean(samples, axis=0), exact_inclusion, atol=0.03)


def test_gibbs_reports_per_variable_diagnostics():
    model, _, _ = make_eight_schools()
    step_fns = {k: blackjax.normal_random_walk for k in ["mu", "tau", "theta"]}
    params = dict(mu=dict(sigma=5.0), tau=dict(sigma=1.0), theta=dict(sigma=5.0 * jnp.eye(8)))
    kernel = gibbs_sampler(model, step_fns=step_fns, step_fn_params=params)
    result = MCMCInference(model, mcmc_kernel=kernel, num_samples=500, num_burn=0).run(jrnd.PRNGKey(0))
    assert set(result["info"]) == {"mu", "tau", "theta"}
    for k in result["info"]:
        rate = jnp.mean(result["info"][k].is_accepted)
        assert 0.0 < rate < 1.0


# ---------------------------------------------------------------------------
# MCMCInference bookkeeping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("num_chains,num_thin", [(1, 1), (3, 1), (2, 5)])
def test_mcmc_inference_output_shapes(guk_model, num_chains, num_thin):
    kernel = mcmc_sampler(guk_model, blackjax.normal_random_walk, dict(sigma=0.5 * jnp.eye(1)))
    engine = MCMCInference(guk_model, mcmc_kernel=kernel, num_chains=num_chains, num_samples=1_000, num_burn=100, num_thin=num_thin)
    result = engine.run(jrnd.PRNGKey(0))
    expected = (1_000 // num_thin,) if num_chains == 1 else (num_chains, 1_000 // num_thin)
    assert result["states"]["mu"].shape == expected
    assert result["info"].is_accepted.shape == expected


def test_mcmc_inference_without_diagnostics(guk_model):
    kernel = mcmc_sampler(guk_model, blackjax.normal_random_walk, dict(sigma=0.5 * jnp.eye(1)))
    result = MCMCInference(guk_model, mcmc_kernel=kernel, num_samples=200, num_burn=0, return_diagnostics=False).run(jrnd.PRNGKey(0))
    assert set(result) == {"states"}


def test_independent_chains_differ(guk_model):
    kernel = mcmc_sampler(guk_model, blackjax.normal_random_walk, dict(sigma=0.5 * jnp.eye(1)))
    result = MCMCInference(guk_model, mcmc_kernel=kernel, num_chains=2, num_samples=200, num_burn=0).run(jrnd.PRNGKey(0))
    assert not jnp.allclose(result["states"]["mu"][0], result["states"]["mu"][1])


# ---------------------------------------------------------------------------
# SMCInference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("diagnostics,trace", [(True, False), (False, False), (False, True)])
@pytest.mark.parametrize("kernel_type", ["rmh", "gibbs"])
def test_smc_posterior_and_evidence(guk_model, guk_data, guk_posterior, diagnostics, trace, kernel_type):
    """Adaptive tempered SMC: posterior moments and log-evidence vs closed form, for all output modes,
    with both a joint random-walk kernel and a Gibbs kernel (the Gibbs-in-SMC evidence path)."""
    if kernel_type == "rmh":
        kernel = mcmc_sampler(guk_model, blackjax.normal_random_walk, dict(sigma=0.5 * jnp.eye(1)))
    else:
        kernel = gibbs_sampler(guk_model, step_fns=dict(mu=blackjax.normal_random_walk), step_fn_params=dict(mu=dict(sigma=0.5)))
    engine = SMCInference(guk_model, mcmc_kernel=kernel, num_particles=2_000, num_mutations=20,
                          return_diagnostics=diagnostics, return_trace=trace)
    result = engine.run(jrnd.PRNGKey(0))

    particles = result["final_state"].particles["mu"]
    mean, sd = guk_posterior
    exact_lml = normal_known_sd_lml(guk_data, GUK["sd"], GUK["mu0"], GUK["sd0"])
    assert particles.shape == (2_000,)
    assert jnp.isclose(jnp.mean(particles), mean, atol=0.05)
    assert jnp.isclose(jnp.std(particles), sd, atol=0.05)
    assert jnp.isclose(result["lml"], exact_lml, atol=0.15)
    assert result["final_state"].tempering_param == 1.0
    if trace:
        assert result["trace"].tempering_param.shape[0] == result["n_iter"]
        assert jnp.all(jnp.diff(result["trace"].tempering_param) > 0)          # tempering schedule is increasing
    if diagnostics:
        assert "final_info" in result


def test_smc_multiple_chains(beta_binomial_model):
    kernel = mcmc_sampler(beta_binomial_model, blackjax.normal_random_walk, dict(sigma=0.1 * jnp.eye(1)))
    result = SMCInference(beta_binomial_model, mcmc_kernel=kernel, num_particles=500, num_mutations=10, num_chains=3).run(jrnd.PRNGKey(0))
    assert result["lml"].shape == (3,)
    assert result["final_state"].particles["theta"].shape == (3, 500)


def test_smc_rejects_trace_and_diagnostics_together(guk_model):
    kernel = mcmc_sampler(guk_model, blackjax.normal_random_walk, dict(sigma=0.5 * jnp.eye(1)))
    with pytest.raises(AssertionError):
        SMCInference(guk_model, mcmc_kernel=kernel, num_particles=10, num_mutations=1, return_diagnostics=True, return_trace=True)


# ---------------------------------------------------------------------------
# VIInference
# ---------------------------------------------------------------------------

def test_meanfield_vi_on_conjugate_model(guk_model, guk_posterior):
    engine = VIInference(guk_model, num_steps=3_000, num_gradient_samples=20, optimizer=optax.adam(1e-2))
    result = engine.run(jrnd.PRNGKey(0))
    draws = engine.sample_from_variational(jrnd.PRNGKey(1), result, num_draws=5_000)
    mean, sd = guk_posterior
    assert draws["mu"].shape == (5_000,)
    assert jnp.isclose(jnp.mean(draws["mu"]), mean, atol=0.05)
    assert jnp.isclose(jnp.std(draws["mu"]), sd, rtol=0.15)

#
def test_model_bijectors_are_detected():
    model, _, _ = make_eight_schools()
    bij = get_model_bijectors(model)
    assert isinstance(bij["tau"], nprb.ExpTransform)
    assert isinstance(bij["mu"], nprb.IdentityTransform)
    assert isinstance(bij["theta"], nprb.IdentityTransform)

#
# ---------------------------------------------------------------------------
# LaplaceInference
# ---------------------------------------------------------------------------

def test_laplace_is_exact_for_gaussian_posterior(guk_model, guk_data, guk_posterior):
    result = LaplaceInference(guk_model).run(jrnd.PRNGKey(0))
    mean, sd = guk_posterior
    assert jnp.isclose(result["mode"]["mu"], mean, atol=1e-4)
    assert jnp.isclose(jnp.sqrt(result["covariance"][0, 0]), sd, rtol=1e-3)
    exact_lml = normal_known_sd_lml(guk_data, GUK["sd"], GUK["mu0"], GUK["sd0"])
    assert jnp.isclose(result["lml"], exact_lml, atol=1e-3)


def test_laplace_with_box_bounds(beta_binomial_model):
    """Bounded optimisation keeps the mode inside the support (as in the RJMCMC example notebook)."""
    result = LaplaceInference(beta_binomial_model, bounds=({"theta": 1e-6}, {"theta": 1 - 1e-6})).run(jrnd.PRNGKey(0))
    # Beta(2+7, 3+13) mode = (a' - 1) / (a' + b' - 2)
    assert jnp.isclose(result["mode"]["theta"], 8 / 23, atol=1e-3)


def _lognormal_scale_model(y):
    model = Model("unknown scale")
    s = model.add_node("s", distribution=dist.TransformedDistribution(dist.Normal(0., 1.), nprb.ExpTransform()))
    model.add_node("y", distribution=dist.Normal, parents=dict(loc=1.0, scale=s), observations=y)
    return model

def test_laplace_mode_with_transformed_variable():
    """The Gaussian is fitted in z = log s, so its centre is the mode of p(y, z) = p(y, e^z) e^z
    (Jacobian included), reported back in the constrained space as exp(z_hat)."""
    y = 1.0 + 2.0 * jrnd.normal(jrnd.PRNGKey(0), (200,))
    model = _lognormal_scale_model(y)
    z_grid = jnp.log(jnp.linspace(1.5, 2.5, 20_001))
    logp_z = jax.vmap(lambda z: model.loglikelihood_fn()({"s": jnp.exp(z)}) + model.logprior_fn()({"s": jnp.exp(z)}) + z)(z_grid)
    result = LaplaceInference(model).run(jrnd.PRNGKey(1))
    assert jnp.isclose(result["mode"]["s"], jnp.exp(z_grid[jnp.argmax(logp_z)]), atol=1e-3)
    assert jnp.isclose(result["unconstrained_mode"]["s"], z_grid[jnp.argmax(logp_z)], atol=1e-3)

def test_laplace_evidence_with_transformed_variable():
    y = 1.0 + 2.0 * jrnd.normal(jrnd.PRNGKey(0), (200,))
    model = _lognormal_scale_model(y)
    grid = jnp.linspace(1e-3, 10, 200_001)
    logp = jax.vmap(lambda v: model.loglikelihood_fn()({"s": v}) + model.logprior_fn()({"s": v}))(grid)
    exact_lml = logsumexp(logp) + jnp.log(grid[1] - grid[0])
    result = LaplaceInference(model).run(jrnd.PRNGKey(1))
    assert jnp.isclose(result["lml"], exact_lml, atol=0.1)


def test_laplace_multiple_chains_agree(guk_model, guk_posterior):
    """Different starting points converge to the same Gaussian; outputs get a leading chain axis."""
    engine = LaplaceInference(guk_model, num_chains=3)
    result = engine.run(jrnd.PRNGKey(0))
    mean, sd = guk_posterior
    assert result["lml"].shape == (3,)
    assert jnp.allclose(result["mode"]["mu"], mean, atol=1e-4)
    assert jnp.allclose(jnp.sqrt(result["covariance"][:, 0, 0]), sd, rtol=1e-3)
    draws = engine.sample_from_laplace(jrnd.PRNGKey(1), result, num_draws=2_000)
    assert draws["mu"].shape == (3, 2_000)
    assert jnp.allclose(jnp.mean(draws["mu"], axis=1), mean, atol=0.05)


def test_laplace_accepts_any_optax_optimizer(guk_model, guk_posterior):
    engine = LaplaceInference(guk_model, optimizer=optax.adam(0.1), max_iter=2_000, tol=1e-5)
    result = engine.run(jrnd.PRNGKey(0))
    mean, _ = guk_posterior
    assert jnp.isclose(result["mode"]["mu"], mean, atol=1e-3)
    assert result["optimizer_info"]["num_iterations"] < 2_000

# ---------------------------------------------------------------------------
# Reversible jump MCMC
# ---------------------------------------------------------------------------

def _negbin_link(rate, dispersion):
    return dict(probs=rate / (rate + 1 / dispersion), total_count=1 / dispersion)


def _poisson_negbin_models(y, a=25., b=10.):
    m0 = Model("Poisson")
    lam = m0.add_node("lam", distribution=dist.Gamma(a, b))
    m0.add_node("Y", distribution=dist.Poisson, parents=dict(rate=lam), observations=y)
    m1 = Model("Negative binomial")
    lam = m1.add_node("lam", distribution=dist.Gamma(a, b))
    kappa = m1.add_node("kappa", distribution=dist.Gamma(1., 10.))
    m1.add_node("Y", distribution=dist.NegativeBinomialProbs, parents=dict(rate=lam, dispersion=kappa),
                link_fn=_negbin_link, observations=y)
    return m0, m1


def _grid_lml_2d(model, lam_grid, kappa_grid):
    f = lambda l, k: model.loglikelihood_fn()({"lam": l, "kappa": k}) + model.logprior_fn()({"lam": l, "kappa": k})
    # lax.map over rows keeps peak memory low (a fully vmapped grid needs several GB)
    logp = jax.lax.map(lambda l: jax.vmap(lambda k: f(l, k))(kappa_grid), lam_grid)
    return logsumexp(logp) + jnp.log(lam_grid[1] - lam_grid[0]) + jnp.log(kappa_grid[1] - kappa_grid[0])


@pytest.mark.slow
def test_reversible_jump_posterior_model_probability():
    """Poisson vs negative binomial: RJMCMC model-index frequencies vs exact/quadrature evidence."""
    disp, rate = 0.12, 2.5
    y = dist.NegativeBinomialProbs(total_count=1 / disp, probs=rate / (rate + 1 / disp)).sample(jrnd.PRNGKey(0), (150,))
    m0, m1 = _poisson_negbin_models(y)

    lml0 = gamma_poisson_lml(y, 25., 10.)
    lml1 = _grid_lml_2d(m1, jnp.linspace(1.0, 5.0, 800), jnp.linspace(1e-5, 1.5, 800))
    exact_pmp1 = 1.0 / (1.0 + jnp.exp(lml0 - lml1))
    assert 0.2 < exact_pmp1 < 0.8   # sanity: the test is only informative for a non-trivial PMP

    mu_aux = 0.08
    up = lambda u: mu_aux * jnp.exp(u)
    down = lambda pos: jnp.log(pos["kappa"] / mu_aux)
    kernel = reversible_jump_mcmc(
        models=[m0, m1],
        auxiliary_proposal_dist=dist.Normal(0., 1.),
        jump_functions=[lambda pos, u: dict(lam=pos["lam"], kappa=up(u), u_0=u),
                        lambda pos: dict(lam=pos["lam"], kappa=pos["kappa"], u_0=down(pos))],
        jacobians=[lambda u: jnp.abs(up(u)), lambda kappa: 1.0 / kappa],
        projections=[up, down],
        within_model_kernels=[dict(mcmc_kernel=blackjax.normal_random_walk,
                                   mcmc_parameters=dict(sigma=0.15 * jnp.eye(m.get_model_size()))) for m in (m0, m1)])
    engine = MCMCInference(MetaModel([m0, m1]), mcmc_kernel=kernel, num_chains=4, num_samples=20_000, num_burn=2_000)
    result = engine.run(jrnd.PRNGKey(0))

    rj_pmp1 = jnp.mean(result["states"]["model_index"])
    assert jnp.isclose(rj_pmp1, exact_pmp1, atol=0.03)

    between = result["info"]["within_model_move"] == 0
    assert 0.05 < jnp.mean(result["info"]["is_accepted"][between]) < 0.95
    # within the Poisson model, lam | y is Gamma(a + sum y, b + n)
    in_m0 = result["states"]["model_index"] == 0
    lam_m0 = result["states"]["lam"][in_m0]
    assert jnp.isclose(jnp.mean(lam_m0), (25. + jnp.sum(y)) / (10. + y.shape[0]), atol=0.02)


def test_reversible_jump_requires_two_models():
    m0, m1 = _poisson_negbin_models(jnp.array([1, 2, 3]))
    with pytest.raises(AssertionError):
        reversible_jump_mcmc(models=[m0, m1, m0], auxiliary_proposal_dist=dist.Normal(0., 1.),
                             jump_functions=[None, None], jacobians=[None, None], projections=[None, None],
                             within_model_kernels=[{}, {}])
