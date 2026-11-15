r""" Tuning of MCMC mutation kernels inside SMC.

Currently only useful for Hamoltionian Monte Carlo, but will be extended to Gibbs and random walk 
kernels in the future.

    params = tuner.init(key, model, particles, weights)
    params = tuner.update(key, params, smc_state, smc_info)
    scalars = tuner.summary(params, smc_info)          # optional, for diagnostics

`params` is a dict that unfolded as arguments for blackjax mutation kernel (i.e., stepsizes). 

"""
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from blackjax import hmc, nuts, normal_random_walk


def weighted_particle_moments(particles, weights, dense: bool = False):
    r""" Weighted mean and (co)variance of the flattened particles.

    Returns:
        mean (D,), and either the variances (D,) or the covariance matrix (D, D)
    """
    jitter = 1e-8
    X = jax.vmap(lambda p: ravel_pytree(p)[0])(particles)    # (N, D)
    mean = weights @ X
    Xc = X - mean
    if dense:
        return mean, (weights[:, None] * Xc).T @ Xc + jitter * jnp.eye(X.shape[1])
    return mean, weights @ Xc**2 + jitter

#
def mean_acceptance(info) -> jnp.ndarray:
    acc = info.update_info.acceptance_rate
    finite = jnp.isfinite(acc)
    return jnp.where(finite.any(), jnp.sum(jnp.where(finite, acc, 0.0)) / jnp.maximum(finite.sum(), 1), 0.0)

#

class SMCKernelTuner:
    r""" Base class. The default implementation keeps fixed parameters (i.e. no adaptation). """

    def __init__(self, parameters: dict = None):
        self.parameters = {} if parameters is None else parameters

    #
    def init(self, key, model, particles, weights) -> dict:
        r""" Model is not always used, but some tuners may need it. """

        return self.parameters

    #
    def update(self, key, params: dict, smc_state, smc_info) -> dict:
        return params

    #
    def summary(self, params: dict, smc_info) -> dict:
        return {}
    
    #
#
class StaticParameters(SMCKernelTuner):
    r""" Non-adaptive SMC: the same mutation parameters at every temperature. """

#
class GibbsFamilyTuner(SMCKernelTuner):
    r""" Tunes the step functions of a bamojax Gibbs sampler, one variable at a time.
 
    Every tuned variable gets its own tuner, which only sees the particles and the acceptance rate of
    that variable. Its parameters are passed to the Gibbs sampler as '<variable>/<parameter>'.
    Variables without a tuner (e.g. those updated with elliptical slice sampling) keep the fixed
    parameters given in `step_fn_params`.
 
    Args:
        gibbs_kernel: the SamplingAlgorithm returned by `gibbs_sampler`
        tuners: dict mapping variable names to tuners, e.g. {'lengthscale': RWTuner(dense=False)}.
            Defaults to the default tuner of each variable's step function, where one exists
            (currently: RWTuner for `normal_random_walk`).
    """
 
    def __init__(self, gibbs_kernel, tuners: dict = None):
        super().__init__()
        step_fns = {getattr(node, 'name', node): fn for node, fn in gibbs_kernel.step.step_fns.items()}
        if tuners is None:
            tuners = {name: default_tuner_for(fn) for name, fn in step_fns.items()}
            tuners = {name: tuner for name, tuner in tuners.items() if tuner is not None}
        for name in tuners:
            if name not in step_fns:
                raise ValueError(f'{name} is not a variable of this Gibbs sampler; available: {list(step_fns)}')
        if not tuners:
            raise NotImplementedError('None of the Gibbs step functions has a default SMC tuner; pass `tuners` explicitly.')
        self.tuners = tuners
 
    #
    def init(self, key, model, particles, weights) -> dict:
        params = {}
        for name, tuner in self.tuners.items():
            node_params = tuner.init(key, model, {name: particles[name]}, weights)
            for k, v in node_params.items():
                params[f'{name}/{k}'] = v
        return params
 
    #
    def node_view(self, name, params, smc_info):
        r""" The parameters and SMC info that belong to one variable.
 
        Parameters are stored as '<variable>/<parameter>'; the prefix is removed here. Blackjax infos
        are NamedTuples, so _replace gives a copy with only this variable's update info.
        """
        prefix = name + '/'
        node_params = {k[len(prefix):]: v for k, v in params.items() if k.startswith(prefix)}
        node_info = smc_info._replace(update_info=smc_info.update_info[name])
        return node_params, node_info
 
    #
    def update(self, key, params, smc_state, smc_info) -> dict:
        new_params = {}
        for name, tuner in self.tuners.items():
            node_params, node_info = self.node_view(name, params, smc_info)
            node_state = smc_state._replace(particles={name: smc_state.particles[name]})
            node_params = tuner.update(key, node_params, node_state, node_info)
            for k, v in node_params.items():
                new_params[f'{name}/{k}'] = v
        return new_params
 
    #
    def summary(self, params, smc_info) -> dict:
        out = {}
        for name, tuner in self.tuners.items():
            node_params, node_info = self.node_view(name, params, smc_info)
            for k, v in tuner.summary(node_params, node_info).items():
                out[f'{name}/{k}'] = v
        return out
 
    #
#


def default_tuner_for(kernel):
    for k, tuner in DEFAULT_TUNERS:
        if kernel is k:
            return tuner()
    return None


class RMHFamilyTuner(SMCKernelTuner):

    def __init__(self, target_acceptance: float = None, gain: float = 2.0, dense: bool = True):
        super().__init__()
        self.target_acceptance = target_acceptance
        self.gain = gain
        self.dense = dense

    #
    def _params(self, particles, weights, log_scale) -> dict:
        D = sum(x[0].size for x in jax.tree_util.tree_leaves(particles))  # static, also under jit
        _, cov = weighted_particle_moments(particles, weights, self.dense)
        root = jnp.linalg.cholesky(cov) if self.dense else jnp.sqrt(cov)
        sigma = jnp.exp(log_scale) * 2.38 / jnp.sqrt(D) * root
        return dict(sigma=sigma[None, ...], _log_scale=jnp.reshape(log_scale, (1, )))

    #
    def init(self, key, model, particles, weights) -> dict:
        return self._params(particles, weights, jnp.zeros(()))

    #
    def update(self, key, params, smc_state, smc_info) -> dict:
        target = self.target_acceptance
        if target is None:
            D = sum(x[0].size for x in jax.tree_util.tree_leaves(smc_state.particles))  # static, also under jit
            target = 0.44 if D == 1 else 0.234
        log_scale = params['_log_scale'][0] + self.gain * (mean_acceptance(smc_info) - target)
        return self._params(smc_state.particles, smc_state.weights, log_scale)

    #
    def summary(self, params, smc_info) -> dict:
        return dict(scale=jnp.exp(params['_log_scale'][0]), mean_acceptance=mean_acceptance(smc_info))

    #
#

class HMCFamilyTuner(SMCKernelTuner):
    r""" Particle-based tuning for gradient-based kernels (HMC, NUTS).

    - inverse mass matrix: the importance-weighted particle (co)variance;
    - step size: log eps <- log eps + gain * (mean acceptance - target acceptance), using the
      acceptance of the previous mutation step;
    - (HMC only) number of leapfrog steps: L = ceil(trajectory_length / eps), clipped to
      [1, max_integration_steps]. The mass matrix whitens the target, so trajectory_length is
      in units of (approximate) posterior standard deviations.

    Args:
        target_acceptance: target mean acceptance rate
        gain: how strongly the step size reacts to the acceptance error
        initial_step_size: defaults to D^(-1/4), sensible for a whitened target
        dense: full particle covariance (True) or diagonal (False) as inverse mass matrix
        trajectory_length: integration time eps * L (HMC only)
        max_integration_steps: upper bound on L (HMC only)
    """

    def __init__(self, 
                 target_acceptance: float, 
                 tune_num_integration_steps: bool,
                 gain: float = 2.0, 
                 initial_step_size: float = None, 
                 dense: bool = False,
                 trajectory_length: float = 1.5, 
                 max_integration_steps: int = 64):
        super().__init__()
        self.target_acceptance = target_acceptance
        self.tune_num_integration_steps = tune_num_integration_steps
        self.gain = gain
        self.initial_step_size = initial_step_size
        self.dense = dense
        self.trajectory_length = trajectory_length
        self.max_integration_steps = max_integration_steps

    #
    def _params(self, particles, weights, step_size) -> dict:
        _, cov = weighted_particle_moments(particles, weights, self.dense)
        params = dict(step_size=jnp.reshape(step_size, (1, )), inverse_mass_matrix=cov[None, ...])
        if self.tune_num_integration_steps:
            L = jnp.clip(jnp.ceil(self.trajectory_length / step_size), 1, self.max_integration_steps)
            params['num_integration_steps'] = jnp.reshape(L.astype(jnp.int32), (1, ))
        return params

    #
    def init(self, key, model, particles, weights) -> dict:
        D = sum(x[0].size for x in jax.tree_util.tree_leaves(particles))  # static, also under jit
        eps0 = D ** (-0.25) if self.initial_step_size is None else self.initial_step_size
        return self._params(particles, weights, jnp.asarray(eps0, dtype=jnp.float64))

    #
    def update(self, key, params, smc_state, smc_info) -> dict:
        log_eps = jnp.log(params['step_size'][0]) + self.gain * (mean_acceptance(smc_info) - self.target_acceptance)
        # The weighted particles after reweighting approximate the *next* mutation target.
        return self._params(smc_state.particles, smc_state.weights, jnp.exp(log_eps))

    #
    def summary(self, params, smc_info) -> dict:
        out = dict(step_size=params['step_size'][0], mean_acceptance=mean_acceptance(smc_info))
        if 'num_integration_steps' in params:
            out['num_integration_steps'] = params['num_integration_steps'][0]
        return out

    #
#

def RMHTuner(target_acceptance: float = None, **kwargs) -> SMCKernelTuner:
    return RMHFamilyTuner(target_acceptance, **kwargs)

#
def GibbsTuner(gibbs_kernel, tuners: dict = None) -> SMCKernelTuner:
    return GibbsFamilyTuner(gibbs_kernel, tuners)

#
def HMCTuner(target_acceptance: float = 0.65, **kwargs) -> SMCKernelTuner:
    return HMCFamilyTuner(target_acceptance, tune_num_integration_steps=True, **kwargs)

#
def NUTSTuner(target_acceptance: float = 0.8, **kwargs) -> SMCKernelTuner:
    return HMCFamilyTuner(target_acceptance, tune_num_integration_steps=False, **kwargs)

#
# Default tuner per blackjax kernel, used for SMCInference(..., adaptive=True).
DEFAULT_TUNERS = [(normal_random_walk, RMHTuner), (hmc, HMCTuner), (nuts, NUTSTuner)]


def get_tuner(mcmc_kernel, adaptive, mcmc_parameters: dict = None) -> SMCKernelTuner:
    r""" Resolve the `adaptive` argument of SMCInference into a tuner object.

    Args:
        mcmc_kernel: the bamojax SamplingAlgorithm used for the mutations
        adaptive: False (fixed parameters), True (default tuner for this kernel), or an SMCKernelTuner
        mcmc_parameters: fixed parameters, used when adaptive is False
    """
    if isinstance(adaptive, SMCKernelTuner):
        return adaptive
    if not adaptive:
        return StaticParameters(mcmc_parameters)
    if hasattr(mcmc_kernel.step, 'step_fns'):    
        return GibbsTuner(mcmc_kernel)
    base_kernel = getattr(mcmc_kernel.step, 'base_kernel', None)
    for kernel, tuner in DEFAULT_TUNERS:
        if base_kernel is kernel:
            return tuner()
    name = getattr(mcmc_kernel.step, '__name__', repr(base_kernel))
    raise NotImplementedError(f'No default SMC tuner for mutation kernel {name}; '
                              f'pass an SMCKernelTuner instance as `adaptive` instead.')

#                              