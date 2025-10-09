from abc import ABC, abstractmethod

from blackjax.base import SamplingAlgorithm
from blackjax.types import PRNGKey
from blackjax.smc.resampling import systematic
from blackjax import window_adaptation, nuts, generate_top_level_api_from, meanfield_vi

import numpyro
import numpyro.distributions as dist
import numpyro.distributions.transforms as nprb

from bamojax.base import Model
from bamojax.samplers import mcmc_sampler
from .modified_blackjax import modified_adaptive_tempered, modified_tempered

tempered_smc = generate_top_level_api_from(modified_tempered)
adaptive_tempered_smc = generate_top_level_api_from(modified_adaptive_tempered)



from typing import Tuple, Callable
import optax
import jaxopt

import jax
import jax.random as jrnd
import jax.numpy as jnp
from jax.tree_util import tree_map, tree_flatten, tree_unflatten
from jax.flatten_util import ravel_pytree

# from tensorflow_probability.substrates import jax as tfp
# tfd = tfp.distributions
# tfb = tfp.bijectors

def run_window_adaptation(model, key: PRNGKey, num_warmup_steps):
    r""" Find optimized parameters for HMC-based inference.

    A wrapper for blackjax HMC window adaptation.

    Args:
        Model: A bamojax model (DAG)
        key: Random seed
        num_warmum_steps: int, number of warmup steps to take

    Returns:
        - A dictionary of the state after performing the warmup steps
        - A set of HMC kernel parameters that result in optimal acceptance rates
    
    """
    logdensity_fn = lambda state: model.loglikelihood_fn()(state) + model.logprior_fn()(state)
    warmup = window_adaptation(nuts, logdensity_fn)
    key_init, key_warmup = jrnd.split(key)
    initial_state = model.sample_prior(key_init)
    (warm_state, warm_parameters), _ = warmup.run(key_warmup, initial_state, num_steps=num_warmup_steps)  
    return warm_state, warm_parameters

#
def get_model_bijectors(model) -> dict:
    r""" Meanfield VI imposes a Gaussian variational distribution. 
    
    In order to use the correct parameter constraints this function determines all relevant bijectors.

    It's a bijector detector! :-)

    Returns:
        A dictionary with bijectors for the variables that use it, and an identity bijector otherwise.
    
    """
    latent_nodes = model.get_latent_nodes()
    bijectors = {}
    for node_name, node in latent_nodes.items():
        if isinstance(node.distribution, dist.TransformedDistribution):
            if len(node.distribution.transforms) == 1:
                transform = node.distribution.transforms[0]
            else:
                transform = nprb.ComposeTransform(node.distribution.transforms)
        else:
            transform = nprb.IdentityTransform()  # Use TensorFlow Probability's Identity bijector
        bijectors[node_name] = transform
    return bijectors

# 

class InferenceEngine(ABC):
    r""" The abstract class for inference engines

    Attributes:
        model: the Bayesian model for which to run the approximate inference
        num_chains: the number of parallel, independent chains
    
    """

    def __init__(self, model: Model, num_chains: int = 1):
        self.model = model
        self.num_chains = num_chains

    #
    @abstractmethod
    def run_single_chain(self, key: PRNGKey):
        pass

    #
    def run(self, key):
        r""" Runs the inference algorithm, optionally vmapped over mutiple chains
        
        """
        if self.num_chains > 1:
            keys = jrnd.split(key, self.num_chains)
            return jax.vmap(self.run_single_chain)(keys)
        return self.run_single_chain(key)
    
    #

#
class MCMCInference(InferenceEngine):
    r""" The MCMC approximate inference class
    
    """

    def __init__(self, 
                 model: Model, 
                 num_chains: int = 1, 
                 mcmc_kernel: SamplingAlgorithm = None, 
                 num_samples: int = 10_000, 
                 num_burn: int = 10_000, 
                 num_warmup: int = 0,
                 num_thin: int = 1, 
                 return_diagnostics: bool = True):
        super().__init__(model, num_chains)
        self.mcmc_kernel = mcmc_kernel
        self.num_samples = num_samples
        self.num_burn = num_burn
        self.num_warmup = num_warmup
        self.num_thin = num_thin
        self.return_diagnostics = return_diagnostics

    #    
    def dense_step(self, key, state):
        """ Take a MCMC step.

        Can be a composite step to reduce autocorrelation and memory consumption (thinning). Rather than
        post-hoc removing samples, a dense step can take N steps, then return only the final state, so 
        that we effective thin by a factor N.

        Args:
            key: Random seed
            state: The current state
        Returns:
            A new state
            (optional) Diagnostic information, depending on the `return_diagnostics` attribute
        
        """
        @jax.jit
        def one_step_fn(state, key):
            state, info = self.mcmc_kernel.step(key, state)
            return state, info

        #
        @jax.jit
        def one_step_state_only_fn(state, key):
            state, _ = self.mcmc_kernel.step(key, state)
            return state, None

        #
        conditional_step = one_step_fn if self.return_diagnostics else one_step_state_only_fn
        
        if self.num_thin > 1:
            keys = jrnd.split(key, self.num_thin)
            state, infos = jax.lax.scan(conditional_step, state, keys)
            info = tree_map(lambda x: x[-1, ...], infos)
            return state, info if self.return_diagnostics else (state, None)
        
        return conditional_step(state, key)
    
    #
    def run_single_chain(self, key: PRNGKey) -> dict:
        r""" Run one MCMC chain

        Depending on different preferences, this 
            - optimizes the MCMC kernel hyperparameters
            - runs `num_burn` burn-in samples, that are discarded
            - performs the actual sampling for `num_samples` / `num_thin` dense steps

        Args:
            key: Random seed
        
        Returns:
            A dictionary containing the resulting collection of states, and optional diagnostic information
        
        """
        def mcmc_body_fn(state, key):
            state, info = self.dense_step(key, state)
            return state, (state, info) 
        
        #
        key, key_init = jrnd.split(key)        

        if self.num_warmup > 0:
            print('Adapting NUTS HMC parameters...', end=" ")
            warm_state, adapted_parameters = run_window_adaptation(self.model, key_init, self.num_warmup) 
            print('done.')
            adapted_kernel = mcmc_sampler(model=self.model, mcmc_kernel=nuts, mcmc_parameters=adapted_parameters)
            self.mcmc_kernel = adapted_kernel
            initial_state = self.mcmc_kernel.init(warm_state.position)
        else:
            initial_state = self.mcmc_kernel.init(self.model.sample_prior(key_init))        

        if self.num_burn > 0:
            num_dense_burn_steps = int(self.num_burn / self.num_thin)
            key, key_burn = jrnd.split(key)
            keys = jrnd.split(key_burn, num_dense_burn_steps)
            initial_state, _ = jax.lax.scan(mcmc_body_fn, initial_state, keys)

        num_dense_steps = int(self.num_samples / self.num_thin)
        key, key_inference = jrnd.split(key)
        keys = jrnd.split(key_inference, num_dense_steps)
        _, (states, info) = jax.lax.scan(mcmc_body_fn, initial_state, keys)
        
        return dict(states=states.position, info=info) if self.return_diagnostics else dict(states=states.position)

    #

#
class SGMCMCInference(InferenceEngine):
    r""" The Stochastic Gradient MCMC approximate inference class
    
    """
    
    def __init__(self, 
                 model: Model,                   
                 sgmcmc_kernel: SamplingAlgorithm, 
                 data_size: int, 
                 batch_size: int, 
                 stepsize: float, 
                 batch_nodes: list, 
                 num_chains: int = 1,
                 num_samples: int = 10_000, 
                 num_burn: int = 10_000, 
                 num_thin: int = 1,  
                 sgmcmc_params: dict = None):
        assert batch_size < data_size, f'Batch size must be smaller than data set size, but found batch size: {batch_size} and data size: {data_size}.'
        super().__init__(model, num_chains)

        if sgmcmc_params is None:
            sgmcmc_params = {}
       
        self.num_samples = num_samples
        self.num_burn = num_burn
        self.num_thin = num_thin
        
        self.sgmcmc_params = sgmcmc_params
        self.data_size = data_size
        self.batch_size = batch_size
        self.batch_nodes = batch_nodes
        self.stepsize = stepsize
        self.grad_fn = self.grad_estimator()
        self.sgmcmc_kernel = sgmcmc_kernel(self.grad_fn, **sgmcmc_params)

    #
    def grad_estimator(self) -> Callable:
        """ The stochastic gradient estimator
        
        Build a simple estimator for the gradient of the log-density, assuming data is mini-batched.

        Returns:
            The gradient of the batched log-density        
        
        """

        logprior_fn = self.model.logprior_fn()
        loglikelihood_fn = self.model.batched_loglikelihood_fn()

        def logdensity_estimator_fn(position, minibatch):
            return logprior_fn(position) + self.data_size / self.batch_size * loglikelihood_fn(position, minibatch)
        
        #
        return jax.grad(logdensity_estimator_fn)

    #
    def get_minibatch(self, data, indices):
        r""" Slice a minibatch of data

        Args:
            data: the data array
            indices: a set of indices into `data` 

        Returns:
            Dynamic slicing of data[indices, ...]
        
        """
        if jnp.ndim(data) == 1:
            data = data[:, jnp.newaxis]    
        slice_size = (1,) + data.shape[1:]  # For (N, p), this will be (1, p)    
        return jnp.squeeze(jax.vmap(lambda i: jax.lax.dynamic_slice(data, (i,) + (0,) * (data.ndim - 1), slice_size))(indices))

    #
    def one_step(self, state, key):
        r""" Take one step of stochastic gradient MCMC

        Args:
            state: Current state
            key: Random seed
        
        Returns:
            The updated state
        
        """
        key_sgmcmc, key_batch = jrnd.split(key)
        idx = jrnd.choice(key_batch, self.data_size, shape=(self.batch_size, ), replace=False)
        minibatch = {node.name: self.get_minibatch(node.observations, idx) for node in self.batch_nodes}
        state = self.sgmcmc_kernel.step(key_sgmcmc, state, minibatch, self.stepsize)   
        return state, state
    
    #
    def run_single_chain(self, key: PRNGKey):
        r""" Run Stochastic Gradient MCMC

        Args: 
            key: PRNGKey
        Returns:
            A dictionary of samples for each variable in the Bayesian model.
        
        """
 
        key, key_init = jrnd.split(key)
        initial_state = self.sgmcmc_kernel.init(self.model.sample_prior(key_init))
        step_fn = jax.jit(self.one_step)

        if self.num_burn > 0:
            key, key_burn = jrnd.split(key)
            keys = jrnd.split(key_burn, self.num_burn)
            initial_state, _ = jax.lax.scan(step_fn, initial_state, keys)

        keys = jrnd.split(key, self.num_samples)
        _, states = jax.lax.scan(step_fn, initial_state, keys)

        return dict(states=states)

    #

#
class SMCInference(InferenceEngine):
    r""" The SMC approximate inference class
    
    """

    def __init__(self, 
                 model: Model,                   
                 mcmc_kernel: SamplingAlgorithm, 
                 num_particles: int, 
                 num_mutations: int, 
                 num_chains: int = 1,
                 mcmc_parameters: dict = None,
                 resampling_fn = systematic, 
                 target_ess: float = 0.5, 
                 return_trace: bool = False, 
                 return_diagnostics: bool = True,
                 num_warmup: int = 0,
                 max_iter: int = 40):
        super().__init__(model, num_chains)
        self.mcmc_kernel = mcmc_kernel
        if mcmc_parameters is None:
            mcmc_parameters = {}
        self.mcmc_parameters = mcmc_parameters
        self.num_particles = num_particles
        self.num_mutations = num_mutations
        self.resampling_fn = resampling_fn
        self.target_ess = target_ess
        self.return_trace = return_trace
        self.return_diagnostics = return_diagnostics
        self.num_warmup = num_warmup
        self.max_iter = max_iter
        assert not (return_diagnostics and return_trace), 'Returning both the trace and diagnostics is not supported.'

    #
    def create_smc_kernel(self) -> SamplingAlgorithm:
        r""" Creates an SMC kernel

        Currently a lightweight wrapper around the blackjax adaptive-tempered SMC object.

        Returns:
            A blackjax SMC `SamplingAlgorithm`
        
        """
        return adaptive_tempered_smc(logprior_fn=self.model.logprior_fn(),
                                     loglikelihood_fn=self.model.loglikelihood_fn(),
                                     mcmc_step_fn=self.mcmc_kernel.step,
                                     mcmc_init_fn=self.mcmc_kernel.init,
                                     mcmc_parameters=self.mcmc_parameters,
                                     resampling_fn=self.resampling_fn,
                                     target_ess=self.target_ess,
                                     num_mcmc_steps=self.num_mutations)
    
    #    
    def tempering_condition(self):
        r""" Checks whether the SMC procedure terminates.

        Returns:
            A boolean deciding on whether the SMC tempering procedure is finished
        
        """
        def cond(carry):
            i, state, *_ = carry
            if self.return_trace:
                return jnp.logical_and(state.lmbda < 1, i < self.max_iter)
            return state.lmbda < 1

        #
        return cond

    #
    def smc_cycle(self, smc_kernel) -> Callable:
        r""" One iteration of the adaptive-tempered SMC algorithm.

        Args:
            smc_kernel: A Blackjax SamplingAlgorithm containing the SMC logic.
        Returns:
            A Callable step function that performs one iteration.
        
        """
        @jax.jit
        def one_step(carry): 
            if self.return_trace:
                i, state, k, curr_log_likelihood, state_hist = carry
            else:               
                if self.return_diagnostics:
                    i, state, k, curr_log_likelihood, _ = carry 
                else:
                    i, state, k, curr_log_likelihood = carry 
            k, subk = jrnd.split(k)
            state, info = smc_kernel.step(subk, state)    
            base_return_tuple = (i + 1, state, k, curr_log_likelihood + info.log_likelihood_increment)
            if self.return_trace:
                state_hist = tree_map(lambda arr, val: arr.at[i].set(val), state_hist, state)
                return base_return_tuple + (state_hist, )            
            
            return base_return_tuple + (info, ) if self.return_diagnostics else base_return_tuple

        #
        return one_step

    #
    
    def run_single_chain(self, key: PRNGKey) -> dict:
        r""" Run one chain of Sequential Monte Carlo.

        This returns SMC particles in different ways, depending on user provided settings.

        Args:
            key: PRNGKey
        Returns:
            A dictionary with the final SMC state, the number of iterations, the log marginal likelihood
            (Optional) diagnostics and the trace of particles over each tempering step
        
        """

        if self.num_warmup > 0:
            key, key_init = jrnd.split(key)
            print('Adapting NUTS HMC parameters...', end=" ")
            _, adapted_parameters = run_window_adaptation(self.model, key_init, self.num_warmup)
            print('done.')
            adapted_kernel = mcmc_sampler(model=self.model, mcmc_kernel=nuts, mcmc_parameters=adapted_parameters)
            self.mcmc_kernel = adapted_kernel
        
        smc_kernel = self.create_smc_kernel()
        smc_cycle = self.smc_cycle(smc_kernel)
        cond = self.tempering_condition()

        key, key_init = jrnd.split(key)        
        keys = jrnd.split(key_init, self.num_particles)
        initial_particles = smc_kernel.init(jax.vmap(self.model.sample_prior)(keys))

        if self.return_trace or self.return_diagnostics:
            # Call SMC once to determine PyTree structures
            key_smc, key_init = jrnd.split(key)
            initial_particles, sample_info = smc_kernel.step(key_init, initial_particles)
            initial_info = tree_map(lambda x: jax.numpy.zeros_like(x), sample_info)
            initial_log_likelihood = sample_info.log_likelihood_increment

        if self.return_trace:
             # Preallocate arrays for state and info history
            trace = tree_map(lambda x: jnp.zeros((self.max_iter,) + x.shape, dtype=x.dtype), initial_particles)
            trace = jax.tree_util.tree_map(lambda arr, val: arr.at[0].set(val), trace, initial_particles)
            n_iter, final_state, _, lml, trace = jax.lax.while_loop(cond, smc_cycle, (1, initial_particles, key_smc, initial_log_likelihood, trace))
            trace = tree_map(lambda x: x[:n_iter], trace)
            return dict(
                n_iter=n_iter, 
                final_state=final_state, 
                lml=lml, 
                trace=trace
            )
        if self.return_diagnostics:            
            n_iter, final_state, _, lml, final_info = jax.lax.while_loop(cond, smc_cycle, (1, initial_particles, key_smc, initial_log_likelihood, initial_info))
            return dict(
                n_iter=n_iter, 
                final_state=final_state, 
                lml=lml, 
                final_info=final_info
            )            
        else:
            n_iter, final_state, _, lml, = jax.lax.while_loop(cond, smc_cycle, (0, initial_particles, key, 0))
            return dict(
                n_iter=n_iter, 
                final_state=final_state, 
                lml=lml
            )  
        
    #

#
class VIInference(InferenceEngine):
    r""" The VI approximate inference class
    
    """

    def __init__(self, 
                 model: Model,                  
                 num_steps: int,
                 num_chains: int = 1,
                 num_gradient_samples: int = 10,
                 optimizer: Callable = optax.sgd,
                 optimizer_chain_args: list = None):
        super().__init__(model, num_chains)
        self.num_steps = num_steps
        self.num_gradient_samples = num_gradient_samples
        if optimizer_chain_args is None:
            self.optimizer = optimizer
        else:
            if not isinstance(optimizer_chain_args, list):
                optimizer_chain_args = [optimizer_chain_args]
            self.optimizer = optax.chain(*optimizer_chain_args, optimizer)
        self.bijectors = get_model_bijectors(self.model)

        self.is_leaf_fn = lambda x: hasattr(x, '__call__') and hasattr(x, '_inverse')
        self.forward_bijectors = lambda x: jax.tree.map(lambda b, v: b(v), self.bijectors, x, is_leaf=self.is_leaf_fn)
        self.backward_bijectors = lambda x: jax.tree.map(lambda b, v: b._inverse(v), self.bijectors, x, is_leaf=self.is_leaf_fn)

        def logdensity_fn(z):
            z = self.forward_bijectors(z)
            return model.loglikelihood_fn()(z) + model.logprior_fn()(z)

        #
        self.logdensity_fn = logdensity_fn

    #   
    def sample_from_variational(self, key: PRNGKey, vi_result: dict, num_draws: int) -> dict:
        r""" Draw samples x ~ q(x | mu, rho)

        Args:
            key: PRNGKey
            vi_result: a dictionary containing the variational approximation
            num_draws: the number of samples to draw from the variational distribution.

        Returns:
            A dictionary with samples from the variational distribution, for each variable in the model.        
        
        """

        if self.num_chains > 1:
            final_state = tree_map(lambda x: x[:, -1, ...], vi_result['states'])
        else:
            final_state = tree_map(lambda x: x[-1, ...], vi_result['states'])

        def sample_fn(key, loc, scale):
            return loc + scale*jrnd.normal(key, shape=(num_draws, ) + loc.shape)

        #
        vi_mu = final_state.mu
        vi_rho = tree_map(lambda x: jnp.exp(x), final_state.rho)

        flat_pytree, treedef = tree_flatten(vi_mu)
        num_leaves = len(flat_pytree)
        keys = jrnd.split(key, num_leaves)
        keys_pytree = tree_unflatten(treedef, keys)

        vi_samples = tree_map(sample_fn, keys_pytree, vi_mu, vi_rho)
        if self.num_chains > 1:
            vi_samples = tree_map(lambda x: jnp.swapaxes(x, 0, 1), vi_samples)

        # apply bijectors
        vi_samples = self.forward_bijectors(vi_samples)
        return vi_samples

    #
    def run_single_chain(self, key: PRNGKey) -> dict:
        r""" Run variational inference.

        Args: 
            key: PRNGKey
        Returns:
            A dictionary with the variational parameters across iterations, and logging results such as the ELBO.
        
        """
        mfvi = meanfield_vi(self.logdensity_fn, self.optimizer, self.num_gradient_samples)
        initial_position = self.model.sample_prior(key=jrnd.PRNGKey(0))  # these are overriden by Blackjax

        initial_position_unconstrained = {k: self.bijectors[k](v) for k, v in initial_position.items()}
        initial_state = mfvi.init(initial_position_unconstrained)

        @jax.jit
        def one_step(state, rng_key):
            state, info = mfvi.step(rng_key, state)
            return state, (state, info)

        #
        keys = jrnd.split(key, self.num_steps)
        _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)
        return dict(
            states=states, 
            info=infos
        )

    #

#
class LaplaceInference(InferenceEngine):
    r""" The Laplace approximate inference class
    
    """

    def __init__(self, 
                 model: Model,   
                 num_chains: int = 1,
                 optimizer: Callable = jaxopt.ScipyMinimize,
                 optimizer_args: dict = None,
                 bounds: dict = None):
        super().__init__(model, num_chains)
        if optimizer_args is None:
            optimizer_args = {}
        self.bounds = bounds

        self.bijectors = get_model_bijectors(model)
        self.is_leaf_fn = lambda x: hasattr(x, '__call__') and hasattr(x, '_inverse')
        self.forward_bijectors = lambda x: jax.tree.map(lambda b, v: b(v), self.bijectors, x, is_leaf=self.is_leaf_fn)
        self.backward_bijectors = lambda x: jax.tree.map(lambda b, v: b._inverse(v), self.bijectors, x, is_leaf=self.is_leaf_fn)

        @jax.jit
        def logdensity_fn(z):
            z = self.forward_bijectors(z)
            return -1.0 * (model.loglikelihood_fn()(z) + model.logprior_fn()(z))

        #
        self.obj_fun = logdensity_fn

        if self.bounds is not None:
            optimizer = jaxopt.ScipyBoundedMinimize(fun=self.obj_fun, **optimizer_args)
        else:
            optimizer = optimizer(fun=self.obj_fun, **optimizer_args)
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args if optimizer_args is not None else {}        
        self.D = model.get_model_size()

    #
    def run_single_chain(self, key):
        def get_unconstrained_init(model, key):
            constrained = tree_map(jnp.asarray, model.sample_prior(key))
            print('constrained sample:', constrained)
            unconstrained = self.backward_bijectors(constrained)
            print('unconstrained sample:', unconstrained)
            return unconstrained
        
        #

        init_params = get_unconstrained_init(self.model, key)
        if self.bounds is not None:
            sol = self.optimizer.run(init_params, bounds=self.bounds)   
        else:
            sol = self.optimizer.run(init_params)   

        # We fit a Gaussian(\hat{\theta}, \Sigma) with 
        # \hat{\theta} = \argmax_\theta p(\theta \mid y)
        # \Sigma^-1 is the Hessian of -\log p(\theta \mid y) at \theta=\hat{\theta}

        mode = self.forward_bijectors(sol.params)
        H = jax.hessian(self.obj_fun)(mode)
        theta_hat_flat, unravel_fn = ravel_pytree(mode)

        def flat_obj_fn(flat_params):
            params = unravel_fn(flat_params)
            return self.obj_fun(params)

        H = jax.hessian(flat_obj_fn)(theta_hat_flat)

        Sigma = jnp.linalg.inv(H)

        if theta_hat_flat.shape == () or theta_hat_flat.shape == (1,):  # Univariate case
            laplace_dist = dist.Normal(loc=theta_hat_flat, scale=jnp.sqrt(Sigma))
        else:
            laplace_dist = dist.MultivariateNormal(loc=theta_hat_flat, covariance_matrix=Sigma)

        _, logdet = jnp.linalg.slogdet(Sigma)

        log_posterior = -1.0 * self.obj_fun(mode)
        lml = log_posterior + 1/2*logdet + self.D/2 * jnp.log(2*jnp.pi)

        return dict(
            distribution=laplace_dist,
            mode=mode,
            flat_mode=theta_hat_flat,
            covariance=Sigma,
            hessian=H,
            unravel_fn=unravel_fn,
            lml=lml
        )
    
    #
#
