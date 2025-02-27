from abc import ABC, abstractmethod

from blackjax.base import SamplingAlgorithm
from blackjax.types import PRNGKey
from blackjax.smc.resampling import systematic
from blackjax import window_adaptation, nuts, generate_top_level_api_from, meanfield_vi

from bamojax.base import Model
from bamojax.samplers import mcmc_sampler
from .modified_blackjax import modified_adaptive_tempered, modified_tempered

tempered_smc = generate_top_level_api_from(modified_tempered)
adaptive_tempered_smc = generate_top_level_api_from(modified_adaptive_tempered)

from typing import Tuple, Callable
import optax

import jax
import jax.random as jrnd
import jax.numpy as jnp
from jax.tree_util import tree_map, tree_flatten, tree_unflatten

def run_window_adaptation(model, key: PRNGKey, num_warmup_steps):
    logdensity_fn = lambda state: model.loglikelihood_fn()(state) + model.logprior_fn()(state)
    warmup = window_adaptation(nuts, logdensity_fn)
    key_init, key_warmup = jrnd.split(key)
    initial_state = model.sample_prior(key_init)
    (warm_state, warm_parameters), _ = warmup.run(key_warmup, initial_state, num_steps=num_warmup_steps)  
    return warm_state, warm_parameters

#

class InferenceEngine(ABC):

    def __init__(self, model: Model, num_chains: int = 1):
        self.model = model
        self.num_chains = num_chains

    #
    @abstractmethod
    def run_single_chain(self, key: PRNGKey):
        pass

    #
    def run(self, key):
        if self.num_chains > 1:
            keys = jrnd.split(key, self.num_chains)
            return jax.vmap(self.run_single_chain)(keys)
        return self.run_single_chain(key)
    
    #

#
class MCMCInference(InferenceEngine):

    def __init__(self, 
                 model: Model, 
                 num_chains: int = 1, 
                 mcmc_kernel: SamplingAlgorithm = None, 
                 num_samples: int = 10_000, 
                 num_burn: int = 10_000, 
                 num_warmup: int = 0,
                 num_thin: int = 1, 
                 return_diagostics: bool = True):
        super().__init__(model, num_chains)
        self.mcmc_kernel = mcmc_kernel
        self.num_samples = num_samples
        self.num_burn = num_burn
        self.num_warmup = num_warmup
        self.num_thin = num_thin
        self.return_diagnostics = return_diagostics

    #    
    def dense_step(self, key, state):
        """ Take an MCMC step.

        Can be a composite step to reduce autocorrelation and memory consumption (thinning).
        
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
        def mcmc_body_fn(state, key):
            state, info = self.dense_step(key, state)
            return state, (state, info) 
        
        #
        key, key_init = jrnd.split(key)        

        if self.num_warmup > 0:
            print('Adapting NUTS HMC parameters')
            warm_state, adapted_parameters = run_window_adaptation(self.model, key_init, self.num_warmup) 
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
        """Build a simple estimator for the gradient of the log-density."""

        logprior_fn = self.model.logprior_fn()
        loglikelihood_fn = self.model.batched_loglikelihood_fn()

        def logdensity_estimator_fn(position, minibatch):
            return logprior_fn(position) + self.data_size / self.batch_size * loglikelihood_fn(position, minibatch)
        
        #
        return jax.grad(logdensity_estimator_fn)

    #
    def get_minibatch(self, data, indices):
        r""" Slice a minibatch of data
        
        """
        if jnp.ndim(data) == 1:
            data = data[:, jnp.newaxis]    
        slice_size = (1,) + data.shape[1:]  # For (N, p), this will be (1, p)    
        return jnp.squeeze(jax.vmap(lambda i: jax.lax.dynamic_slice(data, (i,) + (0,) * (data.ndim - 1), slice_size))(indices))

    #
    def one_step(self, state, key):
        r""" Take one step of stochastic gradient MCMC
        
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

        Args:
            key: PRNGKey
        Returns:
            A dictionary with the final SMC state, the number of iterations, the log marginal likelihood, and optional diagnostics.
        
        """

        if self.num_warmup > 0:
            key, key_init = jrnd.split(key)
            print('Adapting NUTS HMC parameters')
            _, adapted_parameters = run_window_adaptation(self.model, key_init, self.num_warmup)
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
            return dict(n_iter=n_iter, final_state=final_state, lml=lml, trace=trace)
        if self.return_diagnostics:            
            n_iter, final_state, _, lml, final_info = jax.lax.while_loop(cond, smc_cycle, (1, initial_particles, key_smc, initial_log_likelihood, initial_info))
            return dict(n_iter=n_iter, final_state=final_state, lml=lml, final_info=final_info)            
        else:
            n_iter, final_state, _, lml, = jax.lax.while_loop(cond, smc_cycle, (0, initial_particles, key, 0))
            return dict(n_iter=n_iter, final_state=final_state, lml=lml)  
        
    #

#
class VIInference(InferenceEngine):

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
        self.bijectors = self.get_model_bijectors()
        def logdensity_fn(z):
            z = jax.tree.map(lambda f, v: f(v), self.bijectors, z)
            return model.loglikelihood_fn()(z) + model.logprior_fn()(z)

        #
        self.logdensity_fn = logdensity_fn

    #
    def get_model_bijectors(self) -> dict:
        r""" Meanfield VI imposes a Gaussian variational distribution. In order to use the correct parameter constraints this function determines all relevant bijectors.

        Returns:
            A dictionary with bijectors for the variables that use it, and an identity bijector otherwise.
        
        """
        model_pytree = self.model.sample_prior(jrnd.PRNGKey(0))

        model_pytree
        bijectors = {}
        for k in model_pytree.keys():
            if hasattr(self.model.nodes[k].distribution, '_bijector'):                        
                transform = lambda x: self.model.nodes[k].distribution._bijector.forward(x=x)
            else:
                transform = lambda x: x
            bijectors[k] = transform
        return bijectors

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
        initial_state = mfvi.init(initial_position)

        @jax.jit
        def one_step(state, rng_key):
            state, info = mfvi.step(rng_key, state)
            return state, (state, info)

        #
        keys = jrnd.split(key, self.num_steps)
        _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)
        return dict(states=states, info=infos)

    #

#

