import warnings

from typing import NamedTuple, Callable
from jaxtyping import Array
import jax
import jax.random as jrnd
import jax.numpy as jnp
from blackjax.base import SamplingAlgorithm
from blackjax.types import ArrayTree, PRNGKey
from blackjax.smc.resampling import systematic
from blackjax import generate_top_level_api_from, normal_random_walk

from bamojax.modified_blackjax import modified_adaptive_tempered
from bamojax.modified_blackjax import modified_tempered
from bamojax.modified_blackjax import modified_elliptical_slice_nd

tempered_smc = generate_top_level_api_from(modified_tempered)
adaptive_tempered_smc = generate_top_level_api_from(modified_adaptive_tempered)
elliptical_slice_nd = generate_top_level_api_from(modified_elliptical_slice_nd)

from .base import Model

class GibbsState(NamedTuple):

    position: ArrayTree

#
class MCMCState(NamedTuple):

    position: ArrayTree

#
def grad_estimator(logprior_fn: Callable, loglikelihood_fn: Callable, data_size: int, batch_size) -> Callable:
    """Build a simple estimator for the gradient of the log-density."""

    def logdensity_estimator_fn(position, minibatch):
        return logprior_fn(position) + data_size / batch_size * loglikelihood_fn(position, minibatch)
    
    #
    return jax.grad(logdensity_estimator_fn)

#
def gibbs_sampler(model: Model, 
                  step_fns: dict = None, 
                  step_fn_params: dict = None) -> SamplingAlgorithm:
    r""" Constructs a Gibbs sampler as a Blackjax SamplingAlgorithm.

    Args:
        model: The bamojax definition of a Bayesian model.
        step_fns: (optional) a set of step functions to use for updating each variable in turn.
        step_fn_params: (optional) parameters of the step functions

    Returns:
        A SamplingAlgorithm object. This can be used to call the respective .init and .step functions in the inference routines.
    
    """

    def set_step_fns_defaults(step_fns: dict = None, step_fn_params: dict = None):
        r""" Set the step function of each node if not specified. Defaults to a Gaussian random walk with a stepsize of 0.01.
        
        """
        if step_fns is None:
            step_fns = {}
            print('No step functions found; setting defaults.')
        if step_fn_params is None:
            step_fn_params = {}
        
        sorted_free_variables = [node for node in model.get_node_order() if node.is_stochastic() and not node.is_observed() and not node in step_fns]
        for node in sorted_free_variables:
            step_fns[node] = normal_random_walk
            num_elem = 1 if node.shape == () else jnp.prod(jnp.asarray(node.shape))
            step_fn_params[node] = dict(sigma=0.01*jnp.eye(num_elem))
        
        return step_fns, step_fn_params
    
    #
    step_fns, step_fn_params = set_step_fns_defaults(step_fns=step_fns, step_fn_params=step_fn_params)

    def get_nonstandard_gibbs_step(node, position, loglikelihood_fn, step_fns, step_fn_params):
        r""" The Blackjax SamplingAlgorithm is not parametrized in entirely the same way for different algorithms. To not clutter the gibbs_fn, exception cases are handled here.

        """
        mean = node.get_distribution(position).get_mean()
        cov = node.get_distribution(position).get_cov()
        if step_fn_params[node]['name'] == 'elliptical_slice':                    
            step_kernel = step_fns[node](loglikelihood_fn, mean=mean, cov=cov)
            step_substate = step_kernel.init({node.name: position[node]})  
        elif step_fn_params[node]['name'] == 'elliptical_slice_nd':                     
            nd = step_fn_params[node]['nd']
            step_kernel = step_fns[node](loglikelihood_fn, mean=mean, cov=cov, nd=nd)
            step_substate = step_kernel.init({node.name: position[node]})  
        elif step_fn_params[node]['name'] == 'mgrad_gaussian':
            # see issue https://github.com/blackjax-devs/blackjax/issues/237,mgrad does not seem robust
            loglikelihood_fn_mgrad = lambda state: loglikelihood_fn(state[node])
            step_kernel = step_fns[node](logdensity_fn=loglikelihood_fn_mgrad, mean=mean, covariance=cov, **step_fn_params[node]['params'])
            step_substate = step_kernel.init({node.name: position[node.name]}) 
        else:
            raise NotImplementedError
        return step_kernel, step_substate

    #
    def gibbs_fn(model: Model, 
                 key, 
                 state: dict, 
                 *args, 
                 **kwargs) -> dict:
        r""" Updates each latent variable given the current assignment of all other latent variables, according to the assigned step functions.
        
        The signature of the Gibbs function is (key, state, temperature) -> (state, info)

        The Gibbs densities are defined as follows. Let Pa(x) give the parents of the set of variables x, and let Ch(x) give the set of children. Then the density is given by:

        p(x | Pa(x)) \propto p(Ch(x) | Pa(Ch(x))) p(x | Pa(x))

        Args:
            key: PRNGKey
            state: The current assignment of all latent variables.

        Returns:
            state: The updated assignment of all latent variables.
            info: Additional information regarding the updates for every latent variable, such as acceptance rates.
        
        """

        # [TODO]: make the Gibbs function more memory efficient & faster

        # In case we apply likelihood tempering
        temperature = kwargs.get('temperature', 1.0)
        position = state.position.copy()
        
        info = {}
        sorted_free_variables = [node for node in model.get_node_order() if node.is_stochastic() and not node.is_observed()]

        for node in sorted_free_variables:
            # Get conditional densities
            conditionals = []
            children = [c for c in model.get_children(node)]        
            for child in children:
                # Co-parents are all parents of the child, except node
                co_parents = {parent for parent in model.get_parents(child) if parent != node}

                # Values for co-parents are either taken from the position (if latent), or from their respective observations (if observed)
                co_parent_arguments = {k: (position[k] if k in position else k.observations) for k in co_parents}

                def loglikelihood_fn_(substate):
                    dynamic_state = {**co_parent_arguments, node.name: substate[node]}
                    if child.is_leaf():
                        child_value = child.observations
                    else:
                        child_value = position[child]
                    return child.get_distribution(dynamic_state).log_prob(value=child_value)
                
                #            
                co_parents.add(node)
                conditionals.append(loglikelihood_fn_)

            # [TODO] Can we avoid the list and use a vmap instead? Particularly relevant for hierarchical models.
            loglikelihood_fn = lambda val: jnp.sum(jnp.asarray([temperature*ll_fn(val).sum() for ll_fn in conditionals]))

            if 'implied_mvn_prior' in step_fn_params[node]:
                # Some Blackjax step functions are tailored to multivariate Gaussian priors.
                step_kernel, step_substate = get_nonstandard_gibbs_step(node, position, loglikelihood_fn, step_fns, step_fn_params)
            else:
                logprior_fn = lambda substate: node.get_distribution(position).log_prob(value=substate[node]).sum() 
                logdensity_fn = lambda substate_: loglikelihood_fn(substate_) + logprior_fn(substate_)
                step_kernel = step_fns[node](logdensity_fn, **step_fn_params[node])   
                step_substate = step_kernel.init({node.name: position[node]})   
                
            key, subkey = jrnd.split(key)     
            # [TODO]: add functionality to sample specific variables for different numbers of steps
            # def step_body(key, state): step_kernel.step...
            # step_substate, step_info = jax.lax.scan(step_body, keys, state)
            step_substate, step_info = step_kernel.step(subkey, step_substate)
            info[node.name] = step_info
            
            position = {**position, **step_substate.position}
                
            del step_kernel
            del step_substate
            del conditionals
            del children

        del state
        return GibbsState(position=position), info

    #
    def init_fn(position, rng_key=None):
        del rng_key
        return GibbsState(position=position)
    
    #
    def step_fn(key: PRNGKey, state, *args, **kwargs):
        state, info = gibbs_fn(model, key, state, *args, **kwargs)
        return state, info
    
    #
    step_fn.__name__ = 'gibbs_step_fn'
    return SamplingAlgorithm(init_fn, step_fn)

#
def mcmc_sampler(model: Model, mcmc_kernel, mcmc_parameters: dict = None):
    """ Constructs an MCMC sampler from a given Blackjax algorithm.

    This lightweight wrapper ensures the (optional) tempering parameter 'temperature',
    as part of the keyword-arguments of step_fn(..., **kwargs), is passed correctly.

    Args:
        model: A bamojax model definition.
        mcmc_kernel: A Blackjax MCMC algorithm.
        mcmc_parameters: Optional Blackjax MCMC parameters, such as step sizes.
    Returns:
        A Blackjax SamplingAlgorithm object with methods `init_fn` and `step_fn`.
    
    """

    def mcmc_fn(model: Model, 
                key, 
                state: dict, 
                *args, 
                **kwargs) -> dict:
        
        def apply_mcmc_kernel(key_, logdensity_fn, pos):
            kernel_instance = mcmc_kernel(logdensity_fn=logdensity_fn, **mcmc_parameters)
            state_ = kernel_instance.init(pos)
            state_, info_ = kernel_instance.step(key_, state_)
            return state_.position, info_
        
        #
        temperature = kwargs.get('temperature', 1.0)
        position = state.position.copy()

        loglikelihood_fn_ = model.loglikelihood_fn()
        logprior_fn_ = model.logprior_fn()
        tempered_logdensity_fn = lambda state: jnp.squeeze(temperature * loglikelihood_fn_(state) + logprior_fn_(state))
        new_position, mcmc_info = apply_mcmc_kernel(key, tempered_logdensity_fn, position)
        return MCMCState(position=new_position), mcmc_info
    
    #
    def init_fn(position, rng_key=None):
        del rng_key
        return MCMCState(position=position)
    
    #
    def step_fn(key: PRNGKey, state, *args, **kwargs):
        state, info = mcmc_fn(model, key, state, *args, **kwargs)
        return state, info
    
    #
    return SamplingAlgorithm(init_fn, step_fn)

#
def run_sgmcmc_chain(rng_key, step_fn: Callable, initial_state, batch_nodes: list, data_size: int, batch_size: int, stepsize: float, num_samples: int):
    """ The Stochastic-gradient MCMC inference loop.

    Note that the Blackjax implementation of SGMCMC algorithms do not return an 'info' object with diagnostics.

    Args:
        rng_key:
            The jax.random.PRNGKey
        step_fn: 
            A step function that takes a state and returns a new state
        initial_state: 
            The initial state of the sampler
        batch_nodes:
            A list of nodes for which to create a minibatch, e.g. inputs and outputs
        data_size:
            The total data set size
        batch_size:
            The size of a minibatch
        stepsize:
            The SGMCMC stepsize
        num_samples: int
            The number of samples to obtain
    Returns:


    """
    # Note: this minibatch function returns shapes we expect, (N_sub, p) for p-dimensional input, and (N_sub, ) for 1-dimensional input

    def get_minibatch(data, indices):
        if jnp.ndim(data) == 1:
            data = data[:, jnp.newaxis]    
        slice_size = (1,) + data.shape[1:]  # For (N, p), this will be (1, p)    
        return jnp.squeeze(jax.vmap(lambda i: jax.lax.dynamic_slice(data, (i,) + (0,) * (data.ndim - 1), slice_size))(indices))

    #
    @jax.jit
    def one_step(state, key): 
        key_sgld, key_batch = jrnd.split(key)
        idx = jrnd.choice(key_batch, data_size, shape=(batch_size, ), replace=False)
        minibatch = {node.name: get_minibatch(node.observations, idx) for node in batch_nodes}
        state = step_fn(key_sgld, state, minibatch, stepsize)    
        return state, state

    #
    keys = jrnd.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states

#
def sgmcmc_inference_loop(key: Array, 
                          model: Model, 
                          sgkernel: SamplingAlgorithm,                           
                          batch_nodes: list,
                          data_size: int,
                          batch_size: int,
                          stepsize: float,
                          num_samples: int, 
                          sgparams: dict = None,
                          num_burn: int = 0, 
                          num_chains: int = 1, 
                          num_thin: int = 1):
    
    if sgparams is None:
        sgparams = {}

    assert batch_size < data_size, f'Batch size must be smaller than data set size, but found batch size: {batch_size} and data size: {data_size}.'
    
    @jax.jit
    def chain_fun(key: Array):
        key, k_inference, k_init = jrnd.split(key, 3)
        initial_state = sgmcmc_kernel.init(model.sample_prior(k_init))
        states = run_sgmcmc_chain(k_inference, 
                                  step_fn=sgmcmc_kernel.step, 
                                  initial_state=initial_state, 
                                  batch_nodes=batch_nodes, 
                                  data_size=data_size, 
                                  batch_size=batch_size, 
                                  stepsize=stepsize,
                                  num_samples=num_burn+num_samples)
        states = jax.tree_util.tree_map(lambda x: jnp.squeeze(x[num_burn::num_thin, ...]), states)
        return states
    
    #    
    grad_fn = grad_estimator(model.logprior_fn(), model.batched_loglikelihood_fn(), data_size, batch_size)
    sgmcmc_kernel = sgkernel(grad_fn, **sgparams)
    keys = jrnd.split(key, num_chains)
    states = jax.vmap(chain_fun, in_axes=0)(keys)
    states = jax.tree_util.tree_map(lambda x: jnp.squeeze(x), states)
    return states

#


def run_mcmc_chain(rng_key, 
                   step_fn: Callable, 
                   initial_state, 
                   num_samples: int):
    """The MCMC inference loop.

    The inference loop takes an initial state, a step function, and the desired
    number of samples. It returns a list of states.
    
    Args:
        rng_key: 
            The jax.random.PRNGKey
        step_fn: Callable
            A step function that takes a state and returns a new state
        initial_state: 
            The initial state of the sampler
        num_samples: int
            The number of samples to obtain
    Returns: 
        GibbsState [List, "num_samples"]

    """
    @jax.jit
    def one_step(state, rng_key):
        state, info = step_fn(rng_key, state)
        return state, (state, info)

    keys = jrnd.split(rng_key, num_samples)
    _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)

    return states, infos

#
def run_mcmc_chain_no_diagnostics(rng_key, 
                   step_fn: Callable, 
                   initial_state, 
                   num_samples: int,
                   num_thin: int = 1):
    """The MCMC inference loop.

    The inference loop takes an initial state, a step function, and the desired
    number of samples. It returns a list of states.
    
    Args:
        rng_key: 
            The jax.random.PRNGKey
        step_fn: Callable
            A step function that takes a state and returns a new state
        initial_state: 
            The initial state of the sampler
        num_samples: int
            The number of samples to obtain
    Returns: 
        GibbsState [List, "num_samples"]

    """

    @jax.jit
    def one_step(state, rng_key):
        state, _ = step_fn(rng_key, state)
        return state, state
    
    #    
    @jax.jit
    def dense_step(state, rng_key):
        keys_ = jrnd.split(rng_key, num_thin)
        _, states = jax.lax.scan(one_step, state, keys_)
        final_state = jax.tree_util.tree_map(lambda x: x[-1, ...], states)
        return final_state, final_state
    
    #    
    if num_thin == 1:        
        keys = jrnd.split(rng_key, num_samples)
        _, states = jax.lax.scan(one_step, initial_state, keys)
    else:
        keys = jrnd.split(rng_key, int(num_samples / num_thin))
        _, states = jax.lax.scan(dense_step, initial_state, keys)
    return states

#
def mcmc_inference_loop(key: Array, 
                   model: Model, 
                   kernel: SamplingAlgorithm, 
                   num_samples: int, 
                   num_burn: int = 0, 
                   num_chains: int = 1, 
                   num_thin: int = 1,
                   store_diagnostics = True):

    @jax.jit
    def chain_fun(key: Array):
        key, k_inference, k_init = jrnd.split(key, 3)
        initial_state = kernel.init(model.sample_prior(k_init))

        if store_diagnostics:
            states, info = run_mcmc_chain(k_inference, step_fn=kernel.step, initial_state=initial_state, num_samples=num_burn+num_samples)
            states = jax.tree_util.tree_map(lambda x: jnp.squeeze(x[num_burn::num_thin, ...]), states)
            return states, info
        else:
            states = run_mcmc_chain_no_diagnostics(k_inference, step_fn=kernel.step, initial_state=initial_state, num_samples=num_burn+num_samples, num_thin=num_thin)
            num_burn_remove = int(num_burn / num_thin)
            states = jax.tree_util.tree_map(lambda x: jnp.squeeze(x[num_burn_remove:, ...]), states)
            return states   
    
    #    
    keys = jrnd.split(key, num_chains)

    if store_diagnostics:
        states, info = jax.vmap(chain_fun, in_axes=0)(keys)
        states = jax.tree_util.tree_map(lambda x: jnp.squeeze(x), states)
        return states, info
    else:
        states = jax.vmap(chain_fun, in_axes=0)(keys)  
        states = jax.tree_util.tree_map(lambda x: jnp.squeeze(x), states)
        return states    

#
def run_smc(rng_key: PRNGKey, 
            smc_kernel: Callable, 
            initial_state):
    """The sequential Monte Carlo loop, which also stores the MCMC info objects. 
    These contain diagnostics such as acceptance rates, proposals, etc.

    Args:
        key: 
            The jax.random.PRNGKey
        smc_kernel: 
            The SMC kernel object (e.g. SMC, tempered SMC or 
                    adaptive-tempered SMC)
        initial_state: 
            The initial state for each particle
    Returns:
        n_iter: int
            The number of tempering steps
        final_state: 
            The final state of each of the particles
        lml:
            The model log marginal likelihood
        info: 
            The diagnostic information
        
    """

    def cond(carry):
        _, state, *_k = carry
        return state.lmbda < 1

    #
    # Call SMC once to determine the info pytree structure. Note that the jax.lax.while_loop therefore starts at 1.
    rng_key, key_init = jrnd.split(rng_key)
    initial_state, sample_info = smc_kernel(key_init, initial_state)
    initial_info = jax.tree_util.tree_map(lambda x: jax.numpy.zeros_like(x), sample_info)
    initial_log_likelihood = sample_info.log_likelihood_increment
    
    @jax.jit
    def one_step(carry):                
        i, state, k, curr_log_likelihood, _ = carry
        k, subk = jrnd.split(k)
        state, info = smc_kernel(subk, state)        
        return i + 1, state, k, curr_log_likelihood + info.log_likelihood_increment, info

    #
    n_iter, final_state, _, lml, final_info = jax.lax.while_loop(cond, one_step, 
                                                    (1, initial_state, rng_key, initial_log_likelihood, initial_info))

    return n_iter, final_state, lml, final_info

#
def run_smc_no_diagnostics(rng_key: PRNGKey, 
            smc_kernel: Callable, 
            initial_state):
    """The sequential Monte Carlo loop.

    Args:
        key: 
            The jax.random.PRNGKey
        smc_kernel: 
            The SMC kernel object (e.g. SMC, tempered SMC or 
                    adaptive-tempered SMC)
        initial_state: 
            The initial state for each particle
    Returns:
        n_iter: int
            The number of tempering steps
        final_state: 
            The final state of each of the particles
        lml: The model log marginal likelihood
        
    """

    def cond(carry):
        _, state, *_k = carry
        return state.lmbda < 1

    #
    
    @jax.jit
    def one_step(carry):                
        i, state, k, curr_log_likelihood = carry
        k, subk = jrnd.split(k)
        state, info = smc_kernel(subk, state)        
        return i + 1, state, k, curr_log_likelihood + info.log_likelihood_increment

    #
    n_iter, final_state, _, lml = jax.lax.while_loop(cond, one_step, 
                                                    (0, initial_state, rng_key, 0))

    return n_iter, final_state, lml

#
def smc_inference_loop(key, 
                       model, 
                       kernel: SamplingAlgorithm, 
                       num_particles: int, 
                       num_mcmc_steps: int, 
                       num_chains: int = 1, 
                       mcmc_parameters: dict = None, 
                       resampling_fn: Callable = systematic, 
                       target_ess: float = 0.5,
                       store_diagnostics = True):

    # if num_chains > 1:
    #     warnings.warn(f'Number of chains set to {num_chains}, note that these are processed sequentially because AT-SMC uses a fori-loop construction.')

    if mcmc_parameters is None:
        mcmc_parameters = {}

    @jax.jit
    def run_chain(key_):       
        smc = adaptive_tempered_smc(
                    logprior_fn=model.logprior_fn(),
                    loglikelihood_fn=model.loglikelihood_fn(),
                    mcmc_step_fn=kernel.step,
                    mcmc_init_fn=kernel.init,
                    mcmc_parameters=mcmc_parameters,
                    resampling_fn=resampling_fn,
                    target_ess=target_ess,
                    num_mcmc_steps=num_mcmc_steps
                )

        key_smc, key_init = jrnd.split(key_)
        keys = jrnd.split(key_init, num_particles)
        initial_particles = smc.init(jax.vmap(model.sample_prior)(keys))
        if store_diagnostics:
            n_iter, final_state, lml, final_info = run_smc(key_smc, smc.step, initial_particles)        
            return final_state, lml, n_iter, final_info
        else:
            n_iter, final_state, lml = run_smc_no_diagnostics(key_smc, smc.step, initial_particles)    
            return final_state, lml, n_iter            
    
    #        
    keys = jrnd.split(key, num_chains)
    if store_diagnostics:
        final_state, lml, n_iter, final_info = jax.vmap(run_chain)(keys)
        final_info = jax.tree_util.tree_map(lambda x: jnp.squeeze(x), final_info)
    else:
        final_state, lml, n_iter = jax.vmap(run_chain)(keys)        

    n_iter = jax.tree_util.tree_map(lambda x: jnp.squeeze(x), n_iter)
    final_state = jax.tree_util.tree_map(lambda x: jnp.squeeze(x), final_state)
    lml = jax.tree_util.tree_map(lambda x: jnp.squeeze(x), lml)
    
    if store_diagnostics:
        return final_state, lml, n_iter, final_info
    else:
        return final_state, lml, n_iter


#