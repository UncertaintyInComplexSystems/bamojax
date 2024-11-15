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

def gibbs_sampler(model: Model, 
                  step_fns: dict = None, 
                  step_fn_params: dict = None) -> SamplingAlgorithm:
    r""" Constructs a Gibbs sampler as a Blackjax SamplingAlgorithm.

    Args:
        model: The Bayesian model.
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

        # In case we apply likelihood tempering
        temperature = kwargs.get('temperature', 1.0)
        position = state.position.copy()
        
        info = {}
        sorted_free_variables = [node for node in model.get_node_order() if node.is_stochastic() and not node.is_observed()]

        for node in sorted_free_variables:
            # get conditional density
            conditionals = []
            children = [c for c in model.get_children(node)]        
            for child in children:
                co_parents = set()
                for parent in model.get_parents(child):
                    if not parent == node or parent in co_parents:
                        co_parents.add(parent)  

                co_parent_arguments = {}
                for co_parent in co_parents:
                    if co_parent.name in position:
                        co_parent_arguments[co_parent.name] = position[co_parent.name]
                    else:
                        co_parent_arguments[co_parent.name] = co_parent.observations

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
            
            for k, v in step_substate.position.items():
                position[k] = v
                
            del step_substate
            del conditionals

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
    return SamplingAlgorithm(init_fn, step_fn)

#
def run_chain(rng_key, 
              step_fn: Callable, 
              initial_state, 
              num_samples: int):
    """The MCMC inference loop.

    The inference loop takes an initial state, a step function, and the desired
    number of samples. It returns a list of states.
    
    Args:
        rng_key: 
            The jax.random.PRNGKey
        kernel: Callable
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

    keys = jax.random.split(rng_key, num_samples)
    _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)

    return states, infos

#
def inference_loop(key: Array, 
                   model: Model, 
                   kernel: SamplingAlgorithm, 
                   num_samples: int, 
                   num_burn: int = 0, 
                   num_chains: int = 1, 
                   num_thin: int = 1):

    def chain_fun(key: Array):
        key, k_inference, k_init = jrnd.split(key, 3)
        initial_state = kernel.init(model.sample_prior(k_init))
        states, info = run_chain(k_inference, step_fn=kernel.step, initial_state=initial_state, num_samples=num_burn+num_samples)
        states = jax.tree_util.tree_map(lambda x: jnp.squeeze(x[num_burn::num_thin, ...]), states)
        return states, info
    
    #    
    keys = jrnd.split(key, num_chains)
    states, info = jax.vmap(chain_fun, in_axes=0)(keys)
    states = jax.tree_util.tree_map(lambda x: jnp.squeeze(x), states)
    return states, info

#
def run_smc(rng_key: PRNGKey, 
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
        info: SMCinfo
            the SMC info object which contains the log marginal likelihood of 
              the model (for model comparison)
        
    """

    def cond(carry):
        _, state, *_k = carry
        return state.lmbda < 1

    #
    # call once to determine the info pytree structure - start the loop at iteration 1 instead of 0!
    rng_key, key_init = jrnd.split(rng_key)
    initial_state, sample_info = smc_kernel(key_init, initial_state)
    initial_info = jax.tree_util.tree_map(lambda x: jax.numpy.zeros_like(x), sample_info)
    initial_log_likelihood = sample_info.log_likelihood_increment
    
    @jax.jit
    def one_step(carry):                
        i, state, k, curr_log_likelihood, _ = carry
        k, subk = jax.random.split(k)
        state, info = smc_kernel(subk, state)        
        return i + 1, state, k, curr_log_likelihood + info.log_likelihood_increment, info

    #
    n_iter, final_state, _, lml, final_info = jax.lax.while_loop(cond, one_step, 
                                                      (1, initial_state, rng_key, initial_log_likelihood, initial_info))
    del initial_info
    del initial_state

    return n_iter, final_state, lml, final_info

#
def smc_inference_loop(key, 
                       model, 
                       kernel: SamplingAlgorithm, 
                       num_particles: int, 
                       num_mcmc_steps: int, 
                       num_chains: int = 1, 
                       mcmc_parameters: dict = None, 
                       resampling_fn: Callable = systematic, 
                       target_ess: float = 0.5):

    if mcmc_parameters is None:
        mcmc_parameters = {}

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
        n_iter, final_state, lml, final_info = run_smc(key_smc, smc.step, initial_particles)        
        return final_state, lml, n_iter, final_info
    
    #        
    keys = jrnd.split(key, num_chains)
    final_state, lml, n_iter, final_info = jax.vmap(run_chain)(keys)
    n_iter = jax.tree_util.tree_map(lambda x: jnp.squeeze(x), n_iter)
    final_state = jax.tree_util.tree_map(lambda x: jnp.squeeze(x), final_state)
    lml = jax.tree_util.tree_map(lambda x: jnp.squeeze(x), lml)
    final_info = jax.tree_util.tree_map(lambda x: jnp.squeeze(x), final_info)

    return final_state, lml, n_iter, final_info

#