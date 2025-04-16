from typing import NamedTuple
import jax.random as jrnd
import jax.numpy as jnp
from blackjax.base import SamplingAlgorithm
from blackjax.types import ArrayTree, PRNGKey
from blackjax import normal_random_walk, generate_top_level_api_from
from .base import Model
from .modified_blackjax import modified_elliptical_slice_nd

elliptical_slice_nd = generate_top_level_api_from(modified_elliptical_slice_nd)


class GibbsState(NamedTuple):

    position: ArrayTree

#
class MCMCState(NamedTuple):

    position: ArrayTree

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

    def get_nonstandard_gibbs_step(node, 
                                   position, 
                                   loglikelihood_fn, 
                                   step_fns, 
                                   step_fn_params):
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
            # see issue https://github.com/blackjax-devs/blackjax/issues/237,mgrad does not seem robust yet
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
                    child_value = child.observations if child.is_leaf() else position[child]
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
def mcmc_sampler(model: Model, 
                 mcmc_kernel, 
                 mcmc_parameters: dict = None):
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