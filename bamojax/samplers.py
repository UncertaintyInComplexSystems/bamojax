from typing import NamedTuple, Callable
import jax
import jax.random as jrnd
import jax.numpy as jnp
from blackjax.base import SamplingAlgorithm
from blackjax.types import ArrayTree, PRNGKey
from blackjax import normal_random_walk, generate_top_level_api_from
from distrax import Distribution

from .base import Model
from .modified_blackjax import modified_elliptical_slice_nd

elliptical_slice_nd = generate_top_level_api_from(modified_elliptical_slice_nd)


class GibbsState(NamedTuple):

    position: ArrayTree

#
class MCMCState(NamedTuple):

    position: ArrayTree

#
class RJState(NamedTuple):

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
def reversible_jump_mcmc(models: list[Model], 
                         auxiliary_proposal_dist: Distribution,
                         jump_functions: list[Callable],
                         jacobians: list[Callable],
                         projections: list[Callable],
                         within_model_kernels: list[dict],
                         model_move_prob: float = 0.5) -> SamplingAlgorithm:
    """Constructs a reversible jump MCMC algorithm for the given models.
    
    Implementation follows the reversible jump MCMC algorithm described by Hastie & Green (2012).

    Args:
        models: List of models to sample from.
        auxiliary_proposal_dist: Distribution to sample the auxiliary variable from.
        jump_functions: List of functions to transform the position when jumping between models.
        jacobians: List of Jacobian determinant functions for the jump functions
        within_model_kernel: List of within-model sampling algorithms, one for each model.

    Returns:
        SamplingAlgorithm: A sampling algorithm that implements the reversible jump MCMC.

    References:
        Hastie, T., & Green, P. J. (2012). Model choice using reversible jump Markov chain Monte Carlo
    
    """

    assert len(models) == 2, 'Reversible jump MCMC currently only supports two models.'
    assert len(jump_functions) == 2, 'Reversible jump MCMC requires two jump functions for the two models.'
    assert len(jacobians) == 2, 'Reversible jump MCMC requires two Jacobian determinant functions for the two models.'
    assert len(within_model_kernels) == 2, 'Reversible jump MCMC requires within-model sampling algorithms for each models.'

    def make_within_model_kernel(model_index) -> SamplingAlgorithm:
        """Creates a within-model kernel for the specified model index.

        Args:
            model_index: Index of the model for which to create the within-model kernel.

        Returns:
            A SamplingAlgorithm that performs within-model sampling for the specified model.

        """
        return mcmc_sampler(models[model_index], 
                            mcmc_kernel=within_model_kernels[model_index]['mcmc_kernel'], 
                            mcmc_parameters=within_model_kernels[model_index]['mcmc_parameters'])
    
    #    
    def make_within_model_fn(model_index) -> tuple[RJState, dict]:
        """Creates a function that performs a within-model move for the specified model index.

        Args:
            model_index: Index of the model for which to create the within-model move function.

        Returns:
            A function that takes a key, position, and optional arguments, and returns a new RJState and info dictionary.
            The info dictionary imputes nan values for log probabilities and Jacobian determinants, as they are not used in within-model moves.

        """
        def fn( key, position, *args):
            temperature = args[0] if len(args) > 0 else 1.0
            kernel = mcmc_samplers[model_index]
            initial_position = {k: position[k] for k in models[model_index].get_latent_nodes()}            
            within_move_initial_state = kernel.init(position=initial_position)
            new_state, info = kernel.step(key, within_move_initial_state, temperature=temperature)
            new_position = {**position, **new_state.position, 'model_index': model_index}  
            return RJState(position=new_position), {'within_model_move': 1, 
                                                    'is_accepted': info.is_accepted,
                                                    'log_accept_ratio': jnp.log(info.acceptance_rate),
                                                    'step_info': {'log_p_current': jnp.nan,
                                                                  'log_p_proposed': jnp.nan,
                                                                  'logq': jnp.nan,
                                                                  'jacobian_det': jnp.nan}}
        return fn

    #
    def make_model_logprob(model):
        """Creates a function that computes the log probability of the model given a position."""

        latent_keys = model.get_latent_nodes()
        
        def fn(position):
            """Position might contain auxiliary variables and variables from other models, so we extract only the correct latent variables."""
            model_variables = {k: position[k] for k in latent_keys}
            return model.loglikelihood_fn()(model_variables) + model.logprior_fn()(model_variables)
        
        return fn
    
    #
    def reversible_jump_fn(key: PRNGKey, state: RJState, *args, **kwargs):
        """Performs a reversible jump MCMC step.
        
        Args:
            key: Random key for sampling.
            state: Current state of the reversible jump MCMC, containing the model index and position.
            *args: Additional arguments to pass to the within-model sampling functions. NOT USED
            **kwargs: Additional keyword arguments to pass to the within-model sampling functions. NOT USED

        Returns:
            A tuple containing the next state and an info dictionary with details about the move.
                
        """
        
        position = state.position
        model_index = position['model_index']
        key, subkey = jrnd.split(key)
        move_type = jrnd.bernoulli(subkey, p=model_move_prob)
        jacobian_det_up, jacobian_det_down = jacobians

        def do_within_model(_):
            """Perform a standard bamojax within-model move."""

            temperature = kwargs.get('temperature', 1.0)
            return jax.lax.switch(model_index, within_model_fns, key, position, temperature)
                
        #
        def do_between_model(_):
            """Perform a reversible jump between models.
            
            Currently, there is only support for RJMCMC between two models.

            """
            new_model_index = 1 - model_index
            key_aux, key_accept = jrnd.split(key)            
            
            def up_branch(_):
                u = auxiliary_proposal_dist.sample(seed=key_aux)
                new_position = jump_functions[0](position, u)  # make kappa from the auxiliary variable u
                jac_det = jacobian_det_up(u)
                logq = -1.0*auxiliary_proposal_dist.log_prob(u)  # Note the negative sign! To check: is this robust for other proposal distributions?
                return new_position, jac_det, logq

            #
            def down_branch(_):
                new_position = jump_functions[1](position) # discard auxiliary variable and kappa
                jac_det = jacobian_det_down(new_position['kappa'])
                logq = auxiliary_proposal_dist.log_prob(projections[1](new_position)) # log(kappa / mu) where mu is the mean of the auxiliary proposal
                return new_position, jac_det, logq

            #
            new_position, jac_det, logq = jax.lax.cond(model_index == 0, up_branch, down_branch, operand=None)
            new_position['model_index'] = new_model_index  # update model index in the new position
            log_p_current = jax.lax.switch(model_index, model_logprobs, position)
            log_p_proposed = jax.lax.switch(new_model_index, model_logprobs, new_position)
            log_accept_ratio = log_p_proposed - log_p_current + logq + jnp.log(jac_det) 

            accept = jnp.log(jrnd.uniform(key_accept)) < log_accept_ratio
            next_state = jax.lax.cond(accept, lambda _: RJState(new_position), lambda _: state, operand=None)       

            return next_state, {'within_model_move': 0, 
                                'is_accepted': accept, 
                                'log_accept_ratio': log_accept_ratio,
                                'step_info': {'log_p_current': log_p_current,
                                              'log_p_proposed': log_p_proposed,
                                              'logq': logq,
                                              'jacobian_det': jac_det}}
        
        #
        return jax.lax.cond(move_type, do_within_model, do_between_model, operand=None)

    #
    def init_fn(position: ArrayTree, rng_key=None):
        del rng_key
        return RJState(position=position)

    #
    def step_fn(key: PRNGKey, state, *args, **kwargs):
        state, info = reversible_jump_fn(key, state, *args, **kwargs)
        return state, info
    
    #

    within_model_fns = [make_within_model_fn(i) for i in range(len(models))]
    model_logprobs = [make_model_logprob(model) for model in models]
    mcmc_samplers = [make_within_model_kernel(i) for i in range(len(models))]

    step_fn.__name__ = 'reversible_jump_fn'
    return SamplingAlgorithm(init_fn, step_fn)

#