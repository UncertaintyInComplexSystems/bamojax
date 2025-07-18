from jaxtyping import Array, Union
import jax
import jax.numpy as jnp
import distrax as dx
import jax.random as jrnd
from typing import Tuple, Callable
from distrax._src.distributions.distribution import Distribution
from distrax._src.bijectors.bijector import Bijector
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors


class Node:
    r""" The essential element of any Bayesian model is the variable, represented by a node in a DAG. 

    Nodes can consist of stochastic or deterministic variables, and can be observed or latent.

    Hyperparameters of a model are implicitly observed, deterministic nodes.
    
    """
    
    def __init__(self, name: str = 'root', 
                 observations: Array = None, 
                 distribution: Union[Distribution, Bijector] = None, 
                 parents = None, 
                 link_fn: Callable = None,
                 shape: Union[Tuple, int] = None,
                 bijector: Bijector = None):
        self.name = name
        
        if shape is None: 
            shape = ( )
        self.shape = shape
        if bijector is not None:
            self.bijector = bijector
        if observations is not None:
            observations = jnp.asarray(observations) if jnp.isscalar(observations) else observations
            self.observations = observations
        if distribution is not None:
            self.distribution = distribution
            self.parents = {}
            if parents is not None:
                for param, parent in parents.items():
                    self.add_parent(param, parent)            
            if link_fn is None:
                def identity(**kwargs):
                    return kwargs
                link_fn = identity
            self.link_fn = link_fn        

    #
    def is_observed(self) -> bool:
        """ Check if a node is an observed variable.
        
        """
        return hasattr(self, 'observations')

    #
    def is_stochastic(self) -> bool:
        """ Check whether a node is stochastic or deterministic.

        """
        return hasattr(self, 'distribution')
    
    #
    def is_root(self) -> bool:
        """ Check whether a node is a root node.
            
        """
        return not hasattr(self, 'parents') or len(self.parents) == 0
    
    #
    def add_parent(self, param, node):
        """ Add a parent node.

        """
        assert isinstance(node, Node)
        self.parents[param] = node

    #
    def set_step_fn(self, step_fn):
        self.step_fn = step_fn

    #
    def set_step_fn_parameter(self, step_fn_params):
        self.step_fn_params = step_fn_params

    #
    def is_leaf(self):
        """ Check whether a node is observed and has parents.
        
        """
        return hasattr(self, 'parents') and self.is_observed()
    
    #    
    def get_distribution(self, state: dict = None, minibatch: dict = None) -> Distribution:
        r""" Derives the parametrized distribution p(node | Parents=x), where x is derived from the state object.

        Args:
            state: Current assignment of (parent) values.
            minibatch: A additional set of assigned variables, useful for out-of-sample predictions.
        Returns:
            An instantiated distrax distribution object.
        
        """

        # Root-level nodes can be defined as instantiated distrax distributions.
        if isinstance(self.distribution, Distribution):
            return self.distribution
        
        if minibatch is None:
            minibatch = {}

        # Otherwise the distribution is instantiated from the state.
        parent_values = {}
        for parent_name, parent_node in self.parents.items():
            if parent_node in state:
                parent_values[parent_name] = state[parent_node]
            else:
                if parent_name in minibatch:
                    parent_values[parent_name] = minibatch[parent_node]
                else:
                    parent_values[parent_name] = self.parents[parent_name].observations

        transformed_parents = self.link_fn(**parent_values)
        if hasattr(self, 'bijector'):
            return dx.Transformed(self.distribution(**transformed_parents), self.bijector)
        else:
            return self.distribution(**transformed_parents)

    #
    def __repr__(self) -> str:
        return f'{self.name}'
    
    #
    def __hash__(self):
        return hash((self.name))

    #
    def __eq__(self, other):
        if isinstance(other, Node):
            return self.name == other.name
        return NotImplementedError

    #

#
class Model:
    r""" A Bayesian model is represented as a directed acyclic graph, in which nodes are associated with random variables.

    Typical use:

        model = Model('model name')
        _ = model.add_node('x', observations=...)
    
    """

    def __init__(self, name='Bayesian model', verbose=False):
        self.name = name
        self.nodes = {}
        self.root_nodes = list()
        self.leaf_nodes = list()
        self.children = dict()
        self.parents = dict()
        self.verbose = verbose

    #
    def add_node(self, 
                 name: str = 'root', 
                 distribution: Union[Distribution, Bijector] = None, 
                 observations: Array = None, 
                 parents: dict = None, 
                 link_fn: Callable = None,
                 shape: Union[Tuple, int] = None,
                 bijector: Bijector = None) -> Node:
        r""" Adds a node to the Bayesian model DAG

        Args:
          name: The name of the variable.
          distribution: The distrax distribution of the variable given its (transformed) parents.
          observations: If the node is observed; the actual observations.
          parents: The nodes that this node depends on.
          link_fn: A link function combining the inputs to form the input to the corresponding distrax distribution.
          shape: The dimensions of the variable.
          bijector: A bijector can be passed to transform variables.
        Returns:
          New node
        
        """
        if parents is not None:
            new_parents = {}
            for parent_name, parent in parents.items():
                if not isinstance(parent, Node):
                    # Parent is numeric
                    parent_node = self.add_node(name=f'{parent_name}_{name}', observations=parent)
                    new_parents[parent_name] = parent_node
                else:
                    new_parents[parent_name] = parent
            parents = new_parents
        new_node = Node(name=name, distribution=distribution, observations=observations, parents=parents, link_fn=link_fn, shape=shape, bijector=bijector)
        self.nodes[name] = new_node
        if self.verbose: print(f'Adding node ({name})')
        if parents is not None:
            for parent in parents.values():
                self.add_edge(parent, new_node)
        if new_node.is_root():
            self.root_nodes.append(new_node)
        if new_node.is_leaf():
            self.leaf_nodes.append(new_node)
        return new_node

    #    
    def add_edge(self, from_node, to_node):
        r""" Store the dependence between two nodes.

        Args:
            from_node: source node
            to_node: target node
        
        """
        if self.verbose: print(f'Add edge ({from_node}) -> ({to_node})')
        if not from_node in self.children:
            self.children[from_node] = set()
        self.children[from_node].add(to_node)
        if not to_node in self.parents:
            self.parents[to_node] = set()
        self.parents[to_node].add(from_node)

    #
    def get_children(self, node):
        """ Returns the children of a node.

        """
        if node in self.children:
            return self.children[node]
        return []
    
    #
    def get_parents(self, node):
        """ Returns the parents of a node.
        
        """
        if node in self.parents:
            return self.parents[node]
        return []
    
    #
    def get_root_nodes(self):
        """ Return all nodes that are roots.
        
        """
        return self.root_nodes
    
    #
    def get_leaf_nodes(self):
        """ Returns all nodes that are leaves.
        
        """
        return {self.nodes[k]: self.nodes[k] for k in self.nodes.keys() - self.children.keys()}
    
    #
    def get_stochastic_nodes(self):
        """ Returns all stochastic nodes.
        
        """
        return {k: v for k, v in self.nodes.items() if v.is_stochastic()}
    
    #
    def get_latent_nodes(self):
        """ Returns all latent nodes.
        
        """
        return {k: v for k, v in self.nodes.items() if v.is_stochastic() and not v.is_observed()}

    #
    def logprior_fn(self) -> Callable:
        r""" Returns a callable function that provides the log prior of the model given the current state of assigned variables.
        
        """
        
        def logprior_fn_(state) -> float:
            sorted_free_variables = [node for node in self.get_node_order() if node.is_stochastic() and not node.is_observed()]
            logprob = 0.0
            for node in sorted_free_variables:
                if node.is_root():
                    logprob += jnp.sum(node.get_distribution().log_prob(state[node.name]))
                else:
                    logprob += jnp.sum(node.get_distribution(state).log_prob(state[node.name]))
            return logprob
        
        #
        return logprior_fn_

    #
    def loglikelihood_fn(self) -> Callable:
        r""" Returns a callable function that provides the log likelihood of the model given the current state of assigned variables.
        
        """

        def loglikelihood_fn_(state) -> float:
            logprob = 0.0
            for node in self.get_leaf_nodes():
                logprob += jnp.sum(node.get_distribution(state).log_prob(value=node.observations))
            return logprob
        
        #
        return loglikelihood_fn_
    
    #
    def batched_loglikelihood_fn(self) -> Callable:
        r""" Batched loglikelihood function for stochastic-gradient methods.

        Assumes `minibatch` is a dictionary containing a subset of observations for each observed leaf node.

        """

        def loglikelihood_fn_(state, minibatch) -> float:
            logprob = 0.0
            for node in self.get_leaf_nodes():
                logprob += jnp.sum(node.get_distribution(state, minibatch=minibatch).log_prob(value=minibatch[node]))
            return logprob
        
        #
        return loglikelihood_fn_

    #
    def get_model_size(self) -> int:
        r""" Returns the total dimensions of the model. 
        
        As node.distribution can be abstract, we create a concrete instantiation by drawing a sample from the prior and deriving the shape from this sample.
        
        """

        draw = self.sample_prior(key=jrnd.PRNGKey(0)) 
        size = jnp.sum(jnp.array([jnp.size(v) for v in draw.values()]))
        return size
    
    #  
    def get_node_order(self):
        r""" Returns the latent variables in topological order; child nodes are always listed after their parents.
        
        """
        if not hasattr(self, 'node_order'):
            self.node_order = self.__get_topological_order()
        return self.node_order
    
    #
    def __get_topological_order(self) -> list:
        r""" Traverses the directed acyclic graph that defines the Bayesian model and returns its nodes in topological order

        Returns:
            A list of sorted nodes.
        
        """
        def traverse_dag_backwards(node: Node, visited_: dict, order: list):
            if node in visited_:
                return
            visited_[node] = 1

            for parent_node in node.parents.values():
                if parent_node.is_stochastic():
                    traverse_dag_backwards(parent_node, visited_, order)

            order.append(node)

        #
        order = []
        visited = {}
        leaves = self.get_leaf_nodes()
        for leaf in leaves:
            traverse_dag_backwards(leaf, visited, order)
        return order
    
    #
    def sample_prior(self, key) -> dict:
        r""" Samples from the (hierarchical) prior distribution of the model.

        Args:
            key: Random seed
        Returns:
            A state dictionary with one random value for each node.
        
        """
        state = dict()
        sorted_free_variables = [node for node in self.get_node_order() if node.is_stochastic() and not node.is_observed()]
        for node in sorted_free_variables:
            key, subkey = jrnd.split(key)
            if node.is_root(): 
                dist = node.get_distribution()               
            else:
                dist = node.get_distribution(state)
            state[node.name] = dist.sample(seed=subkey, sample_shape=node.shape)   
        return state

    #
    def sample_predictive(self, key, state: dict, input_variables: dict = None) -> dict:
        r""" Sample stochastic observed nodes

        Args:
            key: PRNGKey
            state: a draw from either p(x) or p(x|.)
            input_variables: a dictionary with values for observed non-stochastic nodes
        
        Returns:
            A dictionary which is the same as 'state' but appended with sampled values.
        
        """
        
        for node in self.get_leaf_nodes():
            key, key_obs = jrnd.split(key)
            state[node.name] = node.get_distribution(state, minibatch=input_variables).sample(seed=key_obs, sample_shape=node.shape)
        return state


    def sample_prior_predictive(self, key, **prediction_options) -> dict:
        r""" Sample from the (hierarchical) prior predictive distribution of the model.

        Args:
            key: Random seed
            prediction_options: A dictionary of options which can include minibatched input variables
        
        Returns:
            A dictionary with a random value for all stochastic observed nodes.
        
        """
        key, key_latent = jrnd.split(key)
        state = self.sample_prior(key_latent)
        return self.sample_predictive(key, state, prediction_options)

    #
    def sample_posterior_predictive(self, key, state: dict, input_variables: dict = None) -> dict:
        r""" Sample from the posterior predictive

        Args:
            key: Random key
            state: A draw from the posterior
            input_variables: Potential predictors and other non-stochastic observations

        Returns:
            A dictionary containing values for all stochastic observed nodes, conditioned on the observations.
        
        """

        return self.sample_predictive(key, state, input_variables)

    #
    def print_gibbs(self):
        r""" Print the structure of conditional distributions. 

        
        """

        print('Gibbs structure:')
        sorted_free_variables = [node for node in self.get_node_order() if node.is_stochastic() and not node.is_observed()]
        for node in sorted_free_variables:
            # get prior density        
            if node.is_root():
                prior = f'p({node})'
            else:
                parents = {p for p in self.get_parents(node)}
                prior = f'p({node} | {", ".join([p.name for p in parents])})'

            # get conditional density
            conditionals = []
            children = [c for c in self.get_children(node)]        
            for child in children:
                co_parents = set()
                for parent in self.get_parents(child):
                    if not parent == node or parent in co_parents:
                        co_parents.add(parent)        
                co_parents.add(node)
                conditional = f'p({child} | {", ".join([str(p) for p in co_parents])})'
                conditionals.append(conditional)

            print(f'{str(node):20s}: {" ".join(conditionals)} {prior}')

    #

#
class MetaModel():

    def __init__(self, 
                 model_list):
        self.model_list = model_list
        self.M = len(model_list)
        self.model_sizes = [model.get_model_size() for model in model_list]        
        self.indiv_latent_nodes = [set(model.get_latent_nodes().keys()) for model in model_list]

        # find the number of auxiliary variables needed, as the largest difference in latent node sets between any two models
        self.num_auxiliary = 0
        for i in range(self.M):
            for j in range(i+1, self.M):
                diff_left = set.difference(self.indiv_latent_nodes[i], self.indiv_latent_nodes[j])
                diff_right = set.difference(self.indiv_latent_nodes[j], self.indiv_latent_nodes[i])
                max_diff = jnp.max(jnp.array([len(diff_left), len(diff_right)]))
                if max_diff > self.num_auxiliary:
                    self.num_auxiliary = max_diff
        
        self.latent_variables = set.union(*self.indiv_latent_nodes)
        self.auxiliary_variables = [f'u_{i}' for i in range(self.num_auxiliary)]
        self.meta_state = self.latent_variables.union(set(self.auxiliary_variables))

        def make_model_sample_prior_fn(model_index):
            """Creates a function that samples from the prior of the specified model index."""
            def fn(key):
                sample = self.model_list[model_index].sample_prior(key)
                all_latents = {k: jnp.nan for k in self.latent_variables}
                return {**all_latents, **sample}  
            
            #   
            return fn
        #
        self.model_sample_prior_fns = [make_model_sample_prior_fn(i) for i in range(self.M)]

        def make_model_sample_predictive_fn(model_index):
            """Creates a function that samples from the predictive distribution of the specified model index."""
            def fn(input):
                key, state, input_variables = input
                return self.model_list[model_index].sample_predictive(key, state, input_variables)
            
            #   
            return fn
        
        #
        self.model_sample_predictive_fns = [make_model_sample_predictive_fn(i) for i in range(self.M)]

        def make_logprior_fn(model_index):
            """Creates a function that computes the log prior of the specified model index."""
            def fn(state):
                return self.model_list[model_index].logprior_fn()(state)
            
            #   
            return fn
        
        #
        self.model_logprior_fns = [make_logprior_fn(i) for i in range(self.M)]

        def make_loglikelihood_fn(model_index):
            """Creates a function that computes the log likelihood of the specified model index."""
            def fn(state):
                return self.model_list[model_index].loglikelihood_fn()(state)
            
            #   
            return fn
        
        #
        self.model_loglikelihood_fns = [make_loglikelihood_fn(i) for i in range(self.M)]

    #
    def sample_prior(self, key) -> dict:
        key_model, key_sample = jrnd.split(key)
        model_index = jrnd.randint(key_model, shape=(), minval=0, maxval=self.M)
        sample = jax.lax.switch(model_index, self.model_sample_prior_fns, operand=key_sample)
        auxiliary_values = {f'u_{i}': jnp.nan for i in range(self.num_auxiliary)}  # to keep the same pytree structure across models in reversible jump MCMC
        return {'model_index': model_index, **sample, **auxiliary_values}

    #
    def sample_predictive(self, key, state: dict, input_variables: dict = None) -> dict:
        return jax.lax.switch(state['model_index'], self.model_sample_predictive_fns, 
                              operand=[key, state, input_variables])

    #
    def sample_prior_predictive(self, key, **input_variables) -> dict:
        key_prior, key_predictive = jrnd.split(key)
        prior_sample = self.sample_prior(key_prior)
        return self.sample_predictive(key_predictive, prior_sample, input_variables=input_variables)

    #
    def sample_posterior_predictive(self, key, state: dict, input_variables: dict = None) -> dict:
        return self.sample_predictive(key, state, input_variables=input_variables)

    #
    def logprior_fn(self) -> Callable:

        def fn(state: dict) -> float:
            """Computes the log prior of the state."""
            model_index = state['model_index']
            return jax.lax.switch(model_index, self.model_logprior_fns, operand=state)
        
        #
        return fn

    #
    def loglikelihood_fn(self) -> Callable:

        def fn(state: dict) -> float:   
            """Computes the log likelihood of the state."""
            model_index = state['model_index']
            return jax.lax.switch(model_index, self.model_loglikelihood_fns, operand=state)
        #
        return fn   

    #
#