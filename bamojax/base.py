from jaxtyping import Array, Union
import jax.numpy as jnp
import jax.random as jrnd
from typing import Tuple, Callable, NamedTuple
from distrax._src.distributions.distribution import Distribution
from distrax._src.bijectors.bijector import Bijector
from blackjax.types import ArrayTree, PRNGKey


class Node:
    
    def __init__(self, name: str = 'root', 
                 observations: Array = None, 
                 distribution: Union[Distribution, Bijector] = None, 
                 parents = None, 
                 link_fn: Callable = None,
                 shape: Union[Tuple, int] = None):
        self.name = name
        if shape is None:
            shape = ( )
        self.shape = shape
        if observations is not None:
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
        return hasattr(self, 'observations')

    #
    def is_stochastic(self) -> bool:
        return hasattr(self, 'distribution')
    
    #
    def is_root(self) -> bool:
        return not hasattr(self, 'parents') or len(self.parents) == 0
    
    #
    def add_parent(self, param, node):
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
        return hasattr(self, 'parents') and self.is_observed()
    
    #    
    def get_distribution(self, state: dict = None) -> Distribution:
        r""" Derives the parametrized distribution p(node | Parents=x), where x is derived from the state object.

        Args:
            state: Current assignment of (parent) values.
        Returns:
            An instantiated distrax distribution object.
        
        """

        # Root-level nodes can be defined as instantiated distrax distributions.
        if isinstance(self.distribution, Distribution):
            return self.distribution

        # Otherwise the distribution is instantiated from the state.
        parent_values = {}
        for parent_name, parent_node in self.parents.items():
            if parent_node in state:
                parent_values[parent_name] = state[parent_node]
            else:
                parent_values[parent_name] = self.parents[parent_name].observations

        transformed_parents = self.link_fn(**parent_values)
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

    def __init__(self, name='Bayesian model', verbose=False):
        self.name = name
        self.nodes = {}
        self.root_nodes = list()
        self.leaf_nodes = list()
        self.children = dict()
        self.parents = dict()
        self.verbose = verbose

    #
    def add_node(self, name: str = 'root', 
                 distribution: Union[Distribution, Bijector] = None, 
                 observations: Array = None, 
                 parents: dict = None, 
                 link_fn: Callable = None,
                 shape: Union[Tuple, int] = None) -> Node:
        r""" Adds a node to the Bayesian model DAG

        Args:
          name: The name of the variable.
          distribution: The distribution of the variable given its (transformed) parents.
          observations: If the node is observed; the actual observations.
          parents: The nodes that this node depends on.
          shape: The dimensions of the variable.
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
        new_node = Node(name=name, distribution=distribution, observations=observations, parents=parents, link_fn=link_fn, shape=shape)
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
        if node in self.children:
            return self.children[node]
        return []
    
    #
    def get_parents(self, node):
        if node in self.parents:
            return self.parents[node]
        return []
    
    #
    def get_root_nodes(self):
        return self.root_nodes
    
    #
    def get_leaf_nodes(self):
        return {self.nodes[k]: self.nodes[k] for k in self.nodes.keys() - self.children.keys()}
    
    #
    def get_stochastic_nodes(self):
        return {k: v for k, v in self.nodes.items() if v.is_stochastic()}
    
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
    def get_model_size(self) -> int:
        r""" Returns the total number of latent scalars.
        
        """
        size = 0
        for node in self.nodes.values():
            if node.is_stochastic() and not node.is_observed():
                size += 1 if node.shape == () else jnp.prod(jnp.asarray(node.shape))
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
        
        """
        state = dict()
        sorted_free_variables = [node for node in self.get_node_order() if node.is_stochastic() and not node.is_observed()]
        for node in sorted_free_variables:
            key, subkey = jrnd.split(key)
            if node.is_root():                
                state[node.name] = node.get_distribution().sample(seed=subkey, sample_shape=node.shape)
            else:
                state[node.name] = node.get_distribution(state).sample(seed=subkey, sample_shape=node.shape)   
        return state

    #
    def sample_prior_predictive(self, key) -> dict:
        r""" Sample from the (hierarchical) prior distribution of the model, and sample values from the likelihood given the latent variables.
        
        """
        key, key_latent = jrnd.split(key)
        state = self.sample_prior(key_latent)
        for node in self.get_leaf_nodes():
            key, key_obs = jrnd.split(key)
            state[node.name] = node.get_distribution(state).sample(seed=key_obs, sample_shape=node.shape)
        return state

    #
    def print_gibbs(self):
        r""" Print the structure of conditional distributions. Should be expanded to write Tikz code for graphical models.

        
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