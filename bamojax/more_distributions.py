from typing import Callable, Tuple, Optional
from jaxtyping import Array
from abc import ABC, abstractmethod
import jax
import jax.random as jrnd
import jax.numpy as jnp

from jax.scipy.special import multigammaln

import numpyro as npr
import numpyro.distributions as dist
from numpyro.distributions import Distribution
import numpyro.distributions.transforms as nprb
from numpyro.distributions.constraints import positive_definite

from .base import Node


class MeanFunction(ABC):

    @abstractmethod
    def mean(self, params, x):
        pass

    #
#
class Zero(MeanFunction):

    def mean(self, params, x):
        return jnp.zeros((x.shape[0], ))
    
    #

#
def GaussianProcessFactory(cov_fn: Callable, mean_fn: Callable = Zero(),  nd: Tuple[int, ...] = None, jitter: float = 1e-6):
    r""" Returns an instantiated Gaussian process distribution object. 
    
    This is essentially a dist.MultivariateNormal object, with its mean and covariance determined by the mean and covariance functions of the GP.
    
    Args: 
        cov_fn: The GP covariance function. It assumes a signature of cov_fn(parameters: dict, x: Array, y: Array). 
                This is provided by the `jaxkern` library, but others can be used as well.
        mean_fn: The GP mean function.
        nd: A tuple of integers indicating optional additional output dimensions (for multi-task GPs).
        jitter: A small value for numerical stability.
    Returns:
        A GaussianProcessInstance distrax Distribution object.
    
    """

    class GaussianProcessInstance(Distribution):
        """ An instantiated Gaussian process distribution object, i.e. a multivariate Gaussian.
        
        """

        def __init__(self, input: Node, **params):
            self.input = input

            # In case of composite covariance functions:
            if 'params' in params:
                self.params = params['params']
            else:
                self.params = params

        #
        def sample(self, key, sample_shape=()):
            r""" Sample from the instantiated Gaussian process (i.e. multivariate Gaussian)
            
            """
            x = self.input
            m = x.shape[0]
            output_shape = (m, )

            if nd is not None:
                output_shape = nd + output_shape

            if len(sample_shape) >= 1:
                output_shape = sample_shape + output_shape

            mu = self._get_mean()
            cov = self._get_cov()
            L = jnp.linalg.cholesky(cov)
            z = jrnd.normal(key, shape=output_shape).T
            V = jnp.tensordot(L, z, axes=(1, 0))
            f = jnp.add(mu, jnp.moveaxis(V, 0, -1))
            # if jnp.ndim(f) == 1:
            #     f = f[jnp.newaxis, :]
            return f
        
        #
        def log_prob(self, value):
            mu = self._get_mean()
            cov = self._get_cov()
            return dist.MultivariateNormal(loc=mu, covariance_matrix=cov).log_prob(value=value)

        #
        def sample_predictive_batched(self, key: Array, x_pred: Array, f: Array, num_batches:int = 20):
            r""" Samples from the posterior predictve of the latent f, but in batches to converve memory.

            Args:
                key: PRNGkey
                x_pred: Array
                    The test locations
                f: Array
                    The trained GP to condition on
                num_batches: int
                    The number of batches to predict over.

            Returns:
                Returns samples from the posterior predictive distribution:

                $$
                    \mathbf{f}* \sim p(\mathbf{f}* \mid \mathbf{f}, X, y x^*) = \int p(\mathbf{f}* \mid x^*, \mathbf{f}) p(\mathbf{f} \mid X, y) \,\text{d} \mathbf{f}
                ##
            
            
            """
            if jnp.ndim(x_pred) == 1:
                x_pred = x_pred[:, jnp.newaxis]

            n_pred = x_pred.shape[0]
            data_per_batch = int(n_pred / num_batches)
            fpreds = list()
            for batch in range(num_batches):
                key, subkey = jrnd.split(key)
                lb = data_per_batch*batch
                ub = data_per_batch*(batch + 1)
                fpred_batch = self.sample_predictive(subkey, x_pred[lb:ub, :], f)
                fpreds.append(fpred_batch)

            fpred = jnp.hstack(fpreds)
            return fpred

        #
        def sample_predictive(self, key: Array, x_pred: Array, f: Array):
            r"""Sample latent f for new points x_pred given one posterior sample.

            See Rasmussen & Williams. We are sampling from the posterior predictive for
            the latent GP f, at this point not concerned with an observation model yet.

            We have $[\mathbf{f}, \mathbf{f}^*]^T ~ \mathcal{N}(0, KK)$, where $KK$ can be partitioned as:

            $$
                KK = \begin{bmatrix} K(x,x) & K(x,x^*) \\ K(x,x^*)\top & K(x^*,x^*)\end{bmatrix}
            $$

            This results in the conditional
            $$
            \mathbf{f}^* | x, x^*, \mathbf{f} ~ \mathcal{N}(\mu, \Sigma) \enspace,
             $$ where

            $$
            \begin{align*}
                \mu &= K(x^*, x)K(x,x)^-1 f \enspace,
                \Sigma &= K(x^*, x^*) - K(x^*, x) K(x, x)^-1 K(x, x^*) \enspace.
            \end{align*}                
            $$

            Args:
                key: The jrnd.PRNGKey object
                x_pred: The prediction locations $x^*$
                state_variables: A sample from the posterior

            Returns:
                A single posterior predictive sample $\mathbf{f}^*$

            """
            x = self.input
            n = x.shape[0]
            z = x_pred
            if 'obs_noise' in self.params:
                obs_noise = self.params['obs_noise']
                if jnp.isscalar(obs_noise) or jnp.ndim(obs_noise) == 0:
                    diagonal_noise = obs_noise**2 * jnp.eye(n, )
                else:
                    diagonal_noise = jnp.diagflat(obs_noise)**2
            else:
                diagonal_noise = 0

            mean = mean_fn.mean(params=self.params, x=z)
            Kxx = self.get_cov()
            Kzx = cov_fn(params=self.params, x=z, y=x)
            Kzz = cov_fn(params=self.params, x=z, y=z)

            Kxx += jitter * jnp.eye(*Kxx.shape)
            Kzx += jitter * jnp.eye(*Kzx.shape)
            Kzz += jitter * jnp.eye(*Kzz.shape)

            L = jnp.linalg.cholesky(Kxx + diagonal_noise)
            v = jnp.linalg.solve(L, Kzx.T)

            predictive_var = Kzz - jnp.dot(v.T, v)
            predictive_var += jitter * jnp.eye(*Kzz.shape)
            C = jnp.linalg.cholesky(predictive_var)

            def get_sample(u_, target_):
                alpha = jnp.linalg.solve(L.T, jnp.linalg.solve(L, target_))
                predictive_mean = mean + jnp.dot(Kzx, alpha)
                return predictive_mean + jnp.dot(C, u_)

            #
            if jnp.ndim(f) == 3:            
                _, nu, d = f.shape
                u = jrnd.normal(key, shape=(len(z), nu, d))
                samples = jax.vmap(jax.vmap(get_sample, in_axes=1), in_axes=1)(u, f)
                return samples.transpose([2, 0, 1])
            elif jnp.ndim(f) == 1:
                u = jrnd.normal(key, shape=(len(z),))
                return get_sample(u, f)
            else:
                raise NotImplementedError(f'Shape of target must be (n,) or (n, nu, d)',
                f'but {f.shape} was provided.')
            
        #
        def _get_mean(self):
            """ Returns the mean of the GP at the input locations.
            
            """
            return mean_fn.mean(params=self.params, x=self.input)
        
        #
        def get_mean(self):
            return self._get_mean()
        
        #
        def _get_cov(self):
            """ Returns the covariance of the GP at the input locations.
            
            """
            x = self.input
            m = x.shape[0]
            cov = cov_fn(params=self.params, x=x, y=x)
            jitter_matrix = jitter * jnp.eye(m)
            return cov + jitter_matrix
        
        #
        def get_cov(self):
            return self._get_cov()
        
        #
        @property
        def event_shape(self):
            r""" Event shape in this case is the shape of a single draw of $F = (f(x_1), ..., f(x_n))$
            
            """

            output_shape = (self.input.shape[0], )
            if nd is not None:
                output_shape = nd + output_shape
            return output_shape
        
        #
        @property
        def batch_shape(self):
            return ()
        
        #

    #
    return GaussianProcessInstance

#
def AutoRegressionFactory(ar_fn: Callable):
    r""" Generates an autoregressive distribution with Gaussian emissions.

    This is a generator function that constructs a distrax Distribution object, which can then be queried for its log probability for inference.
    
    Args:
        ar_fn: A Callable function that takes innovations $\epsilon \sim \mathcal{N}(0, \sigma^2)$, and the previous instances $x(t-1), ..., x(t-p)$, and performs whatever computation the user requires.
        
    """

    # TODO: migrate from `distrax` format to `numpyro` format

    class ARInstance(Distribution):
        """ An instantiated autoregressive distribution object.
        
        """

        def __init__(self, **kwargs):
            self.parameters = kwargs
                    
        #
        def _construct_lag_matrix(self, y, y_init):
            r""" Construct $y$, and up to order shifts of it.
            
            """
            order = 1 if jnp.isscalar(y_init) else y_init.shape[0]

            @jax.jit
            def update_fn(carry, i):
                y_shifted = jnp.roll(carry, shift=1)  
                y_shifted = y_shifted.at[0].set(y_init[i])  
                return y_shifted, y_shifted  
            
            #
            _, columns = jax.lax.scan(update_fn, y, jnp.arange(order))
            return columns
        
        #
        def log_prob(self, value):
            r""" Returns the log-density of the complete AR distribution
            
            """
            y_lagged = self._construct_lag_matrix(y=value, y_init=self.parameters['y0'])   
            mu = ar_fn(y_prev=y_lagged, **self.parameters) 
            return dist.Normal(loc=mu, scale=self.parameters['scale']).log_prob(value)

        #
        def _sample_n(self, key, n):
            r""" Sample from the AR distribution

            
            """
            keys = jrnd.split(key, n)
            samples = jax.vmap(self._sample_predictive)(keys)  
            return samples

        #        
        def _sample_predictive(self, key):
            r""" Sample from the AR(p) model.

            Let:

            $$
            \begin{align*}
                \epsilon_t &\sim \mathcal{N}(0, \sigma_y)
                y_t &= f(y_t-1, \theta) + \epsilon_t
            \end{align*}
            $$ for $t = M+1, \ldots, T$.
            
            """
            @jax.jit
            def ar_step(carry, epsilon_t):
                y_t = ar_fn(y_prev=carry, **self.parameters) + epsilon_t
                new_carry = jnp.concatenate([carry[1:], jnp.array([y_t])])
                return new_carry, y_t

            # 
            y_init = self.parameters['y0']
            order = 1 if jnp.isscalar(y_init) else y_init.shape[0]
            innovations = self.parameters['scale'] * jrnd.normal(key, shape=(self.parameters['T'] - order, ))
            _, ys = jax.lax.scan(ar_step, y_init, innovations)
            y = jnp.concatenate([y_init, ys])
            return y
        
        #                        
        @property
        def batch_shape(self):
            return ( )

        #
        @property
        def event_shape(self):
            return (self.T, )

        #

    #
    return ARInstance

#
class GWishart(Distribution):

    support = positive_definite

    def __init__(self, G, dof, scale=None, data=None):
        """Initializes a G-Wishart distribution.

        Args:
          G: binary adjacency matrix
          dof: degrees of freedom
          scale: scale matrix        
        """
        self.G = G
        self.dof = dof
        assert scale is not None or data is not None, 'Provide either a scale matrix or data to compute the scale matrix from.'
        if scale is None:
            # Note that the scale matrix is the empirical scatter matrix, not the covariance matrix; hence the multiplication with n
            n = data.shape[0]
            emp_cov = jnp.cov(data, rowvar=False)
            scale = emp_cov * n
        self.scale = scale
        self.scale_inv = jnp.linalg.inv(scale)
        self.p = scale.shape[0]
        super().__init__(batch_shape=(), event_shape=(self.p, self.p), validate_args=False)
    
    #
    def _sample_G_Wishart(self, key, tol=1e-6, max_iter=100):
        """Draws a sample from the G-Wishart distribution with graph G, scale matrix D, and dof degrees of freedom.

        The procedure implements the algorithm proposed by Alex Lenkoski (2013), based on iterative proportional scaling.

        The key to making this function jittable is to avoid dynamic indexing and keep all arrays and matrices of size (p, ) or (p, p), in the update_W function.

        Args:
            key: JAX PRNG key
            G: binary adjacency matrix of shape (p, p)
            D: scale matrix of shape (p, p)
            dof: degrees of freedom
            tol: tolerance for convergence
            max_iter: maximum number of iterations  
        Returns:
            A sample from the G-Wishart distribution of shape (p, p).

        Literature:
            Lenkoski, A. (2013). A direct sampler for G-Wishart variates. Stat (2):1, pp. 119-128. https://doi.org/10.1002/sta4.23

        """
        # See parametrization of the Wishart distribution in https://github.com/mhinne/BaCon
        # To match the results in Lenkoski (2013), we need to sample K ~ Wishart(dof + p - 1, D^{-1})
        K = dist.Wishart(concentration=self.dof + self.p - 1, scale_matrix=self.scale_inv).sample(key)
        Sigma = jnp.linalg.inv(K)
        W0 = Sigma
        
        def sweep_nodes(W):
            """ Perform one sweep over all nodes to update W.
            
            """
            def update_W(W, j):
                mask_j = self.G[j, :]
                beta_j = jnp.linalg.solve(W, Sigma[:, j] * mask_j)
                update_w = jnp.dot(W, beta_j)
                W_new = W.at[j, :].set(update_w)
                W_new = W_new.at[:, j].set(update_w)
                return W_new, None

            #
            W, _ = jax.lax.scan(update_W, W, xs=jnp.arange(self.p))
            return W

        #
        def cond_fn(state):
            W_old, W_new, iter = state
            diff = jnp.linalg.norm(W_new - W_old)
            return jnp.logical_and(diff > tol, iter < max_iter)
        
        #
        def body_fn(state):
            _, W_new, iter = state
            W_next = sweep_nodes(W_new)
            return (W_new, W_next, iter + 1)

        #
        W1 = sweep_nodes(W0)                          
        init_state = (W0, W1, 0)

        _, W_final, _ = jax.lax.while_loop(cond_fn, body_fn, init_state)

        # Due to numerical issues, we ensure symmetry and zeros:
        return jnp.linalg.inv((W_final + W_final.T)/2)*self.G

    #
    def sample(self, key, sample_shape=( )):
        """Draws samples from the G-Wishart distribution.

        Args:
            key: JAX PRNG key
            sample_shape: shape of the samples to draw  
        Returns:
            Samples from the G-Wishart distribution.
        """
        if sample_shape == ():
            return self._sample_G_Wishart(key)
        keys = jrnd.split(key, jnp.prod(jnp.array(sample_shape)))
        samples = jax.vmap(self._sample_G_Wishart)(keys)
        return samples.reshape(sample_shape + self.event_shape)
    
    #
    def log_prob(self, value):
        """Calculates the log probability of a given value.

        Args:
            value: value to calculate the log probability for  
        Returns:
            Log probability of the given value.
        """
        raise NotImplementedError('Log probability for G-Wishart distribution is intractable and not implemented.')
    
    #

#