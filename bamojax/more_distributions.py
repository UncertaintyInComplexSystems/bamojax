from typing import Callable, Tuple, Optional
from jaxtyping import Array
from abc import ABC, abstractmethod
import jax
import jax.random as jrnd
import jax.numpy as jnp
import distrax as dx
import jaxkern as jk

from jax.scipy.special import multigammaln

from distrax._src.distributions.distribution import Distribution
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

from .base import Node


def euclidean(x, y):
    return jnp.linalg.norm(x - y)

#
def pairwise_distances(dist: Callable, xs, ys):
  return jax.vmap(lambda x: jax.vmap(lambda y: dist(x, y))(xs))(ys)

#
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

# class RBF():
#     r""" Implements the Radial Basis Function Gaussian process covariance function.

#     Its parameters are a lengthscale and output variance.
    
#     """

#     def __init__(self, active_dims: Optional[list[int]] = None, name: str = 'RBF'):
#         self.active_dims = active_dims
#         self.name = name

#     #
#     def __call__(self, params: dict, x: Array, y: Array) -> Array:

#         D = pairwise_distances(dist=euclidean, xs=x, ys=y)
#         K = params['variance']*jnp.exp(-0.5*D**2 / params['lengthscale']**2)        
#         return K.squeeze().T
    
#     #

# #

def GaussianProcessFactory(cov_fn: Callable, mean_fn: Callable = None,  nd: Tuple[int, ...] = None, jitter: float = 1e-6):
    r""" Returns an instantiated Gaussian process distribution object. 
    
    This is essentially a dx.MultivariateNormalFullCovariance object, with its mean and covariance determined by the mean and covariance functions of the GP.
    
    Args: 
        cov_fn: The GP covariance function.
        mean_fn: The GP mean function.
        nd: A tuple of integers indicating optional additional output dimensions (for multi-task GPs).
        jitter: A small value for numerical stability.
    Returns:
        A GaussianProcessInstance distrax Distribution object.
    
    """

    class GaussianProcessInstance(Distribution):

        def __init__(self, input: Node, **params):
            self.input = input
            self.params = params

        #
        def _sample_n(self, key, n):
            r"""
            
            """
            x = self.input
            m = x.shape[0]
            output_shape = (m, )
            if nd is not None:
                output_shape = nd + output_shape

            if n > 1:
                output_shape = (n, ) + output_shape

            mu = self._get_mean()
            cov = self._get_cov()
            L = jnp.linalg.cholesky(cov)
            z = jrnd.normal(key, shape=output_shape).T
            V = jnp.tensordot(L, z, axes=(1, 0))
            f = jnp.add(mu, jnp.moveaxis(V, 0, -1))
            if jnp.ndim(f) == 1:
                f = f[jnp.newaxis, :]
            return f
        
        #
        def log_prob(self, value):
            mu = self._get_mean()
            cov = self._get_cov()
            return dx.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=cov).log_prob(value=value)

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

                f* \sim p(f* | f, X, y x*) = \int p(f* | x*, f) p(f | X, y) df
            
            
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
            """Sample latent f for new points x_pred given one posterior sample.

            See Rasmussen & Williams. We are sampling from the posterior predictive for
            the latent GP f, at this point not concerned with an observation model yet.

            We have [f, f*]^T ~ N(0, KK), where KK is a block matrix:

            KK = [[K(x, x), K(x, x*)], [K(x, x*)^T, K(x*, x*)]]

            This results in the conditional

            f* | x, x*, f ~ N(mu, cov), where

            mu = K(x*, x)K(x,x)^-1 f
            cov = K(x*, x*) - K(x*, x) K(x, x)^-1 K(x, x*)

            Args:
                key: The jrnd.PRNGKey object
                x_pred: The prediction locations x*
                state_variables: A sample from the posterior

            Returns:
                A single posterior predictive sample f*

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
            Kzx = cov_fn.cross_covariance(params=self.params, x=z, y=x)
            Kzz = cov_fn.cross_covariance(params=self.params, x=z, y=z)

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
            return mean_fn.mean(params=self.params, x=self.input)
        
        #
        def get_mean(self):
            return self._get_mean()
        
        #
        def _get_cov(self):
            x = self.input
            m = x.shape[0]
            return cov_fn.cross_covariance(params=self.params, x=x, y=x) + jitter * jnp.eye(m)
        
        #
        def get_cov(self):
            return self._get_cov()
        
        #
        @property
        def event_shape(self):
            r""" Event shape in this case is the shape of a single draw of F = (f(x_1), ..., f(x_n))
            
            """

            output_shape = (self.input.shape[0], )
            if nd is not None:
                output_shape = nd + output_shape
            return output_shape
        
        #
        @property
        def batch_shape(self):
            return self.input.shape[0]
        
        #

    #
    return GaussianProcessInstance

#
def AscendingDistribution(min_u, max_u, num_el):
    r""" Creates a distribution of a sorted array of continuous values in [min_u, max_u].

    To ensure gradient-based methods can work on the model, all transformations must be bijectors.
    A generic sort() does not meet this condition, as it is not invertible. By using the tfb 
    bijector Ascending() in combination with scaling and deriving the expected maximum value of 
    dx.Transformed(Uniform, Ascending()), we can construct, in expectation, the desired random 
    variable. Note that individual draws main contain values that exceed max_u. 

    Args:
        min_u, max_u: The desired range.
        num_el: The length of the desired variate.
    Returns:
        A distribution over arrays of length `num_el`, with values in ascending order.

    
    """

    R = 0.5 + (num_el-1)*(jnp.exp(1) - 1)
    base_distribution = dx.Independent(dx.Uniform(low=jnp.zeros(num_el), high=jnp.ones(num_el)), reinterpreted_batch_ndims=1)

    bijector = tfb.Chain([
        tfb.Scale(scale=(max_u - min_u) / R),  
        tfb.Shift(shift=jnp.array(min_u, dtype=jnp.float64)),  
        tfb.Ascending()               
    ])

    return dx.Transformed(base_distribution, bijector)

#
class Wishart(Distribution):
    """Wishart distribution with parameters `dof` and `scale`."""


    def __init__(self, dof: int, scale: Optional[Array]):
        """Initializes a Wishart distribution.

        Args:
          dof: degrees of freedom
          scale: scale matrix        
        """
        super().__init__()
        p = scale.shape[0]
        assert dof > p - 1, f'DoF must be > p - 1, found DoF = {dof}, and p = {p}.'
        self._dof = dof
        self._scale = scale
        self._p = p

    #
    @property
    def event_shape(self) -> Tuple[int, ...]:
        """Shape of event of distribution samples."""
        return ()
    
    #
    @property
    def batch_shape(self) -> Tuple[int, ...]:
        """Shape of batch of distribution samples."""
        return jax.lax.broadcast_shapes(self._dof.shape, self._scale.shape)
    
    #

    def _sample_n(self, key: Array, n: int) -> Array:
        """See `Distribution._sample_n`."""

        X = jrnd.multivariate_normal(key, mean=jnp.zeros((self._p, )), cov=self._scale, shape=(n, self._dof))
        wishart_matrices = jnp.einsum('ndp,ndq->npq', X, X)
        return wishart_matrices
    
    #
    def log_prob(self, value: Array) -> Array:
        _, logdetV = jnp.linalg.slogdet(self._scale)
        _, logdetK = jnp.linalg.slogdet(value)

        logZ = 0.5*self._dof*self._p*jnp.log(2) + 0.5*self._dof*logdetV + multigammaln(0.5*self._dof, self._p)
        return 0.5*(self._dof - self._p - 1)*logdetK - 0.5*jnp.sum(jnp.diag(jnp.linalg.solve(self._scale, value))) - logZ
    
    #
    def mean(self) -> Array:
        """Calculates the mean."""

        return self._dof*self._scale
    
    #
    def mode(self) -> Array:
        """Calculates the mode."""

        assert self._dof > self._p + 2, f'The mode is only defined for DoF > p + 2, found DoF = {self._dof} and p = {self._p}.'
        return (self._dof - self._p - 1)*self._scale
    
    #
    def variance(self) -> Array:
        """Calculates the variance."""
        
        V_ij_squared = jnp.square(self._scale)  
        V_ii = jnp.diag(self._scale)  
        V_ii_V_jj = jnp.outer(V_ii, V_ii) 
        return self._dof * (V_ij_squared + V_ii_V_jj)

    #
# 