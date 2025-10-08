import jax.numpy as jnp
from bamojax.base import Model
from jax.scipy.stats import norm

import numpyro as npr
import numpyro.distributions as dist
import numpyro.distributions.transforms as nprb

y = jnp.asarray([-3.470e-1, -1.520e+0, 6.760e-1, -1.410e+0, -1.610e+0, 1.430e+0, -3.810e-2, 6.930e-1, -8.290e-1,  1.030e+0])
n = len(y)

model1 = Model('test_model 1')
A = model1.add_node('A', distribution=dist.TransformedDistribution(dist.Normal(loc=0.0, scale=1.0), nprb.ExpTransform()))
B = model1.add_node('B', distribution=dist.Normal(loc=jnp.zeros((n, )), scale=jnp.ones((n, ))))
_ = model1.add_node('C', distribution=dist.Normal, observations=y, parents=dict(loc=B, scale=A))

model2 = Model('test_model 2')
D = model2.add_node('D', distribution=dist.TransformedDistribution(dist.Normal(loc=0.0, scale=1.0), nprb.ExpTransform()))
E = model2.add_node('E', distribution=dist.Normal(loc=0.0, scale=1.0), shape=(n, ))
_ = model2.add_node('F', distribution=dist.Normal, observations=y, parents=dict(loc=E, scale=D))

model3 = Model('test_model 3')
G = model3.add_node('G', distribution=dist.TransformedDistribution(dist.Normal(loc=0.0, scale=1.0), nprb.ExpTransform()))
H = model3.add_node('H', distribution=dist.Normal(loc=0.0, scale=1.0))
_ = model3.add_node('I', distribution=dist.Normal, observations=y, parents=dict(loc=H, scale=G))

model4 = Model('test_model 4 - uninstantiated parent')
J = model4.add_node('J', distribution=dist.TransformedDistribution(dist.Normal(loc=0.0, scale=1.0), nprb.ExpTransform()))
K = model4.add_node('K', distribution=dist.Normal, parents=dict(loc=0.0, scale=1.0), shape=(n, ))
_ = model4.add_node('L', distribution=dist.Normal, observations=y, parents=dict(loc=K, scale=J))

def test_node_shapes():    

    m1 = model1.get_model_size()
    assert m1 == n + 1

    m2 = model2.get_model_size()
    assert m2 == n + 1

    m3 = model3.get_model_size()
    assert m3 == 2

    m4 = model4.get_model_size()
    assert m4 == n + 1

#
def test_derived_densities():

    A_value = 4.0
    B_values = jnp.array([-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    # Compute log prior (using change-of-variables because we assigned it to log A instead of A directy)
    prior_density = norm.logpdf(jnp.log(A_value), loc=0.0, scale=1.0) - jnp.log(A_value) + jnp.sum(norm.logpdf(B_values, loc=jnp.zeros((n, )), scale=jnp.ones((n, ))))
    state = dict(A=A_value, B=B_values)
    likelihood = jnp.sum(norm.logpdf(y, loc=B_values, scale=A_value))

    # depending on hardware these might be different in the final decimal
    assert jnp.isclose(likelihood, model1.loglikelihood_fn()(state))
    assert jnp.isclose(prior_density, model1.logprior_fn()(state))
    
#



