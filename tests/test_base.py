import jax.numpy as jnp
from bamojax.base import Model, Node
import distrax as dx

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

def test_node_shapes():

    y = jnp.asarray([-3.470e-1, -1.520e+0, 6.760e-1, -1.410e+0, -1.610e+0, 1.430e+0, -3.810e-2, 6.930e-1, -8.290e-1,  1.030e+0])
    n = len(y)

    model1 = Model('test_model 1')
    node_A = model1.add_node('A', distribution=dx.Transformed(dx.Normal(loc=0.0, scale=1.0), tfb.Exp()))
    node_B = model1.add_node('B', distribution=dx.Normal(loc=jnp.zeros((n, )), scale=jnp.ones((n, ))))
    _ = model1.add_node('C', distribution=dx.Normal, observations=y, parents=dict(loc=node_B, scale=node_A))

    m1 = model1.get_model_size()
    assert m1 == n + 1

    model2 = Model('test_model 2')
    node_D = model2.add_node('D', distribution=dx.Transformed(dx.Normal(loc=0.0, scale=1.0), tfb.Exp()))
    node_E = model2.add_node('E', distribution=dx.Normal(loc=0.0, scale=1.0), shape=(n, ))
    _ = model2.add_node('F', distribution=dx.Normal, observations=y, parents=dict(loc=node_E, scale=node_D))

    m2 = model2.get_model_size()
    assert m2 == n + 1

    model3 = Model('test_model 3')
    node_G = model3.add_node('G', distribution=dx.Transformed(dx.Normal(loc=0.0, scale=1.0), tfb.Exp()))
    node_H = model3.add_node('H', distribution=dx.Normal(loc=0.0, scale=1.0))
    _ = model3.add_node('I', distribution=dx.Normal, observations=y, parents=dict(loc=node_H, scale=node_G))

    m3 = model3.get_model_size()
    assert m3 == 2

#



