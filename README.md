# bamojax

Welcome to **bamojax**, the Bayesian modelling toolbox implemented using the Jax coding universe. **bamojax** is a probabilistic programming language, similar to Numpyro, PyMC, Stan, JAGS, and BUGS. It relies on [Blackjax](https://blackjax-devs.github.io/blackjax/) for approximate inference, on [Distrax](https://github.com/google-deepmind/distrax) for probability distributions and their essential operations. It adds to this the Directed Acyclic Graph (DAG) structure of a Bayesian model, and automatically derives model priors, likelihoods, and Gibbs inference schemes. This combines the speed from (Black)Jax with a convenient modelling environment.

## Installation

Install **bamojax** using 

```
pip install git+https://github.com/UncertaintyInComplexSystems/bamojax#egg=bamojax
```

**bamojax** has been developed and tested with `Jaxlib` 0.4.34, `Jax` 0.4.35, and Python 3.10.15. 

## Quick tutorial

Let's estimate the latent probability $\theta=p(x_1=\text{heads} \mid \theta)$ that a coin lands up heads, given a set of observations $x_1, \ldots, x_n$:

#### Generate data
``` 
import jax.numpy as jnp
import jax.random as jrnd

key = jrnd.PRNGKey(0)
key, key_data = jrnd.split(key)

true_theta = 0.3
n = 100
x = jrnd.bernoulli(key_data, p=true_theta, shape=(n, ))
```

#### Define Bayesian generative model

Bayesian models in **bamojax** can be instantiated as:

```
from bamojax.base import Model
my_model = Model('The name of the model')
``` 

Subsequently, variables can be added to this model using for example:

```
def link_fn(probs):
    return {'probs': probs}

#
latent_theta = my_model.add_node('theta', distribution=dx.Beta(alpha=1, beta=1))
observations = my_model.add_node('x', distribution=dx.Bernoulli, parents=dict(probs=latent_theta), link_fn, observations=x)
```

#### Perform approximate inference

The next step is to perform inference. Here, we use Adaptive-Tempered Sequential Monte Carlo, as implemented by `Blackjax`:

```
num_particles = 1_000
num_mcmc_steps = 100
num_chains = 1

stepsize = 0.01

mcmc_params = dict(sigma=stepsize*jnp.eye(my_model.get_model_size()))
rmh = mcmc_sampler(my_model, mcmc_kernel=blackjax.normal_random_walk, mcmc_parameters=mcmc_params)

key, key_inference = jrnd.split(key)
final_state, lml, n_iter, final_info = smc_inference_loop(key_inference, model=my_model, kernel=rmh, num_particles=num_particles, num_mcmc_steps=num_mcmc_steps, num_chains=1)

print(jnp.mean(final_state.particles['theta']))
>>> 0.2254779304949821
```

## Modelling in bamojax

**bamojax** aims to make Bayesian modelling and inference with JAX and the Blackjax library more user-friendly, while retaining the speed and flexibility of the latter. To use the framework, a user is required to specify the directed acyclic graph that defines a Bayesian probabilistic model, as well as 
* Distributions,
* Transformations,
* Link functions, and
* Observations.

### The model

To perform these steps, one first initializes the `Model`:

```
my_model = Model('The name of my model')
```

This effectively creates an empty DAG. 

### The variables

Next, one adds nodes to the DAG one by one, or multiple at once in the case of iteration / plate notation: For instance, the following adds a latent random variable with a Gaussian distribution assigned to it, $x\sim \mathcal{N}(0,1)$:

```
a_new_node = my_model.add_node('variable name', distribution=dx.Normal, parents=dict(loc=0.0, scale=1.0))
```

Depending on whether a user provides values for the `distribution=`, and/or `observations=` arguments in the `add_node()` method, **bamojax** derives whether the variable is:
1. Stochastic and latent,
2. Stochastic and observed,
3. Deterministic and observed.

The `parents` argument expects a dictionary, in which the keys must correspond to either the arguments of a `dx.Distribution` object or the arguments of a link function.

### Link functions

An important 'feature' of **bamojax** is that it is straightforward to add any deterministic transformation from the values of a parent variable to the inputs of a child variable. For example, imagine a variable $\theta$ representing a coin flip probability. A typical prior would be the beta distribution, typically parametrized with pseudo-counts $\alpha$ and $\beta$, so that $\theta \sim \text{Beta}(\alpha,\beta)$.  However, we may want to specify a hierarchical prior on the _mode_ and _precision_ of this distribution, rather than on the pseudo-counts. With a link function we can express this:

```
def beta_link_fn(mode, conc):
  a = mode*(conc-2) + 1
  b = (1 - mode)*(conc-2) + 1
  return {'alpha': a, 'beta': b}

omega = my_model.add_node('omega', distribution=dx.Beta, parents=dict(alpha=1.0, beta=1.0))
theta = my_model.add_node('theta', distribution=dx.Beta, parents=dict(mode=omega, conc=15), link_fn=beta_link_fn)
```

The link function `beta_link_fn` takes the mode (given by another node `omega`) and the concentration (given by a scalar, which is implicitly converted to a deterministic and observed node), and returns the standard arguments $\alpha$ and $\beta$ which the `dx.Beta` distribution object recognizes as valid parameters.

Because link functions are written in play Python, they can be of arbitrary complexity. For example, in [bamojax/examples/bnn/bnn_mpl.ipynb](https://github.com/UncertaintyInComplexSystems/bamojax/blob/main/bamojax/examples/bnn/bnn_mpl.ipynb), we use [Flax Linen](https://flax.readthedocs.io/en/latest/) to set up a multilayer perceptron as a link function.

### Inference

**bamojax** does not have its own inference engine, but provides an interface to [Blackjax](https://blackjax-devs.github.io/blackjax/), in addition to some quality-of-life features. Ultimately, control is left entirely to the user. Here is an example where we use Gibbs MCMC:

```
step_fns = dict(beta=normal_random_walk, sigma=normal_random_walk)
step_fn_params = dict(omega=dict(sigma=0.05), theta=dict(sigma=0.5))
gibbs_kernel = gibbs_sampler(my_model, step_fns=step_fns, step_fn_params=step_fn_params)
```

In Gibbs sampling, we specify a kernel for each variable. In this case, we use Gaussian proposal distributions for both $\omega$ and $\theta$.

Now we can set up our inference engine:

```
engine = MCMCInference(model=my_model, num_chains=4, num_samples=100_000, num_burn=100_000, num_thin=50, mcmc_kernel=gibbs_kernel, return_diagnostics=True)
result = engine.run(jrnd.PRNGKey(0))
```

This returns a dictionary with a `states` value containing a dictionary samples for each variable, for the requested number of chains and samples, discarding the specified number of burn-in samples, and storing only every 50th sample. By setting `return_diagnostics=True`, information such as acceptance rates are provided as well. For large models, turning this off can conserve memory consumption.

Alternative inference engines include Sequential Monte Carlo, Variational Inference, and Stochastic-Gradient MCMC methods. Examples on how to use these can be found in the `examples/` folder, as well as on the [Uncertainty in Complex Systems website](https://mhinne.github.io/uncertainty-in-complex-systems).

### Predictions

**bamojax** supports each combination of sampling from the prior or posterior, and the latent variables or the predictive distribution, using any of the following:

|                     |Prior                                         |Posterior                                                                         |
|---------------------|----------------------------------------------|----------------------------------------------------------------------------------|
|**Latent variables** |`my_model.sample_prior(key)`                  |Using `InferenceEngine`                                                           |
|**Predictive**       |`my_model.sample_prior_predictive(key)`       |`my_model.sample_posterior_predictive(key, posterior_samples, input_variables)`   |

When sampling from the posterior predictive, the parameter `input_variables=` can be used to provide for example predictor values, such as when sampling from a regression model or a neural network; $p(y^* \mid X, Y, x^*)$.


## Citing bamojax

To cite **bamojax**, please use

```
@misc{bamojax2025,
  author = {Max Hinne},
  year = {2025},
  title = {{Bamojax: Bayesian modelling in JAX}}
  howpublished  = {\url{https://doi.org/10.5281/zenodo.15038847}}
}
```