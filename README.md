[![Tests](https://github.com/uncertaintyincomplexsystems/bamojax/actions/workflows/test.yml/badge.svg)](https://github.com/uncertaintyincomplexsystems/bamojax/actions/workflows/test.yml)

![bamojax logo](https://github.com/uncertaintyincomplexsystems/bamojax/blob/main/figures/bamojax_logo_2x.png "bamojax")

# Welcome to bamojax!

Welcome to **bamojax**, the Bayesian modelling toolbox implemented using the Jax coding universe. **bamojax** is a probabilistic programming language (PPL), similar to Numpyro, PyMC, Stan, JAGS, and BUGS. It relies on [Blackjax](https://blackjax-devs.github.io/blackjax/) for approximate inference, on [Distrax](https://github.com/google-deepmind/distrax) for probability distributions and their essential operations. 

## What sets bamojax apart?

Existing PPLs, such as PyMC, can export their log density function so it can be sampled using Blackjax. However, this has two downsides:

1. Not all existing probabilistic programming easily allow you to update variables in your model using Gibbs Markov chain Monte Carlo (MCMC) algorithms. For example, if one wants to approximate the posterior over a latent Gaussian process (GP) and its hyperparameters, it can be more efficient to use elliptical slice sampling for the GP, than to apply the No-U-Turn Hamiltonian Monte Carlo Sampler [(Hoffman and Gelman, 2014)](https://jmlr.org/papers/v15/hoffman14a.html) to all variables at once. This effect is even more pronounced when embedding MCMC sampling within Sequential Monte Carlo (see [Hinne, 2025](https://link.springer.com/article/10.3758/s13428-025-02642-1) for more details).
2. It is harder to embed this logdensity into tempered Sequential Monte Carlo algorithms, as for this one needs access to the prior and likelihood separately. 

By implementing your own models and samplers using Blackjax, these problems can be circumvented. However, this is a labour-intensive and error-prone process. Therefore, **bamojax** provides a user-friendly interface around Blackjax, that allows for easy model construction and Gibbs sampling. 

## Installation

Install **bamojax** using 

```
pip install git+https://github.com/UncertaintyInComplexSystems/bamojax#egg=bamojax
```

Please note that **bamojax** has been developed and tested with `Jaxlib` 0.4.34, `Jax` 0.4.35, and Python 3.10. Since the installation for jaxlib depends on your hardware, these dependencies are not automatically installed when setting up **bamojax**.

## Quick tutorial

Let's estimate the latent probability $\theta=p(x_1=\text{heads} \mid \theta)$ that a coin lands up heads, given a set of observations $x_1, \ldots, x_n$:

#### Generate data
``` 
import jax.numpy as jnp
import jax.random as jrnd
from bamojax.base import Model
from bamojax.inference import SMCInference
from bamojax.samplers import mcmc_sampler
import distrax as dx
import blackjax as bjx

key = jrnd.PRNGKey(0)
key, key_data, key_inference = jrnd.split(key, 3)

true_theta = 0.3
n = 10000
x = jrnd.bernoulli(key_data, p=true_theta, shape=(n, ))
```

#### Define Bayesian generative model

Under the good, **bamojax** reprsents the a Bayesian model using a Directed Acyclic Graph (DAG) structure, and automatically derives model priors, likelihoods, and Gibbs inference schemes. These are then combined with the fast inference algorithms implemented in Blackjax. Bayesian models can be instantiated in **bamojax** using:

```
from bamojax.base import Model
my_model = Model('The name of the model')
``` 

Subsequently, variables can be added to this model using for example:

```
def link_fn(probs):
    return {'probs': probs}

#
latent_theta = my_model.add_node('theta', 
                                 distribution=dx.Beta(alpha=1, beta=1))
observations = my_model.add_node('x', 
                                 distribution=dx.Bernoulli, 
                                 parents=dict(probs=latent_theta), 
                                 link_fn=link_fn, 
                                 observations=x)
```

#### Perform approximate inference

The next step is to perform inference. Here, we use Adaptive-Tempered Sequential Monte Carlo, as implemented by Blackjax:

```
num_particles = 1_000
num_mcmc_steps = 100
num_chains = 1

stepsize = 0.01

mcmc_params = dict(sigma=stepsize*jnp.eye(my_model.get_model_size()))
rmh = mcmc_sampler(my_model, 
                   mcmc_kernel=bjx.normal_random_walk, 
                   mcmc_parameters=mcmc_params)

key, key_inference = jrnd.split(key)

engine = SMCInference(model=my_model, 
                      mcmc_kernel=rmh, 
                      num_particles=num_particles, 
                      num_mutations=num_mcmc_steps, 
                      num_chains=num_chains)
result = engine.run(key_inference)


print(jnp.mean(result['final_state'].particles['theta']))            
       
>>> 0.29956531188263924
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

The `parents` argument expects a dictionary, in which the keys must correspond to either the arguments of a Distrax `dx.Distribution` object or the arguments of a link function.

Nodes can furthermore take any Distrax bijector using the `bijector=` argument, for example to transform a real variable to a bounded domain or vice versa. In many cases, the same goal can be achieved by the link function, but sometimes a bijector is simpler to use.

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

Because link functions are written in plain Python, they can be of arbitrary complexity. For example, in [bamojax/examples/bnn/bnn_mpl.ipynb](https://github.com/UncertaintyInComplexSystems/bamojax/blob/main/bamojax/examples/bnn/bnn_mpl.ipynb), we use [Flax Linen](https://flax.readthedocs.io/en/latest/) to set up a multilayer perceptron as a link function.

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

## API

The full **bamojax** API can be found here: https://uncertaintyincomplexsystems.github.io/bamojax/.

## Citing bamojax

If you use Bamojax in your work, please cite the following:

    Max Hinne. (2025). Bamojax: Bayesian Modelling with JAX (Version 0.1.0) [Computer software]. https://doi.org/10.5281/zenodo.15038847

BibTeX:

```
@software{hinne2025bamojax,
  author       = {Hinne, Max},
  title        = {Bamojax: Bayesian Modelling with JAX},
  year         = 2025,
  publisher    = {Zenodo},
  version      = {0.1.0},
  doi          = {10.5281/zenodo.15038847},
  url          = {https://github.com/maxhinne/bamojax}
}
```
