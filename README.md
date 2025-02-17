# bamojax

Welcome to bamojax, the Bayesian modelling toolbox implemented using the Jax coding universe. bamojax is a probabilistic programming language, similar to Numpyro, PyMC, Stan, JAGS, and BUGS. It relies on [Blackjax](https://blackjax-devs.github.io/blackjax/) for approximate inference, on [Distrax](https://github.com/google-deepmind/distrax) for probability distributions and their essential operations. It adds to this the Directed Acyclic Graph (DAG) structure of a Bayesian model, and automatically derives model priors, likelihoods, and Gibbs inference schemes. This combines the speed from (Black)Jax with a convenient modelling environment.

## Installation

Clone the repository using 

```
git clone https://github.com/UncertaintyInComplexSystems/bamojax
```

bamojax has been developed with `Jaxlib` 0.4.34, `Jax` 0.4.35, and Python 3.10.15. For installation of the correct dependencies, we recommend to first create a new Conda environment, and install `jax` and `jaxlib` using:

``` 
conda install jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia
pip install -U "jax[cuda12]"
``` 

After that, the other dependencies can be installed straightforwardly with pip:

```
pip install jaxtyping==0.2.34 distrax==0.1.5 blackjax==1.2.4
```

## Quick tutorial

For this simple tutorial we estimate the latent probability $\theta$ a coin lands up heads, given a set of observations $x_1, \ldots, x_n$:

### Generate data
``` 
import jax.numpy as jnp
import jax.random as jrnd

key = jrnd.PRNGKey(0)
key, key_data = jrnd.split(key)

true_theta = 0.3
n = 100
x = jrnd.bernoulli(key_data, p=true_theta, shape=(n, ))
```

### Define Bayesian generative model

Bayesian models in bamojax can be instantiated as:

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
observations = my_model.add_node('x', distributions=dx.Bernoulli, parents=dict(probs=latent_theta), link_fn, observations=x)
```

The `link_fn` shows how link functions can be defined and used within `bamojax`. Here, since `Distrax` supports either probabilities or logits as input for the Bernoulli distribution, we use the link function to call the `dx.Bernoulli` with the correct variable names.

### Do inference

The next step is to perform inference. Here, we use Adaptive-Tempered Sequential Monte Carlo, as implemented by `Blackjax`:

```
num_particles = 100_000
num_mcmc_steps = 1_000
num_chains = 1

stepsize = 0.01

mcmc_params = dict(sigma=stepsize*jnp.eye(my_model.get_model_size()))
rmh = mcmc_sampler(my_model, mcmc_kernel=blackjax.normal_random_walk, mcmc_parameters=mcmc_params)

key, key_inference = jrnd.split(key)
n_iter, final_state, lml = smc_inference_loop(key_inference, model=my_model, kernel=rmh, num_particles=num_particles, num_mcmc_steps=num_mcmc_steps, num_chains=1)

print(jnp.mean(final_state.particles['theta']))
```


## Citing bamojax

TODO.
