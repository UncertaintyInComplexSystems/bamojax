***THIS REPOSITORY IS WORK-IN-PROGRESS AND COMES WITHOUT ANY WARRANTEES***

# BaMoJax

Welcome to BaMoJax, the Bayesian modelling toolbox implemented using the Jax coding universe. BaMoJax is a simply (and currently limited) probabilistic programming language, similar to Numpyro, PyMC, Stan, JAGS, and BUGS. It primarily relies on [Blackjax](https://blackjax-devs.github.io/blackjax/) for approximate inference, on [Distrax](https://github.com/google-deepmind/distrax) for probability distributions and their essential operations. It adds to this the Directed Acyclic Graph (DAG) structure of a Bayesian model, and automatically derives model priors, likelihoods, and Gibbs inference schemes. 

## Installation

In the future, BaMoJax will be fully pip-installable. For now, we recommend cloning the repository manually. 

BaMoJax has been developed with Jaxlib 0.4.34 and Jax 0.4.35. For installation of the correct dependencies, we recommend these steps:

``
conda install jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia
pip install -U "jax[cuda12]"
``

After that, the other dependencies can be installed straightforwardly with pip.

## Dependencies

BaMoJax is being developed on the following configuration:
- Python 3.10.15
- Jax 0.4.35
- Jaxlib 0.4.34
- Blackjax 1.2.4
- Distrax 0.1.5


## Quick tutorial

## Citing BaMoJax

