---
title: 'Bamojax: Bayesian Modelling with JAX'
tags:
  - Python
  - Jax
  - Bayesian inference
  - Bayesian modelling
authors:
  - name: Max Hinne
    orcid: 0000-0002-9279-6725
    equal-contrib: true
    affiliation: 1
affiliations:
 - name: Radboud University, Nijmegen, The Netherlands
   index: 1
date: 13 March 2025
bibliography: paper.bib

---

# Summary

Bayesian statistics offers a principled and elegant framework for inferring
hidden causes from observed effects. It also provides a rigorous approach to 
hypothesis testing (model comparison), with advantages such as built-in complexity
penalties, and the ability to quantify evidence in favour of the null hypothesis.

However, exact Bayesian inference is computationally intractable in all but the
simplest of cases, and requires _approximate inference_ techniques, such as 
Markov chain Monte Carlo (MCMC) and variational inference. Recent advances in the Python
JAX [@jax] framework have enabled highly efficient implementations of these algorithms,
due to features such as automated differentation and GPU acceleration. These 
developments have the potential to greatly increase the efficiency of statistical 
modelling pipelines.

``bamojax`` (`Bayesian Modelling in Jax') is a probabilistic programming language (PPL) that combines ease-of-use with access to advanced inference algorithms implemented in the Jax ecosystem.


# Statement of need

**Bamojax** is a Bayesian modelling tool based on Python & JAX [@jax]. It
provides an intuitive, intermediate-level interface between defining a Bayesian 
statistical model conceptually, and performing efficient inference using the 
BlackJAX package [@blackjax]. 

Existing probabilistic programming languages, such as PyMC [@pymc], can export (the logarithm of) a probability density function that enables BlackJAX-based inference. However, this has two limitations:

1. It does not support Gibbs sampling, where variables are updated individually using their own MCMC kernels. For example, when approximating the posterior over a latent Gaussian process (GP) and its hyperparameters, elliptical slice sampling [@Murray2010] for the GP can be more efficient than applying No-U-Turn Hamiltonian Monte Carlo (NUTS HMC; [@Hoffman2014]) sampling  to all variables jointly. This becomes even more important when embedding MCMC sampling in Sequential Monte Carlo [@Hinne2025].
2. It makes it harder to apply tempered Sequential Monte Carlo methods that need separate prior and likelihood terms.

While users can circumvent these issues by manually implementing their models using BlackJAX,  this is a labor-intensive and error-prone process. **Bamojax** addresses this gap by providing a user-friendly interface for model construction and Gibbs sampling on top of BlackJAX.

In **Bamojax**, users can define a probabilistic model by specifying variables 
as well as their associated distributions and dependencies, structured using a 
directed acyclic graph (DAG). Under the hood, **Bamojax** translates this DAG and 
collection of probability distributions to the probability densities used in the
approximate inference, leveraging the probability definitions defined in distrax 
[@distrax]. This abstraction allows users to focus on the conceptual model 
formulation, rather than the mathematical or inference details, leading to a more
intuitive, less error-prone, and more efficient development workflow.

**Bamojax** is designed for researchers, students, and practitioners that want to
make use of the extremely fast approximate inference offered by BlackJAX, but 
want to focus on model development instead of implementation.

## Comparison with existing tools

**Bamojax** is comparable to existing modern probabilistic programming languages, like NumPyro [@numpyro], Oryx [https://github.com/jax-ml/oryx], and PyMC [@PyMC], that use the fast JAX backend for efficient computation of Bayesian inference. The aim of **Bamojax** is to stay close to (JAX) Numpy levels of user control in defining probability densities and transformations, while at the same time providing some level of abstraction to the user, by automatically deriving densities from Distrax primitives and the directed acylcic graph structure of a Bayesian model. This allows for convenient integration of new methods with existing tools.

**Bamojax** easily admits Gibbs sampling, where individual model parameters are updated in turn, which in practice can lead to efficiency gains and sampling of discrete variables. These benefits increase further when Gibbs sampling is used with Sequential Monte Carlo [@Hinne2025], which is straightforward to set up here.

Furthermore, **Bamojax** provides a convenient interface to the different sampling algorithms that are available in BlackJAX, giving the user fine-grained control over the inference strategy of their models. This enables users to mix-and-match
BlackJAX MCMC kernels with elements of their probabilistic model, while maintaining
the efficiency of JAX-based inference.

# Acknowledgements

None at this time.

# References

