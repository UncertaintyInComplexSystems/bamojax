
# Functionality

- Add prediction functionality to GP/GWP
- Make info object optional, as it takes quite some memory
- Investigate memory consumption more generally (also interacts with stepsizes as a wrong setting requires too many SMC iterations, which are semi-dynamically allocated)
- Make some observed nodes 'batchable', and derive loglikeihood_fn accordingly (for stochastic gradient methods)
- Add an autoregressive distribution factory to ensure efficient updates of autoregressive parts of the model? Check how PyMC handles this!
- Can Jax derive gradients through the list comprehension in the `gibbs_sampler`?

# Examples

- GMM, GrMM
- EV

# Fixes

- Model size does not always consider shaped arrays correctly.