
# Functionality

- Add prediction functionality to GP/GWP
- Make info object optional, as it takes quite some memory
- Investigate memory consumption more generally (also interacts with stepsizes as a wrong setting requires too many SMC iterations, which are semi-dynamically allocated).
- Make some observed nodes 'batchable', and derive loglikelihood_fn accordingly (for stochastic gradient methods)
- Add an autoregressive distribution factory to ensure efficient updates of autoregressive parts of the model? Check how PyMC handles this!
- Use forked `jaxkern` repository for GP kernels, and see if the random Fourier features approximation works out of the box.
- Implement Variational Inference with a distrax distribution for $q$.
- Try Variational Gibbs Inference: https://jmlr.org/papers/v24/21-1373.html.

# Examples

- GMM, GrMM
- EV

# Fixes

- Model size does not always consider shaped arrays correctly.