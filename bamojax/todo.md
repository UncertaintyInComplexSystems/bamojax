
# Functionality

- Add an autoregressive distribution factory to ensure efficient updates of autoregressive parts of the model? Check how PyMC handles this!
- Implement Variational Inference with a distrax distribution for $q$.
- Try Variational Gibbs Inference: https://jmlr.org/papers/v24/21-1373.html.
- Port marginal likelihood approximators from `uicsmodels`; add others such as path integration and AIS.
- Automated tuning loops, also within SMC.

# Examples

- GMM, GrMM
- Repeated measures GPs and GWPs

# Fixes

- Model size does not always consider shaped arrays correctly.
- Implement burn-in and thinning in-situ rather than post-hoc, to conserve memory.
- Make sure link function arguments, node parents, and minibatch-indexing all properly use node.name