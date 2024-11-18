
# Functionality

- Add prediction functionality to GP/GWP
- Make info object optional, as it takes quite some memory
- Investigate memory consumption more generally (also interacts with stepsizes as a wrong setting requires too many SMC iterations, which are semi-dynamically allocated)
- Make some observed nodes 'batchable', and derive loglikeihood_fn accordingly (for stochastic gradient methods)
- Fix the temperature parameter addition for when using SMC with kernels other than Gibbs!
  * Blackjax changed a whole lot to the approach SMC can be called, making it hard to generalize to other kernels! This is not too high of a priority given 
  that we mostly use it with Gibbs, but would be needed to verify results. Basically, Blackjax now creates a new MCMC kernel in every SMC loop, with a new 
  tempered logdensity to sample from. This means we cannot simply pass the step function, but need to pass the MCMC kernel and _derive_ the step function instead.
  First, we would make a kernel q(new_state|state), then apply new_state = q.step(key, state). Now we must create a kernel q = build_kernel(key, state, tempered_density), 
  and then SMC derives the q.step function itself.
- Add an autoregressive distribution factory to ensure efficient updates of autoregressive parts of the model? Check how PyMC handles this!

# Examples

- GMM, GrMM
- EV