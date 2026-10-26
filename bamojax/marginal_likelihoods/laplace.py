
from jaxtyping import Float
from bamojax.inference import LaplaceInference  # local import avoids a circular import at package load
from bamojax.base import Model

def laplace_approximation(key,
                          model: Model,
                          iid_obs: bool = True,
                          bounds: tuple = None,
                          **engine_args) -> Float:
    r"""Compute the Laplace approximation of the log marginal likelihood of `model`.

    The Laplace approximation replaces the posterior by a Gaussian centred at the mode, with the
    curvature of the negative log posterior as precision. The normalising constant of that Gaussian
    is known in closed form and serves as the marginal likelihood estimate.

    This is a thin functional wrapper around `bamojax.inference.LaplaceInference`, which performs the
    optimisation with optax (L-BFGS by default) in unconstrained space, so latent variables with
    transformed priors are handled correctly.

    Args:
        key: PRNGKey, used to draw the starting point of the optimisation from the prior.
        model: A bamojax model.
        iid_obs: Kept for backwards compatibility; `Model.loglikelihood_fn` already sums over observations.
        bounds: Optional (lower, upper) tuple of dicts with box constraints (see `LaplaceInference`).
        **engine_args: Passed on to `LaplaceInference` (e.g. `max_iter`, `tol`, `optimizer`).

    Returns:
        The log marginal likelihood estimate.

    See https://en.wikipedia.org/wiki/Laplace%27s_approximation

    """
    del iid_obs   

    engine = LaplaceInference(model, bounds=bounds, **engine_args)
    return engine.run(key)['lml']

#

