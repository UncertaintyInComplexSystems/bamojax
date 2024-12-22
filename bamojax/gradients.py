import jax
from typing import Callable
from blackjax.types import ArrayLikeTree, ArrayTree

"""
DISCLAIMER: this code is adapted from Blackjax.

"""

def logdensity_estimator(
    logprior_fn: Callable, loglikelihood_fn: Callable, data_size: int
) -> Callable:
    """Builds a simple estimator for the log-density.

    This estimator first appeared in :cite:p:`robbins1951stochastic`. The `logprior_fn` function has a
    single argument:  the current position (value of parameters). The
    `loglikelihood_fn` takes two arguments: the current position and a batch of
    data; if there are several variables (as, for instance, in a supervised
    learning contexts), they are passed in a tuple.

    This algorithm was ported from :cite:p:`coullon2022sgmcmcjax`.

    Parameters
    ----------
    logprior_fn
        The log-probability density function corresponding to the prior
        distribution.
    loglikelihood_fn
        The log-probability density function corresponding to the likelihood.
    data_size
        The number of items in the full dataset.

    """

    def logdensity_estimator_fn(
        position: ArrayLikeTree, minibatch: ArrayLikeTree
    ) -> ArrayTree:
        """Return an approximation of the log-posterior density.

        Parameters
        ----------
        position
            The current value of the random variables.
        batch
            The current batch of data

        Returns
        -------
        An approximation of the value of the log-posterior density function for
        the current value of the random variables.

        """
        logprior = logprior_fn(position)
        return logprior + loglikelihood_fn(position, minibatch)

    return logdensity_estimator_fn


def grad_estimator(
    logprior_fn: Callable, loglikelihood_fn: Callable, data_size: int
) -> Callable:
    """Build a simple estimator for the gradient of the log-density."""

    logdensity_estimator_fn = logdensity_estimator(
        logprior_fn, loglikelihood_fn, data_size
    )
    return jax.grad(logdensity_estimator_fn)