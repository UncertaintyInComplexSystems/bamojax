r""" Anchored reversible jump sequential Monte Carlo (RJSMC).

See Van Assen & Hinne, Anchored Reversible Jump Sequential Monte Carlo, https://arxiv.org/abs/2610.09930.

Every particle is a *super-particle* containing a variable that indicates the current 'active' model, 
as well as the parameters of all models in the search. Only the active model is used in

    p_beta(k, theta_k) \propto  p(theta_k | k) p(y | theta_k, k)^beta        (uniform prior over models),

with beta the SMC temperature. The inactive parameters are used together with kernel density estimation (KDE)
to create an informed trans-dimensional proposal distribution.
    
One iteration = reweight (adaptive temperature) -> resample -> optional within-model MCMC -> RJ move.

This implementation was ported from its original to bamojax by Claude.AI. It follows the code philosophy of
Blackjax.

"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jrnd
from jax.flatten_util import ravel_pytree
from jax.scipy.special import logsumexp
from jax.scipy.stats import gaussian_kde
from blackjax.smc.resampling import systematic

from bamojax.base import Model
from bamojax.inference import InferenceEngine
from bamojax.samplers import mcmc_sampler


class RJSMCState(NamedTuple):
    positions: tuple        # K pytrees (one per model), leaves have leading dimension N
    model_index: jax.Array  # (N,) active model of each particle
    lmbda: float            # current temperature


def _nan_to_neginf(x):
    return jnp.where(jnp.isnan(x), -jnp.inf, x)


def _pick(matrix, idx):
    """ matrix[i, idx[i]] for every row i. """
    return matrix[jnp.arange(matrix.shape[0]), idx]


class RJSMCInference(InferenceEngine):
    r""" Anchored RJSMC over a list of bamojax models.

    Args:
        models: list of K bamojax models (uniform prior over models).
        num_particles: number of super-particles N (preferably a multiple of K).
        within_model_kernels: optional list of K dicts(mcmc_kernel=..., mcmc_parameters=...), as in
            `reversible_jump_mcmc`; if None, particles move only through the RJ (KDE) proposals.
        num_mutations: within-model MCMC steps per iteration.
        grace: probability of proposing a model uniformly at random; otherwise a model is proposed
            proportionally to its current particle count.
        anchor_weight: KDE weight of particles for which the block is the active one (others get 1).
        target_ess: the temperature increment keeps the normalised ESS of every model's block >= target_ess.
        max_iter: maximum number of tempering iterations (also the length of the trace buffers).

    """

    def __init__(self,
                 models: list[Model],
                 num_particles: int,
                 within_model_kernels: list[dict] = None,
                 num_mutations: int = 0,
                 grace: float = 0.5,
                 anchor_weight: float = 1.0,
                 target_ess: float = 0.9,
                 num_chains: int = 1,
                 max_iter: int = 200):
        super().__init__(models, num_chains)
        self.models = models
        self.K = len(models)
        self.N = num_particles
        self.grace = grace
        self.anchor_weight = anchor_weight
        self.target_ess = target_ess
        self.max_iter = max_iter
        self.num_mutations = num_mutations
        self.logprior_fns = [m.logprior_fn() for m in models]
        self.loglikelihood_fns = [m.loglikelihood_fn() for m in models]
        self.kernels = None
        if within_model_kernels is not None and num_mutations > 0:
            self.kernels = [mcmc_sampler(m, **kw) for m, kw in zip(models, within_model_kernels)]

    #
    def loglikelihoods(self, positions):
        """ (N, K) matrix: log-likelihood of every block under its own model. """
        return jnp.stack([_nan_to_neginf(jax.vmap(f)(pos)) for f, pos in zip(self.loglikelihood_fns, positions)], axis=1)

    #
    def log_target(self, j, pos, lmbda):
        """ Tempered log density of model j for a batch of blocks. """
        f = lambda p: lmbda * self.loglikelihood_fns[j](p) + self.logprior_fns[j](p)
        return _nan_to_neginf(jax.vmap(f)(pos))

    #
    def next_temperature(self, loglik, lmbda):
        """ Largest increment such that the normalised ESS of every model's block stays >= target_ess. """
        def min_ess(delta):
            lw = delta * loglik
            log_ess = 2 * logsumexp(lw, axis=0) - logsumexp(2 * lw, axis=0)
            return jnp.min(jnp.exp(log_ess)) / self.N

        def bisect(_, bounds):
            lo, hi = bounds
            mid = 0.5 * (lo + hi)
            ok = min_ess(mid) >= self.target_ess
            return jnp.where(ok, mid, lo), jnp.where(ok, hi, mid)

        lo, _ = jax.lax.fori_loop(0, 50, bisect, (0.0, 1.0 - lmbda))
        return jnp.where(min_ess(1.0 - lmbda) >= self.target_ess, 1.0 - lmbda, lo)

    #
    def mutate(self, key, positions, lmbda):
        """ num_mutations within-model MCMC steps on every block, each at the current temperature. """
        def mutate_model(j, key, pos):
            kernel = self.kernels[j]

            def one_particle(key, p):
                def body(state, key):
                    state, _ = kernel.step(key, state, temperature=lmbda)
                    return state, None
                state, _ = jax.lax.scan(body, kernel.init(p), jrnd.split(key, self.num_mutations))
                return state.position

            return jax.vmap(one_particle)(jrnd.split(key, self.N), pos)

        keys = jrnd.split(key, self.K)
        return tuple(mutate_model(j, keys[j], pos) for j, pos in enumerate(positions))

    #
    def rj_move(self, key, state):
        """ Anchored jump: propose a model l, draw theta_l from the KDE of model l, accept/reject. """
        N, K, k = self.N, self.K, state.model_index
        key_mix, key_unif, key_pick, key_kde, key_acc = jrnd.split(key, 5)

        # model proposal j(k -> l) = (1 - grace) m_l / N + grace / K  (does not depend on k)
        counts = jnp.bincount(k, length=K)
        explore = jrnd.bernoulli(key_mix, self.grace, (N,))
        l = jnp.where(explore, jrnd.randint(key_unif, (N,), 0, K), k[jrnd.randint(key_pick, (N,), 0, N)])
        log_j = jnp.log((1 - self.grace) * counts / N + self.grace / K)

        proposals, cur, prop = [], [], []
        for j, (pos, key_j) in enumerate(zip(state.positions, jrnd.split(key_kde, K))):
            _, unravel = ravel_pytree(jax.tree.map(lambda x: x[0], pos))
            flat = jax.vmap(lambda p: ravel_pytree(p)[0])(pos)                 # (N, d_j)
            kde = gaussian_kde(flat.T, weights=jnp.where(k == j, self.anchor_weight, 1.0))
            new_flat = kde.resample(key_j, (N,)).T
            new_pos = jax.vmap(unravel)(new_flat)
            proposals.append(new_pos)
            cur.append(self.log_target(j, pos, state.lmbda) - kde.logpdf(flat.T))
            prop.append(self.log_target(j, new_pos, state.lmbda) - kde.logpdf(new_flat.T))
        cur, prop = jnp.stack(cur, axis=1), jnp.stack(prop, axis=1)          # (N, K)

        log_alpha = _nan_to_neginf(log_j[k] - log_j[l] + _pick(prop, l) - _pick(cur, k))
        accept = jnp.log(jrnd.uniform(key_acc, (N,))) < log_alpha

        def replace(j, old, new):
            mask = accept & (l == j)
            return jax.tree.map(lambda o, n: jnp.where(mask.reshape((N,) + (1,) * (o.ndim - 1)), n, o), old, new)

        positions = tuple(replace(j, old, new) for j, (old, new) in enumerate(zip(state.positions, proposals)))
        cross = l != k
        new_state = RJSMCState(positions, jnp.where(accept, l, k), state.lmbda)
        return new_state, jnp.sum(cross), jnp.sum(accept & cross)

    #
    def init(self, key) -> RJSMCState:
        key_perm, key_sample = jrnd.split(key)
        positions = tuple(jax.vmap(m.sample_prior)(jrnd.split(k, self.N))
                          for m, k in zip(self.models, jrnd.split(key_sample, self.K)))
        model_index = jrnd.permutation(key_perm, jnp.arange(self.N) % self.K)  # balanced = uniform model prior
        return RJSMCState(positions, model_index, 0.0)

    #
    def step(self, key, state):
        """ One RJSMC iteration. Returns the new state, the log evidence increment and jump statistics. """
        key_rs, key_mut, key_rj = jrnd.split(key, 3)
        loglik = self.loglikelihoods(state.positions)
        delta = self.next_temperature(loglik, state.lmbda)
        log_w = _nan_to_neginf(delta * _pick(loglik, state.model_index))
        lml_increment = logsumexp(log_w) - jnp.log(self.N)

        idx = systematic(key_rs, jax.nn.softmax(log_w), self.N)
        positions = jax.tree.map(lambda x: x[idx], state.positions)
        lmbda = state.lmbda + delta
        if self.kernels is not None:
            positions = self.mutate(key_mut, positions, lmbda)
        state = RJSMCState(positions, state.model_index[idx], lmbda)
        state, n_attempted, n_accepted = self.rj_move(key_rj, state)
        return state, lml_increment, n_attempted, n_accepted

    #
    def run_single_chain(self, key) -> dict:
        key_init, key = jrnd.split(key)
        state = self.init(key_init)
        trace = dict(lmbda=jnp.zeros(self.max_iter), model_probs=jnp.zeros((self.max_iter, self.K)))

        def cond(carry):
            i, state, *_ = carry
            return (state.lmbda < 1.0) & (i < self.max_iter)

        def body(carry):
            i, state, key, lml, n_att, n_acc, trace = carry
            key, subkey = jrnd.split(key)
            state, lml_inc, att, acc = self.step(subkey, state)
            trace = dict(lmbda=trace['lmbda'].at[i].set(state.lmbda),
                         model_probs=trace['model_probs'].at[i].set(jnp.bincount(state.model_index, length=self.K) / self.N))
            return i + 1, state, key, lml + lml_inc, n_att + att, n_acc + acc, trace

        carry = (0, state, key, 0.0, 0, 0, trace)
        n_iter, state, _, lml, n_att, n_acc, trace = jax.lax.while_loop(cond, body, carry)
        model_probs = jnp.bincount(state.model_index, length=self.K) / self.N
        return dict(n_iter=n_iter,
                    final_state=state,
                    model_probs=model_probs,
                    lml=lml,                                                   # log p(y), uniform model prior
                    lml_per_model=jnp.log(model_probs) + lml + jnp.log(self.K),  # log p(y | k) = log p(k | y) + log p(y) - log p(k)
                    rj_acceptance_rate=n_acc / jnp.maximum(n_att, 1),
                    trace=trace)

    #

#
def get_model_samples(result: dict, model_index: int) -> dict:
    """ Posterior draws of one model: the blocks of the particles for which that model is active. """
    state = result['final_state']
    mask = state.model_index == model_index
    return jax.tree.map(lambda x: x[mask], state.positions[model_index])