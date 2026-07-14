from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
import jax.random as jrnd
import jax.scipy.linalg as jsp_linalg
from blackjax.base import SamplingAlgorithm
from blackjax.types import ArrayTree, PRNGKey

import numpyro.distributions as dist


__all__ = [
    'sample_gwishart',
    'complete_cholesky',
    'drj_step',
    'gwishart_drj',
    'fixed_by_drj',
    'GWishartDRJState',
]


class GWishartDRJState(NamedTuple):
    position: ArrayTree

#

# ----------------------------------------------------------------------------------
# Direct G-Wishart sampler (Lenkoski 2013, Secs. 2.3-2.4), functional & jittable.
# This is a pure-function version of the GWishart.sample implementation in
# bamojax.more_distributions, so it can be called inside a jitted kernel with a
# graph G that changes at runtime (but has fixed shape).
# ----------------------------------------------------------------------------------

def sample_gwishart(key: PRNGKey,
                    G: jnp.ndarray,
                    dof: float,
                    scale: jnp.ndarray,
                    tol: float = 1e-8,
                    max_iter: int = 200) -> jnp.ndarray:
    r"""Draw one sample K ~ W_G(dof, scale) using Lenkoski's direct sampler.

    First samples K* ~ Wishart from the full model, then applies the iterative
    proportional scaling variant of Hastie et al. (2009) (Lenkoski 2013, Sec. 2.4)
    relative to the fixed target Sigma = (K*)^{-1} to place the variate in P_G.

    Args:
        key: JAX PRNG key.
        G: binary adjacency matrix, shape (p, p). Diagonal is ignored.
        dof: G-Wishart degrees of freedom delta, in the parametrization
             p(K) \propto |K|^{(delta-2)/2} exp(-tr(K scale)/2).
        scale: scale matrix D, shape (p, p).
        tol: convergence tolerance of the IPS-style iteration.
        max_iter: maximum number of sweeps.
    Returns:
        A draw K in P_G, shape (p, p), with exact zeros at non-edges.
    """
    p = scale.shape[0]
    scale_inv = jnp.linalg.inv(scale)

    # W_G(delta, D) restricted to the full graph equals Wishart(delta + p - 1, D^{-1})
    # in numpyro's parametrization.
    Kp = dist.Wishart(concentration=dof + p - 1, scale_matrix=scale_inv).sample(key)
    Sigma = jnp.linalg.inv(Kp)

    G_bool = jnp.asarray(G).astype(bool)
    G_no_diag = jnp.logical_and(G_bool, ~jnp.eye(p, dtype=bool))
    idx = jnp.arange(p)

    def one_sweep(W):
        def body(W, j):
            mask_nei = G_no_diag[j]
            m = mask_nei.astype(W.dtype)

            # Solve W_{N_j} beta = Sigma_{N_j, j} in a masked, fixed-size system.
            M_j = W * jnp.outer(m, m) + jnp.diag(1.0 - m)
            rhs_j = Sigma[:, j] * m
            L = jnp.linalg.cholesky(M_j + 1e-10 * jnp.eye(p))
            y = jsp_linalg.solve_triangular(L, rhs_j, lower=True)
            gamma_full = jsp_linalg.solve_triangular(L.T, y, lower=False)
            w_new = W @ gamma_full

            mask_not_j = (idx != j)
            W = W.at[:, j].set(jnp.where(mask_not_j, w_new, W[:, j]))
            W = W.at[j, :].set(jnp.where(mask_not_j, w_new, W[j, :]))
            return W, None

        W, _ = jax.lax.scan(body, W, jnp.arange(p))
        return W

    def cond_fun(state):
        W_prev, W, it = state
        return jnp.logical_and(jnp.max(jnp.abs(W - W_prev)) > tol, it < max_iter)

    def body_fun(state):
        _, W, it = state
        return (W, one_sweep(W), it + 1)

    W1 = one_sweep(Sigma)
    _, W_final, _ = jax.lax.while_loop(cond_fun, body_fun, (jnp.zeros_like(W1), W1, 0))

    W_sym = 0.5 * (W_final + W_final.T)
    G_mask = jnp.maximum(G.astype(W_sym.dtype), jnp.eye(p, dtype=W_sym.dtype))
    K = jnp.linalg.inv(W_sym) * G_mask
    return 0.5 * (K + K.T)

#
# ----------------------------------------------------------------------------------
# Cholesky completion (Roverato 2002; Lenkoski 2013, Eq. 4).
# ----------------------------------------------------------------------------------

def complete_cholesky(Phi: jnp.ndarray, G: jnp.ndarray) -> jnp.ndarray:
    r"""Complete an upper-triangular Cholesky factor with respect to graph G.

    The free elements of Phi are the diagonal and the entries Phi_{ij} with
    (i, j) in G, i < j; these are taken from the input. All remaining
    upper-triangular entries are (re)computed row by row via

        Phi_{ij} = -(1 / Phi_{ii}) * sum_{r < i} Phi_{ri} Phi_{rj},

    which enforces (Phi' Phi)_{ij} = 0 for (i, j) not in G.

    Args:
        Phi: (p, p) matrix whose upper triangle holds correct values at the free
             positions (values elsewhere are ignored and overwritten).
        G: binary adjacency matrix, shape (p, p).
    Returns:
        The completed upper-triangular Cholesky factor.
    """
    p = Phi.shape[0]
    idx = jnp.arange(p)
    G_bool = jnp.asarray(G).astype(bool)
    M0 = jnp.triu(Phi)

    def body(M, i):
        rows_above = (idx < i).astype(M.dtype)[:, None]
        Mm = M * rows_above
        s = Mm.T @ Mm[:, i]                       # s_j = sum_{r<i} Phi_ri Phi_rj
        c = -s / M[i, i]
        not_free = jnp.logical_and(idx > i, ~G_bool[i])
        new_row = jnp.where(not_free, c, M[i])
        return M.at[i].set(new_row), None

    M, _ = jax.lax.scan(body, M0, idx)
    return M

#
def _completion_value(Phi: jnp.ndarray, l, m) -> jnp.ndarray:
    """The value Eq. (4) would assign to Phi[l, m], given the rows above l."""
    p = Phi.shape[0]
    mask = (jnp.arange(p) < l).astype(Phi.dtype)
    return -jnp.dot(mask * Phi[:, l], Phi[:, m]) / Phi[l, l]


#
# ----------------------------------------------------------------------------------
# The double reversible jump move (Lenkoski 2013, Sec. 3.2).
# ----------------------------------------------------------------------------------

def drj_step(key: PRNGKey,
             G: jnp.ndarray,
             dof: float,
             scale: jnp.ndarray,
             n: int,
             U: jnp.ndarray,
             sigma_g: float = 1.0,
             logprior_G_fn: Optional[Callable] = None,
             tol: float = 1e-8,
             max_iter: int = 200):
    r"""One double reversible jump update of (G, K).

    Proposes toggling a uniformly chosen edge e = (l, m), l < m, of the current
    graph G, giving Gt. Both the edge-addition and edge-deletion cases run through
    the same fixed-shape code path:

      1. Refresh K ~ W_G(dof + n, scale + U); Phi = chol(K) (upper triangular).
      2. Draw the auxiliary variate Kt0 ~ W_Gt(dof, scale) from the *prior* under
         the proposed graph; Phi0 = chol(Kt0).
      3. On the posterior side, the free value of the larger graph at (l, m) is
         gamma_post: sampled from N(theta_post, sigma_g^2) when adding, or the
         current Phi[l, m] when deleting; theta_post is the Eq. (4) completion
         value. Completing w.r.t. Gt yields Kt.
      4. On the prior side, the free value gamma_prior is Phi0[l, m] when adding,
         or a fresh draw from N(theta_prior, sigma_g^2) when deleting; completing
         w.r.t. G yields K0.
      5. Accept with probability min(1, alpha), where with s = +1 (add) / -1 (delete):

         log alpha = log p(Gt) - log p(G)
                     - <Kt - K, scale + U> / 2
                     + <Kt0 - K0, scale> / 2
                     + s * (log Phi[l, l] - log Phi0[l, l])
                     + s * ((gamma_post - theta_post)^2
                            - (gamma_prior - theta_prior)^2) / (2 sigma_g^2).

    The G-Wishart normalizing constants I_G(dof, scale) cancel between the
    posterior-side and prior-side reversible jumps, as in the exchange algorithm.
    On acceptance the state moves to (Gt, Kt); on rejection it keeps G with the
    refreshed K (itself a valid conjugate Gibbs draw of K | G, data).

    Args:
        key: PRNG key.
        G: current adjacency matrix, shape (p, p), symmetric binary.
        dof: G-Wishart prior degrees of freedom delta.
        scale: G-Wishart prior scale matrix D, shape (p, p).
        n: number of (zero-mean) Gaussian observations underlying U.
        U: scatter matrix sum_i z_i z_i', shape (p, p).
        sigma_g: std. dev. of the reversible jump proposal on the Cholesky element.
        logprior_G_fn: optional callable G -> log p(G) up to a constant
            (default: uniform over graphs).
        tol, max_iter: settings of the direct G-Wishart sampler.
    Returns:
        (G_new, K_new, info) where info is a dict with 'is_accepted', 'log_alpha',
        'is_add', and the proposed edge indices 'edge_l', 'edge_m'.
    """
    G = jnp.asarray(G)
    p = scale.shape[0]
    rows, cols = jnp.triu_indices(p, k=1)
    num_edges = rows.shape[0]

    key_e, key_k, key_k0, key_ga, key_gb, key_u = jrnd.split(key, 6)

    e = jrnd.randint(key_e, shape=(), minval=0, maxval=num_edges)
    l, m = rows[e], cols[e]
    is_add = 1.0 - G[l, m].astype(scale.dtype)                 # 1: add edge, 0: delete
    s = 2.0 * is_add - 1.0
    new_val = 1 - G[l, m]
    Gt = G.at[l, m].set(new_val).at[m, l].set(new_val)

    post_scale = scale + U

    # --- posterior side: refresh K | G and jump G -> Gt ---
    K = sample_gwishart(key_k, G, dof + n, post_scale, tol=tol, max_iter=max_iter)
    Phi = jnp.linalg.cholesky(K).T                             # upper: Phi' Phi = K
    theta_post = _completion_value(Phi, l, m)
    gamma_post = jnp.where(is_add > 0.5,
                           theta_post + sigma_g * jrnd.normal(key_ga),
                           Phi[l, m])
    Phi_t = complete_cholesky(Phi.at[l, m].set(gamma_post), Gt)
    Kt = Phi_t.T @ Phi_t

    # --- prior side: auxiliary Kt0 ~ W_Gt(dof, scale) and jump Gt -> G ---
    Kt0 = sample_gwishart(key_k0, Gt, dof, scale, tol=tol, max_iter=max_iter)
    Phi0 = jnp.linalg.cholesky(Kt0).T
    theta_prior = _completion_value(Phi0, l, m)
    gamma_prior = jnp.where(is_add > 0.5,
                            Phi0[l, m],
                            theta_prior + sigma_g * jrnd.normal(key_gb))
    Phi_0 = complete_cholesky(Phi0.at[l, m].set(gamma_prior), G)
    K0 = Phi_0.T @ Phi_0

    # --- acceptance ratio (normalizing constants I_G cancel) ---
    def inner(A, B):
        return jnp.sum(A * B)

    log_prior_ratio = 0.0
    if logprior_G_fn is not None:
        log_prior_ratio = logprior_G_fn(Gt) - logprior_G_fn(G)

    log_alpha = (log_prior_ratio
                 - 0.5 * inner(Kt - K, post_scale)
                 + 0.5 * inner(Kt0 - K0, scale)
                 + s * (jnp.log(Phi[l, l]) - jnp.log(Phi0[l, l]))
                 + s * ((gamma_post - theta_post) ** 2
                        - (gamma_prior - theta_prior) ** 2) / (2.0 * sigma_g ** 2))

    accept = jnp.log(jrnd.uniform(key_u)) < log_alpha
    G_new = jnp.where(accept, Gt, G)
    K_new = jnp.where(accept, Kt, K)

    info = {'is_accepted': accept,
            'log_alpha': log_alpha,
            'is_add': is_add,
            'edge_l': l,
            'edge_m': m}
    return G_new, K_new, info

#
# ----------------------------------------------------------------------------------
# Blackjax SamplingAlgorithm factories, in the calling convention that
# bamojax.samplers.gibbs_sampler expects: factory(logdensity_fn, **step_fn_params).
# ----------------------------------------------------------------------------------

def gwishart_drj(logdensity_fn=None,
                 *,
                 dof: float,
                 scale: jnp.ndarray,
                 n: int,
                 U: jnp.ndarray,
                 name: str = 'G',
                 k_name: Optional[str] = 'K',
                 sigma_g: float = 1.0,
                 logprior_G_fn: Optional[Callable] = None,
                 tol: float = 1e-8,
                 max_iter: int = 200) -> SamplingAlgorithm:
    r"""Blackjax ``SamplingAlgorithm`` performing double reversible jump updates of G.

    Designed as a bamojax Gibbs step function: bamojax calls
    ``step_fns[node](logdensity_fn, **step_fn_params[node])``; the generic
    ``logdensity_fn`` is accepted but ignored, because it would require evaluating
    the intractable G-Wishart density p(K | G) -- avoiding exactly that is the point
    of the DRJ construction. All problem structure is passed via keyword arguments.

    If ``k_name`` is given, the kernel also writes the refreshed / accepted precision
    matrix K into the position under that key. bamojax merges the returned position
    into the global Gibbs state, so the K node is updated jointly with G by this
    kernel; assign ``fixed_by_drj`` as the step function of the K node.

    Args:
        logdensity_fn: ignored (present for bamojax/blackjax interface compatibility).
        dof, scale: G-Wishart prior parameters of K | G.
        n, U: number of zero-mean Gaussian observations and their scatter matrix.
        name: position key of the adjacency matrix node.
        k_name: position key of the precision matrix node (None to not emit K).
        sigma_g: reversible jump proposal standard deviation.
        logprior_G_fn: optional callable G -> log p(G); default uniform.
        tol, max_iter: direct sampler settings.
    Returns:
        A ``SamplingAlgorithm`` with the usual ``init(position)`` / ``step(key, state)``.
    """
    del logdensity_fn

    def init_fn(position, rng_key=None):
        del rng_key
        if not isinstance(position, dict):
            position = {name: position}
        return GWishartDRJState(position={name: jnp.asarray(position[name])})

    def step_fn(key: PRNGKey, state, *args, **kwargs):
        G = state.position[name]
        G_new, K_new, info = drj_step(key, G,
                                      dof=dof, scale=scale, n=n, U=U,
                                      sigma_g=sigma_g,
                                      logprior_G_fn=logprior_G_fn,
                                      tol=tol, max_iter=max_iter)
        position = {name: G_new}
        if k_name is not None:
            position[k_name] = K_new
        return GWishartDRJState(position=position), info

    return SamplingAlgorithm(init_fn, step_fn)

#
def fixed_by_drj(logdensity_fn=None, **kwargs) -> SamplingAlgorithm:
    """No-op step function for a node that is updated by another kernel.

    Assign this to the precision matrix node K when ``gwishart_drj`` (with
    ``k_name`` set) already performs the exact conjugate update of K inside the
    graph move, so that bamojax does not apply its default random walk to K.
    """
    del logdensity_fn, kwargs

    def init_fn(position, rng_key=None):
        del rng_key
        return GWishartDRJState(position=position)

    def step_fn(key: PRNGKey, state, *args, **kwargs):
        return state, {'is_accepted': jnp.asarray(True)}

    return SamplingAlgorithm(init_fn, step_fn)

#