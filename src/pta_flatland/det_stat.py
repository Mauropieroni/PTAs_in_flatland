import functools
import numpy as np
import jax
import jax.numpy as jnp

from .utils import _sample_prior
from .model import (
    covariance_matrix,
    woodbury_precompute,
    log_likelihood_woodbury,
    Gamma_matrix,
)

jax.config.update("jax_enable_x64", True)


def init_mlp_params(key, in_dim: int, hidden: list):
    """Initialise MLP parameters: a list of (w, b) pairs, one per Linear
    layer, w of shape (fan_in, fan_out). hidden=[] gives a single linear
    layer (previously the separate mlp_linear class); hidden=[h1, h2, ...]
    gives that many ReLU hidden layers before the final linear readout,
    exactly the architecture the old torch mlp class built.

    Matches torch.nn.Linear's default init (kaiming_uniform_ with a=sqrt(5),
    which reduces to U(-1/sqrt(fan_in), 1/sqrt(fan_in)) for weight and bias
    alike) purely so a freshly-initialised model has the same weight scale
    as before -- the actual draws differ, since torch and JAX use unrelated
    RNGs.
    """
    sizes = [in_dim, *hidden, 1]
    params = []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        key, w_key, b_key = jax.random.split(key, 3)
        bound = 1.0 / jnp.sqrt(fan_in)
        w = jax.random.uniform(w_key, (fan_in, fan_out), minval=-bound, maxval=bound)
        b = jax.random.uniform(b_key, (fan_out,), minval=-bound, maxval=bound)
        params.append((w, b))
    return params


def mlp_apply(params, x):
    """Forward pass through params from init_mlp_params: ReLU after every
    layer but the last, matching the old torch mlp/mlp_linear classes."""
    *hidden_layers, (w_out, b_out) = params
    for w, b in hidden_layers:
        x = jax.nn.relu(x @ w + b)
    return (x @ w_out + b_out).squeeze(-1)


# ---------------------------------------------------------------------------
# Batched log-likelihood  (JAX — hot path, jax.JIT-compiled)
# ---------------------------------------------------------------------------
# IMPORTANT: defined ONCE at module level (phi_j, D_j, Ns passed in as
# explicit arguments) so JAX traces/compiles the XLA graph once per
# argument shape and reuses it across calls to bayes_factor(), instead of
# retracing + recompiling on every call (see the analogous note in model.py).
#
# Uses model.log_likelihood_woodbury, which exploits Gamma_ab's rank-2
# structure (Gamma = Phi^T K Phi) via the Woodbury identity, instead of a
# dense O(N_pul^3) slogdet/solve per MC sample — see tests/test_woodbury.py
# for the proof that this reproduces the dense path exactly. The one
# O(N_pul^2) pass over (phi_j, D_j) is hoisted out of the vmap via
# woodbury_precompute, since it doesn't depend on the sampled parameters.
# ---------------------------------------------------------------------------


@jax.jit
def _log_likes_N(theta, phi_j, D_j, Ns):
    """Batched log-likelihood under H_N (P2r=P2i=0). theta = [sn, sw]."""
    _, S, PhiDPhiT, trD, N = woodbury_precompute(phi_j, D_j)
    return jax.vmap(
        lambda t: log_likelihood_woodbury(t[0], t[1], 0.0, 0.0, Ns, S, PhiDPhiT, trD, N)
    )(theta)


@jax.jit
def _log_likes_S(theta, phi_j, D_j, Ns):
    """Batched log-likelihood under H_S. theta = [sn, sw, P2r, P2i]."""
    _, S, PhiDPhiT, trD, N = woodbury_precompute(phi_j, D_j)
    return jax.vmap(
        lambda t: log_likelihood_woodbury(
            t[0], t[1], t[2], t[3], Ns, S, PhiDPhiT, trD, N
        )
    )(theta)


def bayes_factor(
    phi_j, D_j, Ns, priors, log_uniform=None, key=None, n_mc: int = 50_000
):
    """
    Estimate the Bayesian evidence for H_S and H_N via Monte Carlo integration.

    Z = integral pi(theta) L(theta | D) dtheta
      ≈ (1/N) sum_i L(theta_i),   theta_i ~ pi(theta)

    log Z ≈ logsumexp(log L(theta_i)) - log N

    Parameters
    ----------
    phi_j : JAX array, pulsar angular positions
    D_j   : JAX array (N_pul x N_pul), sufficient-statistic matrix
    Ns    : int, number of time samples used to build D
    priors : dict
        Keys: "sigma2_noise", "sigma2_gwb", and either "P2_max" (uniform disk
        prior: sample |P2| <= P2_max) or both "P2r" and "P2i".
        Each value is either:
          scalar        → parameter fixed to that value
          (min, max)    → uniform prior over [min, max]
    log_uniform : set of str, optional
        Parameter names to sample log-uniformly instead of linearly.
        Default: empty (all range priors are linear-uniform).
    key : JAX PRNGKey (created from seed 0 if None)
    n_mc : int, number of prior samples per model

    Returns
    -------
    dict with keys: log_Z_S, log_Z_N, log_BF
    """
    if log_uniform is None:
        log_uniform = set()
    if key is None:
        key = jax.random.PRNGKey(0)

    key, k_sn, k_sw, k1, k2 = jax.random.split(key, 5)

    sn = _sample_prior(
        k_sn, priors["sigma2_noise"], "sigma2_noise" in log_uniform, n_mc
    )
    sw = _sample_prior(k_sw, priors["sigma2_gwb"], "sigma2_gwb" in log_uniform, n_mc)

    if "P2_max" in priors:
        P2_max = float(priors["P2_max"])
        r = P2_max * jnp.sqrt(jax.random.uniform(k1, (n_mc,)))
        theta = jax.random.uniform(k2, (n_mc,), minval=0.0, maxval=2.0 * jnp.pi)
        P2r = r * jnp.cos(theta)
        P2i = r * jnp.sin(theta)
    else:
        P2r = _sample_prior(k1, priors["P2r"], "P2r" in log_uniform, n_mc)
        P2i = _sample_prior(k2, priors["P2i"], "P2i" in log_uniform, n_mc)

    # H_N evidence: marginalise over (sigma2_noise, sigma2_gwb), P2 = 0
    log_likes_N = _log_likes_N(jnp.stack([sn, sw], axis=1), phi_j, D_j, Ns)
    log_Z_N = float(jax.scipy.special.logsumexp(log_likes_N) - jnp.log(n_mc))

    # H_S evidence: marginalise over (sigma2_noise, sigma2_gwb, P2r, P2i)
    log_likes_S = _log_likes_S(jnp.stack([sn, sw, P2r, P2i], axis=1), phi_j, D_j, Ns)
    log_Z_S = float(jax.scipy.special.logsumexp(log_likes_S) - jnp.log(n_mc))

    return dict(log_Z_S=log_Z_S, log_Z_N=log_Z_N, log_BF=log_Z_S - log_Z_N)


@jax.jit
def df_weights(phi_j, theta_0, theta_1, Ns=None):
    """
    Weights of the deflection (signal-to-noise) statistic.

    w_DF ~ C_0^{-1} (C_1 - C_0) C_0^{-1}

    obtained by maximising

        (<Lambda>_1 - <Lambda>_0)^2 / Var[Lambda]_0 .

    theta_0, theta_1 : dicts with scalar sigma2_noise, sigma2_gwb, P2r, P2i,
                       fixing the null and signal hypotheses respectively.
    Ns : int, optional.  If given, the overall constant is fixed so that
         Var[Tr(w_DF D)]_0 = 1, using
         Var[Tr(W D)]_0 = 2 Tr(W C_0 W C_0) / Ns  for symmetric W.
         This 1/Ns is the same Ns-averaged-D convention as model.py's
         "Cross-correlation convention" note: D is an average over Ns
         samples, so Var[D] -- and anything quadratic in D -- scales as
         1/Ns.

    Returns
    -------
    (N_pul, N_pul) weight matrix.  As for np_weights, Tr(w_DF D) counts each
    off-diagonal pair twice.
    """
    C0 = covariance_matrix(phi_j, **theta_0)
    C1 = covariance_matrix(phi_j, **theta_1)
    C0_inv = jnp.linalg.inv(C0)
    W = C0_inv @ (C1 - C0) @ C0_inv
    if Ns is not None:
        W = W / jnp.sqrt(2.0 * jnp.trace(W @ C0 @ W @ C0) / Ns)
    return W


@functools.partial(jax.jit, static_argnames=("normalize",))
def deflection_batch(phi_j, D_batch, Ns, theta_0, theta_1, normalize=True):
    """
    Deflection detection statistic, Lambda_DF = Tr(w_DF D), evaluated across a
    whole batch of D matrices.

    df_weights depends only on (phi_j, theta_0, theta_1, Ns), not on D, so
    it's built once here and the trace is applied to the whole batch in a
    single jitted pass, rather than rebuilding the weights from scratch for
    every D.

    Parameters
    ----------
    D_batch : (n, N_pul, N_pul) array

    Returns
    -------
    (n,) JAX array
    """
    W = df_weights(phi_j, theta_0, theta_1, Ns if normalize else None)
    return jnp.einsum("ij,nji->n", W, D_batch)


@jax.jit
def map_cross_weights(phi_j, theta_0, Ns):
    """
    Linear weights of the cross-correlation-only sky-map estimator, with the
    monopole fitted jointly (as in NANOGrav 15-yr anisotropy, Eq. 19).

    Data are the N_cc = N_pul (N_pul - 1) / 2 distinct cross-correlations
    rho_ab = D_ab, a < b -- self-correlations (a = b) are excluded, since
    those also carry the pulsar noise sigma2_noise. Their mean is

        E[rho_ab] = sigma2_gwb * Gamma_ab = sum_k T_k,ab P_k ,

    with P = (P0, P2r, P2i) and templates T_k = sigma2_gwb * dGamma/dP_k
    taken from model.Gamma_matrix, so P_hat estimates the physical
    (P0, P2r, P2i); P0 = 1/(2 pi) when the model is correct.  sigma2_noise
    does not enter the mean (it only affects the autocorrelations a = b),
    only the covariance: the full pair covariance of Eq. (41), evaluated
    at C_0 (Eq. 42),

        Sigma_ab,cd = (C_ac C_bd + C_ad C_bc) / Ns ,

    written in this code's convention E[D] = C (no 1/Ns in C) -- see model.py's
    "Cross-correlation convention" note: D is an Ns-sample average rather than
    a sum, so this 1/Ns is the same bookkeeping as df_weights' normalisation.

    The joint maximum-likelihood fit P_hat = M^{-1} T^T Sigma^{-1} rho, with
    M = T^T Sigma^{-1} T, is linear in rho, so each estimator is a weighted
    sum of the cross-correlations.

    Returns
    -------
    w_P0, w_P2r, w_P2i : (N_cc,) arrays
        Weights on the cross-correlations, ordered as the pairs
        (a, b) = np.triu_indices(N_pul, k=1), so that

            P_hat_k = sum_{a<b} w_k,ab rho_ab = w_k @ rho ,
            rho = D[np.triu_indices(N_pul, k=1)] .
    """
    n = phi_j.shape[0]
    a, b = np.triu_indices(n, k=1)
    C0 = covariance_matrix(phi_j, **theta_0)

    Sigma = (C0[a][:, a] * C0[b][:, b] + C0[a][:, b] * C0[b][:, a]) / Ns

    # Templates dGamma/dP_k from the model ORF.  Gamma is linear in
    # (P0, P2r, P2i), and Gamma_matrix bakes in P0 = 1/(2 pi), so
    # dGamma/dP0 = 2 pi Gamma_iso.
    G_iso = Gamma_matrix(phi_j, 0.0, 0.0)
    dG_dP2r, dG_dP2i = jax.jacfwd(Gamma_matrix, argnums=(1, 2))(phi_j, 0.0, 0.0)
    T = (
        theta_0["sigma2_gwb"]
        * jnp.stack([2.0 * jnp.pi * G_iso, dG_dP2r, dG_dP2i], axis=-1)[a, b]
    )

    Sigma_inv_T = jax.scipy.linalg.cho_solve(jax.scipy.linalg.cho_factor(Sigma), T)
    w = jnp.linalg.solve(T.T @ Sigma_inv_T, Sigma_inv_T.T)
    return w[0], w[1], w[2]


@jax.jit
def map_cross_statistic_batch(phi_j, D_batch, Ns, theta_0, theta_1=None):
    """
    Map-based detection statistic from the cross-correlations only,

        Lambda_map = |P2_hat|^2 + |P-2_hat|^2 = 2 (P2r_hat^2 + P2i_hat^2) ,

    Eq. (44), with (P0, P2r, P2i) fitted jointly by map_cross_weights, so
    self-correlations (a = b) never contribute. Null thresholds are
    calibrated empirically. Evaluated across a whole batch of D matrices.

    theta_1 is accepted for interface compatibility with roc._QUAD and is
    ignored: the estimator whitens with the null covariance, as in Eq. (42).

    map_cross_weights depends only on (phi_j, theta_0, Ns), not on D, so
    it's built once here and applied to the whole batch in a single jitted
    pass, rather than rebuilding the weights from scratch for every D.

    Parameters
    ----------
    D_batch : (n, N_pul, N_pul) array

    Returns
    -------
    (n,) JAX array
    """
    n = phi_j.shape[0]
    _, w_P2r, w_P2i = map_cross_weights(phi_j, theta_0, Ns)
    a, b = np.triu_indices(n, k=1)
    rho_batch = D_batch[:, a, b]
    return 2.0 * ((rho_batch @ w_P2r) ** 2 + (rho_batch @ w_P2i) ** 2)


@jax.jit
def np_weights(C0, C1):
    """
    Weights of the Neyman-Pearson statistic for two simple hypotheses.

    w_NP = C_0^{-1} - C_1^{-1}

    C_0, C_1 : JAX arrays (N_pul, N_pul), the covariance matrices for the null
        and signal hypotheses respectively.

    Returns
    -------
    (N_pul, N_pul) weight matrix.  Note that Tr(w_NP D) counts each
    off-diagonal pair twice, unlike a triu-flattened weight vector.
    """

    return jnp.linalg.inv(C0) - jnp.linalg.inv(C1)


@jax.jit
def likelihood_ratio_batch(phi_j, D_batch, Ns, theta_0, theta_1):
    """
    Log likelihood ratio for two simple hypotheses -- the optimal (Neyman-
    Pearson) detection statistic when all parameters are fixed -- evaluated
    across a whole batch of D matrices.

    log L_1/L_0 = (Ns/2) [ ln det C_0 - ln det C_1 + Tr(w_NP D) ] ,

    with the weights w_NP given by np_weights.

    np_weights and the logdet term depend only on (phi_j, theta_0, theta_1),
    not on D, so they're built once here and applied to the whole batch in a
    single jitted pass, rather than being rebuilt from scratch for every D.

    Parameters
    ----------
    D_batch : (n, N_pul, N_pul) array

    Returns
    -------
    (n,) JAX array
    """
    C0 = covariance_matrix(phi_j, **theta_0)
    C1 = covariance_matrix(phi_j, **theta_1)
    W = np_weights(C0, C1)
    logdet_diff = jnp.linalg.slogdet(C0)[1] - jnp.linalg.slogdet(C1)[1]
    return Ns / 2.0 * (logdet_diff + jnp.einsum("ij,nji->n", W, D_batch))
