"""
Core physics model for the 1-D anisotropic PTA toy model (see main.tex).

Model
-----
F_a(phi)  = -sin(phi - phi_a)
P(phi)    = P0 + 2*(P2r*cos(2phi) - P2i*sin(2phi)),  P0 = 1/(2*pi)  [P0 >= 2|P2|]
Gamma_ab  = 1/2*cos(phi_a-phi_b) - pi*(P2r*cos(s_ab) - P2i*sin(s_ab))
C_ab      = sigma2_noise*delta_ab + sigma2_gwb*Gamma_ab
D_ab      = (1/Ns) sum_k x_{a,k}*x_{b,k}   [sufficient statistic, E[D]=C]

"""

# Global imports
import numpy as np
import jax
import jax.numpy as jnp

# Update JAX default to 64-bit floats
jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Physical model  (JAX — hot path, jax.JIT-compiled)
# ---------------------------------------------------------------------------


def response_function(phi, phi_a):
    """F_a(phi) = -sin(phi - phi_a).  Accepts jnp or np arrays.

    main.tex Eq. (9): F_a(phi) = hat_p_a . hat_epsilon(phi).
    """
    return -jnp.sin(phi - phi_a)


def P_of_phi(phi, P2r, P2i):
    """P(phi) = P0 + 2*(P2r*cos(2phi) - P2i*sin(2phi)), P0 = 1/(2*pi) fixed.

    Non-negative iff P0 >= 2|P2|, i.e. hypot(P2r, P2i) <= 1/(4*pi).

    Real parametrization of the Fourier expansion, main.tex Eq. (13),
    truncated to m=0,±2 (only non-zero R_{ab,m}, per Eq. (16)).
    P0 = P_0 = 1/(2*pi) (main.tex, after Eq. (13)), P2r = Re(P_2), P2i = Im(P_2).
    """
    P0 = 1.0 / (2.0 * jnp.pi)
    return P0 + 2.0 * (P2r * jnp.cos(2.0 * phi) - P2i * jnp.sin(2.0 * phi))


@jax.jit
def Gamma_matrix(phi_pulsars, P2r, P2i):
    """
    N_pul x N_pul ORF matrix — fully vectorised, no Python loops.
    Gamma_ab = 1/2 * cos(d_ab) - pi * (P2r*cos(s_ab) - P2i*sin(s_ab))
    where d_ab = phi_a - phi_b,  s_ab = phi_a + phi_b.

    Vectorised form of main.tex Eq. (17) with the iso/aniso split of
    Eqs. (20)-(21): Gamma_iso = (1/2)cos(d_ab) [P0 = 1/(2*pi) already baked
    in], Gamma_aniso = -pi*(P2r*cos(s_ab) - P2i*sin(s_ab)).
    """
    phi_a = phi_pulsars[:, None]
    phi_b = phi_pulsars[None, :]
    d_ab = phi_a - phi_b
    s_ab = phi_a + phi_b
    return 0.5 * jnp.cos(d_ab) - jnp.pi * (P2r * jnp.cos(s_ab) - P2i * jnp.sin(s_ab))


@jax.jit
def covariance_matrix(phi_pulsars, sigma2_noise, sigma2_gwb, P2r, P2i):
    """C_ab = sigma2_noise * I + sigma2_gwb * Gamma_ab.

    Matches main.tex Eq. (26): C_ab = sigma2_noise*delta_ab + sigma2_gwb*Gamma_ab
    (coefficient 1 on the GWB term, Gamma_matrix already carries the correct
    overall normalisation). The overall 1/N_s factor of the TeX convention is
    absorbed into the N_s/2 prefactor of log_likelihood.
    """
    G = Gamma_matrix(phi_pulsars, P2r, P2i)
    return sigma2_noise * jnp.eye(phi_pulsars.shape[0]) + sigma2_gwb * G


@jax.jit
def log_likelihood(C, D, Ns):
    """
    log L = -(Ns/2) * [log det(C) + Tr(C^{-1} D)]
    Autodiff-safe via jnp.linalg.slogdet + jnp.linalg.solve.

    Gaussian Kronecker likelihood derived from main.tex Eqs. (24)-(28),
    using the sufficient statistic D from Eq. (28), normalised by 1/Ns.
    """
    _, logdet = jnp.linalg.slogdet(C)
    C_inv_D = jnp.linalg.solve(C, D)
    return -Ns / 2.0 * (logdet + jnp.trace(C_inv_D))


# ---------------------------------------------------------------------------
# Woodbury fast path for log_likelihood
# ---------------------------------------------------------------------------
# Gamma_ab (Gamma_matrix) has the closed form Gamma = Phi^T K Phi, with
# Phi = [cos(phi); sin(phi)]  (2 x N)   and
# K   = [[0.5*(1-P2r), 0.5*P2i], [0.5*P2i, 0.5*(1+P2r)]]  (2 x 2),
# i.e. Gamma has rank <= 2 regardless of N_pul. So
# C = sigma2_noise*I + sigma2_gwb*Gamma is "diagonal plus rank-2", and both
# logdet(C) and C^{-1} can be obtained from 2x2 linear algebra via the
# Woodbury identity and the matrix determinant lemma, instead of an O(N^3)
# factorisation of the full N x N matrix. woodbury_precompute() does the one
# O(N^2) pass over (phi_j, D) that's independent of the model parameters;
# log_likelihood_woodbury() then costs O(1) per (sigma2_noise, sigma2_gwb,
# P2r, P2i) evaluation. See tests/test_woodbury.py for the equivalence proof
# against the dense covariance_matrix + log_likelihood path above.
# ---------------------------------------------------------------------------


@jax.jit
def _phi_design(phi_pulsars):
    """Phi = [cos(phi); sin(phi)], shape (2, N_pul), s.t. Gamma = Phi^T K Phi."""
    return jnp.stack([jnp.cos(phi_pulsars), jnp.sin(phi_pulsars)])


@jax.jit
def _K_matrix(P2r, P2i):
    """2x2 matrix s.t. Gamma_matrix(phi, P2r, P2i) == Phi^T @ _K_matrix @ Phi."""
    return jnp.array(
        [
            [0.5 - jnp.pi * P2r, jnp.pi * P2i],
            [jnp.pi * P2i, 0.5 + jnp.pi * P2r],
        ]
    )


@jax.jit
def woodbury_precompute(phi_pulsars, D):
    """Precompute the (phi_j, D)-dependent quantities shared by every
    log_likelihood_woodbury() evaluation against this (phi_j, D) pair.

    None of these depend on (sigma2_noise, sigma2_gwb, P2r, P2i), so when
    scanning many parameter values against a single D (e.g. the Bayes-factor
    MC integral in det_stat.py), this should be called once and its outputs
    reused, rather than being recomputed inside the parameter loop/vmap.

    Returns
    -------
    Phi       : (2, N)  cos/sin design matrix
    S         : (2, 2)  Phi @ Phi.T
    PhiDPhiT  : (2, 2)  Phi @ D @ Phi.T
    trD       : scalar  trace(D)
    N         : int     number of pulsars
    """
    Phi = _phi_design(phi_pulsars)
    S = Phi @ Phi.T
    PhiDPhiT = (Phi @ D) @ Phi.T
    trD = jnp.trace(D)
    N = phi_pulsars.shape[0]
    return Phi, S, PhiDPhiT, trD, N


@jax.jit
def log_likelihood_woodbury(
    sigma2_noise, sigma2_gwb, P2r, P2i, Ns, S, PhiDPhiT, trD, N
):
    """
    Same value as log_likelihood(covariance_matrix(phi_j, sigma2_noise,
    sigma2_gwb, P2r, P2i), D, Ns), computed via the Woodbury identity /
    matrix determinant lemma from the precomputed (S, PhiDPhiT, trD, N) =
    woodbury_precompute(phi_j, D). O(1) cost (2x2 linear algebra) instead of
    the O(N^3) dense slogdet + solve.
    """
    K = _K_matrix(P2r, P2i)
    M = sigma2_gwb * K
    W = jnp.linalg.inv(M) + S / sigma2_noise

    _, logdet_M = jnp.linalg.slogdet(M)
    _, logdet_W = jnp.linalg.slogdet(W)
    logdet_C = N * jnp.log(sigma2_noise) + logdet_M + logdet_W

    trace_term = (
        trD / sigma2_noise - jnp.trace(jnp.linalg.solve(W, PhiDPhiT)) / sigma2_noise**2
    )

    return -Ns / 2.0 * (logdet_C + trace_term)


# ---------------------------------------------------------------------------
# MLE helpers  (scipy L-BFGS-B + exact JAX gradients)
# ---------------------------------------------------------------------------
# IMPORTANT: the @jax.jit functions below are defined ONCE at module level
# with phi_j, D_j, and Ns as explicit arguments.  This means JAX traces and
# compiles the XLA graph a single time (per argument shape/dtype) and reuses
# it for every realisation.  Defining @jax.jit inside a factory function
# would create a new Python function object on every call, forcing a full
# retrace + XLA recompilation each time.
# ---------------------------------------------------------------------------


@jax.jit
def _nll_aniso(theta, phi_j, D_j, Ns):
    """Negative log-likelihood for H_S (anisotropic), with positivity barrier.

    theta = [log(sigma2_noise), log(sigma2_gwb), P2r, P2i].
    Barrier enforces P(phi) >= 0, i.e. hypot(P2r, P2i) <= 1/(4*pi)
    (P0 fixed to 1/(2*pi)).
    """
    sn, sw = jnp.exp(theta[0]), jnp.exp(theta[1])
    P2r, P2i = theta[2], theta[3]
    margin = 1.0 / (4.0 * jnp.pi) - jnp.sqrt(P2r**2 + P2i**2 + 1e-30) - 1e-6
    penalty = 1e8 * jnp.maximum(0.0, -margin)
    C = covariance_matrix(phi_j, sn, sw, P2r, P2i)
    return -log_likelihood(C, D_j, Ns) + penalty


@jax.jit
def _nll_iso(theta, phi_j, D_j, Ns):
    """Negative log-likelihood for H_N (isotropic, P2=0 fixed)."""
    sn, sw = jnp.exp(theta[0]), jnp.exp(theta[1])
    C = covariance_matrix(phi_j, sn, sw, 0.0, 0.0)
    return -log_likelihood(C, D_j, Ns)


_vag_aniso = jax.jit(jax.value_and_grad(_nll_aniso))
_vag_iso = jax.jit(jax.value_and_grad(_nll_iso))


def _build_aniso_fun(phi_j, D_j, Ns):
    """Returns a callable (theta_np) -> (value, grad) for H_S.

    theta = [log(sigma2_noise), log(sigma2_gwb), P2r, P2i].
    Soft barrier enforces P(phi) >= 0.
    """

    def fun(theta_np):
        v, g = _vag_aniso(jnp.array(theta_np, dtype=jnp.float64), phi_j, D_j, Ns)
        return float(v), np.array(g, dtype=np.float64)

    return fun


def _build_iso_fun(phi_j, D_j, Ns):
    """Returns a callable (theta_np) -> (value, grad) for H_N (P2=0 fixed)."""

    def fun(theta_np):
        v, g = _vag_iso(jnp.array(theta_np, dtype=jnp.float64), phi_j, D_j, Ns)
        return float(v), np.array(g, dtype=np.float64)

    return fun
