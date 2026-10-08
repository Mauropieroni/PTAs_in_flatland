"""
Tests for the Woodbury-identity fast path for log_likelihood.

Gamma_ab = Phi^T K Phi with Phi = [cos(phi); sin(phi)] (2 x N) and K a 2x2
matrix built from (P2r, P2i) -- see model.py's Gamma_matrix. This means
C = sigma2_noise*I + sigma2_gwb*Gamma is "diagonal plus rank-2", so
log_likelihood(C, D, Ns) can be computed via the Woodbury identity and the
matrix determinant lemma using only 2x2 linear algebra, instead of factorising
the full N x N matrix C. These tests pin down that the fast path reproduces
the existing dense implementation (model.log_likelihood + covariance_matrix)
exactly, value AND gradient, before it is used anywhere.

Run with:  python -m pytest tests/test_woodbury.py -v
"""

import jax
import numpy as np
import jax.numpy as jnp
from numpy.random import default_rng

from pta_flatland.model import (
    covariance_matrix,
    log_likelihood,
    woodbury_precompute,
    log_likelihood_woodbury,
)
from pta_flatland.data_generation import generate_data

jax.config.update("jax_enable_x64", True)


def _dense_ll(phi_j, sn, sw, P2r, P2i, D, Ns):
    C = covariance_matrix(phi_j, sn, sw, P2r, P2i)
    return log_likelihood(C, D, Ns)


def _fast_ll(phi_j, sn, sw, P2r, P2i, D, Ns):
    _, S, PhiDPhiT, trD, N = woodbury_precompute(phi_j, D)
    return log_likelihood_woodbury(sn, sw, P2r, P2i, Ns, S, PhiDPhiT, trD, N)


# ---------------------------------------------------------------------------
# Test 1: values match on random valid (phi, theta, D)
# ---------------------------------------------------------------------------


def test_woodbury_matches_dense_values(N_pulsars=20, n_cases=200, seed=3):
    """log_likelihood_woodbury reproduces log_likelihood(covariance_matrix(...))
    to float64 precision across many random parameter/data draws."""
    rng = default_rng(seed)
    phi_j = jnp.array(rng.uniform(0.0, 2.0 * np.pi, N_pulsars))

    # A single fixed "true" D drawn from a valid covariance, reused for all
    # parameter draws (mirrors how bayes_factor() evaluates many thetas
    # against one fixed data realisation).
    C_true = covariance_matrix(phi_j, 1.3, 0.7, 0.015, -0.010)
    D = generate_data(C_true, 500, jax.random.PRNGKey(seed))

    max_abs = 0.0
    max_rel = 0.0
    for i in range(n_cases):
        sn = rng.uniform(0.2, 3.0)
        sw = rng.uniform(0.05, 3.0)
        # keep hypot(P2r, P2i) < 1/(4*pi) ~ 0.0796 so P(phi) >= 0 and both C
        # and the 2x2 K matrix stay positive-definite (see model.P_of_phi;
        # det(K) = 0.25 - pi^2 * hypot(P2r, P2i)^2)
        P2r = rng.uniform(-0.04, 0.04)
        P2i = rng.uniform(-0.04, 0.04)

        ll_dense = float(_dense_ll(phi_j, sn, sw, P2r, P2i, D, 500))
        ll_fast = float(_fast_ll(phi_j, sn, sw, P2r, P2i, D, 500))

        max_abs = max(max_abs, abs(ll_dense - ll_fast))
        max_rel = max(max_rel, abs(ll_dense - ll_fast) / max(abs(ll_dense), 1.0))

    assert max_abs < 1e-6, f"max abs diff too large: {max_abs:.3e}"
    assert max_rel < 1e-8, f"max rel diff too large: {max_rel:.3e}"


# ---------------------------------------------------------------------------
# Test 2: values match with P2=0 (the H_N / null-model branch used in
# det_stat.py's _log_likes_N)
# ---------------------------------------------------------------------------


def test_woodbury_matches_dense_null_model(N_pulsars=15, seed=4):
    rng = default_rng(seed)
    phi_j = jnp.array(rng.uniform(0.0, 2.0 * np.pi, N_pulsars))
    C_true = covariance_matrix(phi_j, 1.0, 1.0, 0.0, 0.0)
    D = generate_data(C_true, 200, jax.random.PRNGKey(seed))

    for sn, sw in [(0.5, 0.5), (1.0, 1.0), (2.5, 0.3), (0.3, 2.5)]:
        ll_dense = float(_dense_ll(phi_j, sn, sw, 0.0, 0.0, D, 200))
        ll_fast = float(_fast_ll(phi_j, sn, sw, 0.0, 0.0, D, 200))
        assert abs(ll_dense - ll_fast) < 1e-6, (sn, sw, ll_dense, ll_fast)


# ---------------------------------------------------------------------------
# Test 3: batched (vmap) usage matches, exactly as det_stat.py will call it
# ---------------------------------------------------------------------------


def test_woodbury_matches_dense_batched(N_pulsars=25, n_mc=300, seed=5):
    rng = default_rng(seed)
    phi_j = jnp.array(rng.uniform(0.0, 2.0 * np.pi, N_pulsars))
    C_true = covariance_matrix(phi_j, 1.0, 1.0, 0.020, 0.010)
    D = generate_data(C_true, 400, jax.random.PRNGKey(seed))

    sn = jnp.array(rng.uniform(0.2, 3.0, n_mc))
    sw = jnp.array(rng.uniform(0.05, 3.0, n_mc))
    P2r = jnp.array(rng.uniform(-0.04, 0.04, n_mc))
    P2i = jnp.array(rng.uniform(-0.04, 0.04, n_mc))
    Ns = 400

    dense = jax.vmap(lambda a, b, c, d: _dense_ll(phi_j, a, b, c, d, D, Ns))(
        sn, sw, P2r, P2i
    )

    _, S, PhiDPhiT, trD, N = woodbury_precompute(phi_j, D)
    fast = jax.vmap(
        lambda a, b, c, d: log_likelihood_woodbury(a, b, c, d, Ns, S, PhiDPhiT, trD, N)
    )(sn, sw, P2r, P2i)

    err = np.max(np.abs(np.array(dense) - np.array(fast)))
    assert err < 1e-6, f"batched mismatch: {err:.3e}"


# ---------------------------------------------------------------------------
# Test 4: gradients match (this is what MLE fitting differentiates through)
# ---------------------------------------------------------------------------


def test_woodbury_matches_dense_gradient(N_pulsars=12, seed=6):
    rng = default_rng(seed)
    phi_j = jnp.array(rng.uniform(0.0, 2.0 * np.pi, N_pulsars))
    C_true = covariance_matrix(phi_j, 1.2, 0.6, 0.010, -0.020)
    D = generate_data(C_true, 300, jax.random.PRNGKey(seed))
    Ns = 300

    def dense_fn(theta):
        sn, sw = jnp.exp(theta[0]), jnp.exp(theta[1])
        return _dense_ll(phi_j, sn, sw, theta[2], theta[3], D, Ns)

    def fast_fn(theta):
        sn, sw = jnp.exp(theta[0]), jnp.exp(theta[1])
        return _fast_ll(phi_j, sn, sw, theta[2], theta[3], D, Ns)

    theta0 = jnp.array([np.log(1.2), np.log(0.6), 0.010, -0.020])

    v_dense, g_dense = jax.value_and_grad(dense_fn)(theta0)
    v_fast, g_fast = jax.value_and_grad(fast_fn)(theta0)

    assert abs(float(v_dense) - float(v_fast)) < 1e-6
    err = np.max(np.abs(np.array(g_dense) - np.array(g_fast)))
    assert (
        err < 1e-5
    ), f"gradient mismatch: {err:.3e}\n  dense={g_dense}\n  fast ={g_fast}"


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_woodbury_matches_dense_values,
        test_woodbury_matches_dense_null_model,
        test_woodbury_matches_dense_batched,
        test_woodbury_matches_dense_gradient,
    ]
    for fn in tests:
        print(f"{fn.__name__} ... ", end="", flush=True)
        fn()
        print("PASSED")
    print("\nAll Woodbury tests passed.")
