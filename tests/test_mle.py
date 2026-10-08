"""
Tests 7-8: MLE parameter recovery and Wilks model-comparison test.

Run with:  python -m pytest tests/test_mle.py -v
       or:  python tests/test_mle.py
"""

import time
import jax
import numpy as np
import jax.numpy as jnp
from scipy import optimize


from pta_flatland.model import (
    covariance_matrix,
    log_likelihood,
    _build_aniso_fun,
    _build_iso_fun,
)
from pta_flatland.data_generation import generate_data

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

_TRUE = dict(sigma2_noise=1.5, sigma2_gwb=0.3, P2r=0.030, P2i=0.020)  # hypot < 1/(4*pi)
_N_PUL = 8
_NS = 100_000
_SEED = 20


def _make_dataset(N_pulsars=_N_PUL, Ns=_NS, seed=_SEED, true_params=None):
    if true_params is None:
        true_params = _TRUE
    rng = np.random.default_rng(seed)
    phi = jnp.array(rng.uniform(0.0, 2.0 * np.pi, N_pulsars))
    C = covariance_matrix(
        phi,
        true_params["sigma2_noise"],
        true_params["sigma2_gwb"],
        true_params["P2r"],
        true_params["P2i"],
    )
    D = generate_data(C, Ns, jax.random.PRNGKey(seed))
    return phi, C, D


# ---------------------------------------------------------------------------
# Test 7: MLE parameter recovery
# ---------------------------------------------------------------------------


def test_mle_parameter_recovery(N_pulsars=_N_PUL, Ns=_NS, seed=_SEED):
    """
    L-BFGS-B with exact JAX gradients should recover all four true parameters
    to within 10% relative error at Ns=100k.
    """
    TRUE = _TRUE.copy()
    phi, C, D = _make_dataset(N_pulsars, Ns, seed, TRUE)

    assert 1.0 >= 2.0 * np.hypot(TRUE["P2r"], TRUE["P2i"]) + 1e-6

    fun_S = _build_aniso_fun(phi, D, Ns)
    x0_S = np.array(
        [
            np.log(TRUE["sigma2_noise"]),
            np.log(TRUE["sigma2_gwb"]),
            TRUE["P2r"] * 0.3,
            TRUE["P2i"] * 0.3,
        ]
    )
    fun_S(x0_S)  # trigger JIT

    t0 = time.perf_counter()
    res_S = optimize.minimize(
        fun_S,
        x0_S,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
    )
    elapsed = time.perf_counter() - t0

    p = res_S.x
    MLE_S = dict(sigma2_noise=np.exp(p[0]), sigma2_gwb=np.exp(p[1]), P2r=p[2], P2i=p[3])

    print(f"\n  H_S MLE converged={res_S.success}, {res_S.nit} iters, {elapsed:.3f}s")
    for k in TRUE:
        frac = abs(MLE_S[k] - TRUE[k]) / max(abs(TRUE[k]), 1e-9)
        print(f"  {k:<14s}: true={TRUE[k]:.4f}  MLE={MLE_S[k]:.4f}  rel={frac:.3f}")
        assert (
            frac < 0.10
        ), f"MLE of {k} too far: true={TRUE[k]:.4f}, MLE={MLE_S[k]:.4f}, rel={frac:.3f}"


# ---------------------------------------------------------------------------
# Test 8: Wilks model comparison
# ---------------------------------------------------------------------------


def test_wilks_detects_anisotropy(N_pulsars=_N_PUL, Ns=_NS, seed=_SEED):
    """
    2*DeltaLL should exceed chi^2(dof=2) 95th percentile (5.991) when data
    are generated from a truly anisotropic model.
    """
    TRUE = _TRUE.copy()
    phi, C, D = _make_dataset(N_pulsars, Ns, seed, TRUE)

    fun_S = _build_aniso_fun(phi, D, Ns)
    fun_N = _build_iso_fun(phi, D, Ns)

    x0_S = np.array(
        [
            np.log(TRUE["sigma2_noise"]),
            np.log(TRUE["sigma2_gwb"]),
            TRUE["P2r"] * 0.3,
            TRUE["P2i"] * 0.3,
        ]
    )
    x0_N = x0_S[:2].copy()
    fun_S(x0_S)
    fun_N(x0_N)  # warm up JIT

    res_S = optimize.minimize(
        fun_S,
        x0_S,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
    )
    res_N = optimize.minimize(
        fun_N,
        x0_N,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8},
    )

    ll_S = -res_S.fun
    ll_N = -res_N.fun
    two_delta_ll = 2.0 * (ll_S - ll_N)
    chi2_95 = 5.991

    print(f"\n  log L(H_S) = {ll_S:.3f}")
    print(f"  log L(H_N) = {ll_N:.3f}")
    print(f"  2*DeltaLL  = {two_delta_ll:.2f}  (chi^2 dof=2 at 95%: {chi2_95})")

    assert (
        two_delta_ll > chi2_95
    ), f"2*DeltaLL={two_delta_ll:.2f} does not exceed chi^2_95={chi2_95}"


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    _phi_w = jnp.array([0.5, 1.5, 2.5, 3.5])
    _C_w = covariance_matrix(_phi_w, 1.0, 0.1, 0.0, 0.0)
    log_likelihood(_C_w, jnp.eye(4), 100).block_until_ready()

    tests = [test_mle_parameter_recovery, test_wilks_detects_anisotropy]
    for i, fn in enumerate(tests, 7):
        print(f"Test {i}: {fn.__name__} ... ", end="", flush=True)
        fn()
        print("  PASSED")
    print("\nAll MLE tests passed.")
