"""
Tests 1-6: response function, ORF (Gamma), Hellings-Downs matrix,
Fourier coefficients R_ab,m, Gamma via Fourier sum, and data generation.

Run with:  python -m pytest tests/test_physics.py -v
       or:  python tests/test_physics.py
"""

import jax
import numpy as np
import jax.numpy as jnp

from pta_flatland.model import (
    response_function,
    Gamma_matrix,
    covariance_matrix,
)
from pta_flatland.data_generation import generate_data
from helpers import (
    Gamma_ab_numerical,
    Gamma_ab_analytical,
    R_ab_m_numerical,
    R_ab_m_analytical,
)

# ---------------------------------------------------------------------------
# Test 1: Response function
# ---------------------------------------------------------------------------


def test_response_function():
    """F_a(phi_a)=0, F_a(phi_a+pi/2)=-1, F_a(phi_a+pi)=0, and 2pi-periodic."""
    rng = np.random.default_rng(0)
    phi_a = rng.uniform(0.0, 2.0 * np.pi)

    assert abs(float(response_function(phi_a, phi_a))) < 1e-14
    assert abs(float(response_function(phi_a + np.pi / 2, phi_a)) + 1.0) < 1e-14
    assert abs(float(response_function(phi_a + np.pi, phi_a))) < 1e-14

    phi_t = rng.uniform(0.0, 2.0 * np.pi)
    assert (
        abs(
            float(response_function(phi_t + 2 * np.pi, phi_a))
            - float(response_function(phi_t, phi_a))
        )
        < 1e-13
    )


# ---------------------------------------------------------------------------
# Test 2: Analytical vs. numerical Gamma_ab
# ---------------------------------------------------------------------------


def test_gamma_formula(N_pulsars=6, seed=1):
    """Gamma_matrix matches int F_a*F_b*P dphi by quadrature (P0 = 1/(2*pi))."""
    rng = np.random.default_rng(seed)
    phi = jnp.array(rng.uniform(0.0, 2.0 * np.pi, N_pulsars))
    P2r, P2i = 0.030, 0.010  # hypot < 1/(4*pi) ~ 0.0796, keeps P(phi) >= 0

    G_mat = np.array(Gamma_matrix(phi, P2r, P2i))
    max_err = max(
        abs(Gamma_ab_numerical(float(phi[a]), float(phi[b]), P2r, P2i) - G_mat[a, b])
        for a in range(N_pulsars)
        for b in range(N_pulsars)
    )
    assert max_err < 1e-5, f"Gamma_ab quadrature mismatch: {max_err:.2e}"


# ---------------------------------------------------------------------------
# Test 3: Hellings-Downs matrix (P0=1, P2=0)
# ---------------------------------------------------------------------------


def test_hd_matrix(N_pulsars=6, seed=1):
    """With P2=0 the ORF reduces to (1/2)*cos(phi_a - phi_b)."""
    rng = np.random.default_rng(seed)
    phi = jnp.array(rng.uniform(0.0, 2.0 * np.pi, N_pulsars))
    phi_np = np.array(phi)

    G_mat = np.array(Gamma_matrix(phi, 0.0, 0.0))
    H_mat = 0.5 * np.cos(phi_np[:, None] - phi_np[None, :])
    max_err = np.max(np.abs(G_mat - H_mat))

    assert max_err < 1e-14, f"HD matrix mismatch: {max_err:.2e}"


# ---------------------------------------------------------------------------
# Test 4: Fourier coefficients R_ab,m
# ---------------------------------------------------------------------------


def test_R_ab_m_coefficients(seed=2):
    """Analytical R_ab,m matches (1/2pi) int exp(im phi)*F_a*F_b dphi for all m."""
    rng = np.random.default_rng(seed)
    phi_a = rng.uniform(0.0, 2.0 * np.pi)
    phi_b = rng.uniform(0.0, 2.0 * np.pi)

    for m in [0, 2, -2, 1, -1, 3, -3]:
        R_num = R_ab_m_numerical(phi_a, phi_b, m)
        R_ana = R_ab_m_analytical(phi_a, phi_b, m)
        err = abs(R_num - R_ana)
        assert err < 1e-5, f"R_ab,m mismatch at m={m}: err={err:.2e}"


# ---------------------------------------------------------------------------
# Test 5: Gamma_ab = sum_m R_ab,m * P_m
# ---------------------------------------------------------------------------


def test_gamma_from_R_m(seed=2):
    """Gamma_ab reconstructed from Fourier coefficients matches direct formula.

    main.tex: Gamma_ab = 2*pi * sum_m R*_{ab,m} P_m (line above Eq. 17), with
    P0 = 1/(2*pi) fixed and P_{-2} = conj(P_2).
    """
    rng = np.random.default_rng(seed)
    phi_a = rng.uniform(0.0, 2.0 * np.pi)
    phi_b = rng.uniform(0.0, 2.0 * np.pi)
    P0 = 1.0 / (2.0 * np.pi)
    P2r, P2i = 0.020, 0.015  # hypot < 1/(4*pi), keeps P(phi) >= 0
    P2 = P2r + 1j * P2i

    Gamma_sum = (
        2.0
        * np.pi
        * (
            R_ab_m_analytical(phi_a, phi_b, 0) * P0
            + R_ab_m_analytical(phi_a, phi_b, 2) * P2
            + R_ab_m_analytical(phi_a, phi_b, -2) * np.conj(P2)
        )
    )
    Gamma_dir = Gamma_ab_analytical(phi_a, phi_b, P2r, P2i)

    assert abs(Gamma_sum.real - Gamma_dir) < 1e-12


# ---------------------------------------------------------------------------
# Test 6: Data generation
# ---------------------------------------------------------------------------


def test_data_generation(N_pulsars=8, Ns=50_000, seed=10):
    """Sample covariance D converges to true C; C must be positive-definite."""
    rng = np.random.default_rng(seed)
    phi = jnp.array(rng.uniform(0.0, 2.0 * np.pi, N_pulsars))
    sigma2_noise = 2.0
    sigma2_gwb = 0.4
    P2r, P2i = 0.030, 0.015  # hypot < 1/(4*pi), keeps P(phi) >= 0

    C = covariance_matrix(phi, sigma2_noise, sigma2_gwb, P2r, P2i)
    assert bool(jnp.all(jnp.linalg.eigvalsh(C) > 0)), "C not positive-definite"

    D = generate_data(C, Ns, jax.random.PRNGKey(seed))

    C_np = np.array(C)
    D_np = np.array(D)
    rel_err = np.max(np.abs(D_np - C_np)) / np.max(np.abs(C_np))
    assert rel_err < 0.15, f"Sample covariance too far from theory: {rel_err:.4f}"


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_response_function,
        test_gamma_formula,
        test_hd_matrix,
        test_R_ab_m_coefficients,
        test_gamma_from_R_m,
        test_data_generation,
    ]
    for i, fn in enumerate(tests, 1):
        print(f"Test {i}: {fn.__name__} ... ", end="", flush=True)
        fn()
        print("PASSED")
    print("\nAll physics tests passed.")
