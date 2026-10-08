"""
Numerical and analytical scalar helpers used only in tests.

These verify the closed-form expressions in model.py against independent
quadrature and direct-formula implementations.
"""

import numpy as np


def Gamma_ab_numerical(phi_a, phi_b, P2r, P2i, N_quad=4000):
    """Midpoint-rule quadrature of int F_a*F_b*P dphi, P0 = 1/(2*pi) fixed.

    Numerical evaluation of the Gamma_ab integral in main.tex Eq. (11).
    """
    phi = np.linspace(0.0, 2.0 * np.pi, N_quad, endpoint=False)
    dphi = 2.0 * np.pi / N_quad
    Fa = -np.sin(phi - phi_a)
    Fb = -np.sin(phi - phi_b)
    P0 = 1.0 / (2.0 * np.pi)
    Pp = P0 + 2.0 * (P2r * np.cos(2.0 * phi) - P2i * np.sin(2.0 * phi))
    return float(np.sum(Fa * Fb * Pp) * dphi)


def Gamma_ab_analytical(phi_a, phi_b, P2r, P2i):
    """Closed-form scalar ORF, P0 = 1/(2*pi) fixed.

    Scalar evaluation of main.tex Eq. (17), split via Eqs. (20)-(21)
    [\\label{e:Gamma_ab_final}, \\label{e:Gamma_iso}, \\label{e:Gamma_aniso}].
    """
    s = phi_a + phi_b
    return float(
        0.5 * np.cos(phi_a - phi_b) - np.pi * (P2r * np.cos(s) - P2i * np.sin(s))
    )


def R_ab_m_numerical(phi_a, phi_b, m, N_quad=4000):
    """(1/2pi)*int exp(+im*phi)*F_a*F_b dphi by quadrature.

    Numerical evaluation of the R_{ab,m} integral in main.tex Eq. (14).
    """
    phi = np.linspace(0.0, 2.0 * np.pi, N_quad, endpoint=False)
    dphi = 2.0 * np.pi / N_quad
    Fa = -np.sin(phi - phi_a)
    Fb = -np.sin(phi - phi_b)
    return complex(np.sum(np.exp(1j * m * phi) * Fa * Fb) * dphi / (2.0 * np.pi))


def R_ab_m_analytical(phi_a, phi_b, m):
    """Closed-form R_ab,m (non-zero only for m = 0, ±2).

    Analytical result from main.tex Eq. (16).
    """
    if m == 0:
        return complex(0.5 * np.cos(phi_a - phi_b))
    if m == 2:
        return complex(-0.25 * np.exp(1j * (phi_a + phi_b)))
    if m == -2:
        return complex(-0.25 * np.exp(-1j * (phi_a + phi_b)))
    return 0j
