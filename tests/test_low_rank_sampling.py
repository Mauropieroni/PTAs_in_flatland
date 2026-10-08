"""
Equivalence check for data_generation._generate_data_low_rank against the
dense reference generate_data(covariance_matrix(...), ...).

Both draw Ns iid samples from N(0, C) and return D = X^T X / Ns for the
SAME C = sigma2_noise*I + sigma2_gwb*Gamma(P2r, P2i) -- but the low-rank
path never forms the N x N covariance matrix or its Cholesky factor,
exploiting Gamma's exact rank-2 structure instead (same identity
tests/test_woodbury.py checks for the likelihood side). Different sampling
algorithms can't produce bit-identical draws, so this checks what actually
has to hold: E[D] -> C for both (law of large numbers over many replicate
draws), and the two methods' per-replicate D statistics are the same
distribution (KS test), not just the same mean.
"""

import numpy as np
from scipy import stats
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from pta_flatland.model import covariance_matrix
from pta_flatland.data_generation import generate_data, _generate_data_low_rank


@pytest.mark.parametrize(
    "sigma2_noise,sigma2_gwb,P2r,P2i",
    [
        (1.0, 0.05, 0.01, -0.02),
        (0.3, 1.0, 1.0 / (4 * np.pi * np.sqrt(2)), 1.0 / (4 * np.pi * np.sqrt(2))),
        (2.0, 0.0, 0.0, 0.0),  # pure noise, no GWB
    ],
)
def test_low_rank_matches_dense_covariance(sigma2_noise, sigma2_gwb, P2r, P2i):
    n_pul, Ns, n_reps = 12, 5, 20_000
    phi_j = jax.random.uniform(jax.random.PRNGKey(0), (n_pul,), minval=0, maxval=2 * jnp.pi)
    C = covariance_matrix(phi_j, sigma2_noise, sigma2_gwb, P2r, P2i)

    keys = jax.random.split(jax.random.PRNGKey(1), n_reps)
    D_dense = jax.vmap(lambda k: generate_data(C, Ns, k))(keys)
    D_low_rank = jax.vmap(
        lambda k: _generate_data_low_rank(phi_j, sigma2_noise, sigma2_gwb, P2r, P2i, Ns, k)
    )(keys)

    # E[D] -> C for both, to the same Monte Carlo precision
    mean_dense = np.asarray(D_dense).mean(axis=0)
    mean_low_rank = np.asarray(D_low_rank).mean(axis=0)
    np.testing.assert_allclose(mean_dense, np.asarray(C), atol=0.05)
    np.testing.assert_allclose(mean_low_rank, np.asarray(C), atol=0.05)
    np.testing.assert_allclose(mean_low_rank, mean_dense, atol=0.05)

    # Same distribution of trace(D) across replicates (KS test)
    tr_dense = np.trace(np.asarray(D_dense), axis1=1, axis2=2)
    tr_low_rank = np.trace(np.asarray(D_low_rank), axis1=1, axis2=2)
    ks = stats.ks_2samp(tr_dense, tr_low_rank)
    assert ks.pvalue > 0.01, f"trace(D) distributions diverge: {ks}"


def test_low_rank_respects_positivity_bound_edge_case():
    """K = _K_matrix(P2r, P2i) must stay PD (det(K) > 0) exactly at the
    positivity bound hypot(P2r, P2i) == 1/(4*pi), or cholesky(K) fails."""
    n_pul = 8
    phi_j = jax.random.uniform(jax.random.PRNGKey(2), (n_pul,), minval=0, maxval=2 * jnp.pi)
    bound = 1.0 / (4.0 * np.pi)
    P2r = P2i = bound / np.sqrt(2.0)  # hypot(P2r, P2i) == bound exactly

    D = _generate_data_low_rank(phi_j, 1.0, 0.05, P2r, P2i, 10, jax.random.PRNGKey(3))
    assert np.isfinite(np.asarray(D)).all()
