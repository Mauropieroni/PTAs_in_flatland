"""
Data generation for the 1-D anisotropic PTA toy model.

Generates training and test sets of sufficient-statistic matrices D,
saving raw D arrays plus standardisation stats to a single .npz file.
"""

import time
import functools
import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm

from .model import _phi_design, _K_matrix
from .utils import (
    _sample_params,
    _sample_disk,
    _sample_Cl,
    _sample_Cl_isotropic,
    triu_features,
)

jax.config.update("jax_enable_x64", True)

_CHUNK = 10_000  # samples per vmap call — caps peak memory usage


# ---------------------------------------------------------------------------
# Core sampling
# ---------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnums=(1,))
def generate_data(C, Ns, key):
    """Draw Ns samples from N(0, C) and return D = X^T X / Ns.

    General dense-covariance sampler (used by the test suite and anywhere
    a plain covariance matrix is the natural input). The data-generation
    hot path (_generate_chunk below) uses _generate_data_low_rank instead,
    which exploits Gamma's rank-2 structure to avoid ever forming an N x N
    matrix -- see its docstring and tests/test_low_rank_sampling.py for the
    equivalence proof between the two.
    """
    N = C.shape[0]
    samples = jax.random.multivariate_normal(key, jnp.zeros(N), C, shape=(Ns,))
    return samples.T @ samples / Ns


@functools.partial(jax.jit, static_argnums=(5,))
def _generate_data_low_rank(phi_j, sigma2_noise, sigma2_gwb, P2r, P2i, Ns, key):
    """Draw Ns samples from N(0, C), C = sigma2_noise*I + sigma2_gwb*Gamma,
    and return D = X^T X / Ns -- without ever forming the N x N covariance
    matrix or paying its O(N^3) Cholesky factorisation.

    Exploits the same rank-2 structure already used by
    model.woodbury_precompute for the likelihood: Gamma = Phi^T K Phi,
    Phi = [cos(phi); sin(phi)] (2, N), K (2, 2) (model._phi_design /
    model._K_matrix). Then, for Z ~ N(0, I_N), W ~ N(0, I_2) independent
    and L_K @ L_K.T = K (a 2x2 Cholesky factor),

        X = sqrt(sigma2_noise)*Z + sqrt(sigma2_gwb) * W @ L_K.T @ Phi

    has exactly Cov(X) = sigma2_noise*I + sigma2_gwb * Phi.T @ K @ Phi
    = sigma2_noise*I + sigma2_gwb*Gamma = C (no cross term, since Z and W
    are independent). This turns the dominant per-sample cost from O(N^3)
    (dense Cholesky) into O(N) (an N-vector draw and an N x 2 matmul) plus
    the O(N^2) that forming D = X^T X / Ns needs regardless of how X was
    sampled. See tests/test_low_rank_sampling.py for the equivalence proof
    against generate_data(covariance_matrix(...), ...).
    """
    N = phi_j.shape[0]
    Phi = _phi_design(phi_j)  # (2, N)
    K = _K_matrix(P2r, P2i)  # (2, 2)
    L_K = jnp.linalg.cholesky(K)

    key_z, key_w = jax.random.split(key)
    Z = jax.random.normal(key_z, (Ns, N))
    W = jax.random.normal(key_w, (Ns, 2))

    samples = jnp.sqrt(sigma2_noise) * Z + jnp.sqrt(sigma2_gwb) * (W @ L_K.T) @ Phi
    return samples.T @ samples / Ns


@functools.partial(jax.jit, static_argnums=(5,))
def _generate_chunk(phi_j, sn, sw, P2r, P2i, Ns, keys):
    """Sampling for one chunk, fused into a single compiled program."""
    return jax.vmap(
        lambda a, b, c, d, k: _generate_data_low_rank(phi_j, a, b, c, d, Ns, k)
    )(sn, sw, P2r, P2i, keys)


def _generate_batch(phi_j, sn, sw, P2r, P2i, Ns, key):
    """Generate D matrices for n parameter vectors via chunked vmap.

    Every batch (train or test) hits at most one XLA compile per distinct
    chunk shape it actually needs -- padding every chunk to a single global
    shape was tried and reverted (see git history/PR discussion): it turns
    a one-time compile into a recurring ~2x compute cost on every
    smaller-than-_CHUNK batch (test sets are usually well under _CHUNK),
    which loses more than the compile it avoids.

    Returns
    -------
    D : np.ndarray (n, N_pul, N_pul)
    key : updated JAX PRNGKey
    """
    n = sn.shape[0]
    key, subkey = jax.random.split(key)
    all_keys = jax.random.split(subkey, n)

    chunks = []
    for start in tqdm(range(0, n, _CHUNK)):
        sl = slice(start, min(start + _CHUNK, n))
        D_chunk = _generate_chunk(phi_j, sn[sl], sw[sl], P2r[sl], P2i[sl], Ns, all_keys[sl])
        # Pull each chunk to host immediately — this is what bounds peak
        # device memory to _CHUNK rather than n, not a defensive cast.
        chunks.append(np.asarray(D_chunk))

    return np.concatenate(chunks, axis=0), key


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnums=(5,))
def _standardised_features_chunk(phi_j, sn, sw, P2r, P2i, Ns, key, rows, cols, mu, sd):
    """Standardised triu(D) features for one chunk as a single compiled call.

    Same model as _generate_data_low_rank -- X = sqrt(sn) Z + sqrt(sw) W L_K^T Phi,
    D = X^T X / Ns -- but batched over the whole chunk: all normals drawn
    from one key, L_K from a batched 2x2 Cholesky, then the upper triangle
    taken and standardised without leaving the device.
    """
    n, N = sn.shape[0], phi_j.shape[0]
    key_z, key_w = jax.random.split(key)
    Z = jax.random.normal(key_z, (n, Ns, N))
    W = jax.random.normal(key_w, (n, Ns, 2))
    L_K = jnp.linalg.cholesky(jax.vmap(_K_matrix)(P2r, P2i))  # (n, 2, 2)
    X = (
        jnp.sqrt(sn)[:, None, None] * Z
        + jnp.sqrt(sw)[:, None, None]
        * jnp.einsum("nsk,nlk->nsl", W, L_K) @ _phi_design(phi_j)
    )  # (n, Ns, N)
    D = jnp.einsum("nsa,nsb->nab", X, X) / Ns
    return (D[:, rows, cols] - mu) / sd


def training_features(phi_j, Ns, n_train, train_priors, key, mu, sd):
    """Fresh standardised training set, n_train null + n_train signal draws
    from train_priors (disk prior via "P2_max", or fixed/uniform P2r, P2i),
    generated and standardised (with the stored mu, sd) in JAX, chunk by
    chunk. Used by train.train for fresh_data.

    Returns
    -------
    X   : jnp.ndarray (2*n_train, N_pul*(N_pul+1)//2), null rows first
    y   : jnp.ndarray (2*n_train,)  0 for null, 1 for signal
    key : updated JAX PRNGKey
    """
    key, k_sn0, k_sw0, k_sn1, k_sw1, k_p2 = jax.random.split(key, 6)
    if "P2_max" in train_priors:
        P2r, P2i = _sample_disk(train_priors["P2_max"], n_train, k_p2)
    else:
        k_p2r, k_p2i = jax.random.split(k_p2)
        P2r = _sample_params(train_priors["P2r"], n_train, k_p2r)
        P2i = _sample_params(train_priors["P2i"], n_train, k_p2i)
    zeros = jnp.zeros(n_train, dtype=jnp.float64)
    rows, cols = (jnp.asarray(i) for i in np.triu_indices(phi_j.shape[0]))
    mu, sd = jnp.asarray(mu), jnp.asarray(sd)

    def features(sn, sw, pr, pi, key):
        out = []
        for s in range(0, n_train, _CHUNK):
            sl = slice(s, s + _CHUNK)
            key, sub = jax.random.split(key)
            out.append(_standardised_features_chunk(
                phi_j, sn[sl], sw[sl], pr[sl], pi[sl], Ns, sub, rows, cols, mu, sd
            ))
        return out, key

    X0, key = features(
        _sample_params(train_priors["sigma2_noise"], n_train, k_sn0),
        _sample_params(train_priors["sigma2_gwb"], n_train, k_sw0),
        zeros, zeros, key,
    )
    X1, key = features(
        _sample_params(train_priors["sigma2_noise"], n_train, k_sn1),
        _sample_params(train_priors["sigma2_gwb"], n_train, k_sw1),
        P2r, P2i, key,
    )
    y = jnp.concatenate([jnp.zeros(n_train), jnp.ones(n_train)])
    return jnp.concatenate(X0 + X1), y, key


def build_datasets(phi_j, key, n_train, n_test, train_priors, test_scenarios, Ns):
    """
    Generate training and test D matrices.

    Parameters
    ----------
    phi_j          : jnp.ndarray (N_pul,)
    key            : JAX PRNGKey
    n_train        : int  D matrices per class for training
    n_test         : int  D matrices per class per test scenario
    train_priors   : dict  keys: sigma2_noise, sigma2_gwb, P2r, P2i
                           values: scalar (fixed) or [min, max] (uniform)
    test_scenarios : list of dicts  each with sigma2_noise, sigma2_gwb, P2r, P2i, label
    Ns             : int  time samples used to form each D matrix

    Returns
    -------
    datasets : dict
    key      : updated JAX PRNGKey
    """
    key, k_sn_null, k_sw_null, k_sn_sig, k_sw_sig, k_p2 = jax.random.split(key, 6)

    # Sample training parameters
    sn_null = _sample_params(train_priors["sigma2_noise"], n_train, k_sn_null)
    sw_null = _sample_params(train_priors["sigma2_gwb"], n_train, k_sw_null)
    sn_sig = _sample_params(train_priors["sigma2_noise"], n_train, k_sn_sig)
    sw_sig = _sample_params(train_priors["sigma2_gwb"], n_train, k_sw_sig)
    if "P2_max" in train_priors:
        P2r, P2i = _sample_disk(train_priors["P2_max"], n_train, k_p2)
    elif "Cl" in train_priors:
        P2r, P2i = _sample_Cl(train_priors["Cl"], n_train, k_p2)
    elif "Cl_isotropic" in train_priors:
        P2r, P2i = _sample_Cl_isotropic(train_priors["Cl_isotropic"], n_train, k_p2)
    else:
        k_p2r, k_p2i = jax.random.split(k_p2)
        P2r = _sample_params(train_priors["P2r"], n_train, k_p2r)
        P2i = _sample_params(train_priors["P2i"], n_train, k_p2i)

    _p2_bound = 1.0 / (4.0 * jnp.pi)
    if not jnp.all(jnp.hypot(P2r, P2i) <= _p2_bound + 1e-9):
        raise ValueError(
            f"Some training samples violate hypot(P2r, P2i) <= {_p2_bound:.5f}. "
            "Narrow the P2r / P2i prior ranges."
        )

    print("Generating training null …", flush=True)
    t0 = time.perf_counter()
    D_train_null, key = _generate_batch(
        phi_j,
        sn_null,
        sw_null,
        jnp.zeros(n_train, dtype=jnp.float64),
        jnp.zeros(n_train, dtype=jnp.float64),
        Ns,
        key,
    )
    print(f"  Done ({time.perf_counter() - t0:.1f}s)", flush=True)

    print("Generating training signal …", flush=True)
    t0 = time.perf_counter()
    D_train_sig, key = _generate_batch(phi_j, sn_sig, sw_sig, P2r, P2i, Ns, key)
    print(f"  Done ({time.perf_counter() - t0:.1f}s)", flush=True)

    # Standardisation stats from training features
    X_all = np.concatenate([triu_features(D_train_null), triu_features(D_train_sig)])
    mu = X_all.mean(axis=0)
    sig_std = X_all.std(axis=0) + 1e-8

    # Test null (shared across scenarios)
    sn_te = test_scenarios[0]["sigma2_noise"]
    sw_te = test_scenarios[0]["sigma2_gwb"]
    print("Generating test null …", flush=True)
    t0 = time.perf_counter()
    D_test_null, key = _generate_batch(
        phi_j,
        jnp.full(n_test, sn_te, dtype=jnp.float64),
        jnp.full(n_test, sw_te, dtype=jnp.float64),
        jnp.zeros(n_test, dtype=jnp.float64),
        jnp.zeros(n_test, dtype=jnp.float64),
        Ns,
        key,
    )
    print(f"  Done ({time.perf_counter() - t0:.1f}s)", flush=True)

    # Test signal, one batch per scenario
    D_test_sig = {}
    for sc in test_scenarios:
        print(f"Generating test signal [{sc['label']}] …", flush=True)
        t0 = time.perf_counter()
        D_test_sig[sc["label"]], key = _generate_batch(
            phi_j,
            jnp.full(n_test, sc["sigma2_noise"], dtype=jnp.float64),
            jnp.full(n_test, sc["sigma2_gwb"], dtype=jnp.float64),
            jnp.full(n_test, sc["P2r"], dtype=jnp.float64),
            jnp.full(n_test, sc["P2i"], dtype=jnp.float64),
            Ns,
            key,
        )
        print(f"  Done ({time.perf_counter() - t0:.1f}s)", flush=True)

    datasets = {
        "_phi_j": phi_j,
        "_Ns": Ns,
        "_mu": mu,
        "_sig_std": sig_std,
        "_labels": [sc["label"] for sc in test_scenarios],
        "D_train_null": D_train_null,
        "D_train_sig": D_train_sig,
        "D_test_null": D_test_null,
        **{f"D_test_sig_{label}": arr for label, arr in D_test_sig.items()},
    }
    return datasets, key


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


def save_datasets(datasets, path):
    np.savez(path, **datasets)
    print(f"Datasets saved: {path}", flush=True)


def load_datasets(path):
    npz = np.load(path, allow_pickle=False)
    data = {k: npz[k] for k in npz.files}
    data["_Ns"] = int(data["_Ns"])
    return data
