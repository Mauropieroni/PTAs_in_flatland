import os
import pickle
import functools
import numpy as np
import jax
import jax.numpy as jnp


def model_path(method):
    """Path of the saved parameters for an ML detection statistic."""
    import config

    return os.path.join(config.data_out_dir, f"{method}.pkl")


_THETA_KEYS = ("sigma2_noise", "sigma2_gwb", "P2r", "P2i")


def hypotheses(scenario):
    """(theta_0, theta_1) assumed by the simple-hypothesis detection statistics.

    Uses scenario["stat_params"] when present, otherwise the scenario's own
    parameters -- i.e. the statistic is matched to the injected signal.
    theta_0 is the same point with P2r = P2i = 0.
    """
    src = scenario.get("stat_params", scenario)
    missing = [k for k in _THETA_KEYS if k not in src]
    if missing:
        raise KeyError(
            f"scenario {scenario.get('label')!r}: stat_params is missing {missing}"
        )
    theta_1 = {k: float(src[k]) for k in _THETA_KEYS}
    theta_0 = dict(theta_1, P2r=0.0, P2i=0.0)
    return theta_0, theta_1


def save_model(params, path) -> None:
    """Pickle a det_stat.init_mlp_params() pytree to disk, as host arrays."""
    with open(path, "wb") as f:
        pickle.dump(jax.tree_util.tree_map(np.asarray, params), f)
    print(f"  Model saved: {path}", flush=True)


def load_model(path):
    """Unpickle a params pytree saved by save_model, back onto device."""
    with open(path, "rb") as f:
        params = pickle.load(f)
    return jax.tree_util.tree_map(jnp.asarray, params)


def triu_features(D_batch):
    """Extract upper-triangular elements (including diagonal) from D matrices.

    D_batch : (N, N) or (B, N, N)  →  (N*(N+1)//2,) or (B, N*(N+1)//2)
    """
    if D_batch.ndim == 2:
        N = D_batch.shape[0]
        return D_batch[np.triu_indices(N)]
    N = D_batch.shape[-1]
    r, c = np.triu_indices(N)
    return D_batch[:, r, c]


def _sample_prior(key, prior, log_unif: bool, n_mc: int) -> jnp.ndarray:
    """Return (n_mc,) JAX array drawn from prior (used for Bayes factor MC).

    prior : scalar  → fixed value broadcast to n_mc
            [a, b]  → uniform (or log-uniform) draws over [a, b]
    """
    p = jnp.asarray(prior, dtype=jnp.float64)
    if p.size == 1:
        return jnp.full((n_mc,), p)
    lo, hi = p[0], p[1]
    if log_unif:
        return jnp.exp(
            jax.random.uniform(key, (n_mc,), minval=jnp.log(lo), maxval=jnp.log(hi))
        )
    return jax.random.uniform(key, (n_mc,), minval=lo, maxval=hi)


@functools.partial(jax.jit, static_argnums=(1,))
def _sample_disk(P2_max, n, key):
    """Sample n points uniformly from the disk |P2| <= P2_max.

    Uses r = P2_max * sqrt(U) so that area is uniform (not clustered at centre).
    Returns (P2r, P2i) each of shape (n,).
    """
    key_r, key_theta = jax.random.split(key)
    r = P2_max * jnp.sqrt(jax.random.uniform(key_r, shape=(n,), minval=0.0, maxval=1.0))
    theta = jax.random.uniform(key_theta, shape=(n,), minval=0.0, maxval=2.0 * jnp.pi)
    return r * jnp.cos(theta), r * jnp.sin(theta)


@functools.partial(jax.jit, static_argnums=(1,))
def _sample_params(prior, n, key):
    """Return (n,) jnp.ndarray from a prior spec.

    prior : scalar  → broadcast fixed value
            [a, b]  → n uniform draws from [a, b]
    """
    p = jnp.asarray(prior, dtype=jnp.float64)
    if p.size == 1:
        return jnp.full((n,), p)
    return jax.random.uniform(key, (n,), minval=p[0], maxval=p[1])
