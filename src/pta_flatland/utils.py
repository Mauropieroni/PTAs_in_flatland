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
def _sample_Cl(Cl_prior, n, key):
    """Sample n points from a non-negative angular-power-spectrum prior Cl,
    returning the corresponding (P2r, P2i) at the canonical realisation
    P2i = 0, P2r = sqrt(Cl) >= 0.

    Cl = P2r^2 + P2i^2 is this toy model's analogue of the usual angular
    power spectrum C_ell = sum_m |a_lm|^2: unlike a single multipole
    coefficient (P2r, P2i) itself -- the "a_lm"-like quantities, which can
    each be positive or negative -- a power is a sum of squares and so is
    manifestly >= 0 for every physically sensible prior on it (uniform,
    log-uniform, ...). There is therefore no way to build a "rotationally
    symmetric" Cl prior with E[Cl] = 0 the way _sample_disk's P2_max prior
    does on (P2r, P2i): E[Cl] > 0 always, so the marginalised Bayes
    factor's linear-in-D term (proportional to E[P2r], E[P2i] -- see
    composite_weak.py/composite_strong.py) never vanishes, and a linear
    detection statistic keeps real signal to learn regardless of how
    wide/narrow the Cl prior is -- see composite_Cl_*.py.

    P2i is fixed to 0 (not itself sampled) WLOG: the pulsar array phi_j
    carries no preferred frame, so any single-mode anisotropy realisation
    can be described, after a rotation of the sky coordinate, with its
    quadrupole aligned along the P2r axis; P2r = sqrt(Cl) is then exactly
    the non-negative amplitude that Cl parametrises.

    Cl_prior : scalar → Cl fixed to that value (P2i still forced to 0)
               [a, b] → Cl ~ Uniform[a, b], a >= 0
    Returns (P2r, P2i), each shape (n,).
    """
    Cl = _sample_params(Cl_prior, n, key)
    return jnp.sqrt(Cl), jnp.zeros((n,), dtype=jnp.float64)


@functools.partial(jax.jit, static_argnums=(1,))
def _sample_Cl_isotropic(Cl_prior, n, key):
    """Sample n points where Cl ~ Cl_prior (>= 0) and, conditional on Cl,
    (P2r, P2i) are an ISOTROPIC (statistically-isotropic-sky) realisation
    of that power: P2r, P2i ~ N(0, Cl/2) independently, so
    E[P2r^2 + P2i^2 | Cl] = Cl -- the standard assumption for a Gaussian
    random field's individual multipole coefficients, and the "physically
    realistic" alternative to _sample_Cl's canonical-frame P2i=0 shortcut.

    This is EXACTLY as rotationally symmetric as _sample_disk: a zero-mean
    isotropic 2-D Gaussian has a density that depends only on radius,
    same as _sample_disk's uniform-in-area disk, just with a Rayleigh(
    sqrt(Cl/2)) radial profile instead of a uniform one. Since the phase
    is uniform on [0, 2*pi) independently of Cl either way,
    E[P2r] = E[P2i] = 0 EXACTLY here too, regardless of Cl's own prior --
    non-negativity of Cl alone does not prevent E[P2r]=0; what made
    _sample_Cl work was ALSO fixing the phase (P2i=0), not merely giving
    Cl a non-negative prior. This sampler exists specifically to make that
    point numerically: see composite_Cl_isotropic_*.py, which reproduces
    _sample_disk's mlp_linear failure using this sampler.

    Cl_prior : scalar → Cl fixed to that value
               [a, b] → Cl ~ Uniform[a, b], a >= 0
    Returns (P2r, P2i), each shape (n,).
    """
    key_cl, key_r, key_i = jax.random.split(key, 3)
    Cl = _sample_params(Cl_prior, n, key_cl)
    sigma = jnp.sqrt(Cl / 2.0)
    P2r = sigma * jax.random.normal(key_r, (n,))
    P2i = sigma * jax.random.normal(key_i, (n,))
    return P2r, P2i


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
