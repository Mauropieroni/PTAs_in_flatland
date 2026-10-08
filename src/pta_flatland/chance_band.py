"""
Chance band for the linear network's ROC curve in the composite scenarios.

Under the composite (disk) prior, E[D | H1] = E_P2[C(P2)] = C0 = E[D | H0]:
the training data carry no information a statistic linear in D can use, so
the population optimum of the linear network is w = 0. Any weight vector it
ends up with is set by the particular finite training sample, and on the
test set -- where the signal sits at one FIXED P2 -- that w can still land
off the diagonal, on either side of it.

The band is therefore the spread of the test-set ROC over independent
training sets: this script regenerates n_rep fresh training sets (same
priors, pulsars, Ns, n_train as the stored run), refits mlp_linear on each
with train.train_linear_lbfgs, evaluates each on the stored test set, and
saves the pointwise TPR mean/std and quantiles on a common FPR grid.

The training priors must be the ones the stored datasets were generated
with: config.train_priors by default, or --train_priors when the config has
since changed.

Usage:
  python -m pta_flatland.chance_band --config config_files/composite_strong.py
  python -m pta_flatland.chance_band --config ... --n_rep 50 --n_train 12500
  python -m pta_flatland.chance_band --config ... \
      --train_priors '{"sigma2_noise": 0.01, "sigma2_gwb": 1.0, "P2_max": 0.0477}'
"""

import argparse
import json
import importlib.util
import os
import sys
import time

FPR_GRID_POINTS = 201
QUANTILES = {  # pointwise 1 and 2 sigma
    "lo2": 0.02275,
    "lo1": 0.15866,
    "med": 0.5,
    "hi1": 0.84134,
    "hi2": 0.97725,
}


def _load_config(path):
    spec = importlib.util.spec_from_file_location("config", os.path.abspath(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules["config"] = module
    return module


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--n_rep", type=int, default=20, help="training-set replicas")
    parser.add_argument(
        "--n_train",
        type=int,
        default=None,
        help="D matrices per class (default: the config's n_train)",
    )
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--train_priors",
        type=json.loads,
        default=None,
        help="JSON dict overriding config.train_priors",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output .npz (default: <plot_out_dir>/chance_band[_n<N>].npz)",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    config = _load_config(args.config)

    import numpy as np
    import jax
    import jax.numpy as jnp

    from .data_generation import _generate_batch
    from .det_stat import mlp_apply
    from .train import train_linear_lbfgs
    from .utils import _sample_disk, _sample_params, triu_features
    from .roc import _roc

    fpr_grid = np.linspace(0.0, 1.0, FPR_GRID_POINTS)

    def features(phi_j, sn, sw, P2r, P2i, Ns, key):
        """triu(D) features for a batch, generated in chunk-sized pieces so the
        full (n, N, N) D array is never held in memory at once."""
        out, step = [], 10_000
        for s in range(0, len(sn), step):
            sl = slice(s, s + step)
            D, key = _generate_batch(phi_j, sn[sl], sw[sl], P2r[sl], P2i[sl], Ns, key)
            out.append(triu_features(D))
        return np.concatenate(out), key

    def training_set(phi_j, Ns, n_train, key):
        """Same sampling as data_generation.build_datasets' training part."""
        pri = args.train_priors or config.train_priors
        key, k_sn0, k_sw0, k_sn1, k_sw1, k_p2 = jax.random.split(key, 6)
        zeros = jnp.zeros(n_train, dtype=jnp.float64)
        P2r, P2i = _sample_disk(pri["P2_max"], n_train, k_p2)
        X0, key = features(
            phi_j,
            _sample_params(pri["sigma2_noise"], n_train, k_sn0),
            _sample_params(pri["sigma2_gwb"], n_train, k_sw0),
            zeros,
            zeros,
            Ns,
            key,
        )
        X1, key = features(
            phi_j,
            _sample_params(pri["sigma2_noise"], n_train, k_sn1),
            _sample_params(pri["sigma2_gwb"], n_train, k_sw1),
            P2r,
            P2i,
            Ns,
            key,
        )
        return X0, X1, key

    def train_linear(X0, X1, seed):
        """Standardise + split as train.train does, then train.train_linear_lbfgs."""
        X = np.concatenate([X0, X1])
        mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-8
        X = (X - mu) / sd
        n = len(X)
        y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])

        idx = np.random.default_rng(seed).permutation(n)
        n_val = int(n * config.val_split)
        X_tr, y_tr = jnp.asarray(X[idx[n_val:]]), jnp.asarray(y[idx[n_val:]])
        X_val, y_val = jnp.asarray(X[idx[:n_val]]), jnp.asarray(y[idx[:n_val]])
        del X
        params, train_losses, _ = train_linear_lbfgs(X_tr, y_tr, X_val, y_val)
        return params, mu, sd, len(train_losses)

    data_path = os.path.join(config.data_out_dir, "datasets.npz")
    npz = np.load(data_path)
    phi_j, Ns = jnp.asarray(npz["_phi_j"]), int(npz["_Ns"])
    label = str(npz["_labels"][0])
    X_null = triu_features(npz["D_test_null"])
    X_sig = triu_features(npz[f"D_test_sig_{label}"])
    del npz
    n_train = args.n_train or config.n_train
    print(f"train priors: {args.train_priors or config.train_priors}", flush=True)
    print(
        f"{config.run_label}: n_pul={len(phi_j)}, Ns={Ns}, n_train={n_train}, "
        f"n_test={len(X_null)}, n_rep={args.n_rep}",
        flush=True,
    )

    key = jax.random.PRNGKey(args.seed)
    tprs, aucs = [], []
    for rep in range(args.n_rep):
        t0 = time.perf_counter()
        X0, X1, key = training_set(phi_j, Ns, n_train, key)
        params, mu, sd, iters = train_linear(X0, X1, args.seed + rep)
        del X0, X1

        def score(X):
            return np.asarray(mlp_apply(params, jnp.asarray((X - mu) / sd)))

        fpr, tpr, auc = _roc(score(X_null), score(X_sig))
        # fpr is non-decreasing with ties; take the upper envelope at each FPR
        tprs.append(np.interp(fpr_grid, fpr, tpr))
        aucs.append(auc)
        print(
            f"  rep {rep + 1:3d}/{args.n_rep}  AUC={auc:.4f}  iters={iters}"
            f"  ({time.perf_counter() - t0:.0f}s)",
            flush=True,
        )

    tprs = np.array(tprs)
    out = args.out or os.path.join(
        config.plot_out_dir,
        "chance_band.npz" if args.n_train is None else f"chance_band_n{n_train}.npz",
    )
    np.savez(
        out,
        fpr=fpr_grid,
        tpr_all=tprs,
        auc_all=np.array(aucs),
        n_train=n_train,
        n_test=len(X_null),
        mean=tprs.mean(axis=0),
        std=tprs.std(axis=0, ddof=1),
        **{k: np.quantile(tprs, q, axis=0) for k, q in QUANTILES.items()},
    )
    print(
        f"AUC over replicas: mean={np.mean(aucs):.4f}  std={np.std(aucs):.4f}  "
        f"range=[{np.min(aucs):.4f}, {np.max(aucs):.4f}]"
    )
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
