"""
Pipeline orchestrator for the 1-D anisotropic PTA toy model.

Stages (controlled by config flags):
  1. data_gen  — generate D-matrix datasets and save to disk
  2. train     — train ML detection statistics
  3. (always)  — compute ROC curves for all methods in roc_methods

Usage:
  python -m pta_flatland.main --config path/to/cfg.py
  python -m pta_flatland.main --config cfg.py --methods mlp
      # re-run only the listed methods: skips data generation (reuses the
      # saved datasets so every method is still tested on the same data),
      # trains/evaluates only those, and merges them into the existing roc.npz
"""

import argparse
import importlib.util
import os
import sys

_ML_METHODS = ("mlp", "mlp_linear")


def _load_config(path):
    spec = importlib.util.spec_from_file_location("config", os.path.abspath(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Inject under the alias every downstream module imports as `import config`.
    sys.modules["config"] = module
    return module


def main():
    parser = argparse.ArgumentParser(description="PTA 1-D anisotropy pipeline")
    parser.add_argument("--config", required=True, help="path to config .py file")
    parser.add_argument(
        "--methods",
        default=None,
        help="comma-separated subset of roc_methods to re-run; other methods in the "
        "existing roc.npz are kept. Implies no data generation.",
    )
    args, _ = parser.parse_known_args()
    config = _load_config(args.config)

    # Imported after the config is injected into sys.modules, so every
    # downstream module that does `import config` gets this instance.
    import jax
    import jax.numpy as jnp

    from .data_generation import build_datasets, save_datasets, load_datasets
    from .train import train
    from .roc import compute_roc, save_roc, load_roc, plot_roc
    from .utils import save_model, model_path

    data_path = os.path.join(config.data_out_dir, "datasets.npz")
    roc_path = os.path.join(config.plot_out_dir, "roc.npz")

    partial = args.methods is not None
    if partial:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
        unknown = [m for m in methods if m not in config.roc_methods]
        if unknown:
            raise SystemExit(f"--methods {unknown} not in roc_methods {config.roc_methods}")
        if not os.path.exists(data_path):
            raise SystemExit(f"--methods needs existing datasets: {data_path}")
    else:
        methods = list(config.roc_methods)

    key = jax.random.PRNGKey(config.seed)

    # ------------------------------------------------------------------
    # 1. Data generation (skipped for a partial re-run, which must reuse
    #    the datasets the other methods were scored on)
    # ------------------------------------------------------------------
    if config.data_gen and not partial:
        key, phi_key = jax.random.split(key)
        phi_j = jax.random.uniform(
            phi_key, shape=(config.n_pul,), minval=0.0, maxval=2.0 * jnp.pi
        )

        datasets, key = build_datasets(
            phi_j,
            key,
            config.n_train,
            config.n_test,
            config.train_priors,
            config.test_scenarios,
            config.ns,
        )
        save_datasets(datasets, data_path)
    else:
        datasets = load_datasets(data_path)
        phi_j = jnp.array(datasets["_phi_j"])

    # ------------------------------------------------------------------
    # 2. Train ML detection statistics
    # ------------------------------------------------------------------
    ml_methods = [m for m in methods if m in _ML_METHODS]

    if config.train:
        n_feat = datasets["D_train_null"].shape[-1]
        n_feat = n_feat * (n_feat + 1) // 2

        for method in ml_methods:
            print(f"Training {method} …", flush=True)
            hidden = config.hidden_layers if method == "mlp" else []
            key, train_key = jax.random.split(key)
            params, train_losses, val_losses = train(
                train_key, n_feat, hidden, datasets, method=method
            )
            save_model(params, model_path(method))

    # ------------------------------------------------------------------
    # 3. ROC curves
    # ------------------------------------------------------------------
    roc_results = {}
    for method in methods:
        print(f"Computing ROC [{method}] …", flush=True)
        roc_results[method] = compute_roc(method, datasets, phi_j)

    if partial and os.path.exists(roc_path):
        # keep the methods we did not re-run
        roc_results = {**load_roc(roc_path), **roc_results}

    save_roc(roc_results, roc_path)
    plot_roc(roc_results, config.test_scenarios, roc_path.replace(".npz", ".pdf"))


if __name__ == "__main__":
    main()
