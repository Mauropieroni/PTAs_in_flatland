"""
ROC curve computation and plotting for all detection methods.
"""

import numpy as np
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

import config
from .utils import triu_features, load_model, model_path, hypotheses
from .det_stat import (
    mlp_apply,
    bayes_factor,
    likelihood_ratio_batch,
    deflection_batch,
    map_cross_statistic_batch,
)

matplotlib.use("Agg")

# Simple-hypothesis statistics: their weights depend on the assumed (theta_0,
# theta_1), which utils.hypotheses reads off each test scenario -- not on D.
# The _batch variants build those weights once per scenario and apply them
# to the whole D_batch in one jitted pass, instead of rebuilding them from
# scratch for every single D (what a Python loop over the per-sample
# functions would do). map_cross_statistic_batch uses only theta_0
# (direction-agnostic); theta_1 is passed but ignored, which is why the
# interface accepts it as an optional argument. Kept under the
# "map_statistic" key for compatibility with saved ROC data, config files,
# and plotting notebooks that reference that name.
_QUAD = {
    "likelihood_ratio": likelihood_ratio_batch,
    "deflection": deflection_batch,
    "map_statistic": map_cross_statistic_batch,
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _roc(null_stats, signal_stats):
    """Build (fpr, tpr, auc) from two 1-D arrays of test statistics."""
    thresholds = np.sort(np.concatenate([null_stats, signal_stats]))[::-1]
    fpr = np.array([(null_stats >= t).mean() for t in thresholds])
    tpr = np.array([(signal_stats >= t).mean() for t in thresholds])
    fpr = np.concatenate([[0.0], fpr, [1.0]])
    tpr = np.concatenate([[0.0], tpr, [1.0]])
    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auc


# ---------------------------------------------------------------------------
# Per-method statistic computation
# ---------------------------------------------------------------------------


def _stats_bf(D_batch, phi_j, Ns):
    """Compute log BF for each D matrix in D_batch. Returns (n,) array."""
    stats = np.zeros(len(D_batch))
    for i, D in enumerate(tqdm(D_batch)):
        stats[i] = bayes_factor(
            phi_j,
            D,
            Ns,
            priors=config.bf_priors,
            log_uniform=config.bf_log_uniform,
            n_mc=config.n_mc_bf,
        )["log_BF"]
    return stats


def _stats_ml(params, D_batch, datasets):
    """Compute model scores for each D matrix in D_batch. Returns (n,) array."""
    X = triu_features(D_batch)
    X = (X - datasets["_mu"]) / datasets["_sig_std"]
    return np.asarray(mlp_apply(params, jnp.asarray(X)))


def _stats_quad(fn, D_batch, phi_j, Ns, theta_0, theta_1):
    """Evaluate a simple-hypothesis statistic on the whole batch at once via
    one of det_stat's *_batch functions. Returns (n,) array."""
    return fn(phi_j, D_batch, Ns, theta_0, theta_1)


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------


def compute_roc(method, datasets, phi_j):
    """
    Compute ROC curves for one method across all test scenarios.

    Parameters
    ----------
    method   : str        "mlp", "mlp_linear", "bayes_factor",
                          "likelihood_ratio" or "deflection"
    datasets : dict       output of data_generation.build_datasets
    phi_j    : jnp array  pulsar angular positions

    Returns
    -------
    dict : {scenario_label: (fpr, tpr, auc)}
    """
    labels = [str(label) for label in datasets["_labels"]]
    Ns = datasets["_Ns"]
    D_null = datasets["D_test_null"]

    # Simple-hypothesis statistics: the weights depend on the scenario, so the
    # null statistics have to be recomputed for each one rather than shared.
    if method in _QUAD:
        fn = _QUAD[method]
        by_label = {sc["label"]: sc for sc in config.test_scenarios}
        result = {}
        for label in labels:
            theta_0, theta_1 = hypotheses(by_label[label])
            result[label] = _roc(
                _stats_quad(fn, D_null, phi_j, Ns, theta_0, theta_1),
                _stats_quad(
                    fn, datasets[f"D_test_sig_{label}"], phi_j, Ns, theta_0, theta_1
                ),
            )
        return result

    # Load ML model once (reused across all scenarios) -- the params pytree
    # is self-describing, so no separate "build the matching empty
    # architecture" step is needed here.
    if method != "bayes_factor":
        params = load_model(model_path(method))

    # Null statistics (shared across scenarios)
    if method == "bayes_factor":
        null_stats = _stats_bf(D_null, phi_j, Ns)
    else:
        null_stats = _stats_ml(params, D_null, datasets)

    result = {}
    for label in labels:
        D_sig = datasets[f"D_test_sig_{label}"]
        if method == "bayes_factor":
            sig_stats = _stats_bf(D_sig, phi_j, Ns)
        else:
            sig_stats = _stats_ml(params, D_sig, datasets)
        result[label] = _roc(null_stats, sig_stats)

    return result


# ---------------------------------------------------------------------------
# Save / load ROC data
# ---------------------------------------------------------------------------


def save_roc(results, path):
    """Save ROC curves to a .npz file.

    Keys follow the pattern  {method}__{label}__{fpr|tpr|auc}.

    Parameters
    ----------
    results : dict  {method: {label: (fpr, tpr, auc)}}
    path    : output file path (should end in .npz)
    """
    arrays = {}
    for method, by_label in results.items():
        for label, (fpr, tpr, auc) in by_label.items():
            prefix = f"{method}__{label}"
            arrays[f"{prefix}__fpr"] = fpr
            arrays[f"{prefix}__tpr"] = tpr
            arrays[f"{prefix}__auc"] = np.array(auc)
    np.savez(path, **arrays)
    print(f"ROC data saved: {path}", flush=True)


def load_roc(path):
    """Load ROC data saved by save_roc.

    Returns
    -------
    dict  {method: {label: (fpr, tpr, auc)}}
    """
    npz = np.load(path)
    results = {}
    for key in npz.files:
        method, label, field = key.split("__")
        results.setdefault(method, {}).setdefault(label, {})[field] = npz[key]
    return {
        method: {
            label: (fields["fpr"], fields["tpr"], float(fields["auc"]))
            for label, fields in by_label.items()
        }
        for method, by_label in results.items()
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_roc(results, test_scenarios, path):
    """
    Plot ROC curves for all methods and scenarios.

    Parameters
    ----------
    results        : dict  {method: {label: (fpr, tpr, auc)}}
    test_scenarios : list of dicts with keys label, color
    path           : output file path
    """
    n_sc = len(test_scenarios)
    fig, axes = plt.subplots(1, n_sc, figsize=(5 * n_sc, 4), squeeze=False)

    for col, sc in enumerate(test_scenarios):
        ax = axes[0, col]
        label = sc["label"]

        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="random")
        for method, roc_dict in results.items():
            if label not in roc_dict:
                continue
            fpr, tpr, auc = roc_dict[label]
            ax.plot(fpr, tpr, lw=1.8, label=f"{method}  AUC={auc:.3f}")

        ax.axvline(0.05, color="grey", ls=":", lw=0.8, label="FPR=0.05")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(label)
        ax.legend(fontsize=8, loc="lower right")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
