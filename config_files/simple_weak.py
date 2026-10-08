import numpy as np
import os

seed = 0

# ---------------------------------------------------------------------------
# Physical model
# ---------------------------------------------------------------------------

n_pul = 50
ns = 20

priors = {
    "sigma2_noise": 1.0,
    "sigma2_gwb": 0.09,
    "P2r": 1.0 / (4.0 * np.pi * np.sqrt(2.0)),
    "P2i": 1.0 / (4.0 * np.pi * np.sqrt(2.0)),
}

# ---------------------------------------------------------------------------
# Pipeline flags
# ---------------------------------------------------------------------------

roc_methods = ["mlp", "mlp_linear", "bayes_factor"]
data_gen = True
train = True

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

n_train = 50_000  # D matrices per class for training
n_test = 10_000  # D matrices per class per test scenario

train_priors = priors

test_scenarios = [
    dict(
        label="weak",
        sigma2_noise=priors["sigma2_noise"],
        sigma2_gwb=priors["sigma2_gwb"],
        P2r=priors["P2r"],
        P2i=priors["P2i"],
        stat_params=dict(
            sigma2_noise=priors["sigma2_noise"],
            sigma2_gwb=priors["sigma2_gwb"],
            P2r=priors["P2r"],
            P2i=priors["P2i"],
        ),
        color="#D65F5F",
    ),
]
# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

val_split = 0.2  # fraction of training data held out for validation
n_epochs = 100
batch_size = 10000  # 4x the original 4096 -- verified to fit comfortably in
# memory (the full training set already dominates device memory regardless
# of batch_size) and to leave simple_*'s real-signal AUC unchanged, while
# meaningfully reducing composite_*'s AdamW random-walk noise (fewer, larger
# gradient steps per epoch under the same epoch-based early-stop patience).
learning_rate = 1e-3
hidden_layers = [50]  # MLP hidden layer widths
early_stop_patience = 10  # stop if val loss doesn't improve for this many epochs

# ---------------------------------------------------------------------------
# Bayes factor
# ---------------------------------------------------------------------------

bf_priors = priors
bf_log_uniform = set()  # parameter names to sample log-uniformly
n_mc_bf = 1000  # MC samples per evidence integral

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

run_label = "simple_weak"  # subdirectory name; change to separate runs on disk

_base = os.path.dirname(__file__)
data_out_dir = os.path.join(_base, "../out", run_label)
plot_out_dir = os.path.join(_base, "../plots", run_label)
os.makedirs(data_out_dir, exist_ok=True)
os.makedirs(plot_out_dir, exist_ok=True)
