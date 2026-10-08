import os
import numpy as np

# ---------------------------------------------------------------------------
# Physical model
# ---------------------------------------------------------------------------

n_pul = 50
ns = 20
seed = 12

priors = {
    "sigma2_noise": 1.0,
    "sigma2_gwb": 1.0,
    "P2_max": 1.0 / (4.0 * np.pi),  # uniform disk prior: sample |P2| <= P2_max
}

# ---------------------------------------------------------------------------
# Pipeline flags
# ---------------------------------------------------------------------------

data_gen = True
train = True
roc_methods = ["mlp", "mlp_linear", "bayes_factor", "map_statistic"]

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

n_train = 50_000  # D matrices per class for training
n_test = 10_000  # D matrices per class per test scenario

train_priors = priors


test_scenarios = [
    dict(
        label="strong",
        sigma2_noise=priors["sigma2_noise"],
        sigma2_gwb=priors["sigma2_gwb"],
        P2r=priors["P2_max"] / np.sqrt(2.0),
        P2i=priors["P2_max"] / np.sqrt(2.0),
        stat_params=dict(
            sigma2_noise=priors["sigma2_noise"],
            sigma2_gwb=priors["sigma2_gwb"],
            P2r=0.0,
            P2i=0.0,
        ),
        color="#D65F5F",
    ),
]

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

val_split = 0.2  # fraction of training data held out for validation
n_epochs = 300
batch_size = 10000  # same as the simple runs
# The MLP is data-limited here: on a fixed 5e4/class training set it stays
# below the Bayes factor (weak: learns nothing), so it gets a fresh training
# draw (same size) every 10 epochs -- see train.train. The linear network
# uses L-BFGS.
fresh_data_by_method = {"mlp": 1}
learning_rate = 1e-3
weight_decay = 0.01
hidden_layers = [50]  # MLP hidden layer widths
early_stop_patience = 20  # stop if val loss doesn't improve for this many epochs

# ---------------------------------------------------------------------------
# Bayes factor
# ---------------------------------------------------------------------------

# Same prior format as train_priors.
bf_priors = priors
bf_log_uniform = set()  # parameter names to sample log-uniformly
n_mc_bf = 1000  # MC samples per evidence integral

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

run_label = "composite_strong"  # subdirectory name; change to separate runs on disk

_base = os.path.dirname(__file__)
data_out_dir = os.path.join(_base, "../out", run_label)
plot_out_dir = os.path.join(_base, "../plots", run_label)
os.makedirs(data_out_dir, exist_ok=True)
os.makedirs(plot_out_dir, exist_ok=True)
