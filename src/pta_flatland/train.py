"""
Training loop for ML-based detection statistics -- pure JAX (forward pass,
loss, and a hand-rolled AdamW optimizer), replacing the previous
PyTorch-based loop. AdamW is hand-rolled rather than pulled from a library
so nothing sits between this code and the exact update rule (see
_adamw_update): it reproduces torch.optim.AdamW's math term-for-term, see
tests/test_train_jax_matches_torch.py for the cross-check against torch.
"""
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from .utils import triu_features
from .det_stat import init_mlp_params, mlp_apply

# torch.optim.AdamW's defaults -- the old loop only ever overrode lr
# (torch.optim.AdamW(model.parameters(), lr=config.learning_rate)), so these
# are what it was actually running with. Named here so _adamw_update and its
# cross-check test both reference the same numbers.
_BETA1 = 0.9
_BETA2 = 0.999
_EPS = 1e-8
_WEIGHT_DECAY = 0.01


def _bce_with_logits(logits, y):
    """Mean binary cross-entropy from logits -- matches
    torch.nn.BCEWithLogitsLoss()'s default (mean reduction, no pos_weight)."""
    return -(y * jax.nn.log_sigmoid(logits) + (1 - y) * jax.nn.log_sigmoid(-logits)).mean()


def _adamw_init(params):
    return dict(
        m=jax.tree_util.tree_map(jnp.zeros_like, params),
        v=jax.tree_util.tree_map(jnp.zeros_like, params),
        step=0,
    )


def _adamw_update(params, grads, opt_state, lr, weight_decay):
    """One torch.optim.AdamW step: decoupled weight decay applied to the
    pre-update params, then a bias-corrected Adam step -- see
    tests/test_train_jax_matches_torch.py for the term-by-term equivalence
    check against torch.optim.AdamW.

    weight_decay is a genuine argument (not the module-level _WEIGHT_DECAY
    baked in as a Python constant) specifically so it stays a normal JAX
    input wherever this is called from inside @jax.jit -- a plain Python
    global read there would get frozen into the compiled program at the
    first trace and silently ignore any later change to it, unlike lr.
    """
    step = opt_state["step"] + 1
    m = jax.tree_util.tree_map(
        lambda m_, g: _BETA1 * m_ + (1 - _BETA1) * g, opt_state["m"], grads
    )
    v = jax.tree_util.tree_map(
        lambda v_, g: _BETA2 * v_ + (1 - _BETA2) * g * g, opt_state["v"], grads
    )
    bc1 = 1.0 - _BETA1**step
    bc2 = 1.0 - _BETA2**step
    new_params = jax.tree_util.tree_map(
        lambda p, m_, v_: p
        - lr * weight_decay * p
        - lr * (m_ / bc1) / (jnp.sqrt(v_ / bc2) + _EPS),
        params,
        m,
        v,
    )
    return new_params, dict(m=m, v=v, step=step)


@jax.jit
def _train_step(params, opt_state, X_tr, y_tr, batch_idx, lr, weight_decay):
    """One minibatch step, including the batch gather -- fused into a
    single compiled program (previously a separate un-jitted fancy-index
    dispatch before calling this) so each step is one XLA call, not two."""
    x_batch, y_batch = X_tr[batch_idx], y_tr[batch_idx]

    def loss_fn(p):
        return _bce_with_logits(mlp_apply(p, x_batch), y_batch)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    params, opt_state = _adamw_update(params, grads, opt_state, lr, weight_decay)
    # Return the batch's *summed* (not mean) loss, still as a device array --
    # the caller accumulates this across a whole epoch's worth of steps
    # before ever converting to a Python float, so JAX can dispatch/pipeline
    # every step in the epoch without blocking on each one individually.
    return params, opt_state, loss * batch_idx.shape[0]


@jax.jit
def _eval_loss(params, x, y):
    return _bce_with_logits(mlp_apply(params, x), y)


def _method_param(config, name, method, default):
    """config value for `name`, optionally overridden per ML method via a
    `<name>_by_method` dict, e.g.:

        batch_size_by_method = {"mlp_linear": 200_000, "mlp": 4096}

    mlp_linear (convex, logistic loss) wants full-batch for a reproducible
    optimum; mlp (non-convex) wants minibatch stochasticity to actually
    converge -- a single global batch_size can't serve both when
    mlp_linear has no real signal to learn (composite_intermediate.py) and
    minibatch noise would make its result an unreproducible coin flip. Only
    scenarios that need the split set the `_by_method` dict; everything
    else keeps a single plain `name` config value as before.
    """
    overrides = getattr(config, f"{name}_by_method", None)
    if overrides and method in overrides:
        return overrides[method]
    return getattr(config, name, default)


def train_linear_lbfgs(X_tr, y_tr, X_val, y_val, max_iter=5000, gtol=1e-10):
    """Fit the linear network (logistic regression) exactly: full-batch
    L-BFGS from zero init, no weight decay, no early stopping.

    The linear BCE problem is convex with a single optimum, so the fit is
    run to convergence rather than stopped early. With AdamW (lr=1e-2,
    decoupled weight decay, early stopping on the val loss) the weights are
    frozen part-way through training, wherever the per-coordinate Adam steps
    happen to have left them. Under the composite disk prior, where the
    linear population optimum is w = 0, that made the composite test ROC
    land anywhere between AUC ~0.2 and ~0.8 from run to run (see
    chance_band.py). Weight decay is dropped because it stops the fit from
    whitening the features, which is what suppresses the no-signal
    directions; with n_train >> n_features the unregularised optimum is well
    defined.

    Returns (params, train_losses, val_losses) like train(), with one loss
    entry per L-BFGS iteration; params use init_mlp_params' layout.
    """
    from scipy.optimize import minimize

    in_dim = X_tr.shape[1]

    def unpack(theta):
        return [(theta[:-1].reshape(in_dim, 1), theta[-1:])]

    loss_and_grad = jax.jit(
        jax.value_and_grad(lambda t: _bce_with_logits(mlp_apply(unpack(t), X_tr), y_tr))
    )

    def fun(theta):
        loss, grad = loss_and_grad(jnp.asarray(theta))
        return float(loss), np.asarray(grad)

    train_losses, val_losses = [], []

    def record(theta):
        params = unpack(jnp.asarray(theta))
        train_losses.append(float(_eval_loss(params, X_tr, y_tr)))
        val_losses.append(float(_eval_loss(params, X_val, y_val)))

    res = minimize(
        fun,
        np.zeros(in_dim + 1),
        jac=True,
        method="L-BFGS-B",
        callback=record,
        options=dict(maxiter=max_iter, gtol=gtol),
    )
    print(
        f"  L-BFGS: {res.message} after {res.nit} iterations"
        f"  train={train_losses[-1]:.6f}  val={val_losses[-1]:.6f}",
        flush=True,
    )
    return unpack(jnp.asarray(res.x)), train_losses, val_losses


def train(key, in_dim, hidden, datasets, method=None):
    """
    Train a binary classifier on the pre-generated D-matrix datasets.
    A linear network (hidden=[]) is fitted with train_linear_lbfgs instead
    of the AdamW loop below. With config fresh_data = k (per method, see
    _method_param) the training split is regenerated every k epochs.

    Parameters
    ----------
    key      : JAX PRNGKey  for weight init and per-epoch minibatch shuffling
    in_dim   : int          number of (triu) input features
    hidden   : list of int  hidden-layer widths ([] gives a single linear
                             layer -- the old mlp_linear architecture)
    datasets : dict         output of data_generation.build_datasets
    method   : str, optional  "mlp" / "mlp_linear" / ... -- selects
               per-method hyperparameter overrides, see _method_param.

    Returns
    -------
    params       : best params (from the lowest val-loss epoch)
    train_losses : list of float, one per epoch
    val_losses   : list of float, one per epoch
    """
    import config  # deferred: only needed once a real run config is loaded

    weight_decay = _method_param(config, "weight_decay", method, _WEIGHT_DECAY)
    batch_size = _method_param(config, "batch_size", method, config.batch_size)
    n_epochs = _method_param(config, "n_epochs", method, config.n_epochs)
    learning_rate = _method_param(config, "learning_rate", method, config.learning_rate)
    early_stop_patience = _method_param(
        config, "early_stop_patience", method, config.early_stop_patience
    )

    # Build feature matrix and labels
    X = triu_features(
        np.concatenate([datasets["D_train_null"], datasets["D_train_sig"]], axis=0)
    )
    X = (X - datasets["_mu"]) / datasets["_sig_std"]
    n = len(X)
    y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])

    # Shuffle and split
    idx = np.random.default_rng(config.seed).permutation(n)
    n_val = int(n * config.val_split)
    idx_val, idx_tr = idx[:n_val], idx[n_val:]

    X_tr, y_tr = jnp.asarray(X[idx_tr]), jnp.asarray(y[idx_tr])
    X_val, y_val = jnp.asarray(X[idx_val]), jnp.asarray(y[idx_val])
    n_tr = len(X_tr)

    if not hidden:
        return train_linear_lbfgs(X_tr, y_tr, X_val, y_val)

    key, init_key = jax.random.split(key)
    params = init_mlp_params(init_key, in_dim, hidden)
    opt_state = _adamw_init(params)

    train_losses, val_losses = [], []
    best_val = float("inf")
    best_params = params
    patience = 0

    # fresh_data = k (per method, see _method_param): every k epochs replace
    # the training split with a new draw of the same size from
    # config.train_priors, generated and standardised (stored mu, sd) in JAX
    # -- see data_generation.training_features. The network then cannot
    # overfit one finite training set. The validation split stays fixed, so
    # early stopping still compares epochs on the same data. 0/False = off.
    fresh_every = int(_method_param(config, "fresh_data", method, 0))
    if fresh_every:
        from .data_generation import training_features

        data_key = jax.random.fold_in(key, 1)
        phi_j, Ns = jnp.asarray(datasets["_phi_j"]), int(datasets["_Ns"])

    for epoch in range(1, n_epochs + 1):
        if fresh_every and epoch > 1 and (epoch - 1) % fresh_every == 0:
            X_tr, y_tr, data_key = training_features(
                phi_j, Ns, n_tr // 2, config.train_priors, data_key,
                datasets["_mu"], datasets["_sig_std"],
            )

        key, shuffle_key = jax.random.split(key)
        perm = jax.random.permutation(shuffle_key, n_tr)

        # running stays a JAX scalar for the whole epoch -- only converted
        # to a Python float once below, so every step this epoch dispatches
        # without blocking on the previous one's result.
        running = jnp.zeros(())
        for start in range(0, n_tr, batch_size):
            batch_idx = perm[start : start + batch_size]
            params, opt_state, batch_loss_sum = _train_step(
                params, opt_state, X_tr, y_tr, batch_idx, learning_rate, weight_decay
            )
            running = running + batch_loss_sum
        train_losses.append(float(running) / n_tr)

        val_loss = float(_eval_loss(params, X_val, y_val))
        val_losses.append(val_loss)

        if epoch % 10 == 0:
            print(
                f"  epoch {epoch:3d}/{n_epochs}"
                f"  train={train_losses[-1]:.4f}  val={val_loss:.4f}",
                flush=True,
            )

        # params is a fresh pytree each step (JAX arrays are immutable), so
        # this is a cheap reference grab -- no copy.deepcopy needed, unlike
        # the old torch state_dict checkpointing.
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_params = params
            patience = 0
        else:
            patience += 1
            if patience >= early_stop_patience:
                print(
                    f"  Early stop at epoch {epoch} (best val={best_val:.4f})",
                    flush=True,
                )
                break

    return best_params, train_losses, val_losses


def plot_training_curves(train_losses, val_losses, path):
    epochs = range(1, len(train_losses) + 1)
    best_ep = int(np.argmin(val_losses)) + 1
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_losses, label="train")
    ax.plot(epochs, val_losses, label="val")
    ax.axvline(best_ep, color="grey", ls=":", label=f"best epoch ({best_ep})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved: {path}", flush=True)
