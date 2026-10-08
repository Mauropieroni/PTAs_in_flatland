"""
Verifies the hand-rolled JAX training path (det_stat.init_mlp_params /
mlp_apply, train.py's forward pass, loss, and AdamW step) reproduces the
same math as PyTorch's nn.Linear / nn.ReLU / BCEWithLogitsLoss / AdamW.

NOT tested here: that a from-scratch JAX training run reproduces a
from-scratch torch training run bit-for-bit -- torch and JAX use unrelated
RNGs, so weight init and minibatch shuffling necessarily differ and a full
training trajectory can't be made to match. What IS pinned down exactly is
the underlying math: given IDENTICAL weights/inputs copied across
frameworks, the forward pass, the loss, and one AdamW parameter update must
agree to floating-point precision.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from pta_flatland.det_stat import init_mlp_params, mlp_apply
from pta_flatland.train import (
    _bce_with_logits,
    _adamw_init,
    _adamw_update,
    _BETA1,
    _BETA2,
    _EPS,
    _WEIGHT_DECAY,
)


def _torch_mlp(in_dim, hidden):
    """Same architecture det_stat.init_mlp_params/mlp_apply builds: Linear+
    ReLU per hidden width, final Linear with no activation."""
    layers = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers).double()


def _copy_jax_to_torch(params, torch_model):
    """Copy a det_stat params pytree (list of (w, b), w: (fan_in, fan_out))
    into a torch nn.Sequential's Linear layers (weight: (fan_out, fan_in))."""
    linears = [m for m in torch_model if isinstance(m, nn.Linear)]
    assert len(linears) == len(params)
    with torch.no_grad():
        for lin, (w, b) in zip(linears, params):
            lin.weight.copy_(torch.tensor(np.asarray(w).T, dtype=torch.float64))
            lin.bias.copy_(torch.tensor(np.asarray(b), dtype=torch.float64))


@pytest.mark.parametrize("hidden", [[], [4], [8, 4]])
def test_forward_pass_matches_torch(hidden):
    key = jax.random.PRNGKey(0)
    in_dim, batch = 6, 11
    params = init_mlp_params(key, in_dim, hidden)

    torch_model = _torch_mlp(in_dim, hidden)
    _copy_jax_to_torch(params, torch_model)

    x_np = np.random.default_rng(1).normal(size=(batch, in_dim))
    out_jax = np.asarray(mlp_apply(params, jnp.asarray(x_np)))
    out_torch = (
        torch_model(torch.tensor(x_np, dtype=torch.float64)).squeeze(-1).detach().numpy()
    )

    np.testing.assert_allclose(out_jax, out_torch, rtol=1e-10, atol=1e-12)


def test_loss_matches_torch_bce_with_logits():
    rng = np.random.default_rng(2)
    logits_np = rng.normal(size=20)
    y_np = rng.integers(0, 2, size=20).astype(np.float64)

    loss_jax = float(_bce_with_logits(jnp.asarray(logits_np), jnp.asarray(y_np)))
    loss_torch = nn.BCEWithLogitsLoss()(
        torch.tensor(logits_np, dtype=torch.float64),
        torch.tensor(y_np, dtype=torch.float64),
    ).item()

    assert loss_jax == pytest.approx(loss_torch, rel=1e-12, abs=1e-14)


@pytest.mark.parametrize("hidden", [[], [5, 3]])
def test_one_adamw_step_matches_torch(hidden):
    """Forward + loss + backward + one AdamW step, from IDENTICAL initial
    weights and the same batch in both frameworks -- exercises gradients
    and the parameter update, not just the forward pass."""
    key = jax.random.PRNGKey(0)
    in_dim, batch = 5, 9
    lr = 1e-2

    params = init_mlp_params(key, in_dim, hidden)
    torch_model = _torch_mlp(in_dim, hidden)
    _copy_jax_to_torch(params, torch_model)

    rng = np.random.default_rng(3)
    x_np = rng.normal(size=(batch, in_dim))
    y_np = rng.integers(0, 2, size=batch).astype(np.float64)
    x_jax, y_jax = jnp.asarray(x_np), jnp.asarray(y_np)

    # -- jax: one hand-rolled AdamW step --
    def loss_fn(p):
        return _bce_with_logits(mlp_apply(p, x_jax), y_jax)

    loss_jax, grads = jax.value_and_grad(loss_fn)(params)
    opt_state = _adamw_init(params)
    new_params, _ = _adamw_update(params, grads, opt_state, lr, _WEIGHT_DECAY)

    # -- torch: one AdamW step, matching hyperparameters --
    optimizer = torch.optim.AdamW(
        torch_model.parameters(),
        lr=lr,
        betas=(_BETA1, _BETA2),
        eps=_EPS,
        weight_decay=_WEIGHT_DECAY,
    )
    loss_torch = nn.BCEWithLogitsLoss()(
        torch_model(torch.tensor(x_np, dtype=torch.float64)).squeeze(-1),
        torch.tensor(y_np, dtype=torch.float64),
    )
    optimizer.zero_grad()
    loss_torch.backward()
    optimizer.step()

    assert float(loss_jax) == pytest.approx(loss_torch.item(), rel=1e-10)

    linears = [m for m in torch_model if isinstance(m, nn.Linear)]
    for (w, b), lin in zip(new_params, linears):
        np.testing.assert_allclose(
            np.asarray(w).T, lin.weight.detach().numpy(), rtol=1e-8, atol=1e-10
        )
        np.testing.assert_allclose(
            np.asarray(b), lin.bias.detach().numpy(), rtol=1e-8, atol=1e-10
        )
