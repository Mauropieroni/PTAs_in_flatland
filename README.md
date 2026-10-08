# PTAs in Flatland

A 1-D toy model for pulsar timing array (PTA) searches for anisotropy in the
gravitational-wave background (GWB): pulsars sit on a ring instead of the
sky, which keeps the physics (a quadrupole-anisotropic overlap-reduction
function, Gaussian timing-residual statistics) but makes exact Bayesian
inference cheap enough to use as ground truth.

The code compares the exact (Monte-Carlo) Bayes factor against several
cheaper detection statistics for a GWB with weak or strong angular
anisotropy, for both a *simple* (fixed signal amplitude/orientation) and a
*composite* (amplitude/orientation marginalised over a prior) hypothesis
test:

- the classical Neyman-Pearson / deflection / cross-correlation map
  statistics (closed-form, quadratic in the data),
- a linear and a one-hidden-layer MLP classifier, trained to discriminate
  signal from noise directly on the data, as learned surrogates for the
  Bayes factor.

This is the code behind the figures in the accompanying paper (draft in
progress) with James Alvey, Andrea Mitridate and Joe Romano.

## Installation

```bash
pip install -e ".[dev]"
```

`dev` pulls in `pytest` and `torch` (needed only by the test that checks the
hand-rolled JAX training step against PyTorch's reference implementation,
`tests/test_train_jax_matches_torch.py`). For just running the pipeline:

```bash
pip install -e .
```

Requires Python >= 3.11.

## Reproducing the paper figures

```bash
python run_all.py          # generates data, trains the networks, computes
                            # ROC curves for all 4 scenarios (~tens of min)
jupyter notebook make_figures.ipynb   # assembles the 3 final figures
```

`run_all.py` runs the pipeline (`pta_flatland.main`) for the four scenarios
in `config_files/`: `simple_weak`, `simple_strong`, `composite_weak`,
`composite_strong`. Each run writes its datasets/models to `out/<scenario>/`
and its ROC curves to `plots/<scenario>/roc.npz`.

For the composite scenarios, the ROC figure also shows a "chance band" for
the linear classifier (its population optimum is the zero vector under the
composite prior, so its finite-sample fit is pure noise — see
`src/pta_flatland/chance_band.py` for why). Generate it with:

```bash
python -m pta_flatland.chance_band --config config_files/composite_weak.py
python -m pta_flatland.chance_band --config config_files/composite_strong.py
```

`make_figures.ipynb` then reads everything under `out/` and `plots/` and
writes the three figures to `figures/`:

| file | content |
|---|---|
| `figures/plot_roc.pdf` | ROC curves, all methods, simple vs. composite x weak vs. strong |
| `figures/weights.pdf` | linear-classifier weights vs. the analytic Neyman-Pearson weights |
| `figures/bf_vs_net.pdf` | network output vs. exact log Bayes factor, point by point |

Re-running a single method (e.g. after changing just the MLP architecture)
without regenerating data:

```bash
python -m pta_flatland.main --config config_files/simple_weak.py --methods mlp
```

## Physics model

- `N_pul` pulsars at fixed angular positions $\phi_a$ on a ring; response
  $F_a(\phi) = -\sin(\phi - \phi_a)$.
- GWB power as a function of sky angle, truncated to a quadrupole:
  $P(\phi) = P_0 + 2(P_{2r}\cos 2\phi - P_{2i}\sin 2\phi)$, $P_0 = 1/2\pi$.
- Overlap-reduction function $\Gamma_{ab}$ and covariance
  $C_{ab} = \sigma^2_{\rm noise}\delta_{ab} + \sigma^2_{\rm gwb}\Gamma_{ab}$.
- Data are the sufficient statistic $D_{ab} = \frac{1}{N_s}\sum_k x_{a,k}x_{b,k}$,
  Wishart-distributed given $C$.

See the module docstrings in `src/pta_flatland/model.py` and
`src/pta_flatland/det_stat.py` for the exact expressions and the Woodbury-
identity fast path that avoids ever forming/factorising the dense $N\times N$
covariance matrix.

## Package layout

```
src/pta_flatland/
    model.py           physical model: ORF, covariance, (log-)likelihood
    data_generation.py simulate D-matrix datasets
    det_stat.py         detection statistics: Bayes factor, likelihood ratio,
                        deflection, cross-correlation map, MLP
    train.py            JAX training loop (AdamW) + L-BFGS for the linear case
    roc.py               ROC/AUC computation and plotting
    chance_band.py      null-distribution band for the linear classifier
                        (composite hypothesis)
    utils.py             model I/O, feature extraction, prior sampling
    main.py              pipeline CLI (data gen -> train -> ROC)
config_files/            one scenario per file: simple/composite x weak/strong
tests/                   unit tests (physics, Woodbury identity, low-rank
                        sampling, MLE, JAX/PyTorch training parity)
make_figures.ipynb       assembles the 3 paper figures from out/ and plots/
run_all.py                runs all 4 scenarios end to end
```

## Tests

```bash
pytest
```

## License

GPLv3 or later — see [LICENSE](LICENSE).
