# worldcup-2026

Predict the outcomes of the unplayed **FIFA World Cup 2026** fixtures.

## The prediction target (read first)

- **Target:** 3-way **90-minute (regulation) result** — `0 = away win`, `1 = draw`, `2 = home win`.
- The **score does not matter**, only the outcome class.
- This applies to **every** game including knockouts: a tie at 90' is a **Draw** (no extra time, no penalties). One consistent 3-class label across the whole tournament.
- **End product:** per-game W/D/L probabilities for the ~70 unplayed fixtures — the rows with missing scores in `data/raw/results.csv`.
- **Why one label everywhere:** keeps the backtest and the final output on the same definition, and makes draw-calibration relevant in knockouts too.

## How to evaluate

- Primary metric: **3-class log-loss**. Also report **multiclass Brier** and **accuracy**, and watch the **predicted-draw rate** vs. actual.
- The independent-Poisson grid **under-predicts draws**. The fix in progress is the **Dixon-Coles** low-score correction (negative `rho` lifts the 0-0 / 1-1 cells); a direct 3-way classifier is the other option on the table.
- Use the **walk-forward** harness for honest held-out numbers — not a single split.

## Baseline results (number to beat)

Measured 2026-06-13. The **honest, comparable** figure is the walk-forward log-loss.

| Model | log-loss ↓ | Brier ↓ | accuracy | pred-draw vs true |
|---|---|---|---|---|
| Naive base-rate | 1.0510 | — | — | — |
| Poisson, walk-forward (rho=0) | **0.9120** | 0.5379 | 0.576 | 0.000 vs 0.237 |
| Poisson + Dixon-Coles (rho=−0.15) | 0.9091 | 0.5368 | 0.576 | 0.002 vs 0.237 |
| Poisson, single 2022+ split | 0.8809 | 0.5180 | 0.599 | 0.000 vs 0.229 |
| Multinomial logistic (direct 3-way) | 0.9033 | 0.5328 | 0.580 | — |
| **FINAL ensemble** (0.3·DC-Poisson + 0.7·logit, T=0.95) | **0.9024** | 0.5325 | 0.580 | — |

**Current best: 0.9024 walk-forward log-loss** (`src/step3_classifier.py`), a +1.1%
improvement over the 0.9120 Poisson baseline, most of it from switching to a model
that predicts the 3-way label directly.

**Key findings:**
- The "draw blindness" (pred-draw rate ~0.000) was a **measurement artifact of argmax**,
  not a calibration bug. Mean predicted `P(draw)` is 0.219–0.236 vs a 0.237 base rate —
  draws are well-calibrated *in probability*. A draw is simply rarely any single match's
  most-likely class, which is correct. Judge by log-loss/Brier, not the confusion matrix.
- Dixon-Coles barely helps on its own (0.0029); a **direct classifier** is the real win.
- `abs_elo_diff` (draws peak when teams are evenly matched) is a cheap feature that linear
  models can't derive from signed `elo_diff` alone.
- Features are now the bottleneck — `elo_diff` dominates. Further gains need richer data
  (squad/market value, rest/travel), with rising data-collection cost and overfitting risk.

## Project layout

- `data/raw/` — inputs: `results.csv` (~49k historical matches; played rows have scores, future fixtures have NA), `fifa_ranking_2026-06-11.csv`, `goalscorers.csv`, `shootouts.csv`, `former_names.csv` (old→current country names).
- `data/` — derived artifacts: `processed_matches.csv`, `features.csv`, `poisson_model.joblib`.
- `src/` — numbered pipeline (run in order from the project root):
  - `01_load_clean.py` — load raw results, reconcile former country names, split played vs. future.
  - `02_elo.py` — pre-match Elo ratings (start 1500, home advantage 60).
  - `03_elo_baseline.py` — Elo-only baseline.
  - `04_features.py` — rolling recent-form features (last `N=5` matches, no leakage) + host-nation/venue flag → `features.csv`.
  - `05_model.py` — stacked **PoissonRegressor** (one row per attacking side) → expected goals → score grid → W/D/L. Saves `poisson_model.joblib`.
  - `06_simulate.py` — replay played matches for current Elo/form, predict upcoming fixtures (host bonus), rank teams.
  - `07_blend_predict.py` — blend the FIFA ranking snapshot in as a **prior** for current strength (trust Elo for data-rich teams, FIFA for data-poor), then re-predict.
  - `08_predict_final.py` — **FINAL** per-fixture predictions using the Step-3 ensemble + FIFA-blended strength. Writes `data/predictions.csv` (date, teams, p_home/p_draw/p_away, pick). Run with `PYTHONPATH=src`.
- `src/harness.py` — reusable vectorised **walk-forward** evaluator for the 3-way target; supports the Dixon-Coles `rho` correction. Import `walk_forward`.
- `src/step2_dixoncoles.py` — sweeps `rho` on the harness, keeps the value minimising held-out log-loss.
- `src/step3_classifier.py` — direct 3-way classifier (multinomial logistic / HGB) + Poisson ensemble + temperature calibration; the current best model. Reuses `harness` for apples-to-apples eval. Run with `PYTHONPATH=src`.
- `src/eval_baseline.py` — reproduces the `05_model` split and reports all three 3-class metrics (the number to beat).

## Conventions

- Run scripts **from the project root** — paths like `data/raw/...` are relative to it.
- Python via the local `.venv`.
- Label order is fixed everywhere: `0=away, 1=draw, 2=home`. Keep it consistent across training, eval, and output.
- Host nations: `{"United States", "Mexico", "Canada"}`.
