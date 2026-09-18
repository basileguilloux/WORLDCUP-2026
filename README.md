# WORLDCUP-2026

Match-outcome prediction for the 2026 FIFA World Cup. The target is the **3-way 90-minute result** — `0 = away win`, `1 = draw`, `2 = home win` — for the ~70 unplayed fixtures in `data/raw/results.csv`. The score itself is not modelled as an output; goals are only an intermediate quantity.

One consistent label is used everywhere, knockouts included: a tie at 90' is a draw, with no extra time or penalties. That keeps the backtest and the final output on the same definition.

## Setup

```bash
pip install -r requirements.txt
python run_pipeline.py
```

`run_pipeline.py` runs every step below from the repo root and writes `data/power_rankings.csv`, `data/fixture_predictions.csv` and `data/tournament_sim.csv`.

**It does not touch `data/predictions.csv`.** That file is the committed, news-adjusted final output, and refreshing it costs live API calls — see [Team-news adjustment](#team-news-adjustment). Pass `--news` to include that step, or `--skip-news` to be explicit about the default.

Any step can also be run on its own from the repo root:

```bash
python src/08_fixture_predictions.py
python src/10_tournament_sim.py 20000      # optional: number of simulations
```

Tests:

```bash
pip install -r requirements-dev.txt
pytest
```

## Pipeline (src/)

Run in order from the repo root. `harness.py`, `step2_dixoncoles.py`, `step3_classifier.py` and `eval_baseline.py` are deliberately **not** number-prefixed: they are imported as modules by the later stages, and `import 08_foo` is a syntax error in Python.

| # | Script | Reads | Writes |
|---|---|---|---|
| 1 | `01_load_clean.py` | `data/raw/results.csv`, `former_names.csv` | *(report only)* |
| 2 | `02_elo.py` | `data/raw/results.csv` | `data/processed_matches.csv` |
| 3 | `03_elo_baseline.py` | `data/processed_matches.csv` | *(report only)* |
| 4 | `04_features.py` | `data/processed_matches.csv` | `data/features.csv` |
| 5 | `05_model.py` | `data/features.csv` | `data/poisson_model.joblib` |
| — | `eval_baseline.py` | `data/features.csv` | *(report only)* — the number to beat |
| — | `harness.py` | `data/features.csv` | *(report only)* — walk-forward evaluator |
| — | `step2_dixoncoles.py` | `data/features.csv` | *(report only)* — sweeps Dixon-Coles `rho` |
| — | `step3_classifier.py` | `data/features.csv` | *(report only)* — **the winning model** |
| 6 | `06_simulate.py` | `results.csv`, `poisson_model.joblib` | *(report only)* |
| 7 | `07_blend_predict.py` | `results.csv`, `fifa_ranking_2026-06-11.csv` | `data/power_rankings.csv` |
| 8 | `08_fixture_predictions.py` | `features.csv`, `results.csv`, FIFA ranking | `data/fixture_predictions.csv` |
| 9 | `09_team_news.py` | the above + `data/team_news.json` | `data/predictions.csv` *(`--live` only)* |
| 10 | `10_tournament_sim.py` | `features.csv`, `results.csv`, FIFA ranking | `data/tournament_sim.csv` |

What each output is:

- **`data/power_rankings.csv`** — one row per team: expected points across the remaining fixtures, Elo, FIFA-equivalent Elo and the blend weight. A standings-style view, not per-match probabilities.
- **`data/fixture_predictions.csv`** — one row per fixture: `p_home` / `p_draw` / `p_away` from the Step-3 ensemble, with no team-news overlay.
- **`data/predictions.csv`** — the same fixtures with the team-news overlay applied, carrying both pre- and post-adjustment probabilities plus the signed news score, Elo delta and notes per team. **This is the headline output.**
- **`data/tournament_sim.csv`** — one row per team: probability of winning the group, qualifying, and reaching each knockout round through `champion`.

## Method

Each side's expected goals come from a **stacked Poisson regression** (one row per attacking side) on Elo differential, attack/opponent-defence form and home advantage. The score grid is corrected with the **Dixon-Coles** low-score adjustment (`rho = -0.15`, chosen by sweeping held-out log-loss), which lifts the 0-0 and 1-1 cells, then collapsed into W/D/L.

That Poisson + Dixon-Coles model is **both the baseline and a component of the final model**, not the final model itself. The predictions in `fixture_predictions.csv` and `predictions.csv` come from the **Step-3 ensemble** in `step3_classifier.py`: a multinomial logistic classifier that predicts the 3-way label *directly*, blended **0.3 Poisson/Dixon-Coles + 0.7 logistic** and temperature-calibrated (`T = 0.95`).

Walk-forward log-loss (5-fold, ~49k matches — lower is better):

| Model | log-loss |
|---|---|
| Naive base rate | 1.0510 |
| Poisson (`rho = 0`) | 0.9120 |
| Poisson + Dixon-Coles (`rho = -0.15`) | 0.9091 |
| Multinomial logistic alone | 0.9033 |
| **Step-3 ensemble (final)** | **0.9024** |

Most of the gain comes from predicting the 3-way label directly rather than from the Dixon-Coles correction, which is worth only ~0.003 on its own.

A note on draws: the predicted-draw *rate* under argmax is near zero, which looks like draw blindness but is a measurement artifact. Mean predicted `P(draw)` is 0.219–0.236 against a 0.237 base rate, so draws are well calibrated in probability — a draw is simply rarely any single match's single most likely class. Judge the model by log-loss and Brier, not the confusion matrix.

Evaluation is **walk-forward** (`harness.py`, `TimeSeriesSplit`, 5 folds) throughout, so no fold is scored on data that preceded its training window.

## Tournament simulation

The rest of the pipeline stops at per-fixture win/draw/loss probabilities. `src/10_tournament_sim.py` plays the whole 2026 bracket out 20,000 times with the Step 3 ensemble model, to turn those into P(reaches this round) and P(lifts the trophy).

Top of the board across the 20,000 simulated tournaments.

| # | Team | Win group | Reach SF | Reach final | Champion |
|---|------|-----------|----------|--------------|----------|
| 1 | Spain | 77.7% | 47.7% | 33.3% | 22.7% |
| 2 | Argentina | 73.2% | 43.6% | 29.9% | 19.3% |
| 3 | England | 64.9% | 31.6% | 18.3% | 10.0% |
| 4 | France | 59.7% | 30.5% | 18.0% | 9.9% |
| 5 | Portugal | 46.8% | 18.8% | 9.0% | 3.8% |
| 6 | Germany | 46.8% | 17.8% | 8.3% | 3.6% |
| 7 | Japan | 43.1% | 17.0% | 7.9% | 3.3% |
| 8 | Brazil | 43.5% | 16.0% | 7.5% | 3.2% |

Four teams hold 62% of the title probability between them. The hosts trail well behind, Mexico at 2.5%, Canada at 0.6% and the USA at 0.1%. The USA lands in a tough Group B with Turkey, Paraguay and Australia and wins that group only 30% of the time. Full table in `data/tournament_sim.csv`.

`10_tournament_sim.py` **refits the ensemble in-process from `data/features.csv` rather than reading `predictions.csv`** — it needs a probability for knockout pairings that do not exist in the fixture list, so no prediction CSV is an input to it. It therefore ignores the team-news overlay.

The tournament simulation samples each match's outcome class from the calibrated ensemble, then samples a scoreline from the Dixon-Coles grid *conditional* on that class, which supplies the goal differences group tables need without disturbing outcome calibration. Knockout seeding is an approximation of the official bracket, which is not in the data: qualifiers are seeded 1-32 by group record into a standard 1v32 bracket, and a knockout draw at 90' is resolved by a strength-weighted coin flip.

### How the real tournament went

Spain beat Argentina 1-0 after extra time in the final on July 19 2026, Ferran Torres scoring the winner. It's Spain's second World Cup title, after 2010.

The simulation's top two teams by reach-final probability were Spain (33.3%) and Argentina (29.9%), exactly the pair that met in the final. It also ranked Spain first to win it all at 22.7%, ahead of Argentina at 19.3%, so the champion pick landed too.

England then beat France 6-4 in the third-place playoff on July 18. Those were also the model's #3 and #4 ranked teams by champion probability, so all four semifinalists it favored most landed in the right places.

## Team-news adjustment

`09_team_news.py` is an **optional**, transparent, prediction-time overlay. It does not retrain the model and does not add a trained feature.

A Claude agent with web search covers every team and returns a signed score per team, `news_score` in `[-1, +1]` — negative for injuries, suspensions, off-field turmoil or poor preparation; positive for key players returning, strong form or settled preparation. That becomes a signed Elo adjustment, `delta = MAX_SWING * news_score` (default `MAX_SWING = 120`, deliberately larger than the 60-point home advantage), applied to the team's strength **input** and fed through the unchanged Step-3 ensemble. Results are cached in `data/team_news.json` as an audit artifact.

It needs `ANTHROPIC_API_KEY`, read from `~/.worldcup2026.env` (never committed):

```bash
echo 'export ANTHROPIC_API_KEY="sk-ant-..."' > ~/.worldcup2026.env
python run_pipeline.py --news          # or: python src/09_team_news.py --live --refresh
```

Without `--live` the script is a **cache-only dry run** and writes `data/predictions_offline.csv`, leaving `data/predictions.csv` alone — teams missing from the cache default to neutral, so an offline run is a near-no-op overlay and is not a substitute for a live one.

`scripts/refresh_team_news.sh` runs the live refresh every 2 days via a LaunchAgent; see `scripts/worldcup-news.plist.example` for installation. Without the env file it logs a SKIP and makes zero API calls.

## Data

`data/raw/` holds the inputs: ~49k historical results (played rows have scores, 2026 fixtures have NA), a FIFA ranking snapshot, goalscorers, shootouts, and a former-name mapping so each country's history sits under one current name. `data/` holds derived artifacts — the cleaned match table, engineered features, the saved Poisson model, the team-news cache and the four output CSVs above, including `tournament_sim.csv` with the Monte Carlo output described above.

## Status

Complete and evaluated against baselines: data cleaning, Elo, feature engineering, the Poisson/Dixon-Coles model, the Step-3 ensemble, FIFA blending, per-fixture prediction, the optional team-news overlay and the tournament Monte Carlo — with a single entry point (`run_pipeline.py`), a pinned `requirements.txt`, smoke tests in `tests/`, and an MIT license.

Not yet included: CI, and visualizations of the predictions. Known limitation: features are the bottleneck — `elo_diff` dominates, and further gains need richer data (squad/market value, rest and travel) rather than more model tuning.
