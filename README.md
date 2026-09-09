# WORLDCUP-2026

Match-outcome prediction pipeline for the 2026 FIFA World Cup: predicts home win / draw / away win probabilities for international matches, then ranks the 2026 field by expected points across all remaining fixtures.

## Pipeline (src/)

1. `01_load_clean.py` - load and clean the raw match data
2. `02_elo.py` / `03_elo_baseline.py` - compute team Elo ratings and an Elo-only baseline
3. `04_features.py` - build model features (Elo differential, attack/defense strength, home advantage), using only past information (no leakage)
4. `05_model.py` - fit a stacked Poisson regression model (one row per attacking side) to estimate each side's expected goals
5. `eval_baseline.py` / `harness.py` - walk-forward evaluation harness (5-fold, ~49k matches) reporting log-loss, accuracy, Brier score, and confusion matrix against the Elo baseline
6. `step2_dixoncoles.py` - sweeps the Dixon-Coles low-score correlation parameter and keeps the value that minimises held-out log-loss
7. `06_simulate.py` - replays played matches for each team's current Elo/form, then predicts every upcoming fixture and ranks teams by expected points
8. `07_blend_predict.py` - blends the official FIFA ranking into each team's strength (weighted by how many recent matches they've played) and re-predicts the fixtures

## Method

Each side's expected goals are estimated with a Poisson regression on Elo differential, attack/opponent-defense strength, and home advantage. The resulting score grid is optionally corrected with the Dixon-Coles low-score adjustment, then collapsed into win/draw/loss probabilities. Fixture-level probabilities are aggregated into expected points to produce a power ranking of the field - this is an analytical (closed-form) aggregation, not a Monte Carlo bracket simulation. Model evaluation uses walk-forward (time-series) cross-validation to avoid lookahead bias.

## Data

`data/` holds the underlying match history (results, FIFA rankings, goalscorers, former team names) used to fit and evaluate the models, plus the saved model artifact and engineered feature table.

## Status

Core pipeline (data cleaning, Elo, feature engineering, Poisson model, Dixon-Coles tuning, FIFA blending, fixture prediction) is complete and evaluated against baselines. Not yet included: a dependency/requirements file, a single entry-point script to run the full pipeline in order, and a saved output of the final predictions (currently printed to stdout rather than written to a file).
