# WORLDCUP-2026

Match-outcome prediction pipeline for the 2026 FIFA World Cup: predicts home win / draw / away win probabilities for international matches.

## Pipeline (src/)

1. `01_load_clean.py` - load and clean the raw match data
2. `02_elo.py` / `03_elo_baseline.py` - compute team Elo ratings and an Elo-only baseline
3. `04_features.py` - build model features (Elo differential, attack/defense strength, home advantage)
4. `05_model.py` - fit a stacked Poisson regression model (one row per attacking side) to estimate each side's expected goals
5. `step2_dixoncoles.py` - Dixon-Coles low-score correlation correction applied to the score grid
6. `06_simulate.py` - Monte Carlo tournament simulation
7. `07_blend_predict.py` - blend model outputs into final match predictions
8. `harness.py` / `eval_baseline.py` - walk-forward evaluation harness (5-fold, ~49k matches) reporting accuracy, log-loss, and confusion matrix against the Elo baseline

## Method

Each side's expected goals are estimated with a Poisson regression on Elo differential, attack/opponent-defense strength, and home advantage. The resulting score grid is optionally corrected with the Dixon-Coles low-score adjustment, then collapsed into win/draw/loss probabilities. Model evaluation uses walk-forward (time-series) cross-validation to avoid lookahead bias.

## Data

`data/` holds the underlying match history used to fit and evaluate the models.
