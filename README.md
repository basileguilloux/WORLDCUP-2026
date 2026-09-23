# WORLDCUP-2026

## TL;DR

- Predicts the 3-way 90-minute result (away win, draw, home win) for the 2026 World Cup. The model blends Elo, a Dixon-Coles Poisson component and a multinomial logit, then calibrates the blend with a temperature.
- Backtest: 0.9024 walk-forward log-loss on about 49k matches, against 1.0510 for the naive base rate. This is the main claim.
- Tournament check: on the 68 group games played after the forecast commit the log-loss is 0.8851, against 0.9176 for an Elo-only model. A paired bootstrap says that edge is not distinguishable from zero. The margin over a uniform prior (1.0986) is.
- Provenance caveat: no server saw the forecast commit before 18 September, so its 13 June date rests on local git metadata. History was rewritten on 23 September 2026 to remove an internal notes file, which changed every commit hash. The real guarantee is that the inputs contain no result after 11 June. See "The frozen forecast" and "Leakage caveats" below.

Match-outcome prediction for the 2026 FIFA World Cup. The target is the **3-way 90-minute result** — `0 = away win`, `1 = draw`, `2 = home win` — for the ~70 unplayed fixtures in `data/raw/results.csv`. The score itself is not modelled as an output; goals are only an intermediate quantity.

One consistent label is used everywhere, knockouts included: a tie at 90' is a draw, with no extra time or penalties. That keeps the backtest and the final output on the same definition.

## Setup

```bash
pip install -r requirements.txt
python run_pipeline.py
```

`run_pipeline.py` runs every step below from the repo root and writes `data/power_rankings.csv`, `data/fixture_predictions.csv` and `data/tournament_sim.csv`.

**It does not touch `data/predictions.csv`.** That file is the **frozen forecast** (see below). `--news` is retired and exits 1 with an explanation; `--skip-news` is an explicit no-op.

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
| 11 | `11_score_tournament.py` | `predictions.csv`, `wc26_actual_results.csv` | *(report only)* — scores the frozen forecast |

What each output is:

- **`data/power_rankings.csv`** — one row per team: expected points across the remaining fixtures, Elo, FIFA-equivalent Elo and the blend weight. A standings-style view, not per-match probabilities.
- **`data/fixture_predictions.csv`** — one row per fixture: `p_home` / `p_draw` / `p_away` from the Step-3 ensemble, with no team-news overlay.
- **`data/predictions.csv`** — the headline output, and **frozen**. Its `p_*_pre` columns are the forecast; its `p_*_post` columns are a later team-news annotation that is not scored. See [The frozen forecast](#the-frozen-forecast).
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

### How it actually did

The tournament has since been played, so the frozen forecast can be scored against real results. `src/11_score_tournament.py` scores the `p_*_pre` columns against `data/raw/wc26_actual_results.csv` over the **68** group-stage fixtures that kicked off after the forecast was committed (the two 12 June fixtures are excluded — see [Leakage caveats](#leakage-caveats)).

| Model | log-loss ↓ | 95% CI (marginal) | Brier ↓ | accuracy |
|---|---|---|---|---|
| **Frozen forecast (`p_*_pre`)** | **0.8851** | [0.7654, 1.0122] | 0.5248 | 0.618 |
| Elo-only logistic (the `03` baseline) | 0.9176 | [0.7873, 1.0556] | 0.5538 | 0.618 |
| Uniform 1/3 (ln 3) | 1.0986 | — | 0.6667 | n/a |
| *post-news annotation (`p_*_post`), secondary* | *0.8876* | *[0.7663, 1.0161]* | *0.5258* | *0.618* |

Uniform accuracy is n/a: argmax over a three-way tie is arbitrary.

**Paired bootstrap** (10,000 resamples, seed 20260613). Comparing two models by whether their marginal intervals overlap is the wrong test — both models are scored on the same fixtures and rise and fall together on them. Resampling the same fixture indices for both and taking the mean per-fixture log-loss difference cancels that shared difficulty:

| Comparison | mean difference | 95% CI | verdict |
|---|---|---|---|
| Forecast − Elo-only | **−0.0325** | [−0.0753, +0.0099] | spans 0 — not distinguishable |
| post-news − pre-news | +0.0025 | [−0.0015, +0.0077] | spans 0 — not distinguishable |

**Neither difference is established.** The forecast's 0.0325 edge over the Elo-only baseline is in its favour and the paired interval is far tighter than the marginal ones, but it still crosses zero (upper bound +0.0099), so on 68 fixtures the model is not shown to beat Elo alone. The news annotation's effect is likewise indistinguishable from zero, and its point estimate is slightly *worse*.

Against the uniform prior the margin is not in doubt: 0.8851 vs 1.0986, with 61.8% of the 68 results called correctly.

**Calibration.** Marginal outcome rates match the predicted rates closely — predicted home 0.447 / draw 0.236 / away 0.317 against actual home 0.441 / draw 0.279 / away 0.279. The 19 observed draws against roughly 16 expected is within one standard deviation (about 3.5), so it is not evidence of a draw bias either way.

**Sensitivity.** Dropping the four 13 June fixtures as well — the ones sharing a date with the commit, kept in the headline because they kicked off hours after it — leaves **n = 64**, log-loss **0.8667**, Brier 0.5110, accuracy 0.641, and a paired forecast − Elo-only difference of **−0.0364, 95% CI [−0.0799, +0.0065]**. The conclusion is unchanged: slightly better, still not distinguishable.

**On comparing with the 0.9024 backtest.** The tournament figure is nominally better (0.8851), but the two are not on the same footing: the walk-forward number averages over ~49k mostly-lopsided friendlies and qualifiers, while these are World Cup group games between closely matched sides. The fixture mixes differ, so the absolute log-loss levels are not directly comparable, and n = 68 is small regardless. The fair summary is that the model performed in line with expectations, not that it beat its backtest.

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

> **Caveat on two fixtures.** The simulator treats all 70 scoreless rows as unplayed, but two of them — Canada v Bosnia and Herzegovina and United States v Paraguay, both 12 June — had already kicked off before the forecast was committed. Their group-stage contribution to Canada's and the USA's numbers is therefore not a genuine prediction. See [Leakage caveats](#leakage-caveats).

`10_tournament_sim.py` **refits the ensemble in-process from `data/features.csv` rather than reading `predictions.csv`** — it needs a probability for knockout pairings that do not exist in the fixture list, so no prediction CSV is an input to it. It therefore ignores the team-news overlay.

The tournament simulation samples each match's outcome class from the calibrated ensemble, then samples a scoreline from the Dixon-Coles grid *conditional* on that class, which supplies the goal differences group tables need without disturbing outcome calibration. Knockout seeding is an approximation of the official bracket, which is not in the data: qualifiers are seeded 1-32 by group record into a standard 1v32 bracket, and a knockout draw at 90' is resolved by a strength-weighted coin flip.

### How the real tournament went

Spain beat Argentina 1-0 after extra time in the final on July 19 2026, Ferran Torres scoring the winner. It's Spain's second World Cup title, after 2010.

The simulation's top two teams by reach-final probability were Spain (33.3%) and Argentina (29.9%), exactly the pair that met in the final. It also ranked Spain first to win it all at 22.7%, ahead of Argentina at 19.3%, so the champion pick landed too.

England then beat France 6-4 in the third-place playoff on July 18. Those were also the model's #3 and #4 ranked teams by champion probability, so all four semifinalists it favored most landed in the right places.

## The frozen forecast

**The frozen forecast is the `p_home_pre` / `p_draw_pre` / `p_away_pre` columns of `data/predictions.csv`.** Those are the model's own output, with no team-news adjustment. They were produced by the model as committed in **`99a2305`, whose author date is 13 June 2026 12:09 +02:00**, and reproduced **bit-for-bit** by the later run committed in `be71dfd` on 18 September (max difference 1.11e-16 across all 70 fixtures). That date is author-set metadata with no independent corroboration — see [What the 13 June date actually rests on](#what-the-13-june-date-actually-rests-on). The file must not be regenerated.

Provenance is checked in CI-able form by `tests/test_frozen_forecast.py`, which fails if those columns ever drift from `99a2305`.

### What the 13 June date actually rests on

**There is no independent timestamp for 13 June.** The date rests on git metadata written on the author's own machine. Everything below should be read with that in mind.

**History was rewritten on 23 September 2026.** An internal notes file was removed from every commit and the repository was recreated on GitHub. No other file changed. Every commit hash changed as a result. The forecast commit was `32ffb4c` and is now `99a2305`. Every file in it other than the removed one is byte-identical to the original. Author and committer dates were preserved. The rewrite also means the GitHub creation date of this repository attests nothing about the forecast.

Before the rewrite the forecast commit reached GitHub only on 18 September, 90 days after its author date. So no server ever saw it before the tournament ended.

What remains is local evidence. All of it was produced on the same machine and all of it could be forged.

- The **author date** of `99a2305` is 13 June 2026 12:09:22 +02:00. Author dates are plain metadata and can be set to any value.
- A pre-rebase original of the commit, `f296b05` in the pre-rewrite history, is still in the author's local object store. Its **committer date matches its author date**. Its `data/predictions.csv` blob is byte-identical to the one in `99a2305`.
- The author's local **reflog** records that original at `HEAD@{2026-06-13 12:09:22 +0200}`. Git writes reflog timestamps at the moment of the operation, so they are not author-set. They are local only and can be edited.

Independent of any date, and worth more than all of the above:

- Every input (`data/raw/results.csv`, `features.csv`, `processed_matches.csv`, `poisson_model.joblib` and the FIFA snapshot) has **exactly one commit** and **unchanged bytes** since.
- The same holds for `step3_classifier.py`, `harness.py`, `05_model.py`, `04_features.py` and `02_elo.py`. `git diff 99a2305 HEAD` across them is empty.
- Re-running the pipeline today reproduces the numbers to 1.11e-16.
- `results.csv` contains **no tournament result after 11 June**. A forecast produced later from this repository's data could not have known any outcome it predicts, whatever date sits on the commit.

That last point is the real guarantee. The dating is corroborated but not proven. The *information content* of the inputs is verifiable regardless.

### Leakage caveats

These are stated plainly rather than argued away.

1. **Two tournament results are inside the Elo replay.** `results.csv` contains matchday 1 of 11 June — Mexico 2-0 South Africa and South Korea 2-1 Czech Republic — with scores. They feed each team's current Elo. This is real in-tournament information, though it predates every fixture being forecast.
2. **Two forecast fixtures had already kicked off when the forecast was committed.** The commit is 13 June 10:09 UTC. These were played on 12 June:
   - Canada v Bosnia and Herzegovina (Toronto)
   - United States v Paraguay (Inglewood)

   Their results are *not* in `results.csv` (both rows are scoreless), so they did not enter the model. But the forecast for them was committed after they were played and cannot be called a prediction. **Treat these two as out of sample and exclude them from any scoring.** The four fixtures dated 13 June kicked off after the commit: 10:09 UTC is 06:09 in East Rutherford and Foxborough and 03:09 in Santa Clara and Vancouver, hours before any plausible kickoff — though the fixture data carries dates only, not kickoff times, so this rests on venue local time rather than on the data.
3. **The simulator's data is clean; its design choices were not.** `10_tournament_sim.py` was written in September, after the tournament finished, so this needs separating into two claims.

   *The data is clean, and this is checkable.* The simulator reads exactly three files — `data/raw/results.csv`, `data/raw/former_names.csv` and `data/raw/fifa_ranking_2026-06-11.csv` — and refits the model in-process. It reads no prediction file, and it does not read `wc26_actual_results.csv`. In `results.csv` the latest match carrying a score is **11 June 2026**, and the number of scored rows after that date is **zero**; `features.csv` derives from those same played rows and ends on the same date; the FIFA snapshot is dated 11 June. **No tournament result is reachable from the simulator's inputs.** It cannot have fitted to, or been tuned against, an outcome it reports.

   *The design choices are not clean.* The bracket-seeding approximation and the draw-resolution rule were chosen by someone who already knew how the tournament had gone. Nothing in the data leaks, but the structure around it was picked with hindsight, and no audit of the inputs can rule that out. The "How the real tournament went" comparison above should be read with that in mind.
4. **Git commit dates are author-set metadata. They are evidence, not proof.** No server-side timestamp attests the 13 June date. A backdated commit cannot be ruled out from the repository alone.

   The committer date of `99a2305` is 18 September because the commit was replayed by a rebase that day. The author's local reflog records that rebase under the pre-rewrite hashes (`eaddeb6` and `32ffb4c`). The pre-rebase original still exists locally with both dates at 13 June and an identical `predictions.csv` blob. That is why the rebase, rather than a later edit, explains the committer date.

### The post-news columns are an annotation, not the forecast

`p_home_post` / `p_draw_post` / `p_away_post`, and the per-team `news_score`, `sentiment`, `key_out`, `key_back`, `elo_delta` and `notes` columns, are a **team-news annotation added on 18 September 2026**. They move 8 of the 70 fixtures, by up to 6.6 percentage points (largest: Netherlands v Japan, 34.2/32.9/32.9 → 28.1/32.4/39.5).

`data/team_news.json` holds no fetch timestamp or source-date field, and has a single commit, on 18 September. **The retrieval date of that news cannot be established from the repository.** The agent was instructed to restrict itself to news published before each match date, and the notes read as pre-match, but that is an instruction to a model, not a verifiable constraint.

**These columns are not used for evaluation anywhere.** Nothing in `src/` or `tests/` reads `data/predictions.csv`, and the tournament comparison above is scored from `data/tournament_sim.csv`, which is built without them. They are kept for audit, not for scoring.

### Guards

- `run_pipeline.py --news` exits 1 and refuses to run any step.
- `09_team_news.py --live` exits 1 unless given an explicit `--overwrite-frozen`.
- The LaunchAgent that refreshed this every 2 days was **unloaded and deleted on 19 September 2026**. `scripts/refresh_team_news.sh` and `scripts/worldcup-news.plist.example` are retained as documentation, both marked retired; the script now exits non-zero instead of skipping quietly.

## Team-news adjustment

This section documents **how the post-news annotation was produced**. It is not a step to re-run — see the guards above — and it is not part of the frozen forecast.

`09_team_news.py` is a transparent, prediction-time overlay. It does not retrain the model and does not add a trained feature.

An LLM agent with web search covers every team and returns a signed score per team, `news_score` in `[-1, +1]` — negative for injuries, suspensions, off-field turmoil or poor preparation; positive for key players returning, strong form or settled preparation. That becomes a signed Elo adjustment, `delta = MAX_SWING * news_score` (default `MAX_SWING = 120`, deliberately larger than the 60-point home advantage), applied to the team's strength **input** and fed through the unchanged Step-3 ensemble. Results are cached in `data/team_news.json` as an audit artifact.

The live path needed `ANTHROPIC_API_KEY`, read from `~/.worldcup2026.env` (never committed). It is now gated behind `--overwrite-frozen` and should not be run.

Without `--live` the script is a **cache-only dry run**: it writes `data/predictions_offline.csv` and leaves the frozen file alone. Teams missing from the cache default to neutral, so a dry run is a near-no-op overlay, useful only for inspecting the mechanism.

A run either produces a real forecast or fails loudly. `--live` makes a minimal preflight call first, so bad credentials exit non-zero in seconds; authentication errors mid-run abort instead of degrading to neutral; only successful results are cached, so a failure can never poison `data/team_news.json`; and if more than `MAX_DEGRADED_FRACTION` (25%) of teams fell back to neutral, the run aborts rather than publish a flat overlay that looks like a forecast.

## Data

`data/raw/` holds the inputs: ~49k historical results (played rows have scores, 2026 fixtures have NA), a FIFA ranking snapshot, goalscorers, shootouts, and a former-name mapping so each country's history sits under one current name. It also holds `wc26_actual_results.csv` — the 72 actual group-stage results, used only for scoring and never as a model input.

**Source of `wc26_actual_results.csv`:** the same upstream dataset `results.csv` came from — [`martj42/international_results`](https://github.com/martj42/international_results), file `results.csv` on `master`. Retrieved **2026-09-19T13:34:01Z** from `https://raw.githubusercontent.com/martj42/international_results/master/results.csv` (upstream commit `394fe81893`, dated 2026-08-26T21:56:21Z; sha256 of the download `df35268f8fc341ff7fb93d448b4e40356676ac35300a6b4461fd199a99ac1514`). Lineage was checked rather than assumed: the upstream file has identical columns, the two matches already scored in the frozen `results.csv` agree exactly, and all 70 forecast fixtures matched on `(date, home_team, away_team)` with no ambiguity. **`data/raw/results.csv` was not modified** — its 70 fixture rows remain scoreless, which a test enforces. `data/` holds derived artifacts — the cleaned match table, engineered features, the saved Poisson model, the team-news cache and the four output CSVs above, including `tournament_sim.csv` with the Monte Carlo output described above.

## Status

**Complete.** The forecast was made before the tournament, frozen, and has since been scored against real results.

- **Pipeline** — 11 stages, data cleaning through tournament scoring, run end to end by `python run_pipeline.py`.
- **Model** — the Step-3 ensemble (0.3 Dixon-Coles Poisson + 0.7 multinomial logit, T = 0.95), 0.9024 walk-forward log-loss on ~49k historical matches.
- **Result** — 0.8851 log-loss, 0.5248 Brier, 61.8% accuracy over the 68 group-stage fixtures that kicked off after the forecast was committed, against 0.9176 for an Elo-only baseline and 1.0986 for a uniform prior. On a paired bootstrap the edge over Elo alone is not statistically distinguishable; the margin over the uniform prior is. See [How it actually did](#how-it-actually-did).
- **Provenance** — `data/predictions.csv` is frozen, its `p_*_pre` columns pinned by test to the model run in `99a2305`. The scheduled news refresh is retired. Leakage caveats are listed rather than argued away, including the limits of what git dates can prove.
- **Tests** — 31 in `tests/`, no network calls: output schemas, frozen-forecast provenance, the scoring exclusion rule, the paired bootstrap against synthetic cases with known answers, and the team-news failure paths.
- Single entry point, `requirements.txt` with minimum versions, dev deps split into `requirements-dev.txt`, MIT license.
- Tests run in GitHub Actions on every push and pull request.

**Not included.** Visualizations of the predictions. The validated result covers the **group stage only** — the 32 knockout matches were never forecast, and scoring them would first need 90-minute scores, since the upstream dataset records knockout results after extra time.

**Known limitation.** Features are the bottleneck: `elo_diff` dominates, and further gains need richer data (squad or market value, rest and travel) rather than more model tuning.
