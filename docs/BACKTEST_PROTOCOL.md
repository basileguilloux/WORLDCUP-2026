# Title-odds backtest: pre-registered protocol

Written and committed on 29 September 2026, **before any backtest was run** and before the code that runs it existed. It fixes what will be measured, on what, and against what, so the result cannot be steered by choosing among variants after seeing them.

This file must not be edited once results exist. If a change turns out to be unavoidable, it goes in a dated **Deviations** section at the end, with the reason, and the original text stays.

## Why

`data/tournament_sim.csv` (the 2026 title odds) comes from `src/10_tournament_sim.py`, which was written in September 2026, after the tournament. Its inputs are clean but its design was chosen with hindsight, so its 2026 output is an illustration, not evidence. This backtest asks whether the same simulator, run blind on past tournaments, produces useful stage probabilities.

## Tournaments and cutoffs

Ten tournaments. The cutoff is the day before the first match, taken from `data/raw/results.csv`.

| Tournament | `results.csv` name | First match | Cutoff |
|---|---|---|---|
| World Cup 2006 | FIFA World Cup | 2006-06-09 | 2006-06-08 |
| World Cup 2010 | FIFA World Cup | 2010-06-11 | 2010-06-10 |
| World Cup 2014 | FIFA World Cup | 2014-06-12 | 2014-06-11 |
| World Cup 2018 | FIFA World Cup | 2018-06-14 | 2018-06-13 |
| World Cup 2022 | FIFA World Cup | 2022-11-20 | 2022-11-19 |
| Euro 2008 | UEFA Euro | 2008-06-07 | 2008-06-06 |
| Euro 2012 | UEFA Euro | 2012-06-08 | 2012-06-07 |
| Euro 2016 | UEFA Euro | 2016-06-10 | 2016-06-09 |
| Euro 2020 (played 2021) | UEFA Euro | 2021-06-11 | 2021-06-10 |
| Euro 2024 | UEFA Euro | 2024-06-14 | 2024-06-13 |

For each tournament, everything the model sees is **strictly before the cutoff**: the Elo replay, the form features, the model fit and the FIFA release. Matches on the cutoff day itself (the day before kickoff) are therefore also excluded. That is conservative and costs a handful of friendlies. Every match of the tournament is treated as unplayed.

## Tournament formats

Groups are read from the real group-stage fixtures in `results.csv`. The simulator plays every group match, then the knockout bracket.

- **World Cups 2006-2022:** 8 groups of 4, top 2 advance, round of 16 on FIFA's fixed bracket (1A v 2B, 1C v 2D, 1B v 2A, 1D v 2C, 1E v 2F, 1G v 2H, 1F v 2E, 1H v 2G, with the official quarter-final and semi-final pairings).
- **Euros 2008 and 2012:** 4 groups of 4, top 2 advance, straight to quarter-finals (1A v 2B, 1B v 2A, 1C v 2D, 1D v 2C). The semi-final pairings differ: 2008 plays QF1 v QF2 and QF3 v QF4, 2012 plays QF1 v QF3 and QF2 v QF4.
- **Euros 2016, 2020 and 2024:** 6 groups of 4, top 2 plus the 4 best third-placed teams, round of 16. The third-placed teams are placed by UEFA's 15-row allocation table. There are **two** such tables: one in the 2014-16 regulations (Article 17.03, thirds meet 1A/1B/1C/1D), and a different one used for both 2020 and 2024 (2018-20 regulations, Article 21.05, thirds meet 1B/1C/1E/1F). Both are transcribed from UEFA's regulations, and a test checks each against the real round-of-16 pairings.
- **Group ranking inside the simulator:** points, then goal difference, then goals scored, then a random draw. The same rule ranks third-placed teams. This is the rule the 2026 simulator already uses. It ignores head-to-head (the first Euro tiebreak) and fair play, a simplification fixed here in advance.
- **Knockout draws at 90 minutes:** decided by a coin flip weighted by the two sides' regulation win probabilities, as in the 2026 simulator. The third-place playoff is not simulated.

## Home advantage

A host nation gets the home flag only in a match played in its own country, in both groups and knockouts. Every other match is neutral. Hosts: Germany 2006, South Africa 2010, Brazil 2014, Russia 2018, Qatar 2022; Austria and Switzerland 2008; Poland and Ukraine 2012; France 2016; the eleven host associations of 2020 (Azerbaijan, Denmark, England, Germany, Hungary, Italy, Netherlands, Romania, Russia, Scotland, Spain); Germany 2024. Group venues come from the `country` column of `results.csv`. Knockout venues are fixed per bracket slot in the tournament config, from the official schedule, and a test checks them against `results.csv`.

The home flag comes from the host list, not from `results.csv`'s `neutral` column, which marks at least one Euro 2020 game (Wales v Switzerland, played in Baku) as a home game in Wales.

## Models

All four run through the same simulator, format and seed.

1. **Full model** (the shipped one): the Step-3 ensemble, 0.3 Dixon-Coles Poisson (rho = -0.15) + 0.7 multinomial logit, temperature T = 0.95. Both components are refit per tournament on `data/features.csv` rows strictly before the cutoff. Strength is Elo replayed on matches before the cutoff, blended with the FIFA ranking by the shipped rule `w = clip(10 / (10 + recent), 0.2, 0.6)`. `recent` counts a team's matches in the 4 years before the cutoff. FIFA points come from the latest release strictly before the cutoff in `data/raw/fifa_ranking_history.csv`. They are mapped to Elo by a least-squares line fitted over that tournament's participants, as the 2026 pipeline fits it over the 48 World Cup teams. Form is the average goals for and against over each team's last 5 matches before the cutoff.
2. **No FIFA blend:** identical, with strength = raw Elo.
3. **Elo only:** strength = raw Elo. The match engine is the `03_elo_baseline.py` model: a `LogisticRegression` with scikit-learn's defaults, on `elo_diff` alone, refit on the same rows. It has no home feature, so hosts get no home bump. Its intercept was learned on data where the first-listed team is usually at home, so in neutral games it slightly favours the first-listed side (the group fixture's home team, or the first slot of a knockout pairing). That is how the model is defined, and it is kept as is. Group tables need scorelines, so a scoreline is sampled conditional on the Elo-only outcome from the Dixon-Coles grid of model 2. That affects tiebreaks only.
4. **Uniform:** every match is 1/3 home win, 1/3 draw, 1/3 away win. Scorelines come from the Dixon-Coles grid of two identical average teams at a neutral venue.

## Simulation

`N_SIMS = 10000` per tournament per model. Every run uses the same seed, `20260929`, so the models see common random numbers.

## Ground truth

Each team's stage reached is derived from `results.csv`, with `data/raw/shootouts.csv` deciding knockout ties settled on penalties. Categories, in order: group exit, round of 16 (not for Euros 2008 and 2012), quarter-final, semi-final, runner-up, champion. Losing the third-place playoff or winning it both count as semi-final.

## Metrics

For each tournament and model:

- **RPS (primary).** Ranked probability score over the ordered stage categories, per team: `1/(K-1) * sum_k (F_pred(k) - F_obs(k))^2` over the first K-1 cumulative categories. It is averaged over the teams of a tournament, then over the 10 tournaments with equal weight.
- **Champion log score.** `-ln p(actual champion)`, with p floored at `1 / (2 * N_SIMS)` so a champion never reached in simulation scores 9.90, not infinity. Averaged over tournaments.
- **Brier on "reached the semi-final".** Per team, averaged per tournament, then over tournaments.
- **Reliability.** Every team-stage "reach" probability (reach the knockouts, then each later round up to champion) is pooled over all 10 tournaments. These are binned into 10 equal-width bins on [0, 1] and each bin's mean prediction is compared with the observed frequency. Reported for all models. The chart shows the full model and Elo only.

## Uncertainty

- **Primary:** paired bootstrap over whole tournaments. Each of 10,000 resamples draws 10 tournaments with replacement and applies the same draw to both models. Seed `20260929`. The per-tournament metric differences are averaged, and the 95% percentile interval is reported for full minus each other model. With n = 10 these intervals are wide. That is a property of the data, not of the method.
- **Secondary:** the same paired bootstrap over teams pooled across tournaments, for RPS only. Teams within a tournament are not independent (only one can win), so this interval is too narrow and is reported only as a secondary figure.

## Decision rule, fixed now

The headline comparison is **full model versus Elo only on overall RPS**.

- "The full model beats Elo only" only if the tournament-bootstrap 95% interval for (full - Elo only) lies wholly below zero.
- If the interval spans zero, the result is reported as "not distinguishable". That holds whatever the point estimate.
- If it lies wholly above zero, the full model is reported as worse.

The same wording applies to the other comparisons. Whatever the outcome, it is reported in the README.

## Out of scope

- The 2026 tournament is not part of this backtest. Its simulator path keeps its approximate round-of-32 seeding unchanged, so the committed `data/tournament_sim.csv` still reproduces.
- No model parameter is tuned on these tournaments.
