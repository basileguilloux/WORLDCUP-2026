"""Phase 3: Poisson model that predicts expected goals -> exact-score grid.

Trick: instead of one model per team, we stack each match into TWO rows
(one per attacking side). The model learns "given my strength, my form, my
opponent's defence, and whether I'm home -> how many goals do I score?".
Goals are counts, so PoissonRegressor (log link) is the right tool.
"""
import numpy as np
import pandas as pd
from math import exp, factorial
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, log_loss
import joblib
from pathlib import Path

# data/poisson_model.joblib is the frozen artifact committed with the forecast
# (99a2305). This script never overwrites it: the refit goes to a git-ignored
# build path, which 06_simulate.py and 07_blend_predict.py read.
REFIT = Path("data/build/poisson_model.joblib")

df = pd.read_csv("data/features.csv", parse_dates=["date"])

def stack(d):
    """One match -> two attacking rows (home view, away view)."""
    home = pd.DataFrame({"goals": d.home_score, "elo_diff": d.home_elo - d.away_elo,
                         "att": d.home_gf, "opp_def": d.away_ga, "is_home": d.home_advantage})
    away = pd.DataFrame({"goals": d.away_score, "elo_diff": d.away_elo - d.home_elo,
                         "att": d.away_gf, "opp_def": d.home_ga, "is_home": 0})
    return pd.concat([home, away], ignore_index=True)

FEATS = ["elo_diff", "att", "opp_def", "is_home"]

# time split: train on the past, test on 2022+ (honest, no future leakage)
train, test = df[df.date < "2022-01-01"], df[df.date >= "2022-01-01"]
tr = stack(train)
model = make_pipeline(StandardScaler(), PoissonRegressor(alpha=0.1, max_iter=300))
model.fit(tr[FEATS], tr["goals"])

def pmf(k, lam):                       # Poisson probability of exactly k goals
    return exp(-lam) * lam**k / factorial(k)

def outcome_probs(lh, la, maxg=10):    # from two expected-goal numbers -> H/D/A probs
    ph = [pmf(i, lh) for i in range(maxg)]
    pa = [pmf(j, la) for j in range(maxg)]
    h = d = a = 0.0
    for i in range(maxg):
        for j in range(maxg):
            p = ph[i] * pa[j]
            h += p if i > j else 0; d += p if i == j else 0; a += p if i < j else 0
    tot = a + d + h                    # normalise (grid is capped at maxg goals)
    return [a / tot, d / tot, h / tot] # order: away, draw, home

# evaluate on the test years
lh = model.predict(pd.DataFrame({"elo_diff": test.home_elo - test.away_elo,
        "att": test.home_gf, "opp_def": test.away_ga, "is_home": test.home_advantage})[FEATS])
la = model.predict(pd.DataFrame({"elo_diff": test.away_elo - test.home_elo,
        "att": test.away_gf, "opp_def": test.home_ga, "is_home": 0})[FEATS])
probs = np.array([outcome_probs(a, b) for a, b in zip(lh, la)])
ytrue = ((test.home_score > test.away_score).astype(int) * 2
         + (test.home_score == test.away_score).astype(int)).values

print("Phase 3 - Poisson model, tested on 2022+ matches:")
print(f"  accuracy : {accuracy_score(ytrue, probs.argmax(1)):.3f}   (Elo-only baseline was 0.576)")
print(f"  log-loss : {log_loss(ytrue, probs, labels=[0,1,2]):.3f}   (Elo-only baseline was 0.915)")

REFIT.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, REFIT)
print(f"\nsaved {REFIT}  (data/poisson_model.joblib is the frozen copy, left untouched)")
