"""Phase 4: use the trained model to predict the upcoming WC 2026 fixtures.

Steps: (1) replay all played matches to get each team's CURRENT Elo + form,
(2) for every upcoming fixture, predict each side's expected goals (giving the
host nations a home bonus), (3) turn those into score + win/draw/loss odds,
and rank teams by expected points.
"""
import numpy as np
import pandas as pd
from math import exp, factorial
from collections import defaultdict, deque
import joblib

HOSTS = {"United States", "Mexico", "Canada"}
N, START, HOME_ADV = 5, 1500.0, 60.0
model = joblib.load("data/build/poisson_model.joblib")   # refit by 05_model.py

# --- load + clean all matches (same recipe as before) ---
df = pd.read_csv("data/raw/results.csv", parse_dates=["date"])
ren = dict(zip(*[pd.read_csv("data/raw/former_names.csv")[c] for c in ["former", "current"]]))
df["home_team"], df["away_team"] = df.home_team.replace(ren), df.away_team.replace(ren)
df = df.sort_values("date")
played, future = df[df.home_score.notna()], df[df.home_score.isna()]

# --- replay played matches -> current Elo + current form per team ---
elo = defaultdict(lambda: START)
hist = defaultdict(lambda: deque(maxlen=N))
def expected(ra, rb): return 1 / (1 + 10 ** ((rb - ra) / 400))
for r in played.itertuples():
    rh, ra = elo[r.home_team], elo[r.away_team]
    k = 60 if "FIFA World Cup" in r.tournament else (40 if r.tournament != "Friendly" else 20)
    gd = abs(r.home_score - r.away_score); k *= 1 if gd <= 1 else (1.5 if gd == 2 else 1 + gd/5)
    res = 1.0 if r.home_score > r.away_score else (0.5 if r.home_score == r.away_score else 0.0)
    ch = k * (res - expected(rh + (0 if r.neutral else HOME_ADV), ra))
    elo[r.home_team] += ch; elo[r.away_team] -= ch
    hist[r.home_team].append((r.home_score, r.away_score))
    hist[r.away_team].append((r.away_score, r.home_score))

def form(t):
    if not hist[t]: return 1.0, 1.0
    return (np.mean([g[0] for g in hist[t]]), np.mean([g[1] for g in hist[t]]))

FEATS = ["elo_diff", "att", "opp_def", "is_home"]
def xg(team, opp):                              # expected goals for `team` vs `opp`
    gf, _ = form(team); _, oga = form(opp)
    is_home = 1 if team in HOSTS else 0          # host nations count as home
    row = pd.DataFrame([[elo[team] - elo[opp], gf, oga, is_home]], columns=FEATS)
    return float(model.predict(row)[0])

def pmf(k, lam): return exp(-lam) * lam**k / factorial(k)
def match(home, away, maxg=10):
    lh, la = xg(home, away), xg(away, home)
    ph, pa = [pmf(i, lh) for i in range(maxg)], [pmf(j, la) for j in range(maxg)]
    h = sum(ph[i]*pa[j] for i in range(maxg) for j in range(maxg) if i > j)
    d = sum(ph[i]*pa[j] for i in range(maxg) for j in range(maxg) if i == j)
    a = 1 - h - d
    return lh, la, h, d, a

# --- predict every upcoming fixture + tally expected points ---
pts = defaultdict(float)
print(f"Predicting {len(future)} upcoming WC 2026 fixtures:\n")
print(f"{'Match':<34}{'Score':>8}{'Home%':>8}{'Draw%':>7}{'Away%':>7}")
for r in future.itertuples():
    lh, la, h, d, a = match(r.home_team, r.away_team)
    pts[r.home_team] += 3*h + d; pts[r.away_team] += 3*a + d
    print(f"{r.home_team+' v '+r.away_team:<34}{f'{lh:.1f}-{la:.1f}':>8}"
          f"{h*100:>7.0f}{d*100:>7.0f}{a*100:>7.0f}")

print("\nPower ranking by expected points from these fixtures:")
for i, (t, p) in enumerate(sorted(pts.items(), key=lambda x: -x[1])[:12], 1):
    print(f"{i:2}. {t:<16} {p:.2f}  (Elo {elo[t]:.0f})")
