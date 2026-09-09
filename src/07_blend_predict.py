"""Phase 4b: blend official FIFA ranking into each WC team's current strength,
then re-predict the fixtures.

Why blend (not retrain): the FIFA data is a single pre-tournament snapshot, so
it can't be a historical training feature. But it's an excellent *prior* for the
present - especially for teams with few recent matches, where our computed Elo
is shaky. We trust Elo for data-rich teams, FIFA more for data-poor ones.
"""
import numpy as np
import pandas as pd
from math import exp, factorial
from collections import defaultdict, deque
import joblib

HOSTS = {"United States", "Mexico", "Canada"}
N, START, HOME_ADV = 5, 1500.0, 60.0
RECENT_CUTOFF = pd.Timestamp("2022-06-01")     # "recent" = last ~4 years
model = joblib.load("data/poisson_model.joblib")

# --- load + clean matches ---
df = pd.read_csv("data/raw/results.csv", parse_dates=["date"])
ren = dict(zip(*[pd.read_csv("data/raw/former_names.csv")[c] for c in ["former", "current"]]))
df["home_team"], df["away_team"] = df.home_team.replace(ren), df.away_team.replace(ren)
df = df.sort_values("date")
played, future = df[df.home_score.notna()], df[df.home_score.isna()]

# --- replay -> current Elo, form, and recent-match count per team ---
elo = defaultdict(lambda: START)
hist = defaultdict(lambda: deque(maxlen=N))
recent = defaultdict(int)
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
    if r.date >= RECENT_CUTOFF:
        recent[r.home_team] += 1; recent[r.away_team] += 1

# --- map FIFA points onto the Elo scale, then blend ---
rank = pd.read_csv("data/raw/fifa_ranking_2026-06-11.csv")
wc = list(rank.team)
fifa_pts = dict(zip(rank.team, rank.fifa_points))
# least-squares line: fifa_points -> elo (using the 48 WC teams)
a, b = np.polyfit([fifa_pts[t] for t in wc], [elo[t] for t in wc], 1)
fifa_elo = {t: a * fifa_pts[t] + b for t in wc}

def blend_weight(t):                       # more weight on FIFA when few recent matches
    return float(np.clip(10 / (10 + recent[t]), 0.2, 0.6))

blended = {t: (1 - blend_weight(t)) * elo[t] + blend_weight(t) * fifa_elo[t] for t in wc}

# --- prediction helpers using a chosen strength dict ---
def form(t):
    if not hist[t]: return 1.0, 1.0
    return np.mean([g[0] for g in hist[t]]), np.mean([g[1] for g in hist[t]])
FEATS = ["elo_diff", "att", "opp_def", "is_home"]
def xg(team, opp, strength):
    gf, _ = form(team); _, oga = form(opp)
    row = pd.DataFrame([[strength[team] - strength[opp], gf, oga, 1 if team in HOSTS else 0]], columns=FEATS)
    return float(model.predict(row)[0])
def pmf(k, lam): return exp(-lam) * lam**k / factorial(k)
def probs(home, away, strength, maxg=10):
    lh, la = xg(home, away, strength), xg(away, home, strength)
    ph, pa = [pmf(i, lh) for i in range(maxg)], [pmf(j, la) for j in range(maxg)]
    h = sum(ph[i]*pa[j] for i in range(maxg) for j in range(maxg) if i > j)
    d = sum(ph[i]*pa[j] for i in range(maxg) for j in range(maxg) if i == j)
    return lh, la, h, d, 1 - h - d

# --- show the biggest strength corrections from blending ---
print("Biggest strength changes from blending in FIFA ranking:")
diffs = sorted(wc, key=lambda t: abs(blended[t] - elo[t]), reverse=True)[:8]
print(f"{'team':<16}{'Elo':>7}{'FIFA-eq':>9}{'blend':>8}{'wFIFA':>7}{'recent':>8}")
for t in diffs:
    print(f"{t:<16}{elo[t]:>7.0f}{fifa_elo[t]:>9.0f}{blended[t]:>8.0f}{blend_weight(t):>7.2f}{recent[t]:>8}")

# --- recompute power ranking with blended strength, compare to Elo-only ---
def power(strength):
    pts = defaultdict(float)
    for r in future.itertuples():
        _, _, h, d, a = probs(r.home_team, r.away_team, strength)
        pts[r.home_team] += 3*h + d; pts[r.away_team] += 3*a + d
    return pts
old, new = power(elo), power(blended)
print("\nPower ranking WITH FIFA blend (was Elo-only):")
print(f"{'#':>3} {'team':<16}{'pts':>6}{'(Elo-only pts)':>16}")
for i, (t, p) in enumerate(sorted(new.items(), key=lambda x: -x[1])[:12], 1):
    print(f"{i:>3} {t:<16}{p:>6.2f}{old[t]:>16.2f}")

# --- save the full power ranking (blended prediction) to a file ---
ranking = pd.DataFrame(
    [(t, new[t], old.get(t, np.nan), elo[t], fifa_elo.get(t, np.nan), blend_weight(t), recent[t])
     for t in sorted(new, key=lambda x: -new[x])],
    columns=["team", "expected_points_blended", "expected_points_elo_only",
             "elo", "fifa_equivalent_elo", "fifa_blend_weight", "recent_matches"],
)
ranking.to_csv("data/predictions.csv", index=False)
print(f"\nSaved full power ranking to data/predictions.csv ({len(ranking)} teams)")
