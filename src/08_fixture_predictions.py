"""Phase 5: FINAL per-fixture predictions for the ~70 unplayed WC 2026 games.

Model = the Step-3 winner: an ensemble of the Dixon-Coles Poisson (rho=-0.15)
and a direct multinomial-logistic 3-way classifier, blended 0.3/0.7 and
temperature-calibrated (T=0.95). Walk-forward log-loss 0.9024 vs 0.9120 for the
old Poisson-only model.

Strength = current replayed Elo, blended with the FIFA-ranking prior (more FIFA
weight for teams with few recent matches), exactly as in 07_blend_predict.py.
Home advantage uses each fixture's real `neutral` flag — so only the 8 host-at-
home games (USA/Canada/Mexico in their own country) get the home bump; the other
62 neutral-venue games get none.

Output: data/fixture_predictions.csv — per-fixture W/D/L with NO team-news
overlay. data/predictions.csv is the separate frozen forecast and is not
written by any pipeline step.
"""
import numpy as np
import pandas as pd
from collections import defaultdict, deque

from harness import load, outcomes, grid_probs, expected_goals, fit
from step3_classifier import make_clf, match_features, clf_probs, BEST_RHO

W, TEMP = 0.3, 0.95          # ensemble weight on Poisson, temperature (Step-3 winners)
N, START, HOME_ADV = 5, 1500.0, 60.0
RECENT_CUTOFF = pd.Timestamp("2022-06-01")

# ---- 1. replay all played matches -> current Elo + form + recent-match count ----
df = pd.read_csv("data/raw/results.csv", parse_dates=["date"])
ren = dict(zip(*[pd.read_csv("data/raw/former_names.csv")[c] for c in ["former", "current"]]))
df["home_team"], df["away_team"] = df.home_team.replace(ren), df.away_team.replace(ren)
df = df.sort_values("date")
played, future = df[df.home_score.notna()], df[df.home_score.isna()]

elo = defaultdict(lambda: START)
hist = defaultdict(lambda: deque(maxlen=N))
recent = defaultdict(int)
exp_score = lambda ra, rb: 1 / (1 + 10 ** ((rb - ra) / 400))
for r in played.itertuples():
    rh, ra = elo[r.home_team], elo[r.away_team]
    k = 60 if "FIFA World Cup" in r.tournament else (40 if r.tournament != "Friendly" else 20)
    gd = abs(r.home_score - r.away_score); k *= 1 if gd <= 1 else (1.5 if gd == 2 else 1 + gd / 5)
    res = 1.0 if r.home_score > r.away_score else (0.5 if r.home_score == r.away_score else 0.0)
    ch = k * (res - exp_score(rh + (0 if r.neutral else HOME_ADV), ra))
    elo[r.home_team] += ch; elo[r.away_team] -= ch
    hist[r.home_team].append((r.home_score, r.away_score))
    hist[r.away_team].append((r.away_score, r.home_score))
    if r.date >= RECENT_CUTOFF:
        recent[r.home_team] += 1; recent[r.away_team] += 1

def form(t):
    if not hist[t]: return 1.0, 1.0
    return float(np.mean([g[0] for g in hist[t]])), float(np.mean([g[1] for g in hist[t]]))

# ---- 2. blend FIFA-ranking prior into current strength (same as 07) ----
rank = pd.read_csv("data/raw/fifa_ranking_2026-06-11.csv")
wc = list(rank.team)
fifa_pts = dict(zip(rank.team, rank.fifa_points))
a, b = np.polyfit([fifa_pts[t] for t in wc], [elo[t] for t in wc], 1)
fifa_elo = {t: a * fifa_pts[t] + b for t in wc}
blend_w = lambda t: float(np.clip(10 / (10 + recent[t]), 0.2, 0.6))
strength = defaultdict(lambda: START)
for t in set(list(elo) + wc):
    s = elo[t]
    if t in fifa_elo:
        w = blend_w(t); s = (1 - w) * elo[t] + w * fifa_elo[t]
    strength[t] = s

# ---- 3. train the two models on ALL played data ----
feat = load()
poisson = fit(feat)                                   # stacked Poisson regressor
logit = make_clf("logit").fit(match_features(feat), outcomes(feat))

# ---- 4. build a fixtures frame with the exact columns the models expect ----
fx = pd.DataFrame({
    "home_team": future.home_team.values,
    "away_team": future.away_team.values,
    "date": future.date.values,
    "neutral": future.neutral.values,
})
fx["home_elo"] = [strength[t] for t in fx.home_team]
fx["away_elo"] = [strength[t] for t in fx.away_team]
fx["home_gf"], fx["home_ga"] = zip(*[form(t) for t in fx.home_team])
fx["away_gf"], fx["away_ga"] = zip(*[form(t) for t in fx.away_team])
fx["home_advantage"] = (~fx.neutral.astype(bool)).astype(int)   # 1 only for host-at-home

# ---- 5. predict: ensemble the DC-Poisson and the classifier, then temper ----
Pp = grid_probs(*expected_goals(poisson, fx), rho=BEST_RHO)     # [away, draw, home]
Pc = clf_probs(logit, fx)
P = W * Pp + (1 - W) * Pc
P = np.clip(P, 1e-12, 1.0) ** (1.0 / TEMP)
P = P / P.sum(axis=1, keepdims=True)

out = fx[["date", "home_team", "away_team", "neutral"]].copy()
out["p_home"], out["p_draw"], out["p_away"] = P[:, 2], P[:, 1], P[:, 0]
out["pick"] = np.where(P.argmax(1) == 2, "HOME", np.where(P.argmax(1) == 1, "DRAW", "AWAY"))
out = out.sort_values("date")
out.to_csv("data/fixture_predictions.csv", index=False)

pd.set_option("display.width", 200)
print(f"Final predictions for {len(out)} WC 2026 fixtures  "
      f"(ensemble w={W} DC-Poisson + {1-W:.1f} logit, T={TEMP}):\n")
print(f"{'Match':<38}{'Home%':>7}{'Draw%':>7}{'Away%':>7}  pick")
for r in out.itertuples():
    print(f"{r.home_team + ' v ' + r.away_team:<38}"
          f"{r.p_home*100:>6.0f}{r.p_draw*100:>7.0f}{r.p_away*100:>7.0f}  {r.pick}")
print("\nsaved -> data/fixture_predictions.csv")
print(f"mean P [home,draw,away] = {P[:, [2,1,0]].mean(0).round(3)}")
