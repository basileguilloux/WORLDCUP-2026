"""Phase 6: full-tournament Monte Carlo — who wins WC 2026?

The rest of the pipeline stops at per-fixture W/D/L probabilities. This script
plays the whole tournament out N times to turn those into "P(lifts the trophy)".

Match engine
------------
Outcome class (0=away,1=draw,2=home) is sampled from the Step-3 FINAL ensemble
(0.3 * Dixon-Coles Poisson + 0.7 * multinomial logit, T=0.95) — the exact model
behind data/predictions.csv, so the sim inherits its 0.9024 walk-forward log-loss.
A scoreline is then sampled from the DC-Poisson grid CONDITIONAL on that class,
which gives the goal difference / goals for that group tables need without
disturbing the calibrated outcome probabilities.

Tournament
----------
* Groups A-L read off the fixture list; the two already-played games
  (Mexico 2-0 South Africa, South Korea 2-1 Czech Republic) are kept as results.
* Standings: points, then goal difference, then goals for, then a coin flip.
* 32 qualify: 12 winners + 12 runners-up + the 8 best third-placed teams
  (ranked pts / GD / GF, as FIFA does).
* Knockout seeding is an APPROXIMATION of the official R32 bracket, which is not
  in the data: qualifiers are seeded 1-32 (winners by record, then runners-up,
  then thirds) into a standard 1v32 / 2v31 ... bracket. Keeps strong teams apart
  the way the real bracket broadly does, but is not the exact FIFA mapping.
* All knockout games are treated as neutral-venue. A draw at 90' is resolved by
  extra time / penalties, modelled as a coin flip weighted by the two sides'
  regulation win probabilities.
"""
import sys
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from scipy.stats import poisson as pois

from harness import load, outcomes, expected_goals, fit
from step3_classifier import make_clf, match_features, clf_probs, BEST_RHO

N_SIMS = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
SEED = 20260618
W, TEMP = 0.3, 0.95
N, START, HOME_ADV = 5, 1500.0, 60.0
RECENT_CUTOFF = pd.Timestamp("2022-06-01")
MAXG = 10
HOSTS = {"United States", "Mexico", "Canada"}

rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------- 1. strength
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

# ---------------------------------------------------------------- 2. models
feat = load()
poisson_m = fit(feat)
logit = make_clf("logit").fit(match_features(feat), outcomes(feat))

# ---------------------------------------------------------------- 3. groups
adj = defaultdict(set)
for h, aw in zip(future.home_team, future.away_team):
    adj[h].add(aw); adj[aw].add(h)
wc26 = played[(played.date >= "2026-06-01") & (played.tournament == "FIFA World Cup")]
for h, aw in zip(wc26.home_team, wc26.away_team):
    adj[h].add(aw); adj[aw].add(h)

seen, comps = set(), []
for t in sorted(adj):
    if t in seen: continue
    stack_, comp = [t], set()
    while stack_:
        x = stack_.pop()
        if x in comp: continue
        comp.add(x); stack_ += list(adj[x])
    seen |= comp; comps.append(sorted(comp))
comps.sort(key=lambda c: min(future.index[future.home_team.isin(c)].min()
                             if (future.home_team.isin(c)).any() else 10**9, 10**9))
GROUPS = {chr(65 + i): c for i, c in enumerate(comps)}
assert len(GROUPS) == 12 and all(len(v) == 4 for v in GROUPS.values()), GROUPS
TEAMS = sorted(t for g in GROUPS.values() for t in g)
TIDX = {t: i for i, t in enumerate(TEAMS)}
GROUP_OF = {t: g for g, ts in GROUPS.items() for t in ts}

# ---------------------------------------------------------------- 4. engine
def _probs(home, away, home_adv):
    """-> (P_outcome[away,draw,home], conditional score-grid pmfs per class)."""
    fx = pd.DataFrame({
        "home_team": home, "away_team": away,
        "home_elo": [strength[t] for t in home], "away_elo": [strength[t] for t in away],
        "home_advantage": np.asarray(home_adv, int),
    })
    fx["home_gf"], fx["home_ga"] = zip(*[form(t) for t in home])
    fx["away_gf"], fx["away_ga"] = zip(*[form(t) for t in away])
    lh, la = expected_goals(poisson_m, fx)

    ks = np.arange(MAXG)
    ph = pois.pmf(ks[None, :], np.asarray(lh)[:, None])
    pa = pois.pmf(ks[None, :], np.asarray(la)[:, None])
    joint = ph[:, :, None] * pa[:, None, :]
    tau = np.ones_like(joint)
    tau[:, 0, 0] = 1.0 - lh * la * BEST_RHO
    tau[:, 0, 1] = 1.0 + lh * BEST_RHO
    tau[:, 1, 0] = 1.0 + la * BEST_RHO
    tau[:, 1, 1] = 1.0 - BEST_RHO
    joint = np.clip(joint * tau, 0.0, None)
    joint /= joint.sum(axis=(1, 2), keepdims=True)

    i_idx, j_idx = ks[:, None], ks[None, :]
    masks = [i_idx < j_idx, i_idx == j_idx, i_idx > j_idx]          # away, draw, home
    Pp = np.column_stack([(joint * m).sum(axis=(1, 2)) for m in masks])
    Pp /= Pp.sum(1, keepdims=True)

    P = W * Pp + (1 - W) * clf_probs(logit, fx)
    P = np.clip(P, 1e-12, 1.0) ** (1.0 / TEMP)
    P /= P.sum(1, keepdims=True)

    # conditional scoreline pmf within each outcome class, flattened
    cond = []
    for m in masks:
        c = (joint * m).reshape(len(fx), -1)
        cond.append(c / np.maximum(c.sum(1, keepdims=True), 1e-300))
    return P, np.stack(cond, 1)                                      # (n,3,MAXG*MAXG)

# every ordered pair at a neutral venue (knockouts + lookup)
pair_h = [h for h in TEAMS for a in TEAMS]
pair_a = [a for h in TEAMS for a in TEAMS]
PAIR_P, PAIR_C = _probs(pair_h, pair_a, np.zeros(len(pair_h)))
PAIR_P = PAIR_P.reshape(len(TEAMS), len(TEAMS), 3)

# group fixtures, with their real neutral flags (host-at-home gets the bump)
gf = future[["home_team", "away_team", "neutral"]].reset_index(drop=True)
GP, GC = _probs(list(gf.home_team), list(gf.away_team), (~gf.neutral.astype(bool)).astype(int))
GH = np.array([TIDX[t] for t in gf.home_team])
GA = np.array([TIDX[t] for t in gf.away_team])
NG = len(gf)

# already-played group games
PLAYED_G = [(TIDX[r.home_team], TIDX[r.away_team], int(r.home_score), int(r.away_score))
            for r in wc26.itertuples()]

GROUP_IDX = {g: np.array([TIDX[t] for t in ts]) for g, ts in GROUPS.items()}
GKEYS = list(GROUPS)

def _sample(P, C, n):
    """Vectorised: sample outcome class then a scoreline from its conditional grid."""
    cls = (rng.random((n, len(P)))[:, :, None] > np.cumsum(P, 1)[None]).sum(2)   # (n,m)
    m = len(P)
    rows = np.arange(m)
    cond = C[rows[None, :], cls]                                                 # (n,m,100)
    u = rng.random((n, m, 1))
    cell = (u > np.cumsum(cond, 2)).sum(2)
    return cell // MAXG, cell % MAXG                                             # home, away goals

# ---------------------------------------------------------------- 5. simulate
BATCH = 500
title = np.zeros(len(TEAMS)); final = np.zeros(len(TEAMS)); semi = np.zeros(len(TEAMS))
quart = np.zeros(len(TEAMS)); r16 = np.zeros(len(TEAMS)); qual = np.zeros(len(TEAMS))
gwin = np.zeros(len(TEAMS))

done = 0
while done < N_SIMS:
    n = min(BATCH, N_SIMS - done); done += n
    hg, ag = _sample(GP, GC, n)                                                  # (n, NG)

    T = len(TEAMS)
    pts = np.zeros((n, T)); gdf = np.zeros((n, T)); gfr = np.zeros((n, T))
    np.add.at(pts, (slice(None), GH), 3 * (hg > ag) + (hg == ag))
    np.add.at(pts, (slice(None), GA), 3 * (ag > hg) + (hg == ag))
    np.add.at(gdf, (slice(None), GH), hg - ag); np.add.at(gdf, (slice(None), GA), ag - hg)
    np.add.at(gfr, (slice(None), GH), hg);      np.add.at(gfr, (slice(None), GA), ag)
    for ih, ia, sh, sa in PLAYED_G:
        pts[:, ih] += 3 * (sh > sa) + (sh == sa); pts[:, ia] += 3 * (sa > sh) + (sh == sa)
        gdf[:, ih] += sh - sa; gdf[:, ia] += sa - sh
        gfr[:, ih] += sh; gfr[:, ia] += sa

    score = pts * 1e6 + gdf * 1e3 + gfr + rng.random((n, T))        # random final tiebreak
    firsts = np.zeros((n, 12), int); seconds = np.zeros((n, 12), int)
    thirds = np.zeros((n, 12), int); third_sc = np.zeros((n, 12))
    for gi, g in enumerate(GKEYS):
        idx = GROUP_IDX[g]
        order = idx[np.argsort(-score[:, idx], axis=1)]
        firsts[:, gi], seconds[:, gi], thirds[:, gi] = order[:, 0], order[:, 1], order[:, 2]
        third_sc[:, gi] = np.take_along_axis(score, thirds[:, gi:gi + 1], 1)[:, 0]

    best8 = np.argsort(-third_sc, axis=1)[:, :8]
    third_q = np.take_along_axis(thirds, best8, 1)

    # seed 1-32: winners (by record), then runners-up, then qualifying thirds
    def _by_record(arr):
        s = np.take_along_axis(score, arr, 1)
        return np.take_along_axis(arr, np.argsort(-s, axis=1), 1)
    seeds = np.concatenate([_by_record(firsts), _by_record(seconds), _by_record(third_q)], 1)

    rows = np.arange(n)[:, None]
    np.add.at(qual, seeds.ravel(), 1)
    np.add.at(gwin, firsts.ravel(), 1)

    # standard bracket: 1v32, 16v17 | 8v25, 9v24 | ... -> fold pairs each round
    br = seeds.copy()
    for rnd, counter in [(32, r16), (16, quart), (8, semi), (4, final), (2, title)]:
        half = rnd // 2
        A, B = br[:, :half], br[:, rnd - 1:half - 1:-1]
        P = PAIR_P[A, B]                                             # (n,half,3)
        u = rng.random((n, half))[:, :, None]
        cls = (u > np.cumsum(P, 2)).sum(2)
        p_h = P[:, :, 2] / np.maximum(P[:, :, 2] + P[:, :, 0], 1e-12)  # ET/pens on a draw
        home_wins = np.where(cls == 1, rng.random((n, half)) < p_h, cls == 2)
        br = np.where(home_wins, A, B)
        np.add.at(counter, br.ravel(), 1)

# ---------------------------------------------------------------- 6. report
res = pd.DataFrame({
    "team": TEAMS, "elo": [round(strength[t]) for t in TEAMS],
    "group": [GROUP_OF[t] for t in TEAMS],
    "win_group": gwin / N_SIMS, "qualify": qual / N_SIMS,
    "R16": r16 / N_SIMS, "QF": quart / N_SIMS, "SF": semi / N_SIMS,
    "final": final / N_SIMS, "champion": title / N_SIMS,
}).sort_values("champion", ascending=False).reset_index(drop=True)
res.to_csv("data/tournament_sim.csv", index=False)

pd.set_option("display.width", 200)
print(f"\nFIFA World Cup 2026 — {N_SIMS:,} Monte Carlo tournaments\n"
      f"(Step-3 ensemble match engine, seed={SEED})\n")
print(f"{'#':>3} {'Team':<24}{'Elo':>6}{'Grp':>5}{'WinGrp':>8}{'Qual':>7}{'R16':>7}"
      f"{'QF':>7}{'SF':>7}{'Final':>7}{'CHAMP':>8}")
for i, r in enumerate(res.itertuples(), 1):
    print(f"{i:>3} {r.team:<24}{r.elo:>6}{r.group:>5}{r.win_group*100:>7.1f}%{r.qualify*100:>6.1f}%"
          f"{r.R16*100:>6.1f}%{r.QF*100:>6.1f}%{r.SF*100:>6.1f}%{r.final*100:>6.1f}%{r.champion*100:>7.2f}%")
print(f"\nWINNER (most likely): {res.team[0]}  —  {res.champion[0]*100:.1f}%")
print("saved -> data/tournament_sim.csv")
