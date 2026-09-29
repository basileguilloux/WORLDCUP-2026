"""Phase 6: full-tournament Monte Carlo — who wins WC 2026?

The rest of the pipeline stops at per-fixture W/D/L probabilities. This script
plays the whole tournament out N times to turn those into "P(lifts the trophy)".

    python src/10_tournament_sim.py [N_SIMS]      # WC 2026, writes data/tournament_sim.csv

It is also importable: simulate(cutoff_date, fmt, n_sims, seed, model) runs any
format in tournament_formats.py using only data before `cutoff_date`, and
returns per-team stage probabilities. 13_backtest_tournaments.py uses it on ten
past tournaments.

Match engine
------------
Outcome class (0=away,1=draw,2=home) is sampled from the Step-3 FINAL ensemble
(0.3 * Dixon-Coles Poisson + 0.7 * multinomial logit, T=0.95) — the exact model
behind data/predictions.csv, so the sim inherits its 0.9024 walk-forward log-loss.
It REFITS that ensemble in-process from data/features.csv rather than reading any
predictions CSV, so it needs no prediction file and ignores the team-news overlay.
A scoreline is then sampled from the DC-Poisson grid CONDITIONAL on that class,
which gives the goal difference / goals for that group tables need without
disturbing the calibrated outcome probabilities.

`model` swaps the engine, for the backtest baselines:
  full      -- the above, strength = Elo blended with the FIFA ranking (shipped)
  no_blend  -- the above, strength = raw Elo
  elo_only  -- 03_elo_baseline.py's logistic on elo_diff (raw Elo), scorelines
               from the no_blend Dixon-Coles grid conditional on its outcome
  uniform   -- every match 1/3 each way, scorelines from the grid of two
               identical average teams

WC 2026
-------
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
* Strength blends Elo with the 11 June 2026 FIFA snapshot, line fitted over its 48 teams.

Backtest formats (World Cups 2006-2022, Euros 2008-2024)
--------------------------------------------------------
* Groups and the official bracket come from the format; every match is unplayed.
* Standings and third-place ranking as above (no head-to-head, no fair play).
  Best thirds (Euros 2016-2024) are placed by UEFA's allocation table.
* A host gets the home flag only in a match played in its own country, groups
  and knockouts alike; knockout venues come from the format. Everything else is
  neutral. Knockout draws are resolved as above.
* Strength blends Elo with the latest FIFA release before the cutoff
  (fifa_ranking_history.csv), line fitted over the tournament's participants.
"""
import sys
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from scipy.stats import poisson as pois
from sklearn.linear_model import LogisticRegression

from harness import load, outcomes, expected_goals, fit
from step3_classifier import make_clf, match_features, clf_probs, BEST_RHO
from fifa_blend import blend_weight, fifa_to_elo_line, load_history
import tournament_formats as tf

SEED = 20260618
W, TEMP = 0.3, 0.95
N, START, HOME_ADV = 5, 1500.0, 60.0
MAXG = 10
BATCH = 500
MODELS = ("full", "no_blend", "elo_only", "uniform")


# ---------------------------------------------------------------- 1. strength
def load_results():
    """results.csv with former names mapped to current ones, sorted by date."""
    df = pd.read_csv("data/raw/results.csv", parse_dates=["date"])
    ren = dict(zip(*[pd.read_csv("data/raw/former_names.csv")[c] for c in ["former", "current"]]))
    df["home_team"], df["away_team"] = df.home_team.replace(ren), df.away_team.replace(ren)
    return df.sort_values("date")


def replay(df, cutoff, recent_start):
    """Elo, last-N goal history and recent-match counts from matches before `cutoff`."""
    played = df[df.home_score.notna() & (df.date < cutoff)]
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
        if r.date >= recent_start:
            recent[r.home_team] += 1; recent[r.away_team] += 1
    return elo, hist, recent


def blended_strength(fmt, elo, recent, teams, cutoff):
    """Elo mixed with the FIFA ranking (see fifa_blend.py), as the forecast used it.

    WC 2026 uses its own 11 June snapshot and fits the FIFA -> Elo line over the
    snapshot's 48 teams. Backtests use the latest release strictly before the
    cutoff and fit the line over the tournament's participants.
    """
    if fmt.ranking_snapshot:
        rank = pd.read_csv("data/raw/fifa_ranking_2026-06-11.csv")
        wc = list(rank.team)
        fifa_pts = dict(zip(rank.team, rank.fifa_points))
        a, b = np.polyfit([fifa_pts[t] for t in wc], [elo[t] for t in wc], 1)
    else:
        hist = load_history()
        release = hist[hist.date < cutoff]
        release = release[release.date == release.date.max()]
        wc = list(teams)
        fifa_pts = dict(zip(release.team, release.total_points))
        a, b = fifa_to_elo_line([fifa_pts[t] for t in wc], [elo[t] for t in wc])
    fifa_elo = {t: a * fifa_pts[t] + b for t in wc}
    strength = defaultdict(lambda: START)
    for t in set(list(elo) + wc):
        s = elo[t]
        if t in fifa_elo:
            w = blend_weight(recent[t]); s = (1 - w) * elo[t] + w * fifa_elo[t]
        strength[t] = s
    return strength


# ---------------------------------------------------------------- 2. engine
class MatchModel:
    """Per-match outcome probabilities plus conditional scoreline grids."""

    def __init__(self, strength, hist, feat):
        self.strength, self.hist = strength, hist
        self.poisson = fit(feat)
        self.logit = make_clf("logit").fit(match_features(feat), outcomes(feat))

    def form(self, t):
        h = self.hist[t]
        if not h: return 1.0, 1.0
        return float(np.mean([g[0] for g in h])), float(np.mean([g[1] for g in h]))

    def probs(self, home, away, home_adv):
        """-> (P_outcome[away,draw,home], conditional score-grid pmfs per class)."""
        fx = pd.DataFrame({
            "home_team": home, "away_team": away,
            "home_elo": [self.strength[t] for t in home], "away_elo": [self.strength[t] for t in away],
            "home_advantage": np.asarray(home_adv, int),
        })
        fx["home_gf"], fx["home_ga"] = zip(*[self.form(t) for t in home])
        fx["away_gf"], fx["away_ga"] = zip(*[self.form(t) for t in away])
        lh, la = expected_goals(self.poisson, fx)

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

        P = W * Pp + (1 - W) * clf_probs(self.logit, fx)
        P = np.clip(P, 1e-12, 1.0) ** (1.0 / TEMP)
        P /= P.sum(1, keepdims=True)

        # conditional scoreline pmf within each outcome class, flattened
        cond = []
        for m in masks:
            c = (joint * m).reshape(len(fx), -1)
            cond.append(c / np.maximum(c.sum(1, keepdims=True), 1e-300))
        return P, np.stack(cond, 1)                                      # (n,3,MAXG*MAXG)


class EloOnlyModel:
    """03_elo_baseline.py's logistic on elo_diff; scorelines from `grid`'s DC grid."""

    def __init__(self, grid, feat):
        self.grid, self.strength = grid, grid.strength
        self.clf = LogisticRegression().fit((feat.home_elo - feat.away_elo).to_frame("elo_diff"),
                                            outcomes(feat))

    def probs(self, home, away, home_adv):
        diff = np.array([self.strength[h] - self.strength[a] for h, a in zip(home, away)])
        P = self.clf.predict_proba(pd.DataFrame({"elo_diff": diff}))
        col = {c: i for i, c in enumerate(self.clf.classes_)}
        return P[:, [col[0], col[1], col[2]]], self.grid.probs(home, away, home_adv)[1]


class UniformModel:
    """Every match 1/3 each way; scorelines from two identical average teams."""

    def __init__(self, feat):
        self.strength = defaultdict(lambda: START)
        self.grid = MatchModel(self.strength, defaultdict(deque), feat)

    def probs(self, home, away, home_adv):
        _, C = self.grid.probs(home, away, np.zeros(len(home)))
        return np.full((len(home), 3), 1 / 3), C


def make_model(model, fmt, cutoff, elo, hist, recent, feat, teams):
    if model == "full":
        return MatchModel(blended_strength(fmt, elo, recent, teams, cutoff), hist, feat)
    if model == "no_blend":
        return MatchModel(elo, hist, feat)
    if model == "elo_only":
        return EloOnlyModel(MatchModel(elo, hist, feat), feat)
    if model == "uniform":
        return UniformModel(feat)
    raise ValueError(model)


def _sample(rng, P, C, n):
    """Vectorised: sample outcome class then a scoreline from its conditional grid."""
    cls = (rng.random((n, len(P)))[:, :, None] > np.cumsum(P, 1)[None]).sum(2)   # (n,m)
    m = len(P)
    rows = np.arange(m)
    cond = C[rows[None, :], cls]                                                 # (n,m,100)
    u = rng.random((n, m, 1))
    cell = (u > np.cumsum(cond, 2)).sum(2)
    return cell // MAXG, cell % MAXG                                             # home, away goals


def _knockout(rng, P, A, B, n, half):
    """Play one knockout round; a 90' draw goes to a coin weighted by regulation odds."""
    u = rng.random((n, half))[:, :, None]
    cls = (u > np.cumsum(P, 2)).sum(2)
    p_h = P[:, :, 2] / np.maximum(P[:, :, 2] + P[:, :, 0], 1e-12)      # ET/pens on a draw
    home_wins = np.where(cls == 1, rng.random((n, half)) < p_h, cls == 2)
    return np.where(home_wins, A, B)


# ---------------------------------------------------------------- 3. groups
def groups_from_fixtures(future, played):
    """Groups as connected components of the fixture graph, lettered by first fixture."""
    adj = defaultdict(set)
    for h, aw in zip(future.home_team, future.away_team):
        adj[h].add(aw); adj[aw].add(h)
    for h, aw in zip(played.home_team, played.away_team):
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
    return {chr(65 + i): c for i, c in enumerate(comps)}


def host_fixtures(fmt, fixtures):
    """(home, away, home_flag): a host is at home only in its own country; the host
    is put first when it is listed as the away side."""
    home, away, flag = [], [], []
    for r in fixtures.itertuples():
        h, a, f = r.home_team, r.away_team, 0
        if a in fmt.hosts and r.country == a:
            h, a = a, h
        if h in fmt.hosts and r.country == h:
            f = 1
        home.append(h); away.append(a); flag.append(f)
    return home, away, np.array(flag)


# ---------------------------------------------------------------- 4. simulate
def simulate(cutoff_date, fmt, n_sims, seed, model="full"):
    """Play `fmt` out n_sims times using only data before `cutoff_date`.

    Returns one row per team: strength, group, P(win group), P(qualify) and
    P(reach) each later round through `champion` (fmt.reach_columns).
    """
    rng = np.random.default_rng(seed)
    cutoff = pd.Timestamp(cutoff_date)

    df = load_results()
    elo, hist, recent = replay(df, cutoff, fmt.recent_start)
    feat = load()
    feat = feat[feat.date < cutoff]

    m = tf.matches(fmt, df)
    if fmt.seeded:
        future, played_t = m[m.date >= cutoff], m[m.date < cutoff]
        GROUPS = groups_from_fixtures(future, played_t)
        assert len(GROUPS) == 12 and all(len(v) == 4 for v in GROUPS.values()), GROUPS
    else:
        gm = tf.group_matches(fmt, m)
        future, played_t = gm[gm.date >= cutoff], gm[gm.date < cutoff]
        GROUPS = {g: sorted(ts) for g, ts in fmt.groups.items()}
    TEAMS = sorted(t for g in GROUPS.values() for t in g)
    TIDX = {t: i for i, t in enumerate(TEAMS)}
    GROUP_OF = {t: g for g, ts in GROUPS.items() for t in ts}
    assert set(future.home_team) | set(future.away_team) <= set(TEAMS)

    engine = make_model(model, fmt, cutoff, elo, hist, recent, feat, TEAMS)

    if fmt.seeded:
        # every ordered pair at a neutral venue (knockouts + lookup)
        pair_h = [h for h in TEAMS for a in TEAMS]
        pair_a = [a for h in TEAMS for a in TEAMS]
        PAIR_P, PAIR_C = engine.probs(pair_h, pair_a, np.zeros(len(pair_h)))
        PAIR_P = PAIR_P.reshape(len(TEAMS), len(TEAMS), 3)

        gf =future[["home_team", "away_team", "neutral"]].reset_index(drop=True)
        g_home, g_away = list(gf.home_team), list(gf.away_team)
        g_flag = (~gf.neutral.astype(bool)).astype(int)
    else:
        g_home, g_away, g_flag = host_fixtures(fmt, future)
    GP, GC = engine.probs(g_home, g_away, g_flag)
    GH = np.array([TIDX[t] for t in g_home])
    GA = np.array([TIDX[t] for t in g_away])

    # already-played group games
    PLAYED_G = [(TIDX[r.home_team], TIDX[r.away_team], int(r.home_score), int(r.away_score))
                for r in played_t.itertuples()]

    GROUP_IDX = {g: np.array([TIDX[t] for t in ts]) for g, ts in GROUPS.items()}
    GKEYS = list(GROUPS)
    NGR = len(GKEYS)

    if not fmt.seeded:
        ko = _Bracket(fmt, GKEYS, TIDX, engine, len(TEAMS))

    T = len(TEAMS)
    gwin, qual = np.zeros(T), np.zeros(T)
    reach = {c: np.zeros(T) for c in fmt.reach_columns}

    done = 0
    while done < n_sims:
        n = min(BATCH, n_sims - done); done += n
        hg, ag = _sample(rng, GP, GC, n)                                             # (n, NG)

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
        firsts = np.zeros((n, NGR), int); seconds = np.zeros((n, NGR), int)
        thirds = np.zeros((n, NGR), int); third_sc = np.zeros((n, NGR))
        for gi, g in enumerate(GKEYS):
            idx = GROUP_IDX[g]
            order = idx[np.argsort(-score[:, idx], axis=1)]
            firsts[:, gi], seconds[:, gi], thirds[:, gi] = order[:, 0], order[:, 1], order[:, 2]
            third_sc[:, gi] = np.take_along_axis(score, thirds[:, gi:gi + 1], 1)[:, 0]
        np.add.at(gwin, firsts.ravel(), 1)

        if fmt.seeded:
            best8 = np.argsort(-third_sc, axis=1)[:, :8]
            third_q = np.take_along_axis(thirds, best8, 1)

            # seed 1-32: winners (by record), then runners-up, then qualifying thirds
            def _by_record(arr):
                s = np.take_along_axis(score, arr, 1)
                return np.take_along_axis(arr, np.argsort(-s, axis=1), 1)
            seeds = np.concatenate([_by_record(firsts), _by_record(seconds), _by_record(third_q)], 1)
            np.add.at(qual, seeds.ravel(), 1)

            # standard bracket: 1v32, 16v17 | 8v25, 9v24 | ... -> fold pairs each round
            br = seeds.copy()
            for rnd, col in zip([32, 16, 8, 4, 2], fmt.reach_columns):
                half = rnd // 2
                A, B = br[:, :half], br[:, rnd - 1:half - 1:-1]
                br = _knockout(rng, PAIR_P[A, B], A, B, n, half)
                np.add.at(reach[col], br.ravel(), 1)
        else:
            br = ko.first_round(firsts, seconds, thirds, third_sc)
            np.add.at(qual, br.ravel(), 1)
            for rnd, col in zip(fmt.ko_rounds, fmt.reach_columns):
                half = br.shape[1] // 2
                A, B = br[:, 0::2], br[:, 1::2]
                br = _knockout(rng, ko.pair_probs(rnd, A, B), A, B, n, half)
                np.add.at(reach[col], br.ravel(), 1)

    return pd.DataFrame({
        "team": TEAMS, "elo": [round(engine.strength[t]) for t in TEAMS],
        "group": [GROUP_OF[t] for t in TEAMS],
        "win_group": gwin / n_sims, "qualify": qual / n_sims,
        **{c: reach[c] / n_sims for c in fmt.reach_columns},
    })


class _Bracket:
    """The fixed knockout bracket of a backtest format."""

    def __init__(self, fmt, gkeys, tidx, engine, n_teams):
        self.fmt, self.gi = fmt, {g: i for i, g in enumerate(gkeys)}
        teams = sorted(tidx, key=tidx.get)
        pair_h = [h for h in teams for a in teams]
        pair_a = [a for h in teams for a in teams]
        self.neutral = engine.probs(pair_h, pair_a, np.zeros(len(pair_h)))[0].reshape(n_teams, n_teams, 3)
        self.home = engine.probs(pair_h, pair_a, np.ones(len(pair_h)))[0].reshape(n_teams, n_teams, 3)
        self.host_idx = {t: tidx[t] for t in fmt.hosts if t in tidx}

        # combination of qualifying thirds (bitmask of groups) -> group of the third
        # facing each winner in fmt.thirds_winners
        self.third_of = np.full((1 << len(gkeys), max(len(fmt.thirds_winners), 1)), -1)
        for combo, groups in (fmt.thirds_table or {}).items():
            mask = sum(1 << self.gi[g] for g in combo)
            self.third_of[mask] = [self.gi[g] for g in groups]

    def first_round(self, firsts, seconds, thirds, third_sc):
        n = len(firsts)
        rows = np.arange(n)
        if self.fmt.best_thirds:
            best = np.argsort(-third_sc, axis=1)[:, :self.fmt.best_thirds]
            mask = (1 << best).sum(1)
            facing = self.third_of[mask]
            assert (facing >= 0).all()
        cols = []
        for pair in self.fmt.bracket:
            for slot in pair:
                if slot.startswith("3/"):
                    k = self.fmt.thirds_winners.index(slot[2:])
                    cols.append(thirds[rows, facing[:, k]])
                else:
                    cols.append((firsts if slot[0] == "1" else seconds)[:, self.gi[slot[1]]])
        return np.stack(cols, 1)                                         # bracket order

    def pair_probs(self, rnd, A, B):
        """Outcome probabilities for A (listed home) v B, with a host at home in its country."""
        P = self.neutral[A, B]
        for i in range(A.shape[1]):
            h = self.host_idx.get(self.fmt.venue(rnd, i))
            if h is None:
                continue
            a_home, b_home = A[:, i] == h, B[:, i] == h
            P[a_home, i] = self.home[A[a_home, i], B[a_home, i]]
            P[b_home, i] = self.home[B[b_home, i], A[b_home, i]][:, ::-1]
        return P


# ---------------------------------------------------------------- 5. report
def main():
    n_sims = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    fmt = tf.WC2026
    res = simulate(fmt.cutoff_date, fmt, n_sims, SEED)
    res = res.sort_values("champion", ascending=False).reset_index(drop=True)
    res.to_csv("data/tournament_sim.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"\nFIFA World Cup 2026 — {n_sims:,} Monte Carlo tournaments\n"
          f"(Step-3 ensemble match engine, seed={SEED})\n")
    print(f"{'#':>3} {'Team':<24}{'Elo':>6}{'Grp':>5}{'WinGrp':>8}{'Qual':>7}{'R16':>7}"
          f"{'QF':>7}{'SF':>7}{'Final':>7}{'CHAMP':>8}")
    for i, r in enumerate(res.itertuples(), 1):
        print(f"{i:>3} {r.team:<24}{r.elo:>6}{r.group:>5}{r.win_group*100:>7.1f}%{r.qualify*100:>6.1f}%"
              f"{r.R16*100:>6.1f}%{r.QF*100:>6.1f}%{r.SF*100:>6.1f}%{r.final*100:>6.1f}%{r.champion*100:>7.2f}%")
    print(f"\nWINNER (most likely): {res.team[0]}  —  {res.champion[0]*100:.1f}%")
    print("saved -> data/tournament_sim.csv")


if __name__ == "__main__":
    main()
