"""Phase 8: walk-forward backtest of the Elo + FIFA-ranking blend.

The 0.9024 walk-forward log-loss is for the Step-3 ensemble fed RAW Elo. The
forecast in data/predictions.csv came from the same ensemble, trained on raw
Elo, but fed Elo BLENDED with the FIFA ranking at prediction time (see
fifa_blend.py). The blend weights were never validated. This script puts the
blend through the same 5-fold walk-forward, exactly as the forecast used it:

  * the models are trained on raw Elo, as in 08 (training is unchanged);
  * for each test match, the latest FIFA release strictly before its date is
    used, if it is at most 365 days old (the history runs Dec 1992 - Sep 2024);
  * FIFA points are mapped to Elo by a least-squares line fitted, per match
    date, over the release's top 48 teams and their Elo that morning -- the
    analogue of 08 fitting it over the 48 World Cup teams -- and the blend is
    applied to those teams only; everyone else keeps raw Elo, as in 08;
  * `recent` counts each team's matches in the 4 years before the match date.

The team set (top 48) and the shipped weight rule were fixed before any result
was seen, because they mirror 08. Everything else below is reported as
sensitivity, not chosen from.

Every row is POOLED per-match log-loss over the subset, with a paired-bootstrap
95% interval for (blended - raw) on the same matches. (harness.py averages the
five fold log-losses instead; that convention is printed once, only to check
the raw-Elo run still reproduces 0.9024. The five test folds are the same
size, so here the two conventions give the same numbers.) Subsets:

  all test matches      -- 27% have no release under a year old (all of fold
                           1 and part of fold 2 predate 1993; the history
                           ends Sep 2024), so the blend cannot reach them;
  with a FIFA release   -- test matches with a release at most a year old;
  both teams blended    -- both sides in that release's top 48;
  ... major finals      -- that, restricted to major-tournament finals;
  ... before / after the August 2018 change to FIFA's ranking method.

Sensitivity: constant weights 0.2/0.4/0.6 in place of the shipped rule, and
fitting/applying the line over every ranked team instead of the top 48.

Also scores, on the 68 World Cup games in 11_score_tournament.py, the same
model re-run with the blend switched off.

Writes data/blend_backtest.csv (one row per subset x variant).

Run:  python src/12_blend_backtest.py
"""
import importlib
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from fifa_blend import BLEND_MAX, BLEND_MIN, RECENT_YEARS, blend_weight, fifa_to_elo_line, load_history
from harness import expected_goals, fit, grid_probs, load, outcomes
from step3_classifier import BEST_RHO, clf_probs, make_clf, match_features

W, TEMP, N_SPLITS = 0.3, 0.95, 5          # the shipped ensemble (as in 08)
START, HOME_ADV, TOP_N, MAX_AGE = 1500.0, 60.0, 48, pd.Timedelta(days=365)
N_BOOT, BOOT_SEED = 10000, 20260928
OUT = "data/blend_backtest.csv"
SUM_METHOD = pd.Timestamp("2018-08-16")   # first release under FIFA's current (Elo-style) method
MAJOR = {"FIFA World Cup", "UEFA Euro", "Copa América", "African Cup of Nations",
         "AFC Asian Cup", "Gold Cup", "Confederations Cup"}


def fifa_equivalents(df, hist):
    """Per match: each side's FIFA-equivalent Elo (NaN if unranked or no fresh
    release) under two team sets, plus its recent-match count."""
    releases = {d: g for d, g in hist.groupby("date")}
    rel_dates = np.array(sorted(releases), dtype="datetime64[ns]")
    elo = defaultdict(lambda: START)
    seen = defaultdict(deque)
    exp = lambda ra, rb: 1 / (1 + 10 ** ((rb - ra) / 400))
    out = {k: np.full(len(df), np.nan) for k in
           ["h_top", "a_top", "h_all", "a_all", "h_recent", "a_recent", "h_elo_check", "a_elo_check",
            "release"]}

    for date, day in df.groupby("date", sort=True):
        # the FIFA -> Elo lines for this date, from Elo as it stood that morning
        lines = {}
        i = np.searchsorted(rel_dates, np.datetime64(date), side="left") - 1
        if i >= 0 and date - pd.Timestamp(rel_dates[i]) <= MAX_AGE:
            rel = releases[pd.Timestamp(rel_dates[i])]
            for key, sub in [("top", rel.head(TOP_N)), ("all", rel)]:
                a, b = fifa_to_elo_line(sub.total_points.values, [elo[t] for t in sub.team])
                lines[key] = {t: a * p + b for t, p in zip(sub.team, sub.total_points)}
        cutoff = date - pd.DateOffset(years=RECENT_YEARS)
        for t in set(day.home_team) | set(day.away_team):
            q = seen[t]
            while q and q[0] < cutoff:
                q.popleft()
        recent = {t: len(seen[t]) for t in set(day.home_team) | set(day.away_team)}

        for r in day.itertuples():
            j = r.Index
            if lines:
                out["release"][j] = rel_dates[i].astype("datetime64[D]").astype(float)
            for side, team in [("h", r.home_team), ("a", r.away_team)]:
                out[f"{side}_recent"][j] = recent[team]
                for key in ["top", "all"]:
                    if key in lines and team in lines[key]:
                        out[f"{side}_{key}"][j] = lines[key][team]
            # replay Elo exactly as 02_elo.py, to check it matches features.csv
            rh, ra = elo[r.home_team], elo[r.away_team]
            out["h_elo_check"][j], out["a_elo_check"][j] = rh, ra
            k = 60 if "FIFA World Cup" in r.tournament else (40 if r.tournament != "Friendly" else 20)
            gd = abs(r.home_score - r.away_score); k *= 1 if gd <= 1 else (1.5 if gd == 2 else 1 + gd / 5)
            res = 1.0 if r.home_score > r.away_score else (0.5 if r.home_score == r.away_score else 0.0)
            ch = k * (res - exp(rh + (0 if r.neutral else HOME_ADV), ra))
            elo[r.home_team] += ch; elo[r.away_team] -= ch
        for t in list(day.home_team) + list(day.away_team):
            seen[t].append(date)
    return pd.DataFrame(out, index=df.index)


def blended(df, fe, key, weight):
    """Copy of df with home_elo/away_elo blended. `weight` maps recent -> w."""
    d = df.copy()
    for side, col in [("h", "home_elo"), ("a", "away_elo")]:
        f = fe[f"{side}_{key}"].values
        w = np.array([weight(n) for n in fe[f"{side}_recent"].values])
        has = ~np.isnan(f)
        d.loc[has, col] = (1 - w[has]) * d.loc[has, col].values + w[has] * f[has]
    return d


def ensemble(pm, cm, d):
    P = W * grid_probs(*expected_goals(pm, d), rho=BEST_RHO) + (1 - W) * clf_probs(cm, d)
    P = np.clip(P, 1e-12, 1.0) ** (1.0 / TEMP)
    return P / P.sum(axis=1, keepdims=True)


def per_match_ll(y, P):
    return -np.log(np.clip(P[np.arange(len(y)), y], 1e-15, None))


def paired(diff, n_boot=N_BOOT, seed=BOOT_SEED):
    """Mean per-match log-loss difference with a paired-bootstrap 95% interval."""
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(n_boot, len(diff)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


def verdict(lo, hi):
    return "excludes 0" if lo > 0 or hi < 0 else "spans 0"


def main():
    df = load()
    fe = fifa_equivalents(df, load_history())
    drift = max(np.abs(fe.h_elo_check - df.home_elo).max(), np.abs(fe.a_elo_check - df.away_elo).max())
    assert drift < 1e-6, f"Elo replay drifted from features.csv by {drift}"

    variants = {"raw Elo (no blend)": None,
                f"shipped rule clip(10/(10+n),{BLEND_MIN},{BLEND_MAX})": ("top", blend_weight),
                "constant w=0.2": ("top", lambda n: 0.2),
                "constant w=0.4": ("top", lambda n: 0.4),
                "constant w=0.6": ("top", lambda n: 0.6),
                "shipped rule, all ranked teams": ("all", blend_weight)}
    frames = {name: (df if v is None else blended(df, fe, *v)) for name, v in variants.items()}
    SHIPPED = list(variants)[1]

    RAW = "raw Elo (no blend)"
    fold_ll = defaultdict(list)
    ll = {name: np.full(len(df), np.nan) for name in variants}
    for tr, te in TimeSeriesSplit(n_splits=N_SPLITS).split(df):
        train = df.iloc[tr]
        pm = fit(train)
        cm = make_clf("logit").fit(match_features(train), outcomes(train))
        y = outcomes(df.iloc[te])
        for name, d in frames.items():
            l = per_match_ll(y, ensemble(pm, cm, d.iloc[te]))
            ll[name][te] = l
            fold_ll[name].append(l.mean())

    tested = ~np.isnan(ll[SHIPPED])
    fresh = fe.release.notna().values & tested
    both_top = fe.h_top.notna().values & fe.a_top.notna().values & tested
    both_all = fe.h_all.notna().values & fe.a_all.notna().values & tested
    new_method = (fe.release.values >= SUM_METHOD.to_datetime64().astype("datetime64[D]").astype(float))
    subsets = {
        "all test matches": tested,
        "with a FIFA release": fresh,
        "both teams blended": both_top,
        "both blended, major finals": both_top & df.tournament.isin(MAJOR).values,
        "both blended, release before Aug 2018": both_top & ~new_method,
        "both blended, release from Aug 2018": both_top & new_method,
    }

    print("=" * 96)
    print("WALK-FORWARD BACKTEST OF THE ELO + FIFA BLEND  (ensemble trained on raw Elo, as shipped)")
    print("=" * 96)
    print(f"reproduction check, fold-mean convention of harness.py: raw Elo {np.mean(fold_ll[RAW]):.4f}"
          f"  (shipped blend {np.mean(fold_ll[SHIPPED]):.4f})")
    print(f"test matches {tested.sum():,}; without a FIFA release under a year old "
          f"(before 1993, or after Sep 2025) {(tested & ~fresh).sum():,} "
          f"({(tested & ~fresh).sum() / tested.sum():.0%})")
    print("\nPOOLED log-loss per subset; diff = variant - raw on the same matches, paired bootstrap 95% CI")

    ALL_TEAMS = "shipped rule, all ranked teams"
    rows = []

    def report(sub, name, mask):
        base, v = ll[RAW][mask], ll[name][mask]
        d, lo, hi = paired(v - base)
        print(f"   {name:<44}{v.mean():.4f}   diff {d:+.4f}  [{lo:+.4f}, {hi:+.4f}]  {verdict(lo, hi)}")
        rows.append(dict(subset=sub, variant=name, n=int(mask.sum()), log_loss_raw=base.mean(),
                         log_loss_variant=v.mean(), diff=d, ci_low=lo, ci_high=hi))

    for sub, mask in subsets.items():
        print(f"\n{sub} (n={mask.sum():,})   raw {ll[RAW][mask].mean():.4f}")
        for name in variants:
            if name not in (RAW, ALL_TEAMS):
                report(sub, name, mask)

    sub = "both teams blended, line over all ranked teams"
    print(f"\nsensitivity: {sub} (n={both_all.sum():,})   raw {ll[RAW][both_all].mean():.4f}")
    report(sub, ALL_TEAMS, both_all)

    rows.append(tournament_check())
    pd.DataFrame(rows).round(6).to_csv(OUT, index=False)
    print(f"\nsaved -> {OUT}")
    print("=" * 96)


def tournament_check():
    """The 68 scored World Cup games: frozen forecast vs the same model, no blend."""
    fx = importlib.import_module("08_fixture_predictions")
    score = importlib.import_module("11_score_tournament")
    act = pd.read_csv(score.ACTUALS)
    runs = {}
    for blend in (True, False):
        p, _ = fx.predict(blend=blend)
        p["date"] = p.date.astype(str)
        m = score.eligible(p.merge(act[score.KEY + ["home_score", "away_score"]], on=score.KEY))
        runs[blend] = m.reset_index(drop=True)
    frozen = pd.read_csv(score.PREDICTIONS)
    chk = runs[True].merge(frozen, on=score.KEY)
    drift = (chk.p_home - chk.p_home_pre).abs().max()
    assert drift < 1e-12, f"08 with the blend no longer reproduces the frozen forecast ({drift:.2e})"

    y = score.outcomes(runs[True])
    P = {b: r[["p_away", "p_draw", "p_home"]].values for b, r in runs.items()}
    print(f"\nWC 2026 group games scored in 11_score_tournament.py (n={len(y)}), post hoc:")
    print(f"   frozen forecast (blend on)                  {per_match_ll(y, P[True]).mean():.4f}")
    print(f"   same model, blend off                       {per_match_ll(y, P[False]).mean():.4f}")
    d, lo, hi = score.paired_bootstrap(y, P[True], P[False])
    print(f"   blend on - blend off: {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  {verdict(lo, hi)}")
    return dict(subset="WC 2026 scored group games (post hoc)", variant="frozen forecast (shipped rule)",
                n=len(y), log_loss_raw=per_match_ll(y, P[False]).mean(),
                log_loss_variant=per_match_ll(y, P[True]).mean(), diff=d, ci_low=lo, ci_high=hi)


if __name__ == "__main__":
    main()
