"""Phase 9: backtest the tournament simulator on ten past tournaments.

The 2026 title odds (data/tournament_sim.csv) come from a simulator written
after the tournament, so on their own they are illustration, not evidence. This
script runs the same simulator blind on World Cups 2006-2022 and Euros
2008-2024. Each run sees only data from before the day preceding kickoff
(tournament_formats.py), and the result is scored against the stage each team
actually reached.

Everything here follows docs/BACKTEST_PROTOCOL.md, committed before this
script was written. In short:

  models   full (shipped: Elo + FIFA blend, Step-3 ensemble), no FIFA blend,
           Elo only (03's logistic), uniform -- all through 10_tournament_sim
  sims     N_SIMS = 10,000 per tournament and model, seed 20260929 for every run
  metrics  RPS over the ordered stages (primary), -ln p(actual champion),
           Brier on "reached the semi-final", and a reliability table
  CIs      paired bootstrap over whole tournaments (n = 10, so wide);
           team-level bootstrap for RPS as a secondary figure only, since teams
           in one tournament are not independent

Writes data/tournament_backtest.csv (tournament x model x metric),
data/tournament_backtest_teams.csv (per-team probabilities and actual stage)
and figures/tournament_calibration.png.

Run:  python src/13_backtest_tournaments.py
"""
import importlib
from pathlib import Path

import numpy as np
import pandas as pd

import tournament_formats as tf

sim = importlib.import_module("10_tournament_sim")

N_SIMS, SEED = 10000, 20260929
N_BOOT, BOOT_SEED = 10000, 20260929
FLOOR = 1 / (2 * N_SIMS)                     # champion never reached in simulation
MODELS = {"full": "Full model (shipped)", "no_blend": "No FIFA blend",
          "elo_only": "Elo only", "uniform": "Uniform"}
BASE = "full"
BINS = np.linspace(0, 1, 11)
REACH = ["reach_R16", "reach_QF", "reach_SF", "reach_final", "champion"]
OUT, OUT_TEAMS = "data/tournament_backtest.csv", "data/tournament_backtest_teams.csv"
FIGURE = "figures/tournament_calibration.png"


def reach_matrix(fmt, r):
    """P(stage >= k) for k = 0..K-1 over fmt.stages, one row per team."""
    cols = ["qualify"] + list(fmt.reach_columns)
    return np.column_stack([np.ones(len(r))] + [r[c].values for c in cols])


def rps(reach, actual):
    """Ranked probability score over the ordered stages, per team."""
    K = reach.shape[1]
    cdf_pred = 1 - reach[:, 1:]                                   # P(stage <= k), k = 0..K-2
    cdf_obs = (actual[:, None] <= np.arange(K - 1)[None, :]).astype(float)
    return ((cdf_pred - cdf_obs) ** 2).sum(1) / (K - 1)


def team_rows(fmt, model, r, stage):
    reach = reach_matrix(fmt, r)
    actual = np.array([stage[t] for t in r.team])
    sf = fmt.stages.index("SF")
    out = pd.DataFrame({"tournament": fmt.name, "model": model, "team": r.team,
                        "group": r.group, "strength": r.elo, "win_group": r.win_group})
    # simulate() names columns by format ("qualify" = reaching the first knockout
    # round); rename them to the same stage names for every tournament
    src = ["qualify"] + list(fmt.reach_columns)
    dst = ["reach_" + fmt.ko_rounds[0]] + ["reach_" + c for c in fmt.reach_columns[:-1]] + ["champion"]
    named = dict(zip(dst, src))
    for c in REACH:
        out[c] = r[named[c]].values if c in named else np.nan
    out["actual_stage"] = [fmt.stages[k] for k in actual]
    out["rps"] = rps(reach, actual)
    out["sf_brier"] = (reach[:, sf] - (actual >= sf)) ** 2
    out["champion_log_score"] = np.where(actual == len(fmt.stages) - 1,
                                         -np.log(np.maximum(r.champion.values, FLOOR)), np.nan)
    return out, reach, actual


def paired_boot(a, b, n_boot=N_BOOT, seed=BOOT_SEED):
    """Mean of (a - b) with a paired percentile-bootstrap 95% interval."""
    d = np.asarray(a) - np.asarray(b)
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n_boot, len(d)))
    lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def verdict(lo, hi):
    return "excludes 0" if lo > 0 or hi < 0 else "spans 0"


def reliability(preds, obs):
    b = np.clip(np.digitize(preds, BINS[1:-1]), 0, len(BINS) - 2)
    rows = []
    for k in range(len(BINS) - 1):
        m = b == k
        rows.append((BINS[k], BINS[k + 1], int(m.sum()),
                     preds[m].mean() if m.any() else np.nan, obs[m].mean() if m.any() else np.nan))
    return rows


def plot(rel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=150)
    ax.plot([0, 1], [0, 1], color="#999999", lw=1, ls="--", label="perfect calibration")
    for model, color, marker in [("full", "#1f5fa8", "o"), ("elo_only", "#d2691e", "s")]:
        r = [x for x in rel[model] if x[2] > 0]
        pred, obs, n = [x[3] for x in r], [x[4] for x in r], np.array([x[2] for x in r])
        ax.plot(pred, obs, color=color, lw=1.5, alpha=0.8)
        ax.scatter(pred, obs, s=12 + 1.2 * np.sqrt(n) * 6, color=color, marker=marker,
                   edgecolor="white", lw=0.8, zorder=3, label=MODELS[model])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("predicted probability of reaching the stage")
    ax.set_ylabel("observed frequency")
    ax.set_title("Title-odds backtest: reliability, 10 tournaments\n"
                 "(every team x every stage from reaching the knockouts to the title)", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.text(0.98, 0.02, "marker area ~ number of predictions in the bin", ha="right",
            va="bottom", fontsize=7.5, color="#555555", transform=ax.transAxes)
    Path(FIGURE).parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURE)


def main():
    results = sim.load_results()
    shootouts = pd.read_csv("data/raw/shootouts.csv", parse_dates=["date"])

    teams, per_t = [], {m: {"rps": [], "champion_log_score": [], "sf_brier": []} for m in MODELS}
    pooled = {m: ([], []) for m in MODELS}
    for fmt in tf.BACKTEST:
        stage = tf.actual_stages(fmt, results, shootouts)
        line = f"{fmt.name:<16}"
        for model in MODELS:
            r = sim.simulate(fmt.cutoff_date, fmt, N_SIMS, SEED, model)
            rows, reach, actual = team_rows(fmt, model, r, stage)
            teams.append(rows)
            per_t[model]["rps"].append(rows.rps.mean())
            per_t[model]["sf_brier"].append(rows.sf_brier.mean())
            per_t[model]["champion_log_score"].append(rows.champion_log_score.dropna().iloc[0])
            k = np.arange(1, reach.shape[1])
            pooled[model][0].append(reach[:, 1:].ravel())
            pooled[model][1].append((actual[:, None] >= k[None, :]).ravel().astype(float))
            line += f"  {model} rps {rows.rps.mean():.4f}"
        champ = [t for t, k in stage.items() if k == len(fmt.stages) - 1][0]
        print(f"{line}   champion {champ}")
    teams = pd.concat(teams, ignore_index=True)

    names = [f.name for f in tf.BACKTEST]
    out = []
    for model in MODELS:
        for metric, vals in per_t[model].items():
            out += [dict(tournament=t, model=model, metric=metric, value=v) for t, v in zip(names, vals)]
            mean, lo, hi = paired_boot(vals, np.zeros(len(vals)))
            out.append(dict(tournament="ALL", model=model, metric=metric, value=mean,
                            ci_low=lo, ci_high=hi, bootstrap="tournament", n=len(vals)))

    print("\n" + "=" * 100)
    print(f"TITLE-ODDS BACKTEST  {len(names)} tournaments, {N_SIMS:,} sims each, "
          f"paired bootstrap over tournaments ({N_BOOT:,} resamples)")
    print("=" * 100)
    for metric in ["rps", "champion_log_score", "sf_brier"]:
        print(f"\n{metric}  (lower is better)")
        for model, label in MODELS.items():
            r = [x for x in out if x["tournament"] == "ALL" and x["model"] == model and x["metric"] == metric][0]
            print(f"   {label:<22}{r['value']:.4f}   95% CI [{r['ci_low']:.4f}, {r['ci_high']:.4f}]")
        for other in MODELS:
            if other == BASE:
                continue
            d, lo, hi = paired_boot(per_t[BASE][metric], per_t[other][metric])
            print(f"   full - {other:<15}{d:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]  {verdict(lo, hi)}")
            out.append(dict(tournament="ALL", model=f"full - {other}", metric=metric, value=d,
                            ci_low=lo, ci_high=hi, bootstrap="tournament", n=len(names)))

    print("\nsecondary, RPS by team (pooled over tournaments; teams in one tournament are not independent)")
    base = teams[teams.model == BASE].rps.values
    for other in MODELS:
        if other == BASE:
            continue
        d, lo, hi = paired_boot(base, teams[teams.model == other].rps.values)
        print(f"   full - {other:<15}{d:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]  {verdict(lo, hi)}  n={len(base)}")
        out.append(dict(tournament="ALL", model=f"full - {other}", metric="rps", value=d,
                        ci_low=lo, ci_high=hi, bootstrap="team", n=len(base)))

    rel = {m: reliability(np.concatenate(p), np.concatenate(o)) for m, (p, o) in pooled.items()}
    print("\nreliability (all team-stage reach probabilities, pooled)")
    print(f"   {'bin':<12}" + "".join(f"{MODELS[m]:>30}" for m in MODELS))
    for i in range(len(BINS) - 1):
        cells = []
        for m in MODELS:
            lo_, hi_, n, p, o = rel[m][i]
            cells.append(f"{'n=' + str(n):>9} {p:>9.3f} {o:>9.3f}" if n else f"{'n=0':>9} {'':>9} {'':>9}")
        print(f"   [{BINS[i]:.1f}, {BINS[i + 1]:.1f}) " + " ".join(cells))
        for m in MODELS:
            lo_, hi_, n, p, o = rel[m][i]
            out.append(dict(tournament="ALL", model=m, metric=f"reliability_predicted[{lo_:.1f},{hi_:.1f})",
                            value=p, n=n))
            out.append(dict(tournament="ALL", model=m, metric=f"reliability_observed[{lo_:.1f},{hi_:.1f})",
                            value=o, n=n))
    print("   (columns per model: count, mean predicted, observed frequency)")

    cols = ["tournament", "model", "metric", "value", "ci_low", "ci_high", "bootstrap", "n"]
    pd.DataFrame(out)[cols].round(6).to_csv(OUT, index=False)
    teams.round(6).to_csv(OUT_TEAMS, index=False)
    plot(rel)
    print(f"\nsaved -> {OUT}, {OUT_TEAMS}, {FIGURE}")


if __name__ == "__main__":
    main()
