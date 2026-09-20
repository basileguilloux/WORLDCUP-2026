"""Phase 7: score the FROZEN forecast against what actually happened.

Scores the `p_*_pre` columns of data/predictions.csv -- the model's own output,
with no team-news overlay -- against data/raw/wc26_actual_results.csv.

WHAT IS SCORED, AND WHY
-----------------------
Only fixtures whose kickoff FOLLOWED the frozen commit (32ffb4c, author date
2026-06-13 12:09:22 +02:00 = 10:09:22 UTC) are genuine predictions. The fixture
data carries dates but no kickoff times, so eligibility is decided per date:

  * date <  2026-06-13  -> EXCLUDED. The whole day ended before the commit.
                           This removes Canada v Bosnia and United States v
                           Paraguay, both 12 June, which had already been played
                           when the forecast was committed.
  * date == 2026-06-13  -> included. The commit lands at 10:09 UTC, which is
                           06:09 in East Rutherford/Foxborough and 03:09 in
                           Santa Clara/Vancouver -- hours before any plausible
                           kickoff. This rests on venue local time, not on data.
  * date >  2026-06-13  -> included.

The two 11 June matches already carry scores in the frozen results.csv and were
never forecast, so they are not in predictions.csv at all.

Label: 0 = away win, 1 = draw, 2 = home win, at 90 minutes. Every scored fixture
is a group-stage match, so no extra time is involved and the full-time score is
the 90-minute score.

Baselines: a uniform 1/3 prior (log-loss = ln 3) and the repo's own Elo-only
logistic regression from 03_elo_baseline.py, refit here on the same data and
applied to these fixtures.

Run:  python src/11_score_tournament.py
"""
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

PREDICTIONS = "data/predictions.csv"
ACTUALS = "data/raw/wc26_actual_results.csv"
FROZEN_COMMIT_DATE = pd.Timestamp("2026-06-13")   # author date of 32ffb4c (UTC 10:09)
N_BOOT, BOOT_SEED = 10000, 20260613
START, HOME_ADV = 1500.0, 60.0
KEY = ["date", "home_team", "away_team"]


def eligible(df):
    """Fixtures that were still unplayed when the forecast was committed.

    Exposed as a function so the exclusion rule can be tested directly.
    """
    return df[pd.to_datetime(df.date) >= FROZEN_COMMIT_DATE]


def outcomes(d):
    return ((d.home_score > d.away_score).astype(int) * 2
            + (d.home_score == d.away_score).astype(int)).values


def brier_multi(y, p):
    return float(np.mean(np.sum((p - np.eye(3)[y]) ** 2, axis=1)))


def metrics(y, P):
    return {"n": len(y),
            "log_loss": log_loss(y, P, labels=[0, 1, 2]),
            "brier": brier_multi(y, P),
            "accuracy": accuracy_score(y, P.argmax(1))}


def per_match_ll(y, P):
    """-log p assigned to the outcome that actually happened, per fixture."""
    return -np.log(np.clip(P[np.arange(len(y)), y], 1e-15, None))


def bootstrap_ll(y, P, n_boot=N_BOOT, seed=BOOT_SEED):
    """Percentile bootstrap 95% interval for mean log-loss (n is small)."""
    rng = np.random.default_rng(seed)
    per_match = per_match_ll(y, P)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    means = per_match[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_bootstrap(y, P_a, P_b, n_boot=N_BOOT, seed=BOOT_SEED):
    """95% interval for the mean per-fixture log-loss difference (A - B).

    PAIRED: each resample draws one set of fixture indices and applies it to
    BOTH models, so the shared per-fixture difficulty cancels. This is the right
    comparison between two models scored on the same matches -- overlapping
    marginal intervals say little, because the two models rise and fall together
    on the same fixtures.

    Negative => A has the lower (better) log-loss. The interval excluding zero
    is the test of whether the difference is distinguishable from noise.
    """
    rng = np.random.default_rng(seed)
    diff = per_match_ll(y, P_a) - per_match_ll(y, P_b)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


def verdict(lo, hi):
    """An interval excludes zero only if it lies wholly above or wholly below it.

    Note [0, 0] contains zero: a zero-width interval at the origin means the two
    models are identical, which is the strongest possible "not distinguishable".
    """
    excludes = lo > 0 or hi < 0
    return "excludes 0 -> distinguishable" if excludes else "spans 0 -> not distinguishable"


def elo_only_baseline(fixtures):
    """The repo's Elo-only logistic (03_elo_baseline.py), refit and applied here.

    Trained on data/processed_matches.csv (all matches played before the
    tournament) with the single feature elo_diff, then applied to these fixtures
    using Elo replayed from the same frozen results.csv -- no FIFA blend, so this
    is Elo alone, exactly as the baseline script frames it.
    """
    hist = pd.read_csv("data/processed_matches.csv", parse_dates=["date"])
    X = (hist.home_elo - hist.away_elo).to_frame("elo_diff")
    y = outcomes(hist)
    model = LogisticRegression().fit(X, y)

    df = pd.read_csv("data/raw/results.csv", parse_dates=["date"])
    ren = dict(zip(*[pd.read_csv("data/raw/former_names.csv")[c] for c in ["former", "current"]]))
    df["home_team"], df["away_team"] = df.home_team.replace(ren), df.away_team.replace(ren)
    played = df[df.home_score.notna()].sort_values("date")
    elo = defaultdict(lambda: START)
    exp = lambda ra, rb: 1 / (1 + 10 ** ((rb - ra) / 400))
    for r in played.itertuples():
        k = 60 if "FIFA World Cup" in r.tournament else (40 if r.tournament != "Friendly" else 20)
        gd = abs(r.home_score - r.away_score); k *= 1 if gd <= 1 else (1.5 if gd == 2 else 1 + gd / 5)
        res = 1.0 if r.home_score > r.away_score else (0.5 if r.home_score == r.away_score else 0.0)
        ch = k * (res - exp(elo[r.home_team] + (0 if r.neutral else HOME_ADV), elo[r.away_team]))
        elo[r.home_team] += ch; elo[r.away_team] -= ch

    diffs = pd.DataFrame({"elo_diff": [elo[h] - elo[a]
                                       for h, a in zip(fixtures.home_team, fixtures.away_team)]})
    P = model.predict_proba(diffs)
    col = {c: i for i, c in enumerate(model.classes_)}
    return P[:, [col[0], col[1], col[2]]]


def line(label, m, ci=None, acc="{:.3f}"):
    ci_s = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else ""
    acc_s = acc.format(m["accuracy"]) if acc else "n/a"
    print(f"  {label:<34}{m['log_loss']:>9.4f}  {ci_s:<18}{m['brier']:>8.4f}{acc_s:>10}")


def main():
    pred = pd.read_csv(PREDICTIONS)
    act = pd.read_csv(ACTUALS)
    merged = pred.merge(act[KEY + ["home_score", "away_score"]], on=KEY, how="inner")
    if len(merged) != len(pred):
        raise SystemExit(f"only {len(merged)}/{len(pred)} forecast fixtures found in {ACTUALS}")

    scored = eligible(merged).reset_index(drop=True)
    dropped = merged[~merged.index.isin(eligible(merged).index)]

    y = outcomes(scored)
    P_pre = scored[["p_away_pre", "p_draw_pre", "p_home_pre"]].values
    P_post = scored[["p_away_post", "p_draw_post", "p_home_post"]].values
    uniform = np.full_like(P_pre, 1 / 3)
    P_elo = elo_only_baseline(scored)

    print("=" * 84)
    print("FROZEN FORECAST vs ACTUAL RESULTS — WC 2026 group stage")
    print("=" * 84)
    print(f"forecast file : {PREDICTIONS}  (p_*_pre columns — no team-news overlay)")
    print(f"actuals       : {ACTUALS}")
    print(f"frozen commit : 32ffb4c, author date {FROZEN_COMMIT_DATE.date()} 12:09 +02:00 (10:09 UTC)")
    print(f"\nfixtures in forecast            : {len(merged)}")
    print(f"excluded (kicked off pre-commit): {len(dropped)}")
    for r in dropped.itertuples():
        print(f"    {r.date}  {r.home_team} v {r.away_team}")
    print(f"SCORED                          : {len(scored)}")

    outcome_mix = np.bincount(y, minlength=3) / len(y)
    print(f"\nactual outcome mix  home {outcome_mix[2]:.3f}  draw {outcome_mix[1]:.3f}  away {outcome_mix[0]:.3f}")
    print(f"mean predicted      home {P_pre[:,2].mean():.3f}  draw {P_pre[:,1].mean():.3f}  away {P_pre[:,0].mean():.3f}")

    print(f"\n  {'model':<34}{'log-loss':>9}  {'95% CI (boot)':<18}{'Brier':>8}{'accuracy':>10}")
    print("  " + "-" * 80)
    m_pre = metrics(y, P_pre)
    line("FROZEN FORECAST (p_*_pre)", m_pre, bootstrap_ll(y, P_pre))
    line("uniform 1/3 (ln 3)", metrics(y, uniform), acc=None)
    line("Elo-only logistic (03 baseline)", metrics(y, P_elo), bootstrap_ll(y, P_elo))
    print("  " + "-" * 80)
    print("  secondary, NOT the frozen forecast — team-news annotation added 18 Sep 2026,")
    print("  retrieval date unverifiable; shown for completeness only:")
    line("post-news (p_*_post)  [secondary]", metrics(y, P_post), bootstrap_ll(y, P_post))

    print("  (uniform accuracy is n/a: argmax over a three-way tie is arbitrary)")

    # ---- paired bootstrap: the comparison that actually settles things ----
    print(f"\nPAIRED BOOTSTRAP — mean per-fixture log-loss difference, {N_BOOT:,} resamples, seed {BOOT_SEED}")
    print("  (same resampled fixtures for both models, so shared match difficulty cancels)")
    print("  " + "-" * 80)
    d, lo, hi = paired_bootstrap(y, P_pre, P_elo)
    print(f"  forecast - Elo-only : {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  {verdict(lo, hi)}")
    d2, lo2, hi2 = paired_bootstrap(y, P_post, P_pre)
    print(f"  post-news - pre-news: {d2:+.4f}  95% CI [{lo2:+.4f}, {hi2:+.4f}]  {verdict(lo2, hi2)}")

    # ---- sensitivity: also drop the four same-day (13 June) fixtures ----
    strict = scored[pd.to_datetime(scored.date) >= pd.Timestamp("2026-06-14")].reset_index(drop=True)
    ys = outcomes(strict)
    Ps_pre = strict[["p_away_pre", "p_draw_pre", "p_home_pre"]].values
    Ps_elo = elo_only_baseline(strict)
    ms = metrics(ys, Ps_pre)
    ds, los, his = paired_bootstrap(ys, Ps_pre, Ps_elo)
    print(f"\nSENSITIVITY — also excluding the four 13 June fixtures (same day as the commit)")
    print("  " + "-" * 80)
    print(f"  n = {len(strict)}   forecast log-loss {ms['log_loss']:.4f}  "
          f"Brier {ms['brier']:.4f}  accuracy {ms['accuracy']:.3f}")
    print(f"  forecast - Elo-only : {ds:+.4f}  95% CI [{los:+.4f}, {his:+.4f}]  {verdict(los, his)}")

    print(f"\nwalk-forward log-loss on ~49k historical matches was 0.9024 (see README).")
    delta = m_pre["log_loss"] - 0.9024
    print(f"Tournament log-loss is {m_pre['log_loss']:.4f}, {abs(delta):.4f} "
          f"{'worse' if delta > 0 else 'better'} — but the two are not directly comparable:")
    print("the backtest averages over ~49k mostly-lopsided friendlies and qualifiers, while")
    print("these 68 are World Cup group games between closely matched sides. The fixture mix")
    print("differs, so the absolute log-loss levels are not on the same footing.")
    print("=" * 84)


if __name__ == "__main__":
    main()
