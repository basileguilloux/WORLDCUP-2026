"""The tournament scorer must exclude fixtures that had already kicked off.

Canada v Bosnia and Herzegovina and United States v Paraguay were both played on
12 June 2026, before the frozen forecast was committed on 13 June at 10:09 UTC.
Their rows in data/predictions.csv are not predictions, and scoring them would
flatter (or penalise) the model with hindsight.
"""
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
PRE_COMMIT = [("2026-06-12", "Canada", "Bosnia and Herzegovina"),
              ("2026-06-12", "United States", "Paraguay")]


@pytest.fixture(scope="module")
def sc():
    sys.path.insert(0, str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location("score_tournament",
                                                  ROOT / "src/11_score_tournament.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_eligible_drops_pre_commit_fixtures(sc):
    pred = pd.read_csv(ROOT / "data/predictions.csv")
    kept = sc.eligible(pred)
    assert len(pred) == 70
    assert len(kept) == 68, f"expected 68 scorable fixtures, got {len(kept)}"
    remaining = set(map(tuple, kept[["date", "home_team", "away_team"]].values))
    for fx in PRE_COMMIT:
        assert fx not in remaining, f"{fx} kicked off before the commit and must be excluded"


def test_excluded_fixtures_are_exactly_the_two_known_ones(sc):
    pred = pd.read_csv(ROOT / "data/predictions.csv")
    dropped = pred[~pred.index.isin(sc.eligible(pred).index)]
    assert set(map(tuple, dropped[["date", "home_team", "away_team"]].values)) == set(PRE_COMMIT)


def test_cutoff_is_the_frozen_commit_date(sc):
    assert sc.FROZEN_COMMIT_DATE == pd.Timestamp("2026-06-13")


def test_same_day_fixtures_are_kept(sc):
    """13 June fixtures kicked off after the 10:09 UTC commit and must be scored."""
    pred = pd.read_csv(ROOT / "data/predictions.csv")
    kept = sc.eligible(pred)
    assert (kept.date == "2026-06-13").sum() == 4


def test_every_forecast_fixture_has_an_actual_result():
    pred = pd.read_csv(ROOT / "data/predictions.csv")
    act = pd.read_csv(ROOT / "data/raw/wc26_actual_results.csv")
    key = ["date", "home_team", "away_team"]
    merged = pred.merge(act[key + ["home_score", "away_score"]], on=key, how="inner")
    assert len(merged) == len(pred) == 70
    assert merged[["home_score", "away_score"]].notna().all().all()


def test_actuals_file_does_not_replace_the_frozen_input():
    """wc26_actual_results.csv is additive; results.csv must stay scoreless for these."""
    frozen = pd.read_csv(ROOT / "data/raw/results.csv")
    wc = frozen[(frozen.tournament == "FIFA World Cup") & (frozen.date >= "2026-06-01")]
    assert len(wc) == 72
    assert wc.home_score.isna().sum() == 70, "results.csv must keep the 70 fixtures scoreless"


# ---------------------------------------------------------------- paired bootstrap

def _probs_for(true_cls, p_true, n):
    """n rows where the true class gets exactly p_true; remainder split evenly."""
    import numpy as np
    P = np.full((n, 3), 0.0)
    rest = (1.0 - p_true) / 2
    for i in range(n):
        P[i, :] = rest
        P[i, true_cls] = p_true
    return P


def test_paired_bootstrap_recovers_a_known_constant_difference(sc):
    """If A beats B by exactly 1 nat on every fixture, the interval must be [-1, -1].

    Construct it so the arithmetic is exact: model A gives the true class e^-1
    (per-fixture log-loss 1.0), model B gives it e^-2 (log-loss 2.0). Every
    resample, whatever indices it draws, must average to a difference of -1.
    """
    import numpy as np
    n = 40
    y = np.full(n, 2)
    P_a = _probs_for(2, np.exp(-1.0), n)
    P_b = _probs_for(2, np.exp(-2.0), n)

    mean, lo, hi = sc.paired_bootstrap(y, P_a, P_b, n_boot=2000, seed=1)
    assert mean == pytest.approx(-1.0, abs=1e-9)
    assert lo == pytest.approx(-1.0, abs=1e-9)
    assert hi == pytest.approx(-1.0, abs=1e-9)
    assert (lo > 0) == (hi > 0), "a constant non-zero difference must exclude zero"
    assert sc.verdict(lo, hi).startswith("excludes 0")


def test_paired_bootstrap_identical_models_give_zero(sc):
    """Comparing a model with itself must give exactly zero difference."""
    import numpy as np
    n = 30
    y = np.tile([0, 1, 2], n // 3)
    P = _probs_for(1, 0.5, n)
    mean, lo, hi = sc.paired_bootstrap(y, P, P, n_boot=2000, seed=7)
    assert (mean, lo, hi) == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)
    assert sc.verdict(lo, hi).startswith("spans 0")


def test_paired_bootstrap_detects_a_noisy_but_real_difference(sc):
    """A consistently better model over many fixtures must exclude zero."""
    import numpy as np
    rng = np.random.default_rng(0)
    n = 300
    y = rng.integers(0, 3, n)
    P_good = np.full((n, 3), 0.2); P_good[np.arange(n), y] = 0.6
    P_bad = np.full((n, 3), 1 / 3)
    mean, lo, hi = sc.paired_bootstrap(y, P_good, P_bad, n_boot=10000, seed=3)
    assert mean < 0 and hi < 0, "the better model's interval must lie below zero"


def test_paired_bootstrap_is_deterministic_for_a_fixed_seed(sc):
    import numpy as np
    n = 50
    y = np.full(n, 0)
    P_a = _probs_for(0, 0.5, n); P_b = _probs_for(0, 0.4, n)
    a = sc.paired_bootstrap(y, P_a, P_b, n_boot=1000, seed=42)
    b = sc.paired_bootstrap(y, P_a, P_b, n_boot=1000, seed=42)
    assert a == b


def test_simulator_reads_no_post_tournament_data():
    """The clean-data claim in the README must stay true.

    10_tournament_sim.py may only read the three frozen inputs, and none of them
    may contain a match result dated after 11 June 2026.
    """
    src = (ROOT / "src/10_tournament_sim.py").read_text()
    import re
    reads = set(re.findall(r'read_csv\("([^"]+)"', src))
    assert reads == {"data/raw/results.csv", "data/raw/former_names.csv",
                     "data/raw/fifa_ranking_2026-06-11.csv"}, reads
    assert "wc26_actual_results" not in src, "simulator must not read actual results"

    results = pd.read_csv(ROOT / "data/raw/results.csv", parse_dates=["date"])
    played = results[results.home_score.notna()]
    assert (played.date > "2026-06-11").sum() == 0, (
        "results.csv gained a scored match after 11 June; the simulator's inputs "
        "are no longer pre-tournament")
