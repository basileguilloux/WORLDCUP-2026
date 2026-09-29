"""Scoring rules of 13_backtest_tournaments.py, on cases with known answers."""
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="module")
def bt():
    return importlib.import_module("13_backtest_tournaments")


def test_rps_is_zero_for_a_certain_correct_forecast(bt):
    reach = np.array([[1, 1, 1, 0, 0, 0]], float)             # certain QF exit (index 2)
    assert bt.rps(reach, np.array([2]))[0] == 0.0


def test_rps_is_one_for_the_opposite_extreme(bt):
    reach = np.array([[1, 1, 1, 1, 1, 1]], float)             # certain champion
    assert bt.rps(reach, np.array([0]))[0] == pytest.approx(1.0)


def test_rps_penalises_distance_not_just_misses(bt):
    """Predicting SF for a QF exit must beat predicting the title for it."""
    near = np.array([[1, 1, 1, 1, 0, 0]], float)
    far = np.array([[1, 1, 1, 1, 1, 1]], float)
    assert bt.rps(near, np.array([2]))[0] < bt.rps(far, np.array([2]))[0]


def test_rps_hand_computed(bt):
    # K = 4 stages, P(stage >= k) = 1, .5, .2, .1 -> CDF .5, .8, .9; actual stage 1
    reach = np.array([[1, .5, .2, .1]])
    expected = ((.5 - 0) ** 2 + (.8 - 1) ** 2 + (.9 - 1) ** 2) / 3
    assert bt.rps(reach, np.array([1]))[0] == pytest.approx(expected)


def test_paired_bootstrap_of_a_constant_difference(bt):
    d, lo, hi = bt.paired_boot(np.full(10, 0.3), np.full(10, 0.1))
    assert (d, lo, hi) == pytest.approx((0.2, 0.2, 0.2))
    assert bt.verdict(lo, hi) == "excludes 0"


def test_reliability_bins_cover_zero_and_one(bt):
    rows = bt.reliability(np.array([0.0, 0.05, 1.0, 0.95]), np.array([0, 0, 1, 1.]))
    assert sum(r[2] for r in rows) == 4
    assert rows[0][2] == 2 and rows[-1][2] == 2


def test_champion_floor_is_finite(bt):
    assert -np.log(bt.FLOOR) == pytest.approx(np.log(2 * bt.N_SIMS))


# ---------------------------------------------------------------- committed outputs
import pandas as pd  # noqa: E402

BT, TEAMS = ROOT / "data/tournament_backtest.csv", ROOT / "data/tournament_backtest_teams.csv"


@pytest.fixture(scope="module")
def outputs():
    if not BT.exists():
        pytest.skip("run `python src/13_backtest_tournaments.py` first")
    return pd.read_csv(BT), pd.read_csv(TEAMS)


def test_every_tournament_and_model_is_scored(outputs):
    bt, teams = outputs
    assert teams.groupby("model").tournament.nunique().to_dict() == dict.fromkeys(
        ["elo_only", "full", "no_blend", "uniform"], 10)
    assert len(teams) == 4 * 264                                   # 5x32 + 2x16 + 3x24 teams
    champ = teams.groupby(["tournament", "model"]).champion.sum()
    assert np.allclose(champ, 1, atol=1e-6)
    assert (teams.groupby(["tournament", "model"]).actual_stage.apply(
        lambda s: (s == "champion").sum()) == 1).all()


def test_summary_rows_match_team_rows(outputs):
    bt, teams = outputs
    per = bt[(bt.tournament != "ALL") & (bt.metric == "rps")].set_index(["tournament", "model"]).value
    mean = teams.groupby(["tournament", "model"]).rps.mean()
    assert np.allclose(per.sort_index(), mean.sort_index(), atol=1e-6)
    for model in ["full", "no_blend", "elo_only", "uniform"]:
        for metric in ["rps", "champion_log_score", "sf_brier"]:
            vals = bt[(bt.tournament != "ALL") & (bt.model == model) & (bt.metric == metric)].value
            overall = bt[(bt.tournament == "ALL") & (bt.model == model) & (bt.metric == metric)].value
            assert len(vals) == 10 and abs(vals.mean() - overall.iloc[0]) < 1e-5
