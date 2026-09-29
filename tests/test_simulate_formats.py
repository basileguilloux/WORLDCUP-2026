"""simulate() on the backtest formats: host rule, bracket bookkeeping, no look-ahead."""
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import tournament_formats as tf  # noqa: E402


@pytest.fixture(scope="module")
def sim():
    cwd = os.getcwd()
    os.chdir(ROOT)
    yield importlib.import_module("10_tournament_sim")
    os.chdir(cwd)


@pytest.fixture(scope="module")
def results(sim):
    return sim.load_results()


def _flags(sim, fmt, results):
    g = tf.group_matches(fmt, tf.matches(fmt, results))
    home, away, flag = sim.host_fixtures(fmt, g)
    return {(h, a): f for h, a, f in zip(home, away, flag)}


def test_host_flag_only_in_own_country(sim, results):
    f = _flags(sim, tf.BY_NAME["Euro 2020"], results)
    # results.csv marks this as a home game in Wales; it was played in Baku and
    # Wales was not a host
    assert f[("Wales", "Switzerland")] == 0
    assert f[("England", "Croatia")] == 1                       # Wembley
    assert f[("Italy", "Turkey")] == 1                          # Rome
    assert sum(f.values()) == 24          # every non-neutral group row except Wales v Switzerland
    f = _flags(sim, tf.BY_NAME["World Cup 2006"], results)
    assert sum(f.values()) == 3 and all(f[k] for k in f if k[0] == "Germany")


def test_simulate_uses_no_data_from_the_tournament(sim, results):
    for fmt in tf.BACKTEST:
        m = tf.matches(fmt, results)
        assert (m.date > fmt.cutoff_date).all()
        feat = sim.load()
        assert feat[feat.date < fmt.cutoff_date].date.max() < m.date.min()


@pytest.mark.parametrize("name", ["Euro 2012", "Euro 2016"])
def test_stage_probabilities_add_up(sim, name):
    fmt = tf.BY_NAME[name]
    r = sim.simulate(fmt.cutoff_date, fmt, 300, 7)
    cols = ["qualify"] + list(fmt.reach_columns)
    slots = 2 ** len(fmt.ko_rounds)
    assert r[cols].sum().round(9).tolist() == [slots / 2 ** k for k in range(len(cols))]
    assert abs(r.win_group.sum() - len(fmt.groups)) < 1e-9
    for a, b in zip(cols, cols[1:]):
        assert (r[a] >= r[b]).all()


def test_simulate_is_deterministic_for_a_seed(sim):
    fmt = tf.BY_NAME["Euro 2008"]
    a = sim.simulate(fmt.cutoff_date, fmt, 200, 3, "elo_only")
    b = sim.simulate(fmt.cutoff_date, fmt, 200, 3, "elo_only")
    assert a.equals(b)


def test_uniform_model_treats_every_team_alike(sim):
    fmt = tf.BY_NAME["Euro 2008"]
    r = sim.simulate(fmt.cutoff_date, fmt, 4000, 11, "uniform")
    # 16 exchangeable teams: each qualifies half the time, wins 1/16 of titles
    assert np.allclose(r.qualify, 0.5, atol=0.04)
    assert np.allclose(r.champion, 1 / 16, atol=0.02)
