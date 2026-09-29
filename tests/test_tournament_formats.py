"""The backtest tournament configs must agree with the data they describe.

Group letters and brackets in src/tournament_formats.py are transcribed by hand
from the official draws and regulations. These tests check what can be checked
from the repository: the groups against the real group-stage fixtures, the
third-place tables against the structure of the round of 16, and that every
team resolves to one name across results, Elo history and the FIFA ranking.
"""
import os
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import tournament_formats as tf  # noqa: E402
from fifa_blend import load_history  # noqa: E402

IDS = [f.name for f in tf.BACKTEST]


@pytest.fixture(scope="module")
def results():
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        return tf.load_results()
    finally:
        os.chdir(cwd)


def _components(g):
    adj = {}
    for h, a in zip(g.home_team, g.away_team):
        adj.setdefault(h, set()).add(a)
        adj.setdefault(a, set()).add(h)
    seen, out = set(), []
    for t in adj:
        if t in seen:
            continue
        stack, comp = [t], set()
        while stack:
            x = stack.pop()
            if x not in comp:
                comp.add(x)
                stack += list(adj[x])
        seen |= comp
        out.append(frozenset(comp))
    return set(out)


def test_ten_backtest_tournaments():
    assert len(tf.BACKTEST) == 10
    assert [f.cutoff_date.strftime("%Y-%m-%d") for f in tf.BACKTEST] == [
        "2006-06-08", "2010-06-10", "2014-06-11", "2018-06-13", "2022-11-19",
        "2008-06-06", "2012-06-07", "2016-06-09", "2021-06-10", "2024-06-13"]


@pytest.mark.parametrize("fmt", tf.BACKTEST, ids=IDS)
def test_groups_match_the_fixtures(fmt, results):
    m = tf.matches(fmt, results)
    n_groups = len(fmt.groups)
    n_ko = 2 ** len(fmt.ko_rounds) - 1 + (fmt.tournament == "FIFA World Cup")   # + 3rd-place playoff
    assert len(m) == 6 * n_groups + n_ko
    assert m.date.min() == pd.Timestamp(fmt.start) and m.date.max() == pd.Timestamp(fmt.end)
    g = tf.group_matches(fmt, m)
    assert _components(g) == {frozenset(v) for v in fmt.groups.values()}
    assert all(len(v) == 4 for v in fmt.groups.values())
    assert tf.knockout_matches(fmt, m).date.min() > g.date.max()


@pytest.mark.parametrize("fmt", tf.BACKTEST, ids=IDS)
def test_first_round_uses_every_qualifier_once(fmt):
    slots = [s for pair in fmt.bracket for s in pair]
    assert len(slots) == len(set(slots)) == 2 ** len(fmt.ko_rounds)
    letters = sorted(fmt.groups)
    expected = {f"{p}{g}" for g in letters for p in "12"} | {f"3/{w}" for w in fmt.thirds_winners}
    assert set(slots) == expected


@pytest.mark.parametrize("table,winners,bracket", [
    (tf.THIRDS_2016, tf.THIRDS_2016_WINNERS, tf.EURO_R16_2016),
    (tf.THIRDS_2020, tf.THIRDS_2020_WINNERS, tf.EURO_R16_2020),
], ids=["2016", "2020-2024"])
def test_third_place_table_is_complete_and_consistent(table, winners, bracket):
    """15 rows, each qualifying third used once, never against its own group winner,
    and each winner only ever meets the thirds its round-of-16 slot allows."""
    assert set(table) == {"".join(c) for c in combinations("ABCDEF", 4)}
    allowed = {w: set() for w in winners}
    for combo, thirds in table.items():
        assert sorted(thirds) == sorted(combo), combo
        for w, t in zip(winners, thirds):
            assert w != t, combo
            allowed[w].add(t)
    assert all(("1" + w, "3/" + w) in bracket for w in winners)
    # the round-of-16 slot descriptions in the regulations (e.g. "1B v 3A/D/E/F")
    if winners == tf.THIRDS_2016_WINNERS:
        assert allowed == {"A": set("CDE"), "B": set("ACD"), "C": set("ABF"), "D": set("BEF")}
    else:
        assert allowed == {"B": set("ADEF"), "C": set("DEF"), "E": set("ABCD"), "F": set("ABC")}


@pytest.mark.parametrize("fmt", tf.BACKTEST, ids=IDS)
def test_every_team_resolves_in_history_and_fifa_ranking(fmt, results):
    """former_names.csv and fifa_blend.FIFA_TO_RESULTS must put every participant
    under one name: in the fixtures, in the Elo history, and in the FIFA release."""
    teams = {t for v in fmt.groups.values() for t in v}
    m = tf.matches(fmt, results)
    assert set(m.home_team) | set(m.away_team) == teams
    before = results[(results.date < fmt.cutoff_date) & results.home_score.notna()]
    assert teams <= set(before.home_team) | set(before.away_team)
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        hist = load_history()
    finally:
        os.chdir(cwd)
    release = hist[hist.date < fmt.cutoff_date].date.max()
    assert fmt.cutoff_date - release <= pd.Timedelta(days=365)
    assert teams <= set(hist[hist.date == release].team)


def test_fifa_history_holds_nothing_from_2026():
    """The ranking history feeds the backtests only; it must end long before 2026."""
    h = pd.read_csv(ROOT / "data/raw/fifa_ranking_history.csv", parse_dates=["date"])
    assert h.date.max() < pd.Timestamp("2026-06-11")
