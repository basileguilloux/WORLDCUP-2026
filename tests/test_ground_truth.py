"""Ground truth for the title-odds backtest, and the brackets it is scored on.

actual_stages() derives each team's stage from results.csv and shootouts.csv.
The bracket test is the strongest check on src/tournament_formats.py: fed the
real group standings (under the official tiebreaks, which the simulator itself
simplifies), each configured bracket must reproduce every real knockout match,
round by round, at the configured venue.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import tournament_formats as tf  # noqa: E402

IDS = [f.name for f in tf.BACKTEST]

CHAMPIONS = {"World Cup 2006": ("Italy", "France"), "World Cup 2010": ("Spain", "Netherlands"),
             "World Cup 2014": ("Germany", "Argentina"), "World Cup 2018": ("France", "Croatia"),
             "World Cup 2022": ("Argentina", "France"), "Euro 2008": ("Spain", "Germany"),
             "Euro 2012": ("Spain", "Italy"), "Euro 2016": ("Portugal", "France"),
             "Euro 2020": ("Italy", "England"), "Euro 2024": ("Spain", "England")}

# Ties the official criteria settle beyond goals: fair-play / disciplinary points.
# World Cup 2018 group H: Japan over Senegal. Euro 2024 group C: Denmark over Slovenia.
FAIR_PLAY = {("World Cup 2018", "H"): ["Japan", "Senegal"],
             ("Euro 2024", "C"): ["Denmark", "Slovenia"]}


@pytest.fixture(scope="module")
def results():
    df = pd.read_csv(ROOT / "data/raw/results.csv", parse_dates=["date"])
    names = pd.read_csv(ROOT / "data/raw/former_names.csv")
    ren = dict(zip(names.former, names.current))
    df["home_team"], df["away_team"] = df.home_team.replace(ren), df.away_team.replace(ren)
    return df.sort_values("date")


@pytest.fixture(scope="module")
def shootouts():
    return pd.read_csv(ROOT / "data/raw/shootouts.csv", parse_dates=["date"])


def _table(games, teams):
    t = {x: [0, 0, 0] for x in teams}                       # pts, gd, gf
    for r in games.itertuples():
        if r.home_team in t and r.away_team in t:
            for a, b, x, y in [(r.home_team, r.away_team, r.home_score, r.away_score),
                               (r.away_team, r.home_team, r.away_score, r.home_score)]:
                t[a][0] += 3 * (x > y) + (x == y); t[a][1] += x - y; t[a][2] += x
    return t


def _standings(fmt, letter, games):
    """Official order: World Cups rank on points, GD, goals, then head-to-head;
    Euros on points, then head-to-head, then GD and goals."""
    teams = list(fmt.groups[letter])
    overall = _table(games, teams)
    euro = fmt.tournament == "UEFA Euro"
    primary = (lambda x: overall[x][0]) if euro else (lambda x: tuple(overall[x]))
    order = []
    for key in sorted({primary(x) for x in teams}, reverse=True):
        tied = [x for x in teams if primary(x) == key]
        h2h = _table(games, tied)
        tied.sort(key=lambda x: tuple(h2h[x]) + (tuple(overall[x][1:]) if euro else ()), reverse=True)
        order += tied
    fp = FAIR_PLAY.get((fmt.name, letter))
    if fp:
        i = min(order.index(x) for x in fp)
        order[i:i + len(fp)] = fp
    return order, overall


def _slots(fmt, results):
    games = tf.group_matches(fmt, tf.matches(fmt, results))
    slots, thirds = {}, []
    for g in sorted(fmt.groups):
        order, overall = _standings(fmt, g, games)
        slots["1" + g], slots["2" + g] = order[0], order[1]
        thirds.append((tuple(overall[order[2]]), g, order[2]))
    if fmt.best_thirds:
        best = sorted(thirds, reverse=True)[:fmt.best_thirds]
        assert best[-1][0] > sorted(thirds, reverse=True)[fmt.best_thirds][0], "tie for the last third place"
        combo = "".join(sorted(g for _, g, _ in best))
        third_team = {g: t for _, g, t in best}
        for w, g in zip(fmt.thirds_winners, fmt.thirds_table[combo]):
            slots["3/" + w] = third_team[g]
    return slots


@pytest.mark.parametrize("fmt", tf.BACKTEST, ids=IDS)
def test_champion_and_runner_up(fmt, results, shootouts):
    stage = tf.actual_stages(fmt, results, shootouts)
    champion, runner_up = CHAMPIONS[fmt.name]
    assert [t for t, k in stage.items() if fmt.stages[k] == "champion"] == [champion]
    assert [t for t, k in stage.items() if fmt.stages[k] == "runner-up"] == [runner_up]
    counts = pd.Series([fmt.stages[k] for k in stage.values()]).value_counts()
    n_groups = len(fmt.groups)
    assert counts["group"] == 4 * n_groups - 2 ** len(fmt.ko_rounds)
    assert counts["SF"] == 2


@pytest.mark.parametrize("fmt", tf.BACKTEST, ids=IDS)
def test_bracket_reproduces_every_real_knockout_match(fmt, results, shootouts):
    slots = _slots(fmt, results)
    rounds = tf.knockout_rounds(fmt, tf.matches(fmt, results))
    teams = [slots[s] for pair in fmt.bracket for s in pair]         # bracket order
    for rnd in fmt.ko_rounds:
        real = {frozenset((r.home_team, r.away_team)): r for r in rounds[rnd]}
        winners = []
        for i in range(0, len(teams), 2):
            pair = frozenset(teams[i:i + 2])
            assert pair in real, f"{rnd}: bracket gives {sorted(pair)}, not a real match"
            r = real[pair]
            assert r.country == fmt.venue(rnd, i // 2), f"{rnd} {sorted(pair)} played in {r.country}"
            winners.append(tf.match_winner(r, shootouts))
        teams = winners


def test_every_drawn_knockout_has_a_shootout(results, shootouts):
    n = 0
    for fmt in tf.BACKTEST:
        for r in tf.knockout_matches(fmt, tf.matches(fmt, results)).itertuples():
            if r.home_score == r.away_score:
                tf.match_winner(r, shootouts)
                n += 1
    assert n == 33
