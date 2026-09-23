"""Smoke test: every pipeline output CSV exists and has the columns its consumers expect.

These four files used to be one file (data/predictions.csv) written by three
scripts with three different schemas, so the point of this test is to catch a
regression where a script writes the wrong shape to the wrong path.

Run from the repo root:  pytest
Regenerate the inputs with:  python run_pipeline.py
(data/predictions.csv is the committed frozen forecast and no pipeline step writes it.)
"""
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent

# path -> (expected columns, expected row count or None)
EXPECTED = {
    "data/power_rankings.csv": (
        ["team", "expected_points_blended", "expected_points_elo_only", "elo",
         "fifa_equivalent_elo", "fifa_blend_weight", "recent_matches"], 48),
    "data/fixture_predictions.csv": (
        ["date", "home_team", "away_team", "neutral",
         "p_home", "p_draw", "p_away", "pick"], 70),
    "data/predictions.csv": (
        ["date", "home_team", "away_team", "neutral",
         "home_news_score", "home_sentiment", "home_key_out", "home_key_back",
         "home_elo_delta", "home_notes",
         "away_news_score", "away_sentiment", "away_key_out", "away_key_back",
         "away_elo_delta", "away_notes",
         "p_home_pre", "p_draw_pre", "p_away_pre",
         "p_home_post", "p_draw_post", "p_away_post",
         "pick_pre", "pick_post", "max_swing"], 70),
    "data/tournament_sim.csv": (
        ["team", "elo", "group", "win_group", "qualify",
         "R16", "QF", "SF", "final", "champion"], 48),
}


@pytest.mark.parametrize("rel", sorted(EXPECTED))
def test_output_schema(rel):
    path = ROOT / rel
    if not path.exists():
        pytest.skip(f"{rel} not generated yet — run `python run_pipeline.py`")
    cols, n_rows = EXPECTED[rel]
    df = pd.read_csv(path)
    assert list(df.columns) == cols, f"{rel} columns changed"
    if n_rows is not None:
        assert len(df) == n_rows, f"{rel} should have {n_rows} rows, got {len(df)}"


def test_the_outputs_are_distinct_files():
    """Guards the original bug: three scripts writing one path."""
    writers = {
        "src/07_blend_predict.py": "data/power_rankings.csv",
        "src/08_fixture_predictions.py": "data/fixture_predictions.csv",
        "src/10_tournament_sim.py": "data/tournament_sim.csv",
    }
    assert len(set(writers.values())) == len(writers)
    for script, out in writers.items():
        src = (ROOT / script).read_text()
        assert f'"{out}"' in src, f"{script} no longer writes {out}"


def test_probabilities_sum_to_one():
    path = ROOT / "data/fixture_predictions.csv"
    if not path.exists():
        pytest.skip("fixture_predictions.csv not generated yet")
    df = pd.read_csv(path)
    total = df[["p_home", "p_draw", "p_away"]].sum(axis=1)
    assert ((total - 1.0).abs() < 1e-9).all()
