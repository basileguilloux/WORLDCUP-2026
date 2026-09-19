"""The frozen forecast must stay bit-identical to the model run it claims to be.

data/predictions.csv carries two kinds of column:

  * p_*_pre   -- THE FROZEN FORECAST. Must equal, exactly, the model output
                 committed in 32ffb4c (13 June 2026 12:09 +02:00). If this test
                 fails, either the file was regenerated or the model changed,
                 and the README's provenance claim is no longer true.
  * p_*_post  -- a team-news annotation added on 18 September 2026. Not scored,
                 not frozen, and deliberately not checked here.
"""
import io
import subprocess
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
FROZEN_COMMIT = "32ffb4c"          # "final ensemble: multinomial logistic regression and poisson"
KEY = ["date", "home_team", "away_team"]


def _git_show(ref_path):
    proc = subprocess.run(["git", "show", ref_path], cwd=ROOT,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.skip(f"cannot read {ref_path} from git: {proc.stderr.strip()}")
    return pd.read_csv(io.StringIO(proc.stdout))


@pytest.fixture(scope="module")
def frozen():
    return _git_show(f"{FROZEN_COMMIT}:data/predictions.csv")


@pytest.fixture(scope="module")
def current():
    return pd.read_csv(ROOT / "data/predictions.csv")


def test_same_fixture_list(frozen, current):
    assert set(map(tuple, frozen[KEY].values)) == set(map(tuple, current[KEY].values))
    assert len(frozen) == len(current) == 70


def test_pre_news_columns_match_frozen_commit_exactly(frozen, current):
    """The headline forecast must be bit-identical to 32ffb4c's model output."""
    m = frozen.merge(current, on=KEY, how="inner")
    assert len(m) == 70, "fixtures failed to line up"
    for old, new in [("p_home", "p_home_pre"), ("p_draw", "p_draw_pre"), ("p_away", "p_away_pre")]:
        diff = (m[old] - m[new]).abs().max()
        assert diff < 1e-12, (
            f"{new} drifted from {FROZEN_COMMIT}:{old} by {diff:.3e}. The frozen "
            f"forecast is no longer the run the README cites.")


def test_pre_news_probabilities_sum_to_one(current):
    total = current[["p_home_pre", "p_draw_pre", "p_away_pre"]].sum(axis=1)
    assert ((total - 1.0).abs() < 1e-9).all()


def test_post_news_columns_are_a_separate_annotation(frozen, current):
    """Sanity: post-news columns exist and differ from the frozen ones, or the
    annotation would be pointless. 8 fixtures were annotated."""
    moved = (current[["p_home_pre", "p_draw_pre", "p_away_pre"]].values
             != current[["p_home_post", "p_draw_post", "p_away_post"]].values).any(axis=1)
    assert moved.sum() == 8, f"expected 8 annotated fixtures, found {moved.sum()}"
