"""The forecast's inputs must keep the bytes they had in the forecast commit.

The README states that every input of the frozen forecast has one commit and
unchanged bytes since 99a2305. 02_elo.py and 04_features.py regenerate their
CSVs byte for byte, but a joblib pickle records the scikit-learn version, so
05_model.py writes its refit to a git-ignored build path and never touches the
committed model.
"""
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FROZEN_COMMIT = "99a2305"
INPUTS = ["data/raw/results.csv", "data/features.csv", "data/processed_matches.csv",
          "data/poisson_model.joblib", "data/raw/fifa_ranking_2026-06-11.csv"]


@pytest.mark.parametrize("path", INPUTS)
def test_input_unchanged_since_forecast_commit(path):
    proc = subprocess.run(["git", "diff", "--quiet", FROZEN_COMMIT, "HEAD", "--", path], cwd=ROOT)
    if proc.returncode not in (0, 1):
        pytest.skip(f"cannot diff against {FROZEN_COMMIT}")
    assert proc.returncode == 0, f"{path} differs from {FROZEN_COMMIT}"


def test_model_step_does_not_overwrite_the_frozen_model():
    src = (ROOT / "src/05_model.py").read_text()
    assert 'REFIT = Path("data/build/poisson_model.joblib")' in src
    assert "joblib.dump(model, REFIT)" in src
    assert 'dump(model, "data/poisson_model.joblib")' not in src
    assert "data/build/" in (ROOT / ".gitignore").read_text().split()
    for reader in ["src/06_simulate.py", "src/07_blend_predict.py"]:
        assert 'joblib.load("data/build/poisson_model.joblib")' in (ROOT / reader).read_text()
