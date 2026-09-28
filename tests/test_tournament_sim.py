"""10_tournament_sim.py is seeded, so a refactor must leave its output unchanged.

Runs the simulator in a scratch directory (it writes data/tournament_sim.csv
relative to the working directory) and compares with the committed file, team
by team: row order is not stable when teams tie on the sort key.
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def test_tournament_sim_reproduces_committed_output(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data/raw").symlink_to(ROOT / "data/raw")
    (tmp_path / "data/features.csv").symlink_to(ROOT / "data/features.csv")
    subprocess.run([sys.executable, str(ROOT / "src/10_tournament_sim.py")], cwd=tmp_path,
                   check=True, capture_output=True)

    new = pd.read_csv(tmp_path / "data/tournament_sim.csv").set_index("team").sort_index()
    old = pd.read_csv(ROOT / "data/tournament_sim.csv").set_index("team").sort_index()
    assert list(new.index) == list(old.index)
    assert list(new.columns) == list(old.columns)
    num = old.select_dtypes("number").columns
    assert (new[num] - old[num]).abs().max().max() < 1e-12
