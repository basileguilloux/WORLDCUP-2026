"""08_fixture_predictions.predict() must still be the model that made the forecast.

12_blend_backtest.py scores predict(blend=False) as "the same model without the
FIFA blend". That comparison only means something if predict(blend=True) is,
exactly, the model behind the frozen p_*_pre columns.
"""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
KEY = ["date", "home_team", "away_team"]


@pytest.fixture(scope="module")
def fx():
    sys.path.insert(0, str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location("fixture_predictions",
                                                  ROOT / "src/08_fixture_predictions.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def runs(fx):
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        out = {b: fx.predict(blend=b)[0].assign(date=lambda d: d.date.astype(str)) for b in (True, False)}
    finally:
        os.chdir(cwd)
    return out


def test_blended_predict_reproduces_frozen_forecast(runs):
    frozen = pd.read_csv(ROOT / "data/predictions.csv")
    m = runs[True].merge(frozen, on=KEY)
    assert len(m) == 70
    for col in ["p_home", "p_draw", "p_away"]:
        assert (m[col] - m[f"{col}_pre"]).abs().max() < 1e-12


def test_blend_off_changes_the_forecast(runs):
    m = runs[True].merge(runs[False], on=KEY, suffixes=("_on", "_off"))
    assert (m.p_home_on - m.p_home_off).abs().max() > 1e-3


def test_blend_weight_is_the_shipped_rule():
    sys.path.insert(0, str(ROOT / "src"))
    from fifa_blend import blend_weight
    for n in range(0, 80):
        assert blend_weight(n) == float(np.clip(10 / (10 + n), 0.2, 0.6))
