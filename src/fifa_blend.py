"""The Elo + FIFA-ranking blend used at prediction time, and the ranking history
needed to backtest it.

08_fixture_predictions.py and 10_tournament_sim.py do not feed the models raw
Elo. They map FIFA ranking points onto the Elo scale with a least-squares line
fitted over the ranked teams, then mix it in per team:

    strength = (1 - w) * elo + w * fifa_elo,   w = clip(10 / (10 + recent), 0.2, 0.6)

where `recent` is the team's number of matches in the ~4 years before the
forecast. The weights were set by hand, not fitted. 12_blend_backtest.py runs
them through the walk-forward; this module holds the pieces it shares with the
prediction scripts.

Ranking history: every men's FIFA ranking release from 31 Dec 1992 to 19 Sep
2024, compiled by github.com/Dato-Futbol/fifa-ranking (file
ranking_fifa_historical.csv, pinned below by commit and SHA-256). It is not
redistributed here: fetch_history() downloads it to data/raw/ on first use.
"""
import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

BLEND_MIN, BLEND_MAX = 0.2, 0.6
RECENT_YEARS = 4                      # 08 counts matches since 2022-06-01, ~4 years back

HISTORY_PATH = Path("data/raw/fifa_ranking_history.csv")
HISTORY_URL = ("https://raw.githubusercontent.com/Dato-Futbol/fifa-ranking/"
               "6916929cc8fbf3bc49a4d21a9670e5992e6a738b/ranking_fifa_historical.csv")
HISTORY_SHA256 = "d4f4d8d3db8e7560823b31b8db7848e1b85fcd1e1353de1d7e1fbedc6d47226e"

# FIFA spelling -> the spelling used in results.csv (after former_names.csv)
FIFA_TO_RESULTS = {
    "USA": "United States", "Korea Republic": "South Korea", "Korea DPR": "North Korea",
    "IR Iran": "Iran", "China PR": "China", "Chinese Taipei": "Taiwan",
    "Congo DR": "DR Congo", "Zaire": "DR Congo", "Côte d'Ivoire": "Ivory Coast",
    "Cabo Verde": "Cape Verde", "Cape Verde Islands": "Cape Verde",
    "Czechia": "Czech Republic", "Türkiye": "Turkey", "Kyrgyz Republic": "Kyrgyzstan",
    "FYR Macedonia": "North Macedonia", "Swaziland": "Eswatini", "The Gambia": "Gambia",
    "Hong Kong, China": "Hong Kong", "Brunei Darussalam": "Brunei",
    "Curacao": "Curaçao", "Netherlands Antilles": "Curaçao",
    "Aotearoa New Zealand": "New Zealand",
    "St Kitts and Nevis": "Saint Kitts and Nevis", "St. Kitts and Nevis": "Saint Kitts and Nevis",
    "St Lucia": "Saint Lucia", "St. Lucia": "Saint Lucia",
    "St Vincent and the Grenadines": "Saint Vincent and the Grenadines",
    "St. Vincent and the Grenadines": "Saint Vincent and the Grenadines",
    "St. Vincent / Grenadines": "Saint Vincent and the Grenadines",
    "Sao Tome e Principe": "São Tomé and Príncipe", "São Tomé e Príncipe": "São Tomé and Príncipe",
    "US Virgin Islands": "United States Virgin Islands",
    "Yugoslavia": "Serbia", "Serbia and Montenegro": "Serbia",
}


def blend_weight(recent_matches):
    """Weight on the FIFA-equivalent rating: more for teams with few recent matches."""
    return float(np.clip(10 / (10 + recent_matches), BLEND_MIN, BLEND_MAX))


def fifa_to_elo_line(points, elos):
    """Least-squares line mapping FIFA points onto the Elo scale -> (slope, intercept)."""
    a, b = np.polyfit(points, elos, 1)
    return a, b


def fetch_history(path=HISTORY_PATH):
    """Download the pinned ranking history if absent, and verify its hash."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(HISTORY_URL, timeout=60) as r:
            path.write_bytes(r.read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != HISTORY_SHA256:
        raise SystemExit(f"{path}: SHA-256 {digest} does not match the pinned {HISTORY_SHA256}")
    return path


def load_history():
    """All ranking releases as (date, team, points), team names in results.csv spelling."""
    h = pd.read_csv(fetch_history(), parse_dates=["date"])
    h = h[h.total_points.notna() & ~h.team.str.contains(r"\(unranked\)")]
    h = h.assign(team=h.team.replace(FIFA_TO_RESULTS))
    # a rename can collide within one release (none do today); keep the higher entry
    h = h.sort_values("total_points", ascending=False).drop_duplicates(["date", "team"])
    return h[["date", "team", "total_points"]].sort_values(["date", "total_points"],
                                                            ascending=[True, False])
