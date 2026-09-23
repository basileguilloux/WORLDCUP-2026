"""Run the full WORLDCUP-2026 pipeline end to end, in the documented order.

Equivalent to running each script in src/ by hand (see README "Pipeline"
section), but as one command from the repo root:

    python run_pipeline.py

Each step is run as its own process from the repo root, so the relative data
paths inside each script (e.g. "data/raw/results.csv") resolve correctly, and
`from harness import ...` resolves correctly too (Python puts a script's own
directory, src/, on sys.path when that script is the one being executed).

No step writes data/predictions.csv. That file is the frozen forecast: its
p_*_pre columns are bit-identical to the model run committed in 99a2305.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

# (script, what it writes) — None means it only prints / writes no CSV.
STEPS = [
    ("01_load_clean.py", None),
    ("02_elo.py", "data/processed_matches.csv"),
    ("03_elo_baseline.py", None),
    ("04_features.py", "data/features.csv"),
    ("05_model.py", "data/poisson_model.joblib"),
    ("eval_baseline.py", None),
    ("harness.py", None),
    ("step2_dixoncoles.py", None),
    ("step3_classifier.py", None),
    ("06_simulate.py", None),
    ("07_blend_predict.py", "data/power_rankings.csv"),
    ("08_fixture_predictions.py", "data/fixture_predictions.csv"),
    ("10_tournament_sim.py", "data/tournament_sim.csv"),
]


def run(step, args=()):
    print("\n" + "=" * 70)
    print(f"RUNNING {step}" + (f" {' '.join(args)}" if args else ""))
    print("=" * 70)
    result = subprocess.run([sys.executable, str(SRC / step), *args], cwd=ROOT)
    if result.returncode != 0:
        print(f"\n{step} failed (exit code {result.returncode}); stopping pipeline.")
        sys.exit(result.returncode)


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()

    for step, _ in STEPS:
        run(step)

    print("\n" + "=" * 70)
    print("Pipeline complete. Outputs:")
    for out in [o for _, o in STEPS if o]:
        print(f"  {out}")
    print("  data/predictions.csv        (unchanged, frozen forecast)")
    print("=" * 70)


if __name__ == "__main__":
    main()
