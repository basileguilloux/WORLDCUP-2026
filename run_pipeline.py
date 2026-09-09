"""Run the full WORLDCUP-2026 pipeline end to end, in the documented order.

Equivalent to running each script in src/ by hand (see README "Pipeline"
section), but as one command from the repo root:

    python run_pipeline.py

Each step is run as its own process from the repo root, so the relative
data paths inside each script (e.g. "data/raw/results.csv") resolve
correctly, and step2_dixoncoles.py's "from harness import walk_forward"
resolves correctly too (Python puts a script's own directory, src/, on
sys.path when that script is the one being executed).
"""
import subprocess
import sys
from pathlib import Path

STEPS = [
    "01_load_clean.py",
    "02_elo.py",
    "03_elo_baseline.py",
    "04_features.py",
    "05_model.py",
    "eval_baseline.py",
    "harness.py",
    "step2_dixoncoles.py",
    "06_simulate.py",
    "07_blend_predict.py",
]

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"


def main():
    for step in STEPS:
        script = SRC / step
        print("\n" + "=" * 70)
        print(f"RUNNING {step}")
        print("=" * 70)
        result = subprocess.run([sys.executable, str(script)], cwd=ROOT)
        if result.returncode != 0:
            print(f"\n{step} failed (exit code {result.returncode}); stopping pipeline.")
            sys.exit(result.returncode)

    print("\n" + "=" * 70)
    print("Pipeline complete. Final power ranking saved to data/predictions.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()
