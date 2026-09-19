"""Run the full WORLDCUP-2026 pipeline end to end, in the documented order.

Equivalent to running each script in src/ by hand (see README "Pipeline"
section), but as one command from the repo root:

    python run_pipeline.py              # everything except the team-news step
    python run_pipeline.py --skip-news  # explicit no-op; same as the default
    python run_pipeline.py --news       # RETIRED: exits 1 and explains why

Each step is run as its own process from the repo root, so the relative data
paths inside each script (e.g. "data/raw/results.csv") resolve correctly, and
`from harness import ...` resolves correctly too (Python puts a script's own
directory, src/, on sys.path when that script is the one being executed).

THE TEAM-NEWS STEP IS RETIRED. data/predictions.csv is the frozen
pre-tournament forecast that the README scores against the real 2026 results,
so no run from here may rewrite it. --news is kept only to fail loudly and
explain why, pointing at the one deliberate override; the pipeline itself has
no way to pass that override through.
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

NEWS_STEP = ("09_team_news.py", "data/predictions.csv")

FROZEN_REFUSAL = """ERROR: --news is retired and will not run.

data/predictions.csv is the FROZEN pre-tournament forecast. The 2026 tournament
finished in July and the README scores that forecast against the real results,
so rewriting it would leak post-tournament news into a file whose entire value
is that it predates the event.

There is deliberately no way to override this from run_pipeline.py. If you truly
intend to discard the frozen forecast, run the one command that says so:

    python src/09_team_news.py --live --overwrite-frozen

Re-run without --news to run everything else."""


def run(step, args=()):
    print("\n" + "=" * 70)
    print(f"RUNNING {step}" + (f" {' '.join(args)}" if args else ""))
    print("=" * 70)
    result = subprocess.run([sys.executable, str(SRC / step), *args], cwd=ROOT)
    if result.returncode != 0:
        print(f"\n{step} failed (exit code {result.returncode}); stopping pipeline.")
        sys.exit(result.returncode)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--news", action="store_true",
                   help="RETIRED: exits 1 and explains why. data/predictions.csv is "
                        "the frozen pre-tournament forecast.")
    g.add_argument("--skip-news", action="store_true",
                   help="explicitly skip the team-news step (this is the default).")
    opts = ap.parse_args()

    # Refuse before running anything, so the failure costs nothing.
    if opts.news:
        print(FROZEN_REFUSAL, file=sys.stderr)
        sys.exit(1)

    for step, _ in STEPS:
        run(step)

    print("\n" + "=" * 70)
    print("SKIPPED 09_team_news.py (team-news overlay) — retired.")
    print("  data/predictions.csv was NOT touched: it is the frozen")
    print("  pre-tournament forecast. See the README.")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("Pipeline complete. Outputs:")
    for out in [o for _, o in STEPS if o]:
        print(f"  {out}")
    print("  data/predictions.csv        (unchanged — frozen pre-tournament forecast)")
    print("=" * 70)


if __name__ == "__main__":
    main()
