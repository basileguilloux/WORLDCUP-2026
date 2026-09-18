"""Run the full WORLDCUP-2026 pipeline end to end, in the documented order.

Equivalent to running each script in src/ by hand (see README "Pipeline"
section), but as one command from the repo root:

    python run_pipeline.py              # everything except the team-news step
    python run_pipeline.py --news       # also refresh the team-news overlay
    python run_pipeline.py --skip-news  # explicit no-op; same as the default

Each step is run as its own process from the repo root, so the relative data
paths inside each script (e.g. "data/raw/results.csv") resolve correctly, and
`from harness import ...` resolves correctly too (Python puts a script's own
directory, src/, on sys.path when that script is the one being executed).

THE TEAM-NEWS STEP IS OPT-IN. 09_team_news.py is the only writer of the
committed data/predictions.csv, and refreshing it costs live Anthropic API
calls, so the default run never touches that file. Pass --news to include it;
that requires ANTHROPIC_API_KEY (sourced from ~/.worldcup2026.env if present,
which is only read when --news is passed) and fails loudly if it is missing.
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
ENV_FILE = Path.home() / ".worldcup2026.env"

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


def load_key_from_env_file():
    """Read ANTHROPIC_API_KEY out of ~/.worldcup2026.env. Only called with --news."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "environment"
    if not ENV_FILE.exists():
        return None
    for line in ENV_FILE.read_text().splitlines():
        m = re.match(r'\s*(?:export\s+)?ANTHROPIC_API_KEY\s*=\s*["\']?([^"\'\s]+)', line)
        if m:
            os.environ["ANTHROPIC_API_KEY"] = m.group(1)
            return str(ENV_FILE)
    return None


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
                   help="also run the team-news overlay (live API calls; rewrites "
                        "data/predictions.csv). Requires ANTHROPIC_API_KEY.")
    g.add_argument("--skip-news", action="store_true",
                   help="explicitly skip the team-news step (this is the default).")
    opts = ap.parse_args()

    if opts.news:
        # Only read the env file when the user actually asked for the news step.
        source = load_key_from_env_file()
        if source is None:
            print("ERROR: --news needs ANTHROPIC_API_KEY, which is not set.\n"
                  f"       Export it, or put it in {ENV_FILE}:\n"
                  '           export ANTHROPIC_API_KEY="sk-ant-..."\n'
                  "       Re-run without --news to run everything else.", file=sys.stderr)
            sys.exit(1)
        print(f"[news] ANTHROPIC_API_KEY loaded from {source}")

    for step, _ in STEPS:
        run(step)

    if opts.news:
        run(NEWS_STEP[0], ["--live", "--refresh"])
    else:
        print("\n" + "=" * 70)
        print("SKIPPED 09_team_news.py (team-news overlay).")
        print("  data/predictions.csv was NOT touched — it keeps its committed,")
        print("  news-adjusted contents. Pass --news to refresh it (needs an API key).")
        print("=" * 70)

    written = [out for _, out in STEPS if out] + ([NEWS_STEP[1]] if opts.news else [])
    print("\n" + "=" * 70)
    print("Pipeline complete. Outputs:")
    for out in written:
        print(f"  {out}")
    if not opts.news:
        print("  data/predictions.csv        (unchanged — team-news step skipped)")
    print("=" * 70)


if __name__ == "__main__":
    main()
