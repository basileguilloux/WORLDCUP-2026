#!/usr/bin/env bash
# ============================================================================
# RETIRED -- kept for reference only. DO NOT SCHEDULE.
#
# The 2026 tournament is over and data/predictions.csv is now the FROZEN
# pre-tournament forecast that the README scores against the real results.
# Re-running this would leak post-tournament news into that file and destroy
# the only thing that makes it worth keeping.
#
# The LaunchAgent that ran this every 2 days was unloaded and deleted on
# 2026-09-19. scripts/worldcup-news.plist.example is retained as a record of
# how it was installed, and is likewise retired.
#
# This script now refuses to do anything quietly: every path that used to be a
# silent no-op exits non-zero, so any future scheduled use fails visibly. Even
# with a key present it will NOT overwrite the frozen forecast, because
# 09_team_news.py requires an explicit --overwrite-frozen.
# ============================================================================
#
# Refresh the WC-2026 team-news overlay for EVERY team and rewrite data/predictions.csv.
#
# Requires an API key. Put it in ~/.worldcup2026.env (NOT committed):
#     export ANTHROPIC_API_KEY="sk-ant-..."

set -euo pipefail

# Repo root is derived from this script's own location, so the checkout can live
# anywhere and be named anything. ${BASH_SOURCE[0]:-$0} keeps working if the
# script is sourced by a shell that does not set BASH_SOURCE (e.g. zsh).
SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
REPO="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)"
LOG="$REPO/data/team_news_refresh.log"
ENV_FILE="$HOME/.worldcup2026.env"
PY_SCRIPT="$REPO/src/09_team_news.py"
VENV_PY="$REPO/.venv/bin/python"

cd "$REPO"
mkdir -p "$(dirname "$LOG")"

log() { echo "$*" >> "$LOG"; }
die() { log "ERROR: $*"; log "===== aborted ====="; echo "ERROR: $*" >&2; exit 1; }

log "===== $(date '+%Y-%m-%d %H:%M:%S') refresh start ====="
log "repo: $REPO"

# Fail loudly on a broken checkout rather than silently doing nothing.
[ -f "$PY_SCRIPT" ] || die "missing $PY_SCRIPT (is \$REPO correct? repo=$REPO)"
PYTHON="$VENV_PY"
[ -x "$PYTHON" ] || PYTHON="$(command -v python3)" || die "no python found (.venv missing and no python3 on PATH)"

# The key never lives in the repo -- only in ~/.worldcup2026.env.
if [ ! -f "$ENV_FILE" ]; then
  die "$ENV_FILE not found, so no ANTHROPIC_API_KEY. No API calls made. (This script is RETIRED; see the header.)"
fi
# shellcheck source=/dev/null
source "$ENV_FILE"

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  die "$ENV_FILE exists but sets no ANTHROPIC_API_KEY. No API calls made."
fi

# --refresh re-fetches every team so the swing tracks the latest news, not a stale cache.
# NOTE: no --overwrite-frozen here, by design. 09_team_news.py will refuse to
# overwrite the frozen data/predictions.csv and exit non-zero. Anyone reviving
# this must add that flag deliberately, having read why it is frozen.
log "running: $PYTHON src/09_team_news.py --live --refresh"
PYTHONPATH=src "$PYTHON" src/09_team_news.py --live --refresh >> "$LOG" 2>&1
log "===== $(date '+%Y-%m-%d %H:%M:%S') refresh done ====="
