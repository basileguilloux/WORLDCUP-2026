#!/usr/bin/env bash
# Refresh the WC-2026 team-news overlay for EVERY team and rewrite data/predictions.csv.
# Run by launchd every 2 days (com.worldcup2026.teamnews). Safe to run by hand too.
#
# Requires an API key. Put it in ~/.worldcup2026.env (NOT committed):
#     export ANTHROPIC_API_KEY="sk-ant-..."
# Without it, this logs a SKIP and makes zero API calls.
#
# Install the LaunchAgent: see scripts/worldcup-news.plist.example.

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
  log "SKIP: $ENV_FILE not found, so no ANTHROPIC_API_KEY. No API calls made."
  log "===== done (skipped) ====="
  exit 0
fi
# shellcheck source=/dev/null
source "$ENV_FILE"

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  log "SKIP: $ENV_FILE exists but sets no ANTHROPIC_API_KEY. No API calls made."
  log "===== done (skipped) ====="
  exit 0
fi

# --refresh re-fetches every team so the swing tracks the latest news, not a stale cache.
log "running: $PYTHON src/09_team_news.py --live --refresh"
PYTHONPATH=src "$PYTHON" src/09_team_news.py --live --refresh >> "$LOG" 2>&1
log "===== $(date '+%Y-%m-%d %H:%M:%S') refresh done ====="
