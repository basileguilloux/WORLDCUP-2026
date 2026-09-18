#!/bin/zsh
# Refresh the WC-2026 team-news overlay for EVERY team and rewrite data/predictions.csv.
# Run by launchd every 2 days (com.worldcup2026.teamnews). Safe to run by hand too.
#
# Requires an API key. Put it in ~/.worldcup2026.env (NOT committed):
#     export ANTHROPIC_API_KEY="sk-ant-..."
# Without it, this logs a SKIP and makes zero API calls.

REPO="/Users/basileguilloux/worldcup-2026"
LOG="$REPO/data/team_news_refresh.log"
cd "$REPO" || exit 1

[ -f "$HOME/.worldcup2026.env" ] && source "$HOME/.worldcup2026.env"

echo "===== $(date '+%Y-%m-%d %H:%M:%S') refresh start =====" >> "$LOG"
if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo "SKIP: ANTHROPIC_API_KEY not set (create ~/.worldcup2026.env). No API calls made." >> "$LOG"
  echo "===== done (skipped) =====" >> "$LOG"
  exit 0
fi

# --refresh re-fetches every team so the swing tracks the latest news, not a stale cache.
PYTHONPATH=src "$REPO/.venv/bin/python" src/08_team_news.py --live --refresh >> "$LOG" 2>&1
echo "===== $(date '+%Y-%m-%d %H:%M:%S') refresh done =====" >> "$LOG"
