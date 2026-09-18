"""Phase 6: TEAM-NEWS OVERLAY — a transparent, two-sided, prediction-time adjustment.

Does NOT retrain and does NOT add a trained feature. The validated Steps 1-3 model
is untouched; this only adjusts each team's strength INPUT for the 2026 fixtures.

  1. A team-news AGENT (Claude `claude-opus-4-8` + `web_search`) covers EVERY team and,
     per fixture, returns a SIGNED net-sentiment score from recent news (restricted to
     before the match date):
        {"team", "news_score" -1..+1, "sentiment", "key_players_out",
         "key_players_back", "notes"}
     news_score is two-sided: NEGATIVE for injuries / suspensions / off-field turmoil /
     poor preparation; POSITIVE for key players returning, strong form & momentum, and
     excellent preparation. 0 = nothing notable. Output is cached to data/team_news.json
     (the audit artifact).
  2. news_score -> a SIGNED Elo adjustment on that team's strength input:
        delta = MAX_SWING * news_score        (negative news lowers the rating, positive raises it)
  3. The adjusted ratings feed the EXISTING calibrated Step-3 ensemble. No retraining.

Output: data/predictions.csv with news_score, sentiment, key_out/back, applied Elo delta
and notes per team, PLUS both pre- and post-adjustment probabilities.

Run:
  PYTHONPATH=src python src/08_team_news.py            # use cached news (no API calls)
  PYTHONPATH=src python src/08_team_news.py --live     # agent covers every uncached (team,date)
  PYTHONPATH=src python src/08_team_news.py --live --refresh   # re-fetch EVERY team (ignore cache)
A LaunchAgent (scripts/refresh_team_news.sh) runs `--live --refresh` every 2 days.
"""
import json
import os
import re
import sys
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from harness import load, outcomes, grid_probs, expected_goals, fit
from step3_classifier import make_clf, match_features, clf_probs, BEST_RHO

# ----------------------------------------------------------------------------
# TUNABLE: how influential the overlay is. MAX_SWING Elo points are applied at
# news_score = ±1.0 (catastrophe / dream scenario). 120 is deliberately strong —
# bigger than the 60-pt home advantage — so real news visibly moves predictions.
# Lower it to soften, raise it to bite harder.
# ----------------------------------------------------------------------------
MAX_SWING = 120.0
SWING_FLAG = 0.15             # flag a fixture if any class prob moves > this (audit cue)
W, TEMP = 0.3, 0.95           # Step-3 ensemble winners (unchanged)
N, START, HOME_ADV = 5, 1500.0, 60.0
RECENT_CUTOFF = pd.Timestamp("2022-06-01")
NEWS_CACHE = "data/team_news.json"
MODEL = "claude-opus-4-8"

# ============================================================================
# 1. TEAM-NEWS AGENT  (Claude + web_search -> signed structured JSON, cached)
# ============================================================================
AGENT_SYSTEM = """You are a high-quality football team-news analyst for FIFA World Cup 2026 match prediction.

For a given national team and match date, search recent news RESTRICTED TO THE DAYS BEFORE
the match date and produce a single SIGNED assessment of how the team's chances for that
specific match are affected by news, relative to its normal full-strength baseline.

Consider BOTH directions and weigh everything by how much it actually affects the result:

NEGATIVE signals (lower the score):
- Confirmed injuries or suspensions to important players (a first-choice keeper, star forward,
  or key playmaker matters far more than a fringe squad player).
- Off-field disruption: legal trouble, theft/robbery of players, internal disputes, a coach
  sacking days before the match, illness sweeping the camp, travel/logistics chaos.
- Poor preparation or form: heavy defeats in warm-ups, low morale, fatigue, a dead-rubber game
  where the side is expected to rotate / rest starters.

POSITIVE signals (raise the score):
- Key players returning to fitness or available again after doubt.
- Strong recent form and momentum; confidence from good warm-up results.
- Excellent, settled preparation; full-strength squad; tactical cohesion; a must-win edge.

Then output a SINGLE net score capturing the balance of these.

Return STRUCTURED JSON ONLY — no prose, no markdown, no code fences. Exactly these keys:
{"team": string, "news_score": number from -1.0 to 1.0, "sentiment": "positive"|"neutral"|"negative",
 "key_players_out": integer, "key_players_back": integer, "notes": string}

news_score scale (signed, two-sided):
   1.0  dream scenario: full strength + returning stars + excellent prep & momentum
   0.5  clearly boosted: a key player back and/or strong form
   0.2  modest positive
   0.0  nothing notable / business as usual
  -0.2  modest negative: one important player doubtful, minor disruption
  -0.5  clearly weakened: several key players out, or notable off-field trouble
  -1.0  disaster: decimated squad and/or major turmoil, or deliberately resting most starters
key_players_out / key_players_back: counts of starter-level players unavailable / returning.
notes: one or two sentences naming who/what and why it matters, so a human can audit the score."""


def _agent_user(team, match_date):
    return (f"Team: {team}\n"
            f"Match date: {match_date} (only use news published before this date).\n"
            "Assess the NET news effect on this team for its World Cup 2026 fixture on that date, "
            "weighing positive and negative signals. Return JSON only.")


def _extract_json(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _neutral(team, why="no data — neutral (no swing)"):
    return {"team": team, "news_score": 0.0, "sentiment": "neutral",
            "key_players_out": 0, "key_players_back": 0, "notes": why}


def _validate(rec, team):
    """Coerce a parsed record into the signed contract; default ANY bad/missing field to neutral.

    Backward-compatible: an old-style record with `availability_score` (0-1) but no
    `news_score` is converted to news_score = availability_score - 1.0 (<= 0)."""
    if not isinstance(rec, dict):
        return _neutral(team)
    score = rec.get("news_score")
    if score is None and "availability_score" in rec:           # legacy cache entry
        try:
            score = float(rec["availability_score"]) - 1.0
        except (TypeError, ValueError):
            score = 0.0
    try:
        score = float(score if score is not None else 0.0)
    except (TypeError, ValueError):
        score = 0.0
    score = min(1.0, max(-1.0, score))                          # clamp to [-1, 1]
    def _int(k):
        try:
            return max(0, int(rec.get(k, 0)))
        except (TypeError, ValueError):
            return 0
    sentiment = rec.get("sentiment")
    if sentiment not in ("positive", "neutral", "negative"):
        sentiment = "positive" if score > 0.05 else "negative" if score < -0.05 else "neutral"
    return {"team": team, "news_score": score, "sentiment": sentiment,
            "key_players_out": _int("key_players_out"), "key_players_back": _int("key_players_back"),
            "notes": str(rec.get("notes", "")) or "no notes returned"}


def agent_gather(team, match_date, client):
    """One live agent call: Claude + web_search -> validated signed record. Neutral on failure."""
    try:
        msgs = [{"role": "user", "content": _agent_user(team, match_date)}]
        tools = [{"type": "web_search_20260209", "name": "web_search"}]
        resp = None
        for _ in range(6):                                      # resume server-tool loop on pause_turn
            resp = client.messages.create(
                model=MODEL, max_tokens=2000,
                thinking={"type": "adaptive"},
                system=AGENT_SYSTEM, tools=tools, messages=msgs,
            )
            if resp.stop_reason == "pause_turn":
                msgs = [msgs[0], {"role": "assistant", "content": resp.content}]
                continue
            break
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        return _validate(_extract_json(text), team)
    except Exception as e:
        return _neutral(team, f"agent error ({type(e).__name__}) — neutral default")


def load_cache():
    if os.path.exists(NEWS_CACHE):
        with open(NEWS_CACHE) as f:
            return json.load(f)
    return {}


def get_news(team, match_date, cache, client, refresh=False):
    """Cache-first lookup keyed by (date, team). --live fills/refreshes; offline -> neutral."""
    key = f"{match_date}|{team}"
    if not refresh and key in cache:
        return _validate(cache[key], team)
    if client is not None:
        rec = agent_gather(team, match_date, client)
        cache[key] = rec
        return rec
    return _validate(cache[key], team) if key in cache else _neutral(team)


# ============================================================================
# 2. CURRENT STRENGTH  (replay + FIFA blend — identical recipe to 08_predict_final)
# ============================================================================
def build_strength_and_form():
    df = pd.read_csv("data/raw/results.csv", parse_dates=["date"])
    ren = dict(zip(*[pd.read_csv("data/raw/former_names.csv")[c] for c in ["former", "current"]]))
    df["home_team"], df["away_team"] = df.home_team.replace(ren), df.away_team.replace(ren)
    df = df.sort_values("date")
    played, future = df[df.home_score.notna()], df[df.home_score.isna()]

    elo = defaultdict(lambda: START)
    hist = defaultdict(lambda: deque(maxlen=N))
    recent = defaultdict(int)
    expc = lambda ra, rb: 1 / (1 + 10 ** ((rb - ra) / 400))
    for r in played.itertuples():
        rh, ra = elo[r.home_team], elo[r.away_team]
        k = 60 if "FIFA World Cup" in r.tournament else (40 if r.tournament != "Friendly" else 20)
        gd = abs(r.home_score - r.away_score); k *= 1 if gd <= 1 else (1.5 if gd == 2 else 1 + gd / 5)
        res = 1.0 if r.home_score > r.away_score else (0.5 if r.home_score == r.away_score else 0.0)
        ch = k * (res - expc(rh + (0 if r.neutral else HOME_ADV), ra))
        elo[r.home_team] += ch; elo[r.away_team] -= ch
        hist[r.home_team].append((r.home_score, r.away_score))
        hist[r.away_team].append((r.away_score, r.home_score))
        if r.date >= RECENT_CUTOFF:
            recent[r.home_team] += 1; recent[r.away_team] += 1

    rank = pd.read_csv("data/raw/fifa_ranking_2026-06-11.csv")
    wc = list(rank.team)
    fifa_pts = dict(zip(rank.team, rank.fifa_points))
    a, b = np.polyfit([fifa_pts[t] for t in wc], [elo[t] for t in wc], 1)
    fifa_elo = {t: a * fifa_pts[t] + b for t in wc}
    blend_w = lambda t: float(np.clip(10 / (10 + recent[t]), 0.2, 0.6))
    strength = defaultdict(lambda: START)
    for t in set(list(elo) + wc):
        s = elo[t]
        if t in fifa_elo:
            w = blend_w(t); s = (1 - w) * elo[t] + w * fifa_elo[t]
        strength[t] = s

    def form(t):
        if not hist[t]:
            return 1.0, 1.0
        return float(np.mean([g[0] for g in hist[t]])), float(np.mean([g[1] for g in hist[t]]))
    return strength, form, future


# ============================================================================
# 3. PREDICTION  (Step-3 ensemble, unchanged — only the elo INPUT is adjusted)
# ============================================================================
def predict_probs(poisson, logit, fx):
    Pp = grid_probs(*expected_goals(poisson, fx), rho=BEST_RHO)   # [away, draw, home]
    Pc = clf_probs(logit, fx)
    P = W * Pp + (1 - W) * Pc
    P = np.clip(P, 1e-12, 1.0) ** (1.0 / TEMP)
    return P / P.sum(axis=1, keepdims=True)


def pick(P):
    return np.where(P.argmax(1) == 2, "HOME", np.where(P.argmax(1) == 1, "DRAW", "AWAY"))


# ============================================================================
# main
# ============================================================================
def main():
    live = "--live" in sys.argv
    refresh = "--refresh" in sys.argv
    client = None
    if live:
        try:
            import anthropic
            client = anthropic.Anthropic()
            print(f"[live] team-news agent enabled — covering EVERY team "
                  f"(model {MODEL} + web_search){', full refresh' if refresh else ''}\n")
        except Exception as e:
            print(f"[live] could not init Anthropic client ({e}); using cache only\n")

    cache = load_cache()
    strength, form, future = build_strength_and_form()
    feat = load()
    poisson = fit(feat)
    logit = make_clf("logit").fit(match_features(feat), outcomes(feat))

    home_t, away_t = future.home_team.values, future.away_team.values
    dates = [str(pd.Timestamp(d).date()) for d in future.date.values]
    base = pd.DataFrame({"home_team": home_t, "away_team": away_t, "date": dates,
                         "neutral": future.neutral.values})
    base["home_elo"] = [strength[t] for t in home_t]
    base["away_elo"] = [strength[t] for t in away_t]
    base["home_gf"], base["home_ga"] = zip(*[form(t) for t in home_t])
    base["away_gf"], base["away_ga"] = zip(*[form(t) for t in away_t])
    base["home_advantage"] = (~base.neutral.astype(bool)).astype(int)

    # cover every fixture, every team — signed news_score -> signed Elo delta
    if live:
        n = len(base) * 2
        print(f"[live] gathering team news for {n} (team, match) slots across {len(base)} fixtures...")
    rows = []
    for r in base.itertuples():
        hn = get_news(r.home_team, r.date, cache, client, refresh)
        an = get_news(r.away_team, r.date, cache, client, refresh)
        rows.append((hn, an))
    h_score = np.array([hn["news_score"] for hn, _ in rows])
    a_score = np.array([an["news_score"] for _, an in rows])
    h_delta = MAX_SWING * h_score          # signed: + raises rating, - lowers it
    a_delta = MAX_SWING * a_score

    if live:
        with open(NEWS_CACHE, "w") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)

    P_pre = predict_probs(poisson, logit, base)
    adj = base.copy()
    adj["home_elo"] = base.home_elo + h_delta
    adj["away_elo"] = base.away_elo + a_delta
    P_post = predict_probs(poisson, logit, adj)

    out = base[["date", "home_team", "away_team", "neutral"]].copy()
    out["home_news_score"] = h_score.round(2)
    out["home_sentiment"] = [hn["sentiment"] for hn, _ in rows]
    out["home_key_out"] = [hn["key_players_out"] for hn, _ in rows]
    out["home_key_back"] = [hn["key_players_back"] for hn, _ in rows]
    out["home_elo_delta"] = h_delta.round(1)
    out["home_notes"] = [hn["notes"] for hn, _ in rows]
    out["away_news_score"] = a_score.round(2)
    out["away_sentiment"] = [an["sentiment"] for _, an in rows]
    out["away_key_out"] = [an["key_players_out"] for _, an in rows]
    out["away_key_back"] = [an["key_players_back"] for _, an in rows]
    out["away_elo_delta"] = a_delta.round(1)
    out["away_notes"] = [an["notes"] for _, an in rows]
    out["p_home_pre"], out["p_draw_pre"], out["p_away_pre"] = P_pre[:, 2], P_pre[:, 1], P_pre[:, 0]
    out["p_home_post"], out["p_draw_post"], out["p_away_post"] = P_post[:, 2], P_post[:, 1], P_post[:, 0]
    out["pick_pre"], out["pick_post"] = pick(P_pre), pick(P_post)
    out["max_swing"] = np.abs(P_post - P_pre).max(axis=1)
    out.to_csv("data/predictions.csv", index=False)

    # ---- coverage report (every team) ----
    all_teams = sorted(set(home_t) | set(away_t))
    covered = {t for t in all_teams
               if any(f"{d}|{t}" in cache for d in set(dates))}
    print(f"COVERAGE: {len(covered)}/{len(all_teams)} teams have at least one news read.")
    missing = [t for t in all_teams if t not in covered]
    if missing:
        print(f"  not yet covered ({len(missing)}): {', '.join(missing)}")
        print("  -> run with --live (needs `anthropic` installed + ANTHROPIC_API_KEY) to cover them.\n")
    else:
        print("  every team covered.\n")

    # ---- movers ----
    movers = out[out.max_swing > 0.005].sort_values("max_swing", ascending=False)
    if movers.empty:
        print("No fixture moved (cache has no non-zero news_score). Run --live to populate.")
    else:
        print(f"Fixtures the overlay meaningfully moved (top {min(10, len(movers))}):\n")
        for r in movers.head(10).itertuples():
            flag = "  ⚠ LARGE SWING — sanity-check MAX_SWING" if r.max_swing > SWING_FLAG else ""
            print(f"{r.home_team} v {r.away_team}  ({r.date}){flag}")
            print(f"    H/D/A  pre : {r.p_home_pre*100:4.0f}/{r.p_draw_pre*100:4.0f}/{r.p_away_pre*100:4.0f}"
                  f"   post: {r.p_home_post*100:4.0f}/{r.p_draw_post*100:4.0f}/{r.p_away_post*100:4.0f}"
                  f"   (max move {r.max_swing*100:.0f} pts, pick {r.pick_pre}→{r.pick_post})")
            if abs(r.home_news_score) > 0.01:
                print(f"    {r.home_team}: {r.home_sentiment} {r.home_news_score:+.2f} "
                      f"({r.home_key_out} out, {r.home_key_back} back, {r.home_elo_delta:+.0f} Elo) — {r.home_notes}")
            if abs(r.away_news_score) > 0.01:
                print(f"    {r.away_team}: {r.away_sentiment} {r.away_news_score:+.2f} "
                      f"({r.away_key_out} out, {r.away_key_back} back, {r.away_elo_delta:+.0f} Elo) — {r.away_notes}")
            print()

    big = movers[movers.max_swing > SWING_FLAG]
    if not big.empty:
        print(f"⚠  {len(big)} fixture(s) swung > {SWING_FLAG*100:.0f} pts — "
              f"if those look too strong, lower MAX_SWING (currently {MAX_SWING:.0f}).")
    print("saved -> data/predictions.csv  (signed news_score, Elo delta, notes + pre/post probs per team)")


if __name__ == "__main__":
    main()
