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

Output columns: news_score, sentiment, key_out/back, applied Elo delta and notes
per team, PLUS both pre- and post-adjustment probabilities.

WHERE IT WRITES (deliberate — an offline run must never clobber the real output):
  --live  -> data/predictions.csv          the FROZEN forecast (gated)
  offline -> data/predictions_offline.csv  cache-only dry run, safe to throw away
Offline, any team missing from the cache defaults to neutral (0, no swing), so an
offline run is mostly a no-op overlay and is NOT a substitute for the live one.

data/predictions.csv IS FROZEN. Its p_*_pre columns are the forecast, bit-identical
to the model run committed in 99a2305 on 13 June 2026; the p_*_post columns are this
overlay, added 18 September. Regenerating the file would destroy that provenance, so
--live refuses to run without the explicit --overwrite-frozen flag.

FAILURE HANDLING (a run either writes a real forecast or fails loudly):
  * --live makes a minimal preflight call first; bad credentials exit non-zero
    in seconds, before any work.
  * Authentication / permission errors mid-run are never swallowed: the run
    aborts non-zero and writes nothing.
  * Only successful agent results are cached. Transient failures (rate limit,
    5xx, timeout) degrade to neutral FOR THAT RUN ONLY and are never persisted.
  * If more than MAX_DEGRADED_FRACTION of teams fell back to neutral, a live run
    aborts rather than publish a flat overlay that looks like a forecast.

Run:
  python src/09_team_news.py                    # cached dry run -> predictions_offline.csv
  python src/09_team_news.py --live --overwrite-frozen             # deliberate refresh
  python src/09_team_news.py --live --overwrite-frozen --refresh   # re-fetch EVERY team
The LaunchAgent that used to run this every 2 days is RETIRED; see
scripts/refresh_team_news.sh.
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
OUT_LIVE = "data/predictions.csv"           # real output: only a --live run may write here
OUT_OFFLINE = "data/predictions_offline.csv"  # cache-only dry run
MODEL = "claude-opus-4-8"

# Refuse to write more than this fraction of TEAMS from degraded (error / neutral
# fallback) records. A run that cannot actually reach the agent for most of the
# field produces a flat, all-neutral overlay that looks like a real forecast, so
# it is treated as a failed run rather than written out.
MAX_DEGRADED_FRACTION = 0.25

# data/predictions.csv is the FROZEN forecast (see README). Its p_*_pre columns
# are bit-identical to the model run committed in 99a2305 on 13 June 2026, and
# regenerating the file would destroy that provenance.
FROZEN_MSG = (
    "data/predictions.csv is the FROZEN forecast and must not be refreshed: its "
    "p_*_pre columns are bit-identical to the model run committed in 99a2305 on "
    "13 June 2026, and regenerating the file would destroy that provenance.\n"
    "If you genuinely intend to overwrite it, re-run with --overwrite-frozen."
)


class NewsError(RuntimeError):
    """Fatal condition: abort the run and write nothing."""


def _auth_error_types():
    """(AuthenticationError, PermissionDeniedError) if the SDK is importable, else ()."""
    try:
        import anthropic
        return (anthropic.AuthenticationError, anthropic.PermissionDeniedError)
    except Exception:
        return ()

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
    """One live agent call -> (validated record, status).

    status is "ok" for a real answer or "fallback" for a transient failure
    (rate limit, 5xx, timeout) that degrades to neutral FOR THIS RUN ONLY.
    Authentication / permission errors are NOT transient and are never
    swallowed: they raise NewsError so the caller can abort and write nothing.
    """
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
        return _validate(_extract_json(text), team), "ok"
    except _auth_error_types() as e:
        raise NewsError(
            f"authentication/permission failure from the Anthropic API while fetching "
            f"{team} ({type(e).__name__}). Aborting: nothing was written.\n"
            f"Check ANTHROPIC_API_KEY."
        ) from e
    except Exception as e:
        # Transient only. Degrades to neutral for this run and is NEVER cached.
        return _neutral(team, f"transient agent error ({type(e).__name__}) — neutral for this run"), "fallback"


def preflight(client):
    """One minimal call before any work, so a bad key fails in seconds, not after a full run.

    Auth / permission failures are fatal (NewsError). Anything else is treated as
    possibly transient: warn and let the run proceed.
    """
    try:
        client.messages.create(model=MODEL, max_tokens=1,
                               messages=[{"role": "user", "content": "ping"}])
    except _auth_error_types() as e:
        raise NewsError(
            f"preflight failed: the Anthropic API rejected the credentials "
            f"({type(e).__name__}). Aborting before any work; nothing was written.\n"
            f"Check ANTHROPIC_API_KEY (the team-news step reads it from the "
            f"environment, or from ~/.worldcup2026.env via scripts/refresh_team_news.sh)."
        ) from e
    except Exception as e:
        print(f"[live] preflight warning: {type(e).__name__}: {e}\n"
              f"       not an auth failure, continuing.")
    else:
        print("[live] preflight OK — credentials accepted.")


def load_cache():
    if os.path.exists(NEWS_CACHE):
        with open(NEWS_CACHE) as f:
            return json.load(f)
    return {}


def get_news(team, match_date, cache, client, refresh=False):
    """Cache-first lookup keyed by (date, team) -> (record, status).

    status: "ok"       fresh, successful agent result (the only kind ever cached)
            "cached"   served from a previous run's cached result
            "fallback" neutral stand-in: transient error, or no data at all
    """
    key = f"{match_date}|{team}"
    if not refresh and key in cache:
        return _validate(cache[key], team), "cached"
    if client is not None:
        rec, status = agent_gather(team, match_date, client)
        if status == "ok":
            cache[key] = rec          # only real results are persisted
        return rec, status
    if key in cache:
        return _validate(cache[key], team), "cached"
    return _neutral(team), "fallback"


# ============================================================================
# 2. CURRENT STRENGTH  (replay + FIFA blend — identical recipe to 08_fixture_predictions)
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
def make_client():
    """Construct the Anthropic client. Separated out so tests can inject a fake."""
    import anthropic
    return anthropic.Anthropic()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    live = "--live" in argv
    refresh = "--refresh" in argv
    overwrite_frozen = "--overwrite-frozen" in argv

    # Fail before any work (and before any API spend) if the run could not be
    # written out anyway.
    if live and not overwrite_frozen:
        raise NewsError(FROZEN_MSG)

    client = None
    if live:
        try:
            client = make_client()
        except Exception as e:
            raise NewsError(
                f"--live could not initialise the Anthropic client ({type(e).__name__}: {e}). "
                f"Aborting; nothing was written.\nIs ANTHROPIC_API_KEY set?") from e
        print(f"[live] team-news agent enabled — covering EVERY team "
              f"(model {MODEL} + web_search){', full refresh' if refresh else ''}")
        preflight(client)
        print()

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
    status_by_team = defaultdict(set)
    for r in base.itertuples():
        hn, hs = get_news(r.home_team, r.date, cache, client, refresh)
        an, as_ = get_news(r.away_team, r.date, cache, client, refresh)
        status_by_team[r.home_team].add(hs)
        status_by_team[r.away_team].add(as_)
        rows.append((hn, an))

    # ---- sanity guard: refuse to publish a mostly-degraded overlay ----
    all_teams = sorted(status_by_team)
    degraded = [t for t, st in status_by_team.items() if "fallback" in st]
    frac = len(degraded) / len(all_teams) if all_teams else 0.0
    degraded_msg = (
        f"{len(degraded)}/{len(all_teams)} teams ({frac:.0%}) fell back to neutral "
        f"records, above the {MAX_DEGRADED_FRACTION:.0%} limit (MAX_DEGRADED_FRACTION). "
        f"A mostly-neutral overlay is a failed run, not a forecast.\n"
        f"  affected: {', '.join(sorted(degraded)[:12])}"
        f"{' ...' if len(degraded) > 12 else ''}")
    if frac > MAX_DEGRADED_FRACTION:
        # Fatal for a live run, which writes the real data/predictions.csv. A dry
        # run only writes the throwaway offline file, so it warns and continues.
        if live:
            raise NewsError(degraded_msg + "\nAborting; nothing was written.")
        print(f"[offline] WARNING: {degraded_msg}\n")
    h_score = np.array([hn["news_score"] for hn, _ in rows])
    a_score = np.array([an["news_score"] for _, an in rows])
    h_delta = MAX_SWING * h_score          # signed: + raises rating, - lowers it
    a_delta = MAX_SWING * a_score

    if live:
        # Reached only after the degraded-fraction guard passed. `cache` holds
        # successful results only — fallbacks are never persisted.
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

    # Offline runs write to a separate file so a cache-only dry run can never
    # overwrite the committed, news-adjusted data/predictions.csv.
    out_path = OUT_LIVE if live else OUT_OFFLINE
    out.to_csv(out_path, index=False)
    if not live:
        print(f"[offline] cache-only dry run -> {out_path}  "
              f"({OUT_LIVE} is the frozen forecast and was left untouched)\n")

    # ---- coverage report: REAL results only ----
    # "Covered" means a fresh, successful agent result in THIS run. Cached
    # records and neutral fallbacks are reported separately so a run can never
    # look better than it was.
    fresh = [t for t, st in status_by_team.items() if "ok" in st]
    cached_only = [t for t, st in status_by_team.items() if "ok" not in st and "cached" in st]
    # A team plays 3 fixtures, so these buckets can overlap: a team may have one
    # date served from cache and another that fell back. Counts are "teams with
    # at least one date of this kind", which is why they need not sum to 48.
    print(f"COVERAGE: {len(fresh)}/{len(all_teams)} teams have a real result from this run.")
    if cached_only:
        print(f"  no live read, >=1 date from cache ({len(cached_only)}): "
              f"{', '.join(sorted(cached_only))}")
    if degraded:
        print(f"  >=1 date fell back to neutral ({len(degraded)}): {', '.join(sorted(degraded))}")
    if not fresh:
        print("  -> no live reads. Run with --live (needs `anthropic` + ANTHROPIC_API_KEY).")
    print()

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
    print(f"saved -> {out_path}  (signed news_score, Elo delta, notes + pre/post probs per team)")


if __name__ == "__main__":
    try:
        main()
    except NewsError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
