"""09_team_news.py must fail loudly rather than publish a fake forecast.

Every test here uses a mocked client and makes NO network calls. Each asserts
that the run exits non-zero AND that data/predictions.csv and data/team_news.json
are byte-for-byte unchanged -- the regression these guards exist to prevent is a
failed run silently overwriting the frozen forecast with all-neutral records.

The module is loaded by path because `import 09_team_news` is a syntax error.
"""
import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS = ROOT / "data/predictions.csv"
CACHE = ROOT / "data/team_news.json"


def _load_module():
    sys.path.insert(0, str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location("team_news", ROOT / "src/09_team_news.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tn():
    return _load_module()


@pytest.fixture
def untouched():
    """Snapshot the two protected files; assert they are identical afterwards."""
    def digest(p):
        return hashlib.md5(p.read_bytes()).hexdigest() if p.exists() else None
    before = {p: digest(p) for p in (PREDICTIONS, CACHE)}
    yield
    for p, was in before.items():
        assert digest(p) == was, f"{p.name} was modified by a run that should have aborted"


class _AuthError(Exception):
    """Stands in for anthropic.AuthenticationError."""


class FakeMessages:
    def __init__(self, exc=None):
        self.exc = exc
        self.calls = 0

    def create(self, **kw):
        self.calls += 1
        if self.exc:
            raise self.exc
        raise AssertionError("unexpected successful call in a failure test")


class FakeClient:
    def __init__(self, exc=None):
        self.messages = FakeMessages(exc)


def _patch_auth_types(tn, monkeypatch):
    """Make the module treat _AuthError as an auth failure, with no SDK installed."""
    monkeypatch.setattr(tn, "_auth_error_types", lambda: (_AuthError,))


def test_invalid_key_fails_at_preflight(tn, monkeypatch, untouched):
    """An invalid key must abort during preflight, before any team is fetched."""
    _patch_auth_types(tn, monkeypatch)
    client = FakeClient(_AuthError("401 invalid x-api-key"))
    monkeypatch.setattr(tn, "make_client", lambda: client)

    with pytest.raises(tn.NewsError) as exc:
        tn.main(["--live", "--overwrite-frozen"])

    assert "preflight" in str(exc.value).lower()
    assert client.messages.calls == 1, "preflight must abort before fetching any team"


def test_auth_error_midrun_aborts(tn, monkeypatch, untouched):
    """Auth failure after a clean preflight must abort, not degrade to neutral."""
    _patch_auth_types(tn, monkeypatch)

    def boom(team, match_date, client):
        raise tn.NewsError(f"authentication/permission failure while fetching {team}")

    monkeypatch.setattr(tn, "make_client", lambda: FakeClient())
    monkeypatch.setattr(tn, "preflight", lambda client: None)
    monkeypatch.setattr(tn, "agent_gather", boom)

    with pytest.raises(tn.NewsError) as exc:
        tn.main(["--live", "--overwrite-frozen", "--refresh"])
    assert "authentication" in str(exc.value).lower()


def test_degraded_fraction_guard(tn, monkeypatch, untouched):
    """Too many neutral fallbacks must abort instead of publishing a flat overlay."""
    monkeypatch.setattr(tn, "make_client", lambda: FakeClient())
    monkeypatch.setattr(tn, "preflight", lambda client: None)
    # every team degrades to a transient-error fallback -> 100% > 25%
    monkeypatch.setattr(tn, "agent_gather",
                        lambda team, d, c: (tn._neutral(team, "transient"), "fallback"))

    with pytest.raises(tn.NewsError) as exc:
        tn.main(["--live", "--overwrite-frozen", "--refresh"])
    msg = str(exc.value)
    assert "MAX_DEGRADED_FRACTION" in msg and "nothing was written" in msg


def test_live_refuses_to_touch_frozen_forecast(tn, untouched):
    """--live without --overwrite-frozen must refuse before doing any work."""
    with pytest.raises(tn.NewsError) as exc:
        tn.main(["--live"])
    assert "--overwrite-frozen" in str(exc.value)


def test_fallbacks_are_never_cached(tn, monkeypatch):
    """A transient failure must not be persisted to team_news.json."""
    monkeypatch.setattr(tn, "agent_gather",
                        lambda team, d, c: (tn._neutral(team, "transient"), "fallback"))
    cache = {}
    rec, status = tn.get_news("Brazil", "2026-06-13", cache, FakeClient(), refresh=True)
    assert status == "fallback"
    assert cache == {}, "a failed read must never be written to the cache"


def test_successful_results_are_cached(tn, monkeypatch):
    good = {"team": "Brazil", "news_score": -0.3, "sentiment": "negative",
            "key_players_out": 2, "key_players_back": 0, "notes": "ok"}
    monkeypatch.setattr(tn, "agent_gather", lambda team, d, c: (good, "ok"))
    cache = {}
    rec, status = tn.get_news("Brazil", "2026-06-13", cache, FakeClient(), refresh=True)
    assert status == "ok"
    assert cache["2026-06-13|Brazil"] == good


def test_agent_gather_converts_auth_error(tn, monkeypatch):
    """agent_gather itself must raise, not swallow an auth error into a neutral record."""
    _patch_auth_types(tn, monkeypatch)
    client = FakeClient(_AuthError("401 invalid x-api-key"))
    with pytest.raises(tn.NewsError):
        tn.agent_gather("Brazil", "2026-06-13", client)


def test_agent_gather_degrades_on_transient_error(tn, monkeypatch):
    """A non-auth error degrades to neutral for this run, and is flagged as a fallback."""
    _patch_auth_types(tn, monkeypatch)
    client = FakeClient(TimeoutError("read timeout"))
    rec, status = tn.agent_gather("Brazil", "2026-06-13", client)
    assert status == "fallback"
    assert rec["news_score"] == 0.0


def test_run_pipeline_news_flag_is_retired(untouched):
    """`run_pipeline.py --news` must refuse, exit non-zero, and run no steps."""
    import subprocess

    proc = subprocess.run([sys.executable, str(ROOT / "run_pipeline.py"), "--news"],
                          cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "frozen" in proc.stderr.lower()
    assert "--overwrite-frozen" in proc.stderr, "must point at the deliberate override"
    assert "RUNNING" not in proc.stdout, "must refuse before running any pipeline step"


def test_run_pipeline_has_no_overwrite_frozen_passthrough():
    """There must be no way to force the frozen overwrite from run_pipeline.py."""
    src = (ROOT / "run_pipeline.py").read_text()
    assert "--overwrite-frozen" in src, "the refusal message should name the override"
    # ...but only inside the refusal text, never as an argument actually passed on.
    assert 'run(NEWS_STEP[0]' not in src, "run_pipeline must not invoke the news step at all"
