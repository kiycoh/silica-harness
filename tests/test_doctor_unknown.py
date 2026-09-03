# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alessandro Carosia

"""`unknown` is a doctor status of its own, never folded into `ok`.

The report's job is to say what is live. Three of its checks could not answer
and said "ok" anyway: a hosted provider is never probed, and a local endpoint
that is still loading its model answers 503 on every path, which the exception
handler never sees. That is how a run reported "rerank ready" and marked rerank
unreachable in the same session.

An unprobed or half-answering endpoint reads as unknown: not a failure, since
nothing is known to be broken, and not ok, since nothing is known to work.
"""
from types import SimpleNamespace


from silica.onboarding import checks


class _Resp:
    def __init__(self, status_code):
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _config(**over):
    base = dict(model="m", provider="lmstudio", provider_base_url="",
                rerank_base_url="", rerank_model="")
    base.update(over)
    return SimpleNamespace(**base)


class TestTheStatusExists:
    def test_unknown_renders_with_its_own_glyph(self):
        """Sharing ok's glyph would put it back where it started."""
        assert "unknown" in checks._STATUS_GLYPH
        assert checks._STATUS_GLYPH["unknown"] != checks._STATUS_GLYPH["ok"]

    def test_unknown_is_not_a_failure(self):
        """Nothing is known to be broken, so it must not fail the exit code."""
        results = [checks.CheckResult("x", "unknown", "not probed")]
        assert checks.has_failures(results) is False

    def test_a_real_failure_still_fails(self):
        results = [checks.CheckResult("x", "unknown", "not probed"),
                   checks.CheckResult("y", "fail", "broken")]
        assert checks.has_failures(results) is True


class TestTheChatEndpoint:
    def test_a_hosted_provider_is_unknown_not_ok(self, monkeypatch):
        """It is never probed, so `ok` is a claim the check did not make."""
        result = checks.check_chat_endpoint(_config(provider="openrouter"))

        assert result.status == "unknown"
        assert "not probed" in result.detail

    def test_a_loading_endpoint_is_unknown_not_reachable(self, monkeypatch):
        """llama.cpp opens the port before the weights are in and answers 503
        on every path, so "the socket accepted" is not "the model is up"."""
        monkeypatch.setattr(checks.httpx, "get", lambda *a, **k: _Resp(503))

        result = checks.check_chat_endpoint(_config())

        assert result.status == "unknown"
        assert "loading" in result.detail.lower()

    def test_a_serving_endpoint_is_still_ok(self, monkeypatch):
        monkeypatch.setattr(checks.httpx, "get", lambda *a, **k: _Resp(200))

        assert checks.check_chat_endpoint(_config()).status == "ok"

    def test_a_dead_endpoint_still_fails(self, monkeypatch):
        def _boom(*a, **k):
            raise RuntimeError("connection refused")

        monkeypatch.setattr(checks.httpx, "get", _boom)

        result = checks.check_chat_endpoint(_config())
        assert result.status == "fail"
        assert "unreachable" in result.detail

    def test_an_error_status_that_is_not_503_still_counts_as_serving(self, monkeypatch):
        """A 404 on /models means the route is absent, not the server: some
        OpenAI-compatible servers do not implement it and work fine."""
        monkeypatch.setattr(checks.httpx, "get", lambda *a, **k: _Resp(404))

        assert checks.check_chat_endpoint(_config()).status == "ok"


class TestTheReranker:
    def test_a_loading_reranker_is_unknown_not_unreachable(self, monkeypatch):
        """"unreachable" sends the operator to start a server that is already
        starting; the honest reading is that it cannot be judged yet."""
        monkeypatch.setattr(checks.httpx, "post", lambda *a, **k: _Resp(503))
        monkeypatch.setattr(checks, "has_local_rerank", lambda: False)

        result = checks.check_rerank(_config(rerank_base_url="http://x:1235",
                                             rerank_model="bge"))

        assert result.status == "unknown"
        assert "loading" in result.detail.lower()

    def test_a_serving_reranker_is_still_ok(self, monkeypatch):
        monkeypatch.setattr(checks.httpx, "post", lambda *a, **k: _Resp(200))

        result = checks.check_rerank(_config(rerank_base_url="http://x:1235",
                                             rerank_model="bge"))
        assert result.status == "ok"

    def test_a_dead_reranker_still_warns(self, monkeypatch):
        def _boom(*a, **k):
            raise RuntimeError("connection refused")

        monkeypatch.setattr(checks.httpx, "post", _boom)
        monkeypatch.setattr(checks, "has_local_rerank", lambda: False)

        result = checks.check_rerank(_config(rerank_base_url="http://x:1235",
                                             rerank_model="bge"))
        assert result.status == "warn"
        assert "unreachable" in result.detail


class TestEveryStatusRenders:
    def test_no_check_can_emit_a_status_the_report_cannot_draw(self, monkeypatch):
        """A status without a glyph is a KeyError in render_report, i.e. the
        doctor crashing on the run that needed it."""
        for status in ("ok", "warn", "fail", "unknown"):
            assert status in checks._STATUS_GLYPH


class TestTheEmbeddingsEndpoint:
    def test_a_loading_embedder_is_unknown_not_a_rejection(self, monkeypatch):
        """The one probe without a 503 branch told the operator to change a
        correct SILICA_EMBEDDING_MODEL while llama-server was merely loading."""
        monkeypatch.setattr(checks.httpx, "post", lambda *a, **k: _Resp(503))

        result = checks.check_embeddings(
            _config(embedding_base_url="http://localhost:1234/v1",
                    embedding_model="emb"))

        assert result.status == "unknown"
        assert "loading" in result.detail.lower()

    def test_a_real_rejection_still_warns(self, monkeypatch):
        monkeypatch.setattr(checks.httpx, "post", lambda *a, **k: _Resp(404))

        result = checks.check_embeddings(
            _config(embedding_base_url="http://localhost:1234/v1",
                    embedding_model="emb"))

        assert result.status == "warn"


class TestTheExitCode:
    """0 = every row ok, 1 = a row failed, 2 = nothing failed but a row needs
    reading. Before this `warn` and `unknown` exited 0, so a script could not
    tell "clean" from "a fallback was taken" without parsing the table: the
    hold is the third verdict the report already drew but never returned."""

    def test_clean_is_zero(self):
        assert checks.exit_code([checks.CheckResult("a", "ok", "")]) == 0

    def test_a_warning_is_a_hold_not_a_pass(self):
        results = [checks.CheckResult("a", "ok", ""),
                   checks.CheckResult("b", "warn", "fallback taken")]
        assert checks.exit_code(results) == 2

    def test_unknown_is_a_hold_not_a_pass(self):
        assert checks.exit_code([checks.CheckResult("a", "unknown", "not probed")]) == 2

    def test_a_failure_beats_a_hold(self):
        results = [checks.CheckResult("a", "warn", ""),
                   checks.CheckResult("b", "fail", "broken")]
        assert checks.exit_code(results) == 1

    def test_the_payload_carries_the_same_verdict(self):
        """The MCP tool and --json read report_payload, not the exit code: an
        agent sent to silica_doctor must see the hold without re-deriving the
        policy from the rows. One resolver, three surfaces."""
        ok = checks.CheckResult("a", "ok", "")
        assert checks.report_payload([ok])["verdict"] == "ok"
        assert checks.report_payload([ok, checks.CheckResult("b", "warn", "")])["verdict"] == "hold"
        assert checks.report_payload([ok, checks.CheckResult("b", "fail", "")])["verdict"] == "fail"


class TestTheDeadRerankerHint:
    def test_names_the_autostart_variable(self, monkeypatch):
        """`silica mcp` already runs ensure_local_servers; it only starts the
        reranker when SILICA_RERANK_SERVE_CMD names a command. "start the
        reranker" sent the operator to a terminal for something the
        harness would do itself if told how (2026-09-03)."""
        def dead(*a, **k):
            raise ConnectionError("refused")
        monkeypatch.setattr(checks.httpx, "post", dead)
        monkeypatch.setattr(checks, "has_local_rerank", lambda: False)

        result = checks.check_rerank(_config(rerank_base_url="http://x:1235",
                                             rerank_model="bge"))

        assert result.status == "warn"
        assert "SILICA_RERANK_SERVE_CMD" in result.hint
