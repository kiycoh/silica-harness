"""Cross-encoder reranker: reorder logic (kernel/rerank) + client abstention
(agent/providers.Reranker). The reranker is a precision pass over an already-fused
pool; when absent or erroring it must be a pure no-op that preserves the pool order.
"""
from __future__ import annotations

import httpx

from silica.kernel.recall.rerank import (_best_window, best_window, best_windows,
                                          best_window_spans, rerank_related,
                                          window_weights)
from silica.agent.providers import FallbackReranker, LocalReranker, Reranker, get_reranker


class _Fake:
    def __init__(self, scores):
        self._scores = scores
        self.seen = None

    def scores(self, query, documents):
        self.seen = (query, list(documents))
        return self._scores


_ITEMS = [{"path": "a"}, {"path": "b"}, {"path": "c"}]
_DOC = lambda it: it["path"]


def test_reorders_by_score_within_pool():
    out = rerank_related(_Fake([0.2, 0.1, 0.9]), "q", _ITEMS, k=3, document_of=_DOC)
    assert [i["path"] for i in out] == ["c", "a", "b"]


def test_membership_belongs_to_first_stage():
    # Reorder-only (gate 2a): the pool is truncated to k BEFORE scoring, so the
    # reranker can neither evict a first-stage top-k member nor pull one in.
    # Re-tested 2026-08-25 (G4 depth-50): the deeper pool lost 4 gold contexts
    # and gained 0 on locomo (bench/g_depth.json) — membership stays first-stage.
    fake = _Fake([0.1, 0.9])
    out = rerank_related(fake, "q", _ITEMS, k=2, document_of=_DOC)
    assert [i["path"] for i in out] == ["b", "a"]   # reordered within top-k
    assert fake.seen[1] == ["a", "b"]               # c was never scored


def test_granularity_gate_skips_reranker_on_long_docs():
    # Gate 2b: when the median document dwarfs the cross-encoder window, its
    # ordering is noise — skip the call, keep first-stage order.
    from silica.kernel.recall import rerank as rr_mod

    long_doc = "x" * (rr_mod._RERANK_WINDOW_FACTOR * rr_mod._WINDOW_CHARS + 1)
    fake = _Fake([0.9, 0.1])
    out = rerank_related(fake, "q", _ITEMS, k=2, document_of=lambda it: long_doc)
    assert [i["path"] for i in out] == ["a", "b"]
    assert fake.seen is None                        # cross-encoder never called


def test_granularity_gate_lets_short_docs_rerank():
    fake = _Fake([0.1, 0.9])
    out = rerank_related(fake, "q", _ITEMS, k=2, document_of=_DOC)
    assert [i["path"] for i in out] == ["b", "a"]
    assert fake.seen is not None


def test_rerank_gate_probe_receives_median_and_fired(monkeypatch):
    from silica.kernel.recall import rerank as rr_mod

    seen = []
    monkeypatch.setattr(rr_mod, "RERANK_GATE_PROBE", seen.append)
    rerank_related(_Fake([0.1, 0.9]), "q", _ITEMS, k=2, document_of=_DOC)
    assert seen and seen[0]["fired"] is False
    assert seen[0]["median_len"] == 1               # docs are 1-char paths


def test_abstains_when_reranker_returns_none_keeps_order():
    out = rerank_related(_Fake(None), "q", _ITEMS, k=3, document_of=_DOC)
    assert [i["path"] for i in out] == ["a", "b", "c"]


def test_no_reranker_is_passthrough_truncated():
    out = rerank_related(None, "q", _ITEMS, k=1, document_of=_DOC)
    assert [i["path"] for i in out] == ["a"]


def test_empty_query_is_noop():
    out = rerank_related(_Fake([0.9, 0.0, 0.0]), "", _ITEMS, k=3, document_of=_DOC)
    assert [i["path"] for i in out] == ["a", "b", "c"]


def test_mismatched_score_length_abstains():
    # A malformed reranker reply must not silently drop/misalign candidates.
    out = rerank_related(_Fake([0.9]), "q", _ITEMS, k=3, document_of=_DOC)
    assert [i["path"] for i in out] == ["a", "b", "c"]


# --- query-aware document window (long-note blind-spot fix) ----------------

def test_window_surfaces_query_passage_past_head_slice():
    # The relevant turn sits far past a naive head slice; the window must find it
    # so the cross-encoder scores the right text instead of opening chatter.
    body = ("weather smalltalk " * 80) + "I attend yoga classes for my anxiety " \
           + ("unrelated filler " * 80)
    assert len(body) > 800
    win = _best_window(body, "how often do I attend yoga for anxiety?", 200)
    assert "yoga classes for my anxiety" in win


def test_window_is_noop_when_body_fits():
    assert _best_window("short body", "anything", 800) == "short body"


def test_window_falls_back_to_head_without_query_terms():
    long = "x" * 2000
    assert _best_window(long, "a an of", 500) == long[:500]  # no >3-char terms


# --- multi-window selection (multi-window spec 2026-07-15) -----------------

def test_best_windows_do_not_overlap():
    from silica.kernel.recall.rerank import best_windows

    body = ("zebra zebra ") + ("filler " * 100) + ("zebra " * 5) + ("filler " * 100)
    wins = best_windows(body, "zebra", 60, 2)
    assert len(wins) == 2
    spans = sorted((body.find(w), body.find(w) + len(w)) for w in wins)
    assert all(a_end <= b_start for (_, a_end), (b_start, _) in zip(spans, spans[1:]))


def test_best_windows_output_in_document_order():
    from silica.kernel.recall.rerank import best_windows

    # The denser region sits LAST in the document, so greedy picks it first;
    # the output must still follow document order (chat chronology).
    body = ("zebra zebra ") + ("filler " * 100) + ("zebra " * 5) + ("filler " * 100)
    wins = best_windows(body, "zebra", 60, 2)
    assert [w.count("zebra") for w in wins] == [2, 5]


def test_best_windows_whole_text_when_it_fits_the_budget():
    from silica.kernel.recall.rerank import best_windows

    text = "zebra " * 30  # 180 chars <= 2 * 100
    assert best_windows(text, "zebra", 100, 2) == [text]


def test_best_windows_head_slice_without_usable_terms():
    from silica.kernel.recall.rerank import best_windows

    text = "y" * 1000
    assert best_windows(text, "a an of it", 100, 2) == [text[:100]]


def test_best_windows_drops_second_window_without_hits():
    from silica.kernel.recall.rerank import best_windows

    # All hits in the head: never pad with irrelevant text, fewer windows is normal.
    body = ("zebra " * 4) + ("filler " * 200)
    wins = best_windows(body, "zebra", 60, 2)
    assert len(wins) == 1
    assert wins[0].count("zebra") == 4


def test_best_windows_n1_equals_best_window():
    from silica.kernel.recall.rerank import best_window, best_windows

    body = ("weather smalltalk " * 80) + "I attend yoga classes for my anxiety " \
           + ("unrelated filler " * 80)
    q = "how often do I attend yoga for anxiety?"
    assert best_windows(body, q, 200, 1) == [best_window(body, q, 200)]
    assert best_windows("short body", q, 800, 1) == [best_window("short body", q, 800)]
    long = "x" * 2000
    assert best_windows(long, "a an of", 500, 1) == [best_window(long, "a an of", 500)]


# --- link_query (orphan/link-path query gate) -----------------------------

def test_link_query_uses_informative_title_alone(monkeypatch):
    # Query-shape A/B (bench/local_rerank_excerpt_sweep.json): the bare title
    # reordered best; any body text in the query erased the gain.
    from silica.kernel.recall import rerank as rr_mod

    monkeypatch.setattr(rr_mod, "_read_body",
                        lambda p, **kw: ("Normatività", "body " * 300))
    assert rr_mod.link_query("Concepts/Normatività") == "Normatività"


def test_link_query_dated_title_falls_back_to_body_window(monkeypatch):
    # "2026-08-09" has no 3-letter word — rejected with no date regex; the
    # fallback is the measured bare body head (title-blind ablation: no
    # syntactic surrogate beat it, bench/local_rerank_title_blind.json).
    from silica.kernel.recall import rerank as rr_mod

    body = "meeting notes " * 100
    monkeypatch.setattr(rr_mod, "_read_body", lambda p, **kw: ("2026-08-09", body))
    assert rr_mod.link_query("inbox/2026-08-09") == body[:rr_mod._WINDOW_CHARS].strip()


def test_link_query_dated_title_with_a_word_stays_a_title(monkeypatch):
    from silica.kernel.recall import rerank as rr_mod

    monkeypatch.setattr(rr_mod, "_read_body",
                        lambda p, **kw: ("2026-08-09 riunione", "body text"))
    assert rr_mod.link_query("inbox/2026-08-09 riunione") == "2026-08-09 riunione"


def test_link_query_default_capture_names_fall_back(monkeypatch):
    from silica.kernel.recall import rerank as rr_mod

    for junk in ("Untitled", "Untitled 3", "New Note 12", "senza titolo 2"):
        monkeypatch.setattr(rr_mod, "_read_body", lambda p, **kw: (junk, "body text"))
        assert rr_mod.link_query("inbox/x") == "body text", junk


def test_link_query_unreadable_note_is_empty(monkeypatch):
    # '' propagates to rerank_related, which abstains on an empty query.
    from silica.kernel.recall import rerank as rr_mod

    monkeypatch.setattr(rr_mod, "_read_body", lambda p, **kw: ("", ""))
    assert rr_mod.link_query("inbox/x") == ""


# --- client ---------------------------------------------------------------

def test_client_parses_results_into_input_order(monkeypatch):
    def fake_post(url, **kwargs):
        return httpx.Response(200, request=httpx.Request("POST", url), json={"results": [
            {"index": 2, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.5},
            {"index": 1, "relevance_score": 0.1},
        ]})

    rr = Reranker(base_url="http://x/v1", model="m")
    monkeypatch.setattr(httpx, "post", fake_post)
    scores = rr.scores("q", ["d0", "d1", "d2"])
    assert scores == [0.5, 0.1, 0.9]   # remapped to input order


def test_client_abstains_on_transport_error(monkeypatch):
    def boom(*a, **k):
        raise httpx.ConnectError("down")

    rr = Reranker(base_url="http://x/v1", model="m")
    monkeypatch.setattr(httpx, "post", boom)
    assert rr.scores("q", ["d0"]) is None   # abstain, never raise


class _NoCfg:
    rerank_base_url = ""
    rerank_model = ""
    rerank_api_key = ""


def _extra(monkeypatch, present: bool):
    from silica.agent import providers as p
    monkeypatch.setattr(p, "has_local_rerank", lambda: present)


def test_get_reranker_disabled_without_config_or_extra(monkeypatch):
    _extra(monkeypatch, False)
    assert get_reranker(_NoCfg()) is None


def test_get_reranker_enabled_with_config(monkeypatch):
    """Config alone (no [rerank] extra) → a plain served Reranker, no fallback
    to wrap it in."""
    _extra(monkeypatch, False)

    class _Cfg:
        rerank_base_url = "http://x/v1"
        rerank_model = "bge-reranker"
        rerank_api_key = "k"
    rr = get_reranker(_Cfg())
    assert isinstance(rr, Reranker) and rr.url == "http://x/v1/rerank"


def test_get_reranker_falls_back_to_local_when_extra_installed(monkeypatch):
    """Neither LM Studio nor Ollama serves cross-encoders: with the [rerank]
    extra present and no endpoint configured, rerank works with no user setup."""
    _extra(monkeypatch, True)
    rr = get_reranker(_NoCfg())
    assert isinstance(rr, LocalReranker)


def test_served_endpoint_and_extra_wraps_in_fallback(monkeypatch):
    """Both a configured endpoint (default: local llama-server) and the extra
    installed → wrap in FallbackReranker, so a down endpoint degrades to the
    in-process cross-encoder per call instead of losing the pass entirely."""
    _extra(monkeypatch, True)

    class _Cfg:
        rerank_base_url = "http://x/v1"
        rerank_model = "bge-reranker"
        rerank_api_key = "k"
    rr = get_reranker(_Cfg())
    assert isinstance(rr, FallbackReranker)
    assert isinstance(rr.primary, Reranker) and rr.primary.url == "http://x/v1/rerank"
    assert isinstance(rr.secondary, LocalReranker)


def test_fallback_reranker_prefers_primary_when_up():
    primary = _Fake([0.1, 0.9])
    secondary = _Fake([0.5, 0.5])
    rr = FallbackReranker(primary, secondary)
    assert rr.scores("q", ["a", "b"]) == [0.1, 0.9]


def test_fallback_reranker_uses_secondary_when_primary_abstains():
    primary = _Fake(None)
    secondary = _Fake([0.5, 0.5])
    rr = FallbackReranker(primary, secondary)
    assert rr.scores("q", ["a", "b"]) == [0.5, 0.5]


def test_local_reranker_scores_in_input_order(monkeypatch):
    from silica.agent import providers as p

    class _Encoder:
        def predict(self, pairs):
            assert pairs == [["q", "d0"], ["q", "d1"]]  # [query, doc] cross-encoder pairs
            return [0.25, 0.75]
    monkeypatch.setattr(p, "_load_cross_encoder", lambda model: _Encoder())
    assert LocalReranker(model="m").scores("q", ["d0", "d1"]) == [0.25, 0.75]


def test_local_reranker_abstains_on_error(monkeypatch):
    """Same contract as the served client: abstain, never raise, never drop."""
    from silica.agent import providers as p

    def boom(model):
        raise RuntimeError("no weights")
    monkeypatch.setattr(p, "_load_cross_encoder", boom)
    assert LocalReranker(model="m").scores("q", ["d0"]) is None


def test_local_reranker_empty_input():
    assert LocalReranker(model="m").scores("", ["d"]) is None
    assert LocalReranker(model="m").scores("q", []) is None


def test_rerank_updates_scores_to_match_the_new_order():
    """Reordering while keeping first-stage scores made /find print
    0.033, 0.032, 0.031, 0.031, 0.032 — numbers the order contradicts.
    After a rerank the score IS the cross-encoder relevance."""
    from types import SimpleNamespace

    items = [SimpleNamespace(path=p, score=s)
             for p, s in (("a", 0.03), ("b", 0.02), ("c", 0.01))]
    out = rerank_related(_Fake([0.2, 0.1, 0.9]), "q", items, k=3,
                         document_of=lambda it: it.path)
    assert [i.path for i in out] == ["c", "a", "b"]
    assert [i.score for i in out] == [0.9, 0.2, 0.1]


def test_rerank_leaves_items_without_a_score_field_alone():
    out = rerank_related(_Fake([0.2, 0.9]), "q",
                         [{"path": "a"}, {"path": "b"}], k=2, document_of=_DOC)
    assert [i["path"] for i in out] == ["b", "a"]
    assert "score" not in out[0] and "score" not in out[1]


# --- G3: idf-weighted window scan (offline-signals-map §3) ---

def test_idf_weights_recentre_window_on_rare_term():
    filler = "pad " * 50                      # no query terms ("pad" is len 3)
    a = "model model model " + "pad " * 45    # 3 hits of the common term
    b = "sagredo " + "pad " * 47              # 1 hit of the rare term
    text = filler + a + filler + b + filler
    q = "model sagredo"
    unweighted = best_window(text, q, 200)
    assert "model" in unweighted and "sagredo" not in unweighted
    weighted = best_window(text, q, 200, {"model": 0.2, "sagredo": 5.0})
    assert "sagredo" in weighted


def test_empty_weights_stay_bit_identical():
    text = "alpha " * 30 + "gradient descent " * 5 + "omega " * 300
    q = "gradient descent"
    assert best_windows(text, q, 120, 2) == best_windows(text, q, 120, 2, {})
    assert best_windows(text, q, 120, 2) == best_windows(text, q, 120, 2, None)


def test_best_window_spans_offsets_are_verbatim_slices():
    text = "alpha " * 40 + "gradient descent here " + "omega " * 300
    spans = best_window_spans(text, "gradient descent", 150, 2)
    assert spans
    for p, s in spans:
        assert text[p:p + len(s)] == s


def test_window_weights_fail_open_without_store(monkeypatch):
    import silica.kernel.recall.lexical as lex

    def boom():
        raise RuntimeError("no vault bound")

    monkeypatch.setattr(lex, "get_lexical_store", boom)
    assert window_weights("gradient descent") == {}
