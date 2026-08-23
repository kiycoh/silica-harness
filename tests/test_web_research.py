"""web_search tool + web_research orchestrator (ADR-0015 staged acquisition).

No real network (the _no_network fixture below fails any unstubbed httpx call)
and no real LLM (run_agent is monkeypatched). Asserts: the DuckDuckGo primary
lane, the Mojeek, Tavily and Wikipedia backstops behind it, the lane line that
names a fallback on the note, compact result mapping, sensitivity, and the inbox
findings note.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import httpx
import pytest

from silica.config import CONFIG
from silica.sources import web_research as wr
from silica.tools import TOOLS


@pytest.fixture(autouse=True)
def _fresh_turn_state():
    """`_LANES` and the dead-lane counter are per-turn module globals, cleared by
    `_reset_turn()` at loop entry. A test calling `wr.web_search` directly never
    enters a loop, so without this a run of failed searches carries into the next
    test and the guard fires on a search that was meant to reach a lane."""
    wr._reset_turn()
    yield
    wr._reset_turn()


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Every lane is a stub or an error. Mojeek scrapes with httpx.get, so a
    test that stubs only httpx.post would reach the real Mojeek from a
    challenged DDG — the fixture turns that into a failure, not a slow pass."""

    def boom(url, *a, **kw):
        raise AssertionError(f"test reached the network: {url}")

    monkeypatch.setattr(wr.httpx, "get", boom)
    monkeypatch.setattr(wr.httpx, "post", boom)


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    """call_llm goes to a live provider, not through wr.httpx — a test that
    banks a quote would otherwise really compose (measured: a local
    llama-server answered one). Composition swallows this into the §3.6
    fallback, which is exactly the one-shot path those tests exercise; tests
    about composition install their own fake over this one."""

    def boom(*a, **kw):
        raise RuntimeError("test reached the LLM")

    monkeypatch.setattr(wr, "call_llm", boom)


# --- web_search tool --------------------------------------------------------

def test_web_search_registered_and_sensitive():
    assert "web_search" in TOOLS
    assert TOOLS["web_search"].sensitive is True


_DDG_HTML = """
<div class="result results_links web-result">
  <a rel="nofollow" class="result__a"
     href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fa.test%2Fpage&amp;rut=abc">Title <b>One</b></a>
  <a class="result__snippet" href="//duckduckgo.com/l/?uddg=x">Snippet <b>text</b> one.</a>
</div>
<div class="result result--ad">
  <a rel="nofollow" class="result__a"
     href="https://duckduckgo.com/y.js?ad_domain=x&amp;u3=enc">Ad title</a>
  <a class="result__snippet">Buy stuff.</a>
</div>
"""


class _FakeDDGResp:
    status_code = 200
    text = _DDG_HTML

    def raise_for_status(self):
        return self


def test_web_search_without_key_uses_duckduckgo(monkeypatch):
    """No key is not an error: the default backend scrapes DDG's HTML endpoint,
    unwraps the redirect hrefs, and drops the ad (whose href has no uddg)."""
    monkeypatch.setattr(CONFIG, "tavily_api_key", "")
    seen = {}

    def fake_post(url, data=None, headers=None, timeout=None):
        seen["url"] = url
        seen["data"] = data
        seen["headers"] = headers
        return _FakeDDGResp()

    monkeypatch.setattr(wr.httpx, "post", fake_post)

    items = json.loads(wr.web_search("graph theory"))

    assert seen["url"] == wr._DDG_URL
    assert seen["data"]["q"] == "graph theory"
    assert "Mozilla" in seen["headers"]["User-Agent"]  # browser UA, not httpx's
    assert items == [
        {"title": "Title One", "url": "https://a.test/page", "content": "Snippet text one."}
    ]


class _Challenged(_FakeDDGResp):
    """DDG's rate-limit answer: 202, which raise_for_status waves through as
    success, with a bare JavaScript shell for a body."""

    status_code = 202
    text = "prove you are human"


# Mojeek's results list, plus a chrome list carrying an <h2> anchor of its own:
# the parser is anchored on ul.results-standard, so the chrome must not become a
# hit. Selectors per searxng's mojeek engine (see _mojeek_search).
_MOJEEK_HTML = """
<ul class="nav-standard">
  <li><h2><a href="https://www.mojeek.com/about/">About Mojeek</a></h2></li>
</ul>
<ul class="results-standard">
  <li>
    <h2><a href="https://m1.test/page">Mojeek <b>One</b></a></h2>
    <a class="ob" href="https://m1.test/page">m1.test/page</a>
    <p class="s">Snippet one &amp; a bit.</p>
  </li>
  <li>
    <h2><a href="https://m2.test/">Mojeek Two</a></h2>
    <a class="ob" href="https://m2.test/">m2.test</a>
    <p class="s">Snippet two.</p>
  </li>
</ul>
"""


class _FakeMojeekResp:
    status_code = 200
    text = _MOJEEK_HTML


class _MojeekCaptcha:
    """Mojeek's challenge: HTTP *200* with a captcha page, so the status code
    cannot be the guard (measured 2026-07-30, every UA tried)."""

    status_code = 200
    text = '<html><head><title>Captcha</title></head><body>...</body></html>'


def _serve_lanes(monkeypatch, *, mojeek=None, wikipedia=None) -> list[tuple]:
    """Stub the two lanes that go through `_web_fetch._fetch`.

    Mojeek runs through it as well as Wikipedia — it is the fetcher that
    revalidates every redirect hop, and a raw `follow_redirects=True` get is
    exactly the SSRF hole it exists to close — so httpx.get no longer sees that
    lane at all. One fake serves both and dispatches on the URL; an Exception
    stands for a lane that is down, and a lane left unstubbed must not run.
    Returns the (url, headers) pairs it was called with.
    """
    seen: list[tuple] = []

    def fake_fetch(url, headers=None):
        seen.append((url, headers))
        resp = mojeek if url.startswith(wr._MOJEEK_URL) else wikipedia
        if resp is None:
            raise AssertionError(f"lane must not run: {url}")
        if isinstance(resp, Exception):
            raise resp
        return resp, url

    monkeypatch.setattr(wr._web_fetch, "_fetch", fake_fetch)
    return seen


def test_web_search_falls_back_to_mojeek_when_ddg_challenges(monkeypatch):
    """The keyless default keeps the open web when DDG challenges: Mojeek is its
    own crawl with its own rate limits, so it is tried before the encyclopedia."""
    monkeypatch.setattr(CONFIG, "tavily_api_key", "")
    monkeypatch.setattr(wr.httpx, "post", lambda *a, **k: _Challenged())
    seen = _serve_lanes(monkeypatch, mojeek=_FakeMojeekResp())

    items = json.loads(wr.web_search("graph theory"))

    assert seen[0][0] == f"{wr._MOJEEK_URL}?q=graph+theory"
    assert items == [
        {"title": "Mojeek One", "url": "https://m1.test/page",
         "content": "Snippet one & a bit."},
        {"title": "Mojeek Two", "url": "https://m2.test/", "content": "Snippet two."},
    ]


def test_mojeek_runs_ahead_of_a_set_key(monkeypatch):
    """Same posture as DDG-first: a keyless lane on its own index beats billing a
    vendor, so a healthy Mojeek means Tavily is never posted to."""
    monkeypatch.setattr(CONFIG, "tavily_api_key", "k-123")
    posted = []

    def fake_post(url, **kw):
        posted.append(url)
        return _Challenged()  # only DDG should be posted to at all

    monkeypatch.setattr(wr.httpx, "post", fake_post)
    _serve_lanes(monkeypatch, mojeek=_FakeMojeekResp())

    items = json.loads(wr.web_search("graph theory"))

    assert posted == [wr._DDG_URL]
    assert [i["url"] for i in items] == ["https://m1.test/page", "https://m2.test/"]


def test_mojeek_parse_does_not_depend_on_anchor_order(monkeypatch):
    """searxng's selectors say the title anchor is a sibling of the url anchor
    but not which comes first, and that is not verifiable from a captcha'd IP —
    so the parser takes the url from whichever anchor leads and the title only
    from the one inside the <h2>, and holds either way round."""
    swapped = _MOJEEK_HTML.replace(
        '<h2><a href="https://m1.test/page">Mojeek <b>One</b></a></h2>\n'
        '    <a class="ob" href="https://m1.test/page">m1.test/page</a>',
        '<a class="ob" href="https://m1.test/page">m1.test/page</a>\n'
        '    <h2><a href="https://m1.test/page">Mojeek <b>One</b></a></h2>',
    )
    assert swapped != _MOJEEK_HTML  # the replace matched

    class _Swapped(_FakeMojeekResp):
        text = swapped

    _serve_lanes(monkeypatch, mojeek=_Swapped())

    assert wr._mojeek_search("graph theory")[0] == {
        "title": "Mojeek One",
        "url": "https://m1.test/page",
        "content": "Snippet one & a bit.",
    }


def test_mojeek_captcha_and_empty_page_both_raise(monkeypatch):
    """A 200 captcha and a 200 whose markup no longer parses must both raise: a
    silent [] would spend the loop's whole budget on a lane that stopped
    answering and never reach the ones that still do."""
    _serve_lanes(monkeypatch, mojeek=_MojeekCaptcha())
    with pytest.raises(ValueError, match="challenged"):
        wr._mojeek_search("graph theory")

    class _Renamed(_FakeMojeekResp):
        text = _MOJEEK_HTML.replace("results-standard", "results-v2")

    _serve_lanes(monkeypatch, mojeek=_Renamed())
    with pytest.raises(ValueError, match="no parseable results"):
        wr._mojeek_search("graph theory")


class _FakeWPResp:
    def json(self):
        return {
            "query": {
                "search": [
                    {"title": "Graph theory",
                     "snippet": 'a <span class="searchmatch">graph</span> is a &amp; b'},
                    {"title": "PageRank", "snippet": "link analysis"},
                ]
            }
        }


def test_web_search_falls_back_to_wikipedia_when_ddg_challenges(monkeypatch):
    """Measured: DDG 202s from the third consecutive query and 20s of backoff
    does not clear it, while the loop is budgeted for 8-10. A challenge must
    degrade the lane, not end the research turn. Keyless with Mojeek challenged
    too, the encyclopedia is what is left."""
    monkeypatch.setattr(CONFIG, "tavily_api_key", "")
    monkeypatch.setattr(wr.httpx, "post", lambda *a, **k: _Challenged())
    seen = _serve_lanes(monkeypatch, mojeek=_MojeekCaptcha(), wikipedia=_FakeWPResp())

    items = json.loads(wr.web_search("graph theory"))

    # Both lanes now share one fetcher, and mojeek is tried first, so the
    # encyclopedia's call is picked out by URL rather than by position.
    wp = next(s for s in seen if s[0].startswith("https://en.wikipedia.org/w/api.php?"))
    assert "list=search" in wp[0] and "srsearch=graph+theory" in wp[0]
    assert "silica-harness" in wp[1]["User-Agent"]  # Wikimedia UA policy
    assert items == [
        {"title": "Graph theory",
         "url": "https://en.wikipedia.org/wiki/Graph_theory",
         "content": "a graph is a & b"},
        {"title": "PageRank",
         "url": "https://en.wikipedia.org/wiki/PageRank",
         "content": "link analysis"},
    ]


def test_web_search_double_failure_names_the_tavily_escape_hatch(monkeypatch):
    """When the fallback is down too, the surfaced error is DDG's, the
    one that tells the user their way out, and Wikipedia's would bury it."""
    monkeypatch.setattr(CONFIG, "tavily_api_key", "")
    monkeypatch.setattr(wr.httpx, "post", lambda *a, **k: _Challenged())
    monkeypatch.setattr(wr.httpx, "get", lambda *a, **k: _MojeekCaptcha())

    def boom(*a, **k):
        raise ValueError("cannot resolve 'en.wikipedia.org'")

    monkeypatch.setattr(wr._web_fetch, "_fetch", boom)
    with pytest.raises(ValueError, match="TAVILY"):
        wr.web_search("anything")


def _all_lanes_down(monkeypatch) -> list[int]:
    """Stub every lane into failing. Returns the list that counts DDG dials."""
    def boom(*a, **k):
        raise ValueError("cannot resolve 'en.wikipedia.org'")

    dialled: list[int] = []
    monkeypatch.setattr(CONFIG, "tavily_api_key", "")
    monkeypatch.setattr(wr.httpx, "post", lambda *a, **k: (dialled.append(1), _Challenged())[1])
    monkeypatch.setattr(wr.httpx, "get", lambda *a, **k: _MojeekCaptcha())
    monkeypatch.setattr(wr._web_fetch, "_fetch", boom)
    return dialled


def test_web_search_stops_dialling_once_every_lane_is_down(monkeypatch):
    """A search that exhausts every lane pays up to four HTTP timeouts, and the
    convergence guard in run_agent cannot catch the pattern because it keys on
    identical arguments and a research loop never repeats a query. At a ceiling
    of 48 that is half an hour of dialling a dead stack, so the tool gives up on
    its own after _DEAD_LANES_LIMIT and fails without touching the network."""
    dialled = _all_lanes_down(monkeypatch)

    for _ in range(wr._DEAD_LANES_LIMIT):
        with pytest.raises(ValueError, match="DuckDuckGo answered"):
            wr.web_search("q")
    spent = len(dialled)
    assert spent == wr._DEAD_LANES_LIMIT

    with pytest.raises(ValueError, match="exhausted every lane"):
        wr.web_search("q")
    assert len(dialled) == spent, "the guard fired after dialling anyway"


def test_one_answering_lane_resets_the_dead_lane_guard(monkeypatch):
    """Consecutive, not cumulative: a rate limit that lifts mid-run must not
    leave the loop counting down to a shutdown it no longer needs."""
    dialled = _all_lanes_down(monkeypatch)

    for _ in range(wr._DEAD_LANES_LIMIT - 1):
        with pytest.raises(ValueError, match="DuckDuckGo answered"):
            wr.web_search("q")

    monkeypatch.setattr(wr.httpx, "post", lambda *a, **k: _FakeDDGResp())
    assert json.loads(wr.web_search("q"))          # DDG is back
    monkeypatch.setattr(wr.httpx, "post", lambda *a, **k: (dialled.append(1), _Challenged())[1])

    # Back to a dead stack: the counter restarts, so this is a lane error again.
    with pytest.raises(ValueError, match="DuckDuckGo answered"):
        wr.web_search("q")


class _FakeTavilyResp:
    def raise_for_status(self):
        return self

    def json(self):
        return {
            "results": [
                {"title": "T1", "url": "https://a.test", "content": "c1", "score": 0.9},
                {"title": "T2", "url": "https://b.test", "content": "c2"},
            ]
        }


def test_web_search_prefers_ddg_over_tavily_when_a_key_is_set(monkeypatch):
    """A key is a backstop, not a switch: DDG is the primary lane, so a healthy
    DDG answers and Tavily is never billed."""
    monkeypatch.setattr(CONFIG, "tavily_api_key", "k-123")
    posted = []

    def fake_post(url, **kw):
        posted.append(url)
        return _FakeDDGResp()

    monkeypatch.setattr(wr.httpx, "post", fake_post)

    items = json.loads(wr.web_search("graph theory"))

    assert posted == [wr._DDG_URL]
    assert [i["url"] for i in items] == ["https://a.test/page"]


def test_web_search_posts_to_tavily_when_ddg_challenges(monkeypatch):
    """With a key, Tavily takes over from a challenged DDG once the keyless
    Mojeek lane is challenged too, and still ahead of the Wikipedia lane, which
    stays the last resort."""
    monkeypatch.setattr(CONFIG, "tavily_api_key", "k-123")
    monkeypatch.setattr(wr.httpx, "get", lambda *a, **k: _MojeekCaptcha())
    seen = {}

    def fake_post(url, json=None, data=None, headers=None, timeout=None):
        if url == wr._DDG_URL:
            return _Challenged()
        seen["url"] = url
        seen["body"] = json
        seen["timeout"] = timeout
        return _FakeTavilyResp()

    monkeypatch.setattr(wr.httpx, "post", fake_post)
    monkeypatch.setattr(wr._web_fetch, "_fetch", _wikipedia_must_not_run)

    items = json.loads(wr.web_search("graph theory"))

    assert seen["url"] == wr._TAVILY_URL
    assert seen["body"]["api_key"] == "k-123"
    assert seen["body"]["query"] == "graph theory"
    assert seen["body"]["max_results"] == wr._MAX_RESULTS
    assert items == [
        {"title": "T1", "url": "https://a.test", "content": "c1"},
        {"title": "T2", "url": "https://b.test", "content": "c2"},
    ]


def test_web_search_falls_through_tavily_to_wikipedia(monkeypatch):
    """A key that is expired or a Tavily outage must not end the turn: the
    keyless lane is still there behind it."""
    monkeypatch.setattr(CONFIG, "tavily_api_key", "k-123")
    monkeypatch.setattr(wr.httpx, "get", lambda *a, **k: _MojeekCaptcha())

    def fake_post(url, **kw):
        if url == wr._DDG_URL:
            return _Challenged()
        raise httpx.HTTPError("tavily 401")

    monkeypatch.setattr(wr.httpx, "post", fake_post)
    monkeypatch.setattr(wr._web_fetch, "_fetch", lambda url, headers=None: (_FakeWPResp(), url))

    items = json.loads(wr.web_search("graph theory"))

    assert [i["url"] for i in items] == [
        "https://en.wikipedia.org/wiki/Graph_theory",
        "https://en.wikipedia.org/wiki/PageRank",
    ]


def _wikipedia_must_not_run(url, headers=None):
    raise AssertionError(f"Wikipedia lane ran while Tavily was available: {url}")


# --- web_research orchestrator ----------------------------------------------

def _patch_run_agent(monkeypatch, body, tool_results=None):
    """Fake run_agent: replay a web_search trace the way the real loop does —
    a ToolCompleteEvent per call *and* the same payload appended to `messages`
    — then return the body."""
    from silica.agent.events import ToolCompleteEvent

    captured = {}

    def fake_run_agent(messages, model, tool_progress_callback=None, constraints=None, **kw):
        captured["constraints"] = constraints
        captured["messages"] = messages
        captured["model"] = model
        for i, items in enumerate(tool_results or []):
            call_id = f"c{i}"
            payload = json.dumps(items)
            if tool_progress_callback is not None:
                tool_progress_callback(ToolCompleteEvent(
                    name="web_search", args={"query": "q"}, call_id=call_id,
                    result=payload, duration_s=0.0, iteration=i + 1,
                ))
            messages.append(
                {"role": "tool", "tool_call_id": call_id, "content": payload}
            )
        return body

    monkeypatch.setattr(wr, "run_agent", fake_run_agent)
    return captured


def test_web_research_writes_inbox_note_with_deterministic_frontmatter(tmp_vault, monkeypatch):
    monkeypatch.setattr(CONFIG, "tavily_api_key", "k")
    _patch_run_agent(
        monkeypatch,
        body="Findings about graph theory [1][2].",
        tool_results=[[
            {"title": "T1", "url": "https://a.test", "content": "c1"},
            {"title": "T2", "url": "https://b.test", "content": "c2"},
        ]],
    )

    note_rel = wr.web_research("graph theory")

    assert note_rel.startswith(f"{CONFIG.inbox_dir}/")
    body = (Path(CONFIG.vault_path) / note_rel).read_text(encoding="utf-8")
    today = datetime.date.today().isoformat()
    assert 'title: "graph theory"' in body
    assert "source: web-research" in body
    assert f"fetched: {today}" in body
    assert "tags: [inbox, web-research]" in body
    assert "Findings about graph theory" in body


def test_web_research_appends_sources_when_model_omits_them(tmp_vault, monkeypatch):
    monkeypatch.setattr(CONFIG, "tavily_api_key", "k")
    _patch_run_agent(
        monkeypatch,
        body="Findings with no sources section.",  # model forgot ## Sources
        tool_results=[[
            {"title": "T1", "url": "https://a.test", "content": "c1"},
        ]],
    )

    body = (Path(CONFIG.vault_path) / wr.web_research("x")).read_text(encoding="utf-8")
    assert body.count("## Sources") == 1
    assert "https://a.test" in body


def test_web_research_keeps_model_sources_section(tmp_vault, monkeypatch):
    monkeypatch.setattr(CONFIG, "tavily_api_key", "k")
    _patch_run_agent(
        monkeypatch,
        body="Findings [1].\n\n## Sources\n1. T1 — https://a.test",
        tool_results=[[{"title": "T1", "url": "https://a.test", "content": "c1"}]],
    )

    body = (Path(CONFIG.vault_path) / wr.web_research("x")).read_text(encoding="utf-8")
    assert body.count("## Sources") == 1  # not doubled


def test_web_research_constrains_loop_to_search_and_fetch(tmp_vault, monkeypatch):
    monkeypatch.setattr(CONFIG, "tavily_api_key", "k")
    captured = _patch_run_agent(
        monkeypatch,
        body="Findings.",
        tool_results=[[{"title": "T1", "url": "https://a.test", "content": "c"}]],
    )

    wr.web_research("x", max_searches=7)

    assert captured["constraints"].tools == (
        "web_search", "web_fetch", "remember", "find_in_page", "plan"
    )
    assert captured["constraints"].max_iterations == 7


def test_steering_off_restores_the_pre_plan_loop(tmp_vault, monkeypatch):
    """Gate arm A (spec §6): flipping _STEERING must remove both the tool
    and the prompt step, or arm A tells the model to call a tool it does
    not have."""
    monkeypatch.setattr(CONFIG, "tavily_api_key", "k")
    monkeypatch.setattr(wr, "_STEERING", False)
    captured = _patch_run_agent(monkeypatch, body="Findings.")
    wr.web_research("q")
    assert captured["constraints"].tools == (
        "web_search", "web_fetch", "remember", "find_in_page"
    )
    assert "plan(" not in captured["messages"][0]["content"]


def test_steering_on_puts_the_plan_step_in_the_prompt(tmp_vault, monkeypatch):
    monkeypatch.setattr(CONFIG, "tavily_api_key", "k")
    captured = _patch_run_agent(monkeypatch, body="Findings.")
    wr.web_research("q")
    assert "plan(" in captured["messages"][0]["content"]


def _patch_run_agent_calling_web_search(monkeypatch, calls):
    """Fake run_agent that goes through the real web_search, so the lanes on the
    note are the ones that actually answered rather than a hand-set list."""

    def fake_run_agent(messages, model, tool_progress_callback=None, constraints=None, **kw):
        for _ in range(calls):
            wr.web_search("graph theory")
        return "Findings [1]."

    monkeypatch.setattr(wr, "run_agent", fake_run_agent)


def test_web_research_note_names_the_lanes_that_answered(tmp_vault, monkeypatch):
    """The loud half: a note whose sources came from a fallback says which lane
    and how many calls, so a thin answer is legible as a challenged primary lane
    rather than as a thin web."""
    monkeypatch.setattr(CONFIG, "tavily_api_key", "")
    monkeypatch.setattr(wr.httpx, "post", lambda *a, **k: _Challenged())
    _serve_lanes(monkeypatch, mojeek=_FakeMojeekResp())
    _patch_run_agent_calling_web_search(monkeypatch, calls=2)

    body = (Path(CONFIG.vault_path) / wr.web_research("graph theory")).read_text(
        encoding="utf-8"
    )

    assert "Search lanes: mojeek 2. DuckDuckGo was challenged" in body


def test_web_research_note_stays_quiet_when_ddg_answered(tmp_vault, monkeypatch):
    """No banner on a healthy note: the line is the fallback's, not a status
    report on every turn."""
    monkeypatch.setattr(CONFIG, "tavily_api_key", "")
    monkeypatch.setattr(wr.httpx, "post", lambda *a, **k: _FakeDDGResp())
    _patch_run_agent_calling_web_search(monkeypatch, calls=2)

    body = (Path(CONFIG.vault_path) / wr.web_research("graph theory")).read_text(
        encoding="utf-8"
    )

    assert "Search lanes" not in body


def test_lane_line_is_per_turn_not_cumulative(tmp_vault, monkeypatch):
    """A second turn must not inherit the first one's lanes: web_research clears
    the record before the loop, so a recovered DDG stops reporting a fallback."""
    monkeypatch.setattr(CONFIG, "tavily_api_key", "")
    monkeypatch.setattr(wr.httpx, "post", lambda *a, **k: _Challenged())
    _serve_lanes(monkeypatch, mojeek=_FakeMojeekResp())
    _patch_run_agent_calling_web_search(monkeypatch, calls=1)
    wr.web_research("graph theory")

    monkeypatch.setattr(wr.httpx, "post", lambda *a, **k: _FakeDDGResp())
    body = (Path(CONFIG.vault_path) / wr.web_research("second turn")).read_text(
        encoding="utf-8"
    )

    assert "Search lanes" not in body


def test_web_research_prompt_tells_the_model_to_fetch():
    assert "web_fetch" in wr._RESEARCH_SYSTEM_PROMPT


def test_collect_sources_picks_up_a_fetched_url():
    """web_fetch returns prose, not JSON; its Source: line is the citation."""
    results = [
        json.dumps([{"title": "T1", "url": "https://a.test", "content": "c"}]),
        "Source: https://b.test/article\n\nBody text.",
    ]
    assert wr._collect_sources(results) == [
        ("https://a.test", "T1"),
        ("https://b.test/article", "https://b.test/article"),
    ]


def test_collect_sources_ignores_prose_without_a_source_line():
    assert wr._collect_sources(["just some text\nno header"]) == []


# --- compaction cannot reach the trace the leaf and the citations are built from


def _patch_run_agent_then_compact(monkeypatch, body, fetches):
    """run_agent double that behaves like the real loop on a long research run.

    It emits a ToolCompleteEvent per tool call (as silica/agent/loop.py does,
    before anything can rewrite the message), appends the same result to
    `messages`, and then lets the *real* compaction sweep run over that history.
    Past the recency floor the sweep replaces each fat web_fetch result with an
    elision stub in place, which is exactly what a caller reading `messages`
    after run_agent returns would find.
    """
    from silica.agent.compaction import COMPACT_FLOOR_TURNS, compact_read_history
    from silica.agent.events import ToolCompleteEvent
    from silica.tools import TOOLS

    def fake_run_agent(messages, model, tool_progress_callback=None, constraints=None, **kw):
        for i, (url, text) in enumerate(fetches):
            call_id = f"call-{i}"
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "web_fetch",
                                 "arguments": json.dumps({"url": url})},
                }],
            })
            if tool_progress_callback is not None:
                tool_progress_callback(ToolCompleteEvent(
                    name="web_fetch", args={"url": url}, call_id=call_id,
                    result=text, duration_s=0.1, iteration=i + 1,
                ))
            messages.append({"role": "tool", "tool_call_id": call_id, "content": text})
        messages.append({"role": "assistant", "content": body})
        compact_read_history(
            messages, set(), prompt_tokens=10**9, budget=0,
            floor_turns=COMPACT_FLOOR_TURNS, tools=TOOLS,
        )
        return body

    monkeypatch.setattr(wr, "run_agent", fake_run_agent)


_PAGES = [
    (f"https://p{i}.test/article", f"Source: https://p{i}.test/article\n\nPage {i} title\n\n"
     + f"body of page {i}. " * 40)
    for i in range(5)
]


def test_web_research_leaf_survives_context_compaction(tmp_vault, monkeypatch):
    """web_fetch is `collapse="lazy"` and ~7.5k tokens a call, so a handful of
    fetches trips run_agent's compaction sweep, which rewrites the old tool
    results in `messages` to elision stubs *in place*. A leaf built by reading
    `messages` after the loop returns is then a list of stubs, not the pages."""
    from silica.kernel.recall.paths import SOURCES_DIR

    monkeypatch.setattr(CONFIG, "tavily_api_key", "k")
    _patch_run_agent_then_compact(monkeypatch, body="Findings.", fetches=_PAGES)

    note_rel = wr.web_research("deep topic")
    leaf = (Path(CONFIG.vault_path) / SOURCES_DIR / note_rel.rsplit("/", 1)[-1]
            ).read_text(encoding="utf-8")

    assert "result elided" not in leaf
    for i in range(len(_PAGES)):
        assert f"body of page {i}." in leaf


def test_web_research_citations_survive_context_compaction(tmp_vault, monkeypatch):
    """Same sweep, other casualty: the ADR-0015 ## Sources fallback is built
    from the same trace, so the elided fetches lose their URLs entirely."""
    monkeypatch.setattr(CONFIG, "tavily_api_key", "k")
    _patch_run_agent_then_compact(monkeypatch, body="Findings.", fetches=_PAGES)

    body = (Path(CONFIG.vault_path) / wr.web_research("deep topic")).read_text(
        encoding="utf-8")

    assert body.count("## Sources") == 1
    for url, _ in _PAGES:
        assert url in body


def test_web_research_still_forwards_progress_events_to_its_caller(tmp_vault, monkeypatch):
    """The recorder wraps the caller's callback; it must not swallow it."""
    monkeypatch.setattr(CONFIG, "tavily_api_key", "k")
    _patch_run_agent_then_compact(monkeypatch, body="Findings.", fetches=_PAGES[:1])

    seen = []
    wr.web_research("x", tool_progress_callback=seen.append)
    assert [e.call_id for e in seen] == ["call-0"]


def test_main_agent_default_toolset_excludes_web_fetch():
    from unittest.mock import patch
    from types import SimpleNamespace
    from silica.agent.loop import run_agent

    # "web_fetch" in TOOLS alone doesn't pin edit 3a: tests/test_web_fetch.py
    # imports the module at collection time too, so TOOLS would be populated
    # even without wr's own import. Assert the attribute wr._web_fetch itself
    # carries, which only holds if web_research.py did the import (edit 3a).
    assert wr._web_fetch.web_fetch.__name__ == "web_fetch"

    captured = {}

    def fake_call_llm(model, messages, tools=None, cancel=None):
        captured["tools"] = tools
        return SimpleNamespace(
            assistant_message={"role": "assistant", "content": "ok"},
            tool_calls=[], text="ok", reasoning=None, usage={},
        )

    with patch("silica.agent.loop.call_llm", fake_call_llm):
        run_agent(messages=[{"role": "user", "content": "hi"}], model="m")

    names = {t["function"]["name"] for t in (captured["tools"] or [])}
    assert "web_fetch" not in names


def test_web_research_no_findings_raises_and_writes_nothing(tmp_vault, monkeypatch):
    monkeypatch.setattr(CONFIG, "tavily_api_key", "k")
    _patch_run_agent(monkeypatch, body="(silica: maximum iterations reached)")

    with pytest.raises(ValueError, match="no findings"):
        wr.web_research("x")
    inbox = Path(CONFIG.vault_path) / CONFIG.inbox_dir
    assert not inbox.exists() or not list(inbox.glob("*.md"))


def test_web_research_runs_without_a_key(tmp_vault, monkeypatch):
    """No Tavily key no longer fails fast: web_search falls back to DDG."""
    monkeypatch.setattr(CONFIG, "tavily_api_key", "")
    _patch_run_agent(
        monkeypatch,
        body="Findings.",
        tool_results=[[{"title": "T", "url": "https://a.test", "content": "c"}]],
    )

    note_rel = wr.web_research("x")
    assert (Path(CONFIG.vault_path) / note_rel).exists()


def test_web_research_empty_body_raises_and_writes_nothing(tmp_vault, monkeypatch):
    monkeypatch.setattr(CONFIG, "tavily_api_key", "k")
    _patch_run_agent(monkeypatch, body="")

    with pytest.raises(ValueError, match="no findings"):
        wr.web_research("x")
    inbox = Path(CONFIG.vault_path) / CONFIG.inbox_dir
    assert not inbox.exists() or not list(inbox.glob("*.md"))


def test_web_research_sources_section_nonempty_when_no_sources(tmp_vault, monkeypatch):
    monkeypatch.setattr(CONFIG, "tavily_api_key", "k")
    _patch_run_agent(monkeypatch, body="Findings with no sources and no trace.", tool_results=[])

    body = (Path(CONFIG.vault_path) / wr.web_research("x")).read_text(encoding="utf-8")
    assert body.count("## Sources") == 1
    assert "(no sources captured)" in body


def test_web_research_title_with_colon_is_valid_yaml(tmp_vault, monkeypatch):
    """A concept containing a colon must produce parseable YAML frontmatter."""
    import yaml

    monkeypatch.setattr(CONFIG, "tavily_api_key", "k")
    _patch_run_agent(
        monkeypatch,
        body="Findings about RAG.",
        tool_results=[[{"title": "T1", "url": "https://a.test", "content": "c1"}]],
    )

    note_rel = wr.web_research("RAG: a survey")
    body = (Path(CONFIG.vault_path) / note_rel).read_text(encoding="utf-8")

    # Extract the frontmatter block between the first two --- delimiters
    parts = body.split("---\n", 2)
    assert len(parts) >= 3, "frontmatter delimiters not found"
    fm_block = parts[1]
    fm = yaml.safe_load(fm_block)
    assert fm["title"] == "RAG: a survey"
    # Ensure the malformed bare form is not present
    assert "title: RAG: a survey\n" not in body


# --- ADR-0015 / ADR-0009 boundary, as wired in production --------------------

def test_main_agent_default_toolset_excludes_web_search():
    """With web_search registered (module imported), run_agent without
    constraints must NOT expose it to the main agent."""
    from unittest.mock import patch
    from types import SimpleNamespace
    from silica.agent.loop import run_agent

    assert "web_search" in TOOLS  # registered by importing this module's target

    captured = {}

    def fake_call_llm(model, messages, tools=None, cancel=None):
        captured["tools"] = tools
        return SimpleNamespace(
            assistant_message={"role": "assistant", "content": "ok"},
            tool_calls=[], text="ok", reasoning=None, usage={},
        )

    with patch("silica.agent.loop.call_llm", fake_call_llm):
        run_agent(messages=[{"role": "user", "content": "hi"}], model="m")

    names = {t["function"]["name"] for t in (captured["tools"] or [])}
    assert "web_search" not in names


# --- /fetch ------------------------------------------------------------------

def _patch_web_fetch(monkeypatch, text, title=""):
    """Stand in for the fetch at the seam `fetch_to_inbox` actually calls.

    `title=""` means the page declared none, which is the first-line fallback
    these tests were written against.
    """
    import silica.sources.web_fetch as wf
    monkeypatch.setattr(wf, "fetch_page", lambda url: wf.Page(text, title))


def test_fetch_to_inbox_writes_a_note_titled_after_the_page(tmp_vault, monkeypatch):
    _patch_web_fetch(
        monkeypatch,
        "Source: https://a.test/post\n\nOn Graph Theory\n\nBody prose here.",
    )

    note_rel = wr.fetch_to_inbox("https://a.test/post")

    assert note_rel.startswith(f"{CONFIG.inbox_dir}/")
    body = (Path(CONFIG.vault_path) / note_rel).read_text(encoding="utf-8")
    today = datetime.date.today().isoformat()
    assert 'title: "On Graph Theory"' in body
    assert "source: web-fetch" in body
    assert "tags: [inbox, web-fetch]" in body
    assert f"fetched: {today}" in body
    assert "Body prose here." in body
    # the ADR-0015 sources guarantee: a ## Sources block naming this URL, not
    # merely the URL appearing somewhere in the fetched text we echo verbatim
    assert "## Sources" in body
    assert "1. On Graph Theory — https://a.test/post" in body


def test_fetch_to_inbox_prefers_the_declared_title_over_the_first_line(
    tmp_vault, monkeypatch
):
    """The note title is the reranker's query and the filename slug. When the
    page declares one, the lead sentence must not win it."""
    _patch_web_fetch(
        monkeypatch,
        "Source: https://a.test/post\n\nGraphs are everywhere, and this post "
        "argues that they are also unavoidable.\n\nBody prose here.",
        title="On Graph Theory",
    )

    note_rel = wr.fetch_to_inbox("https://a.test/post")

    assert note_rel == f"{CONFIG.inbox_dir}/On Graph Theory.md"
    body = (Path(CONFIG.vault_path) / note_rel).read_text(encoding="utf-8")
    assert 'title: "On Graph Theory"' in body
    assert "1. On Graph Theory — https://a.test/post" in body
    # the lead sentence is still in the body, just not in the title
    assert "Graphs are everywhere" in body


def test_fetch_to_inbox_assumes_https_for_a_bare_domain(tmp_vault, monkeypatch):
    """`/fetch en.wikipedia.org/wiki/X` is how humans type URLs. The scheme is
    inferred here at the user-facing seam, so the strict guard in web_fetch
    still validates the full https form (and agent calls stay strict)."""
    import silica.sources.web_fetch as wf
    seen = {}

    def fake(url):
        seen["url"] = url
        return wf.Page("Source: https://a.test/post\n\nOn Graph Theory\n\nBody.", "")

    monkeypatch.setattr(wf, "fetch_page", fake)

    body = (Path(CONFIG.vault_path) / wr.fetch_to_inbox("a.test/post")).read_text(
        encoding="utf-8"
    )

    assert seen["url"] == "https://a.test/post"
    # the citation carries the URL actually fetched, not the schemeless input
    assert "1. On Graph Theory — https://a.test/post" in body


def test_fetch_to_inbox_sources_block_cannot_be_spoofed_by_the_page(tmp_vault, monkeypatch):
    """For /fetch the note body IS the fetched page, so a page that happens to
    contain its own `## Sources` heading (any markdown README does) would
    otherwise suppress ours and leave a reviewer looking at an attacker-authored
    Sources section. ADR-0015 makes sources mandatory, not content-dependent."""
    _patch_web_fetch(
        monkeypatch,
        "Source: https://raw.example.test/README.md\n\nAwesome Thing\n\n"
        "Prose.\n\n## Sources\n1. Somebody else — https://evil.test/theirs\n",
    )

    body = (Path(CONFIG.vault_path) / wr.fetch_to_inbox(
        "https://raw.example.test/README.md")).read_text(encoding="utf-8")

    assert "1. Awesome Thing — https://raw.example.test/README.md" in body


def test_fetch_to_inbox_falls_back_to_its_own_namespace(tmp_vault, monkeypatch):
    """A title that slugifies to nothing must not land on web-research.md and
    push the other command's next note to `web-research 2.md`."""
    _patch_web_fetch(monkeypatch, "Source: https://a.test/p\n\n***\n\nBody.")

    assert wr.fetch_to_inbox("https://a.test/p") == f"{CONFIG.inbox_dir}/web-fetch.md"


def test_fetch_to_inbox_writes_a_source_leaf(tmp_vault, monkeypatch):
    from silica.kernel.recall.paths import SOURCES_DIR

    _patch_web_fetch(monkeypatch, "Source: https://a.test/post\n\nTitle\n\nBody.")
    note_rel = wr.fetch_to_inbox("https://a.test/post")

    leaf = Path(CONFIG.vault_path) / SOURCES_DIR / note_rel.rsplit("/", 1)[-1]
    assert leaf.exists()
    assert "Body." in leaf.read_text(encoding="utf-8")


def test_fetch_to_inbox_rejects_a_page_with_no_body(tmp_vault, monkeypatch):
    """A login wall can extract down to nothing but our own Source: header."""
    _patch_web_fetch(monkeypatch, "Source: https://a.test/post\n\n")
    with pytest.raises(ValueError, match="nothing readable"):
        wr.fetch_to_inbox("https://a.test/post")
    inbox = Path(CONFIG.vault_path) / CONFIG.inbox_dir
    assert not inbox.exists() or not list(inbox.glob("*.md"))


def test_fetch_to_inbox_propagates_the_fetch_error(tmp_vault, monkeypatch):
    import silica.sources.web_fetch as wf

    def boom(url):
        raise ValueError("403 at https://a.test: bot wall")

    monkeypatch.setattr(wf, "fetch_page", boom)
    with pytest.raises(ValueError, match="403"):
        wr.fetch_to_inbox("https://a.test/post")
    inbox = Path(CONFIG.vault_path) / CONFIG.inbox_dir
    assert not inbox.exists() or not list(inbox.glob("*.md"))


def test_fetch_to_inbox_does_not_attach_a_stale_leaf_after_nucleate(tmp_vault, monkeypatch):
    """A note's sources/ leaf outlives /nucleate consuming the note itself
    (by design). A later /fetch that happens to produce the same title must
    not inherit that unrelated, stale leaf for its own new note."""
    from silica.kernel.recall.paths import SOURCES_DIR

    _patch_web_fetch(monkeypatch, "Source: https://a.test/first\n\nSame Title\n\nPage A body.")
    note_a = wr.fetch_to_inbox("https://a.test/first")
    (Path(CONFIG.vault_path) / note_a).unlink()  # simulate /nucleate consuming the note

    _patch_web_fetch(monkeypatch, "Source: https://b.test/second\n\nSame Title\n\nPage B body.")
    note_b = wr.fetch_to_inbox("https://b.test/second")

    leaf = Path(CONFIG.vault_path) / SOURCES_DIR / note_b.rsplit("/", 1)[-1]
    content = leaf.read_text(encoding="utf-8")
    assert "Page B body." in content
    assert "Page A body." not in content


# --- the CLI lines, which interpolate untrusted text into Rich markup --------


def _run_cli(cmd: str) -> str:
    """Dispatch one REPL command, returning what the user would see."""
    from silica import cli
    from silica.ui.console import CONSOLE

    with CONSOLE.capture() as capture:
        assert cli._expand_workflow_shortcut(cmd) == ""  # handled inline
    return capture.get()


def test_fetch_failure_line_survives_a_url_that_looks_like_rich_markup(
    tmp_vault, monkeypatch
):
    """A URL carrying `[/x]` used to raise MarkupError out of the very except
    that exists to report the failure, so the user got a traceback."""
    import silica.sources.web_fetch as wf

    def boom(url):
        raise ValueError(f"cannot resolve {url!r}: nope")

    monkeypatch.setattr(wf, "fetch_page", boom)

    out = _run_cli("/fetch https://a.test/[/x]")
    assert "fetch failed" in out
    assert "[/x]" in out  # shown verbatim, not swallowed as a closing tag


def test_fetch_success_line_shows_the_real_path(tmp_vault, monkeypatch):
    """slugify strips `\\ / : * ? " < > |` but not brackets, so a page titled
    `[bold red]Foo` yields a note_rel whose markup Rich would silently eat,
    telling the user a path that is not the file's name."""
    _patch_web_fetch(monkeypatch, "Source: https://a.test/p\n\n[bold red]Foo\n\nBody.")

    out = _run_cli("/fetch https://a.test/p")
    assert "[bold red]Foo.md" in out


def test_web_search_failure_line_survives_rich_markup(tmp_vault, monkeypatch):
    """The identical defect two lines away in the sibling command."""
    monkeypatch.setattr(CONFIG, "tavily_api_key", "k")

    def boom(*a, **kw):
        raise ValueError("no findings for 'x [/y] z'")

    monkeypatch.setattr(wr, "web_research", boom)

    out = _run_cli('/web-search "x"')
    assert "web-search failed" in out
    assert "[/y]" in out


# --- remember: the evidence bank and its verbatim guardian (spec §3.2/§3.5) --

_QUOTE_PAGE = (
    "Source: https://s.test/a\n\nAlpha Title\n\n"
    "Graphs beat lists for this workload. Second sentence,\nwrapped by the "
    "renderer, still one claim."
)


def _fetched(page=_QUOTE_PAGE, call_id="f1"):
    """Feed the guardian the event the real loop emits after a web_fetch."""
    from silica.agent.events import ToolCompleteEvent

    wr._harvest_page(ToolCompleteEvent(
        name="web_fetch", args={"url": "https://s.test/a"}, call_id=call_id,
        result=page, duration_s=0.0, iteration=1,
    ))


def test_remember_registered_and_sensitive():
    assert TOOLS["remember"].sensitive is True


def test_remember_accepts_a_verbatim_quote_and_rejects_a_paraphrase():
    _fetched()

    out = wr.remember("https://s.test/a", "Graphs beat lists for this workload.", "core")

    assert "[Q1]" in out
    with pytest.raises(ValueError, match="verbatim"):
        wr.remember("https://s.test/a", "Lists are worse than graphs here.", "para")
    assert list(wr._BANK) == ["Q1"]


def test_remember_tolerates_reflowed_whitespace():
    """The fetched text wraps mid-sentence; the model quotes it on one line.
    Whitespace is the renderer's, not the author's, so it must not fail the
    verbatim check."""
    _fetched()

    out = wr.remember(
        "https://s.test/a",
        "Second sentence, wrapped by the renderer, still one claim.",
        "wrap",
    )

    assert "[Q1]" in out


def test_remember_rejects_a_url_never_fetched():
    with pytest.raises(ValueError, match="no page fetched"):
        wr.remember("https://never.test/x", "anything", "w")


def test_remember_is_idempotent_for_the_same_quote():
    """A retrying model must not mint a second ID for the same evidence."""
    _fetched()

    wr.remember("https://s.test/a", "Graphs beat lists for this workload.", "a")
    again = wr.remember("https://s.test/a", "Graphs  beat lists for this workload.", "b")

    assert "already banked as [Q1]" in again
    assert len(wr._BANK) == 1


def test_bank_and_pages_are_per_turn():
    _fetched()
    wr.remember("https://s.test/a", "Graphs beat lists for this workload.", "a")

    wr._reset_turn()

    assert not wr._BANK and not wr._PAGES


def test_bind_citations_renumbers_by_first_appearance_and_reorders_sources():
    bank = {
        "Q1": wr._Quote("https://b.test", "quote b", "w"),
        "Q2": wr._Quote("https://a.test", "quote a", "w"),
    }
    collected = [
        ("https://c.test", "C"), ("https://b.test", "B"), ("https://a.test", "A"),
    ]

    body, sources, audit = wr._bind_citations("x [Q2] y [Q1] z [Q2].", collected, bank)

    assert body == "x [1] y [2] z [1]."
    # Cited pages first, in first-citation order; the uncited page stays listed
    # (ADR-0015: sources are every page the run opened), just after them.
    assert sources == [
        ("https://a.test", "A"), ("https://b.test", "B"), ("https://c.test", "C"),
    ]
    assert audit == ""


def test_quotes_from_one_page_share_one_source_number():
    """Citations name sources, not bank rows: [Q1] and [Q2] off the same page
    are the same [1], singly or in a combined marker."""
    bank = {
        "Q1": wr._Quote("https://a.test", "one", "w"),
        "Q2": wr._Quote("https://a.test", "two", "w"),
    }

    body, sources, _ = wr._bind_citations(
        "x [Q1] y [Q2] z [Q1, Q2].", [("https://a.test", "A")], bank
    )

    assert body == "x [1] y [1] z [1]."
    assert sources == [("https://a.test", "A")]


def test_a_phantom_marker_is_removed_and_audited():
    body, sources, audit = wr._bind_citations("A claim [Q9]. More.", [], {})

    assert body == "A claim. More."
    assert "1 marker(s) named no banked quote" in audit


def test_web_research_binds_citations_and_leaves_the_bank_in_the_leaf(
    tmp_vault, monkeypatch
):
    """End to end: the note carries [n] markers bound to the mechanical Sources
    block, a phantom marker is stripped and audited, a disobedient model-written
    Sources section is cut, and the leaf carries the bank for /nucleate."""
    from silica.agent.events import ToolCompleteEvent

    def fake_run_agent(messages, model, tool_progress_callback=None,
                       constraints=None, **kw):
        tool_progress_callback(ToolCompleteEvent(
            name="web_fetch", args={"url": "https://s.test/a"}, call_id="f1",
            result=_QUOTE_PAGE, duration_s=0.0, iteration=1,
        ))
        wr.remember(
            "https://s.test/a", "Graphs beat lists for this workload.", "core claim"
        )
        return (
            "Graphs win [Q1], allegedly always [Q7].\n\n"
            "## Sources\n1. stale hand-written line"
        )

    monkeypatch.setattr(wr, "run_agent", fake_run_agent)

    note_rel = wr.web_research("graph workloads")
    body = (Path(CONFIG.vault_path) / note_rel).read_text(encoding="utf-8")

    assert "Graphs win [1], allegedly always." in body
    assert "[Q1]" not in body and "[Q7]" not in body
    assert "stale hand-written line" not in body
    assert "1. https://s.test/a — https://s.test/a" in body
    assert "Citation audit: 1 marker(s)" in body
    assert body.count("## Sources") == 1

    from silica.kernel.recall.paths import SOURCES_DIR

    leaf = (
        Path(CONFIG.vault_path) / SOURCES_DIR / Path(note_rel).name
    ).read_text(encoding="utf-8")
    assert "## Evidence bank" in leaf
    assert "[Q1] https://s.test/a" in leaf
    assert "> Graphs beat lists for this workload." in leaf
    assert "why: core claim" in leaf


# --- outline + per-section writer (spec §3.3/§3.4) and the §3.6 fallback -----

def test_parse_outline_reads_sections_and_ids():
    text = (
        "## How they are trained [Q3, Q7, Q11]\n"
        "Some stray prose the model added.\n"
        "## Where they fail [Q2]\n"
    )

    assert wr._parse_outline(text) == [
        ("How they are trained", ["Q3", "Q7", "Q11"]),
        ("Where they fail", ["Q2"]),
    ]


def test_parse_outline_ignores_headings_without_ids():
    assert wr._parse_outline("## Intro\n\nJust prose, no bank IDs anywhere.") == []


_TWO_PAGE_BANK = {
    "Q1": wr._Quote("https://a.test", "alpha quote", "why a"),
    "Q2": wr._Quote("https://b.test", "beta quote", "why b"),
}


def _sequential_llm(monkeypatch, replies):
    """wr.call_llm fake answering `replies` in order; returns the calls seen.

    A reply that is an Exception is raised instead of returned."""
    from types import SimpleNamespace

    calls: list[list[dict]] = []

    def fake(model, messages, **kw):
        assert kw.get("tools") is None, "outline/writer calls must be tool-less"
        calls.append(messages)
        reply = replies[len(calls) - 1]
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(text=reply)

    monkeypatch.setattr(wr, "call_llm", fake)
    return calls


def test_compose_findings_writes_each_section_from_its_own_quotes(monkeypatch):
    """The outline call sees the bank index (no quote texts); each section call
    sees only its own quotes; headings are added mechanically."""
    calls = _sequential_llm(monkeypatch, [
        "## Alpha side [Q1]\n## Beta side [Q2]",
        "Alpha prose [Q1].",
        "Beta prose [Q2].",
    ])

    body = wr._compose_findings("the question", _TWO_PAGE_BANK)

    assert body == (
        "## Alpha side\n\nAlpha prose [Q1].\n\n## Beta side\n\nBeta prose [Q2]."
    )
    outline_user = calls[0][-1]["content"]
    assert "Q1 | https://a.test | why a" in outline_user
    assert "alpha quote" not in outline_user          # index only, spec §3.3
    alpha_user = calls[1][-1]["content"]
    assert "alpha quote" in alpha_user
    assert "beta quote" not in alpha_user             # only its own quotes, §3.4
    assert "## Alpha side [Q1]" in alpha_user         # the whole outline as context


def test_compose_findings_returns_none_when_outline_unparsable(monkeypatch):
    _sequential_llm(monkeypatch, ["I could not decide on sections, sorry."])

    assert wr._compose_findings("q", _TWO_PAGE_BANK) is None


def test_compose_findings_returns_none_when_a_section_call_fails(monkeypatch):
    _sequential_llm(monkeypatch, [
        "## Alpha side [Q1]\n## Beta side [Q2]",
        "Alpha prose [Q1].",
        RuntimeError("provider down"),
    ])

    assert wr._compose_findings("q", _TWO_PAGE_BANK) is None


def test_compose_findings_strips_a_writer_echoed_heading(monkeypatch):
    """Measured on the first live replay: the writer sometimes opens with the
    section heading despite the prompt, and the mechanical heading made it a
    duplicate. An echoed heading is dropped; an unrelated one is prose."""
    calls = _sequential_llm(monkeypatch, [
        "## Alpha side [Q1]",
        "## Alpha side\n\nAlpha prose [Q1].\n\n### A sub-point\n\nMore.",
    ])

    body = wr._compose_findings("q", _TWO_PAGE_BANK)

    assert body == (
        "## Alpha side\n\nAlpha prose [Q1].\n\n### A sub-point\n\nMore."
    )
    assert calls  # composition really ran


def test_bind_citations_collapses_adjacent_duplicate_markers():
    """[Q3][Q3] (measured live) and [Q3] [Q3] are one citation, not two."""
    bank = {"Q3": wr._Quote("https://a.test", "one", "w")}

    body, _, _ = wr._bind_citations(
        "bypassed [Q3][Q3]. also [Q3] [Q3]. kept [Q3], [Q3].",
        [("https://a.test", "A")], bank,
    )

    assert body == "bypassed [1]. also [1]. kept [1], [1]."


def test_compose_findings_drops_sections_citing_only_phantoms(monkeypatch):
    """An outline section naming only unknown IDs gets no writer call; one
    whose every section is phantom is unparsable in effect."""
    _sequential_llm(monkeypatch, ["## Ghosts [Q9]\n## More ghosts [Q8]"])

    assert wr._compose_findings("q", _TWO_PAGE_BANK) is None


def _run_agent_banking_one_quote(monkeypatch, body="One-shot body [Q1]."):
    """Fake loop: fetches a page, banks Q1, returns `body` as its final message."""
    from silica.agent.events import ToolCompleteEvent

    def fake_run_agent(messages, model, tool_progress_callback=None,
                       constraints=None, **kw):
        tool_progress_callback(ToolCompleteEvent(
            name="web_fetch", args={"url": "https://s.test/a"}, call_id="f1",
            result=_QUOTE_PAGE, duration_s=0.0, iteration=1,
        ))
        wr.remember(
            "https://s.test/a", "Graphs beat lists for this workload.", "core"
        )
        return body

    monkeypatch.setattr(wr, "run_agent", fake_run_agent)


def test_web_research_composes_from_bank_when_quotes_banked(tmp_vault, monkeypatch):
    """/web-search end to end: the composed sections replace the loop's
    one-shot body, and the composed markers bind to the Sources block."""
    _run_agent_banking_one_quote(monkeypatch)
    _sequential_llm(monkeypatch, [
        "## The claim [Q1]",
        "Graphs carry the day [Q1].",
    ])

    note_rel = wr.web_research("graph workloads")
    body = (Path(CONFIG.vault_path) / note_rel).read_text(encoding="utf-8")

    assert "## The claim\n\nGraphs carry the day [1]." in body
    assert "One-shot body" not in body
    assert "1. https://s.test/a — https://s.test/a" in body


def test_web_research_keeps_oneshot_body_when_composition_fails(
    tmp_vault, monkeypatch
):
    """§3.6: a dead composition path costs nothing — the loop's final message
    is the note body, exactly as before the outline existed."""
    _run_agent_banking_one_quote(monkeypatch)
    _sequential_llm(monkeypatch, [RuntimeError("provider down")])

    note_rel = wr.web_research("graph workloads")
    body = (Path(CONFIG.vault_path) / note_rel).read_text(encoding="utf-8")

    assert "One-shot body [1]." in body


def test_web_research_skips_composition_when_bank_is_empty(tmp_vault, monkeypatch):
    """§3.6: no banked quotes, no extra LLM calls — today's path verbatim.

    Counted with a recording fake, not an in-fake raise: composition swallows
    every exception into its fallback, so a raise would pass vacuously."""
    _patch_run_agent(monkeypatch, "Plain findings.", tool_results=[
        json.dumps([{"title": "A", "url": "https://a.test", "content": ""}])
    ])
    calls = _sequential_llm(monkeypatch, [])

    note_rel = wr.web_research("plain concept")
    body = (Path(CONFIG.vault_path) / note_rel).read_text(encoding="utf-8")

    assert "Plain findings." in body
    assert calls == []


def test_web_turn_never_composes(monkeypatch):
    """Spec §5: outline and per-section writer run only in /web-search. The
    /web turn answers in direct prose even with quotes banked."""
    monkeypatch.setattr(
        wr, "call_llm",
        lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError("/web must not compose")
        ),
    )
    turn = wr.WebTurn("q")
    wr._BANK["Q1"] = wr._Quote("https://a.test", "alpha", "w")

    out = turn.attribute("Answer [Q1].", [])

    assert out.startswith("Answer [1].")


# --- plan: the live steering plan (spec-web-research-plan-steering) ---------


def test_plan_registered_and_sensitive():
    assert TOOLS["plan"].sensitive is True


def test_plan_rejects_text_without_section_headings():
    with pytest.raises(ValueError, match="no `## ` section headings"):
        wr.plan("just prose, no structure")
    assert wr._PLAN == ""


def test_plan_rejects_unknown_ids_listing_the_known_ones():
    wr._BANK["Q1"] = wr._Quote(url="https://s.test/a", quote="q", why="w")
    with pytest.raises(ValueError, match=r"unknown quote ID\(s\): Q7"):
        wr.plan("## Training [Q7]")
    assert wr._PLAN == ""


def test_plan_rejection_names_an_empty_bank():
    with pytest.raises(ValueError, match=r"\(none yet\)"):
        wr.plan("## Training [Q1]")


def test_plan_accepts_gap_headings_and_names_them():
    wr._BANK["Q1"] = wr._Quote(url="https://s.test/a", quote="q", why="w")
    out = wr.plan("## Training [Q1]\n## Failure modes\n## Cost")
    assert out == "saved: 3 sections, 1 with evidence; gaps: Failure modes; Cost"


def test_plan_confirms_full_coverage():
    wr._BANK["Q1"] = wr._Quote(url="https://s.test/a", quote="q", why="w")
    assert wr.plan("## Training [Q1]") == "saved: 1 section, all with evidence"


def test_plan_replaces_the_previous_plan():
    wr.plan("## First")
    wr.plan("## Second")
    assert wr._PLAN == "## Second"


def test_reset_turn_clears_the_plan():
    wr._PLAN = "## Stale"
    wr._reset_turn()
    assert wr._PLAN == ""


def test_live_plan_leaves_composition_untouched(tmp_vault, monkeypatch):
    """§2 invariant: the composer sees only (concept, bank). A saved plan
    must neither divert composition nor be passed to it. Recording fake, not
    asserts inside the fake: web_research swallows composer failures by
    design (§3.6), so an AssertionError raised in there is vacuous."""
    monkeypatch.setattr(CONFIG, "tavily_api_key", "k")
    calls = []

    def fake_run_agent(messages, model, tool_progress_callback=None,
                       constraints=None, **kw):
        wr._PAGES["https://s.test/a"] = "Source: https://s.test/a\nGraphs beat lists."
        wr.remember("https://s.test/a", "Graphs beat lists.", "core")
        wr.plan("## Data structures [Q1]\n## Open gap")
        return "Findings [Q1]."

    monkeypatch.setattr(wr, "run_agent", fake_run_agent)

    def recording_compose(concept, bank):
        calls.append((concept, dict(bank)))
        return "## Data structures\n\nProse [Q1]."

    monkeypatch.setattr(wr, "_compose_findings", recording_compose)
    wr.web_research("the question")
    assert calls == [("the question", {"Q1": wr._BANK["Q1"]})]


def test_guardian_forgives_quote_style_not_words():
    """Pages render apostrophes curly; models copy them back ASCII. Reading that
    as a paraphrase killed a whole research run (18 rejections, no findings)."""
    from silica.sources.web_research import _squash

    page = "the verifier looks at all the states it’s previously been in, " \
           "so the branch is ‘pruned’ — that is, it converges."
    assert _squash("the states it's previously been in") in _squash(page)
    assert _squash("the branch is 'pruned' - that is") in _squash(page)
    # every word must still be there verbatim: style is forgiven, wording is not
    assert _squash("the verifier prunes branches already explored") not in _squash(page)
    assert _squash("the states it has previously been in") not in _squash(page)
    # guillemet and CJK quote styles fold too (European and CJK sources)
    assert _squash('he called it "fine"') in _squash("he called it «fine».")
    assert _squash('the 「lease」 expires') in _squash('the "lease" expires')


def test_find_in_page_rejects_unfetched_url_naming_the_fetched():
    wr._PAGES["https://s.test/a"] = "Source: https://s.test/a\nbody"
    with pytest.raises(ValueError) as err:
        wr.find_in_page("https://other.test", "x")
    assert "https://s.test/a" in str(err.value)


def test_find_in_page_returns_matching_lines_with_context():
    wr._PAGES["https://s.test/a"] = (
        "Source: https://s.test/a\n"
        "intro line\n"
        "the RVV 1.0 spec was ratified in 2021\n"
        "closing line\n"
        "unrelated\n"
    )
    out = wr.find_in_page("https://s.test/a", "ratified in 2021")
    assert "the RVV 1.0 spec was ratified in 2021" in out
    assert "intro line" in out and "closing line" in out  # ±1 line of context
    assert "unrelated" not in out


def test_find_in_page_forgives_typography_and_reports_misses():
    wr._PAGES["https://s.test/a"] = "Source: https://s.test/a\nit’s ‘pruned’ here\n"
    assert "pruned" in wr.find_in_page("https://s.test/a", "it's 'pruned'")
    assert wr.find_in_page("https://s.test/a", "absent").startswith("no line")


def test_find_in_page_folds_nearby_hits_and_caps_windows():
    lines = [f"filler {i}" for i in range(40)]
    for i in (3, 4, 30):        # 3 and 4 fold into one window
        lines[i] = f"needle at {i}"
    wr._PAGES["https://s.test/a"] = "Source: https://s.test/a\n" + "\n".join(lines)
    out = wr.find_in_page("https://s.test/a", "needle")
    assert "needle at 3" in out and "needle at 30" in out
    assert "(1 more matching lines not shown)" in out  # 4 folded, counted
