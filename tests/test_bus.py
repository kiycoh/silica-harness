"""Tests for silica/agent/bus.py — EventBus pub-sub mechanics."""
from __future__ import annotations

import threading


from silica.agent.bus import EventBus


def test_subscribe_and_publish():
    bus = EventBus()
    received = []
    bus.subscribe("foo/bar", received.append)
    bus.publish("foo/bar", "hello")
    assert received == ["hello"]


def test_wildcard_matches_subtopics():
    bus = EventBus()
    received = []
    bus.subscribe("work/*", received.append)
    bus.publish("work/feedback", "A")
    bus.publish("work/complete", "B")
    bus.publish("work/cancelled", "C")
    assert received == ["A", "B", "C"]


def test_wildcard_does_not_match_parent():
    bus = EventBus()
    received = []
    bus.subscribe("work/*", received.append)
    bus.publish("work", "X")          # exact "work" — not "work/..."
    assert received == []


def test_exact_match_does_not_catch_wildcard():
    bus = EventBus()
    exact = []
    wildcard = []
    bus.subscribe("work/feedback", exact.append)
    bus.subscribe("work/*", wildcard.append)
    bus.publish("work/feedback", "ev")
    assert exact == ["ev"]
    assert wildcard == ["ev"]


def test_unsubscribe_stops_delivery():
    bus = EventBus()
    received = []
    bus.subscribe("x", received.append)
    bus.unsubscribe("x", received.append)
    bus.publish("x", "hello")
    assert received == []


def test_unsubscribe_unknown_fn_is_noop():
    bus = EventBus()
    bus.subscribe("x", lambda e: None)
    bus.unsubscribe("x", lambda e: None)   # different object — no error


def test_subscriber_exception_is_isolated(caplog):
    import logging
    bus = EventBus()
    good = []

    def bad(event):
        raise RuntimeError("boom")

    bus.subscribe("t", bad)
    bus.subscribe("t", good.append)

    with caplog.at_level(logging.ERROR, logger="silica.agent.bus"):
        bus.publish("t", "ev")

    assert good == ["ev"]
    assert any("subscriber failed" in r.message for r in caplog.records)


def test_multiple_subscribers_all_called():
    bus = EventBus()
    results = []
    bus.subscribe("t", lambda e: results.append(f"A:{e}"))
    bus.subscribe("t", lambda e: results.append(f"B:{e}"))
    bus.publish("t", 1)
    assert results == ["A:1", "B:1"]


def test_thread_safety():
    """Concurrent subscribe + publish must not deadlock or corrupt state."""
    bus = EventBus()
    received = []
    lock = threading.Lock()

    def producer():
        for i in range(50):
            bus.publish("t", i)

    def subscriber_adder():
        for _ in range(50):
            bus.subscribe("t", lambda e, _l=lock, _r=received: (
                _l.acquire(), _r.append(e), _l.release()
            ))

    threads = [threading.Thread(target=producer) for _ in range(3)]
    threads += [threading.Thread(target=subscriber_adder) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # No assertion on exact contents — just must not crash or deadlock.


def test_no_subscribers_publish_is_noop():
    bus = EventBus()
    bus.publish("empty/topic", "ev")   # must not raise
