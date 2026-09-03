from __future__ import annotations


from silica.config import SilicaConfig


def test_worker_max_concurrent_default():
    cfg = SilicaConfig()
    assert cfg.worker_max_concurrent == 4


def test_worker_max_concurrent_from_env(monkeypatch):
    monkeypatch.setenv("SILICA_WORKER_MAX_CONCURRENT", "9")
    cfg = SilicaConfig()
    assert cfg.worker_max_concurrent == 9
