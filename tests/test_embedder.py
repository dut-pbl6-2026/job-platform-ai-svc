"""Embedder selection (unit): the model is required; the hashing embedder is an explicit, visible opt-in."""
import pytest

from app import main
from app.config import Settings
from app.embed import HashingEmbedder


class Loaded:
    name = "loaded-model"
    dim = 384


def test_loaded_model_is_used(monkeypatch):
    monkeypatch.setattr(main, "OnnxEmbedder", lambda *a, **k: Loaded())
    assert main.build_embedder(Settings()).name == "loaded-model"


def test_load_failure_stops_startup_by_default(monkeypatch):
    def broken(*a, **k):
        raise OSError("no model files")

    monkeypatch.setattr(main, "OnnxEmbedder", broken)
    with pytest.raises(OSError):
        main.build_embedder(Settings(embedder_fallback=""))


def test_load_failure_with_the_hashing_opt_in_is_degraded(monkeypatch):
    def broken(*a, **k):
        raise OSError("no model files")

    monkeypatch.setattr(main, "OnnxEmbedder", broken)
    emb = main.build_embedder(Settings(embedder_fallback="hashing"))
    assert isinstance(emb, HashingEmbedder) and emb.name == "hashing-256"


def test_fallback_is_only_read_from_the_explicit_variable(monkeypatch):
    monkeypatch.delenv("EMBEDDER_FALLBACK", raising=False)
    assert Settings(_env_file=None).embedder_fallback == ""
    monkeypatch.setenv("EMBEDDER_FALLBACK", "hashing")
    assert Settings(_env_file=None).embedder_fallback == "hashing"
