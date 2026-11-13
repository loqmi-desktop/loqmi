"""Tests for the routing pipeline (core/router.py) with mocked inference."""

import pytest

from core import router as core_router


class _FakeInference:
    """Replaces core.inference.run with a canned token stream."""

    def __init__(self):
        self.calls = []

    def __call__(self, prompt, model, category, chat_id):
        self.calls.append({"prompt": prompt, "category": category, "chat_id": chat_id})
        yield {"type": "token", "text": "hello"}
        yield {"type": "token", "text": " world"}


@pytest.fixture
def fake_pipeline(monkeypatch):
    from core import classifier, model_manager

    fake_inference = _FakeInference()
    classify_calls = []

    def fake_classify(prompt):
        classify_calls.append(prompt)
        return "code"

    monkeypatch.setattr(classifier, "classify", fake_classify)
    monkeypatch.setattr(model_manager, "load", lambda category: f"model-{category}")
    monkeypatch.setattr(core_router.inference, "run", fake_inference)
    return {"classify_calls": classify_calls, "inference": fake_inference}


def test_classified_routing(fake_pipeline):
    events = list(core_router.handle("write me a function", chat_id="c1"))

    assert fake_pipeline["classify_calls"] == ["write me a function"]
    assert fake_pipeline["inference"].calls[0]["category"] == "code"

    tokens = [e for e in events if e["type"] == "token"]
    assert "".join(t["text"] for t in tokens) == "hello world"

    stats = [e for e in events if e["type"] == "stats"]
    assert len(stats) == 1
    assert stats[0]["category"] == "code"
    assert "model_name" in stats[0]


def test_override_bypasses_classifier(fake_pipeline):
    events = list(
        core_router.handle("write me a function", chat_id="c1", override_category="reasoning")
    )

    assert fake_pipeline["classify_calls"] == []  # classifier never invoked
    assert fake_pipeline["inference"].calls[0]["category"] == "reasoning"
    stats = [e for e in events if e["type"] == "stats"]
    assert stats[0]["category"] == "reasoning"
    assert stats[0]["classify_ms"] == 0.0


def test_inference_error_is_raised_through(fake_pipeline, monkeypatch):
    def boom(prompt, model, category, chat_id):
        raise RuntimeError("model exploded")
        yield  # pragma: no cover — makes this a generator

    monkeypatch.setattr(core_router.inference, "run", boom)
    with pytest.raises(RuntimeError, match="model exploded"):
        list(core_router.handle("hi", chat_id="c1"))
