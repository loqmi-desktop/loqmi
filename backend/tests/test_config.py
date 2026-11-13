"""Sanity checks on the model registry in config.py."""

from config import MODELS, get_model_config, get_recommended_model


def test_registry_has_expected_categories():
    assert set(MODELS.keys()) == {"chat", "code", "reasoning", "document"}


def test_every_category_has_a_recommended_model():
    for category in MODELS:
        assert get_recommended_model(category) is not None, f"{category} has no recommended model"


def test_model_ids_are_unique_per_category():
    for category, models in MODELS.items():
        ids = [m["id"] for m in models]
        assert len(ids) == len(set(ids)), f"duplicate ids in '{category}'"


def test_model_entries_have_required_fields():
    for category, models in MODELS.items():
        for model in models:
            assert model.get("filename"), f"{category}/{model.get('id')} missing filename"
            assert model.get("url", "").startswith("https://"), f"{category}/{model['id']} has bad url"
            assert model.get("size_gb", 0) > 0, f"{category}/{model['id']} has bad size"


def test_get_model_config_roundtrip():
    cfg = get_recommended_model("chat")
    found = get_model_config("chat", cfg["id"])
    assert found is not None
    assert found["id"] == cfg["id"]


def test_get_model_config_returns_none_for_unknown():
    assert get_model_config("chat", "does-not-exist") is None
