import time

from src.model_registry import list_models, load_model, register_model


def test_register_and_load_roundtrip(prepared_dataset, tmp_path, monkeypatch):
    from src import model_registry as mr

    monkeypatch.setattr(mr, "REGISTRY_DIR", tmp_path / "registry")
    monkeypatch.setattr(mr, "REGISTRY_INDEX", tmp_path / "registry" / "index.jsonl")

    payload = {
        "model": object(),
        "scaler": prepared_dataset.scaler,
        "label_encoder": prepared_dataset.label_encoder,
        "feature_names": prepared_dataset.feature_names,
        "class_names": prepared_dataset.class_names,
        "model_name": "Test",
    }
    entry = register_model(payload, metric_f1=0.85, metric_accuracy=0.9, model_name="Test")
    assert entry.version_id.startswith("v")
    listed = list_models()
    assert any(e.version_id == entry.version_id for e in listed)
    loaded = load_model(entry.version_id)
    assert loaded is not None
    assert loaded["model_name"] == "Test"


def test_registry_prunes_to_max_versions(prepared_dataset, tmp_path, monkeypatch):
    from src import model_registry as mr

    monkeypatch.setattr(mr, "REGISTRY_DIR", tmp_path / "registry")
    monkeypatch.setattr(mr, "REGISTRY_INDEX", tmp_path / "registry" / "index.jsonl")

    payload = {
        "model": object(),
        "scaler": prepared_dataset.scaler,
        "label_encoder": prepared_dataset.label_encoder,
        "feature_names": prepared_dataset.feature_names,
        "class_names": prepared_dataset.class_names,
        "model_name": "Test",
    }
    for _ in range(5):
        register_model(payload, metric_f1=0.8, metric_accuracy=0.8, model_name="Test", max_versions=3)
        time.sleep(1.05)

    entries = list_models()
    assert len(entries) == 3
    pkls = list((tmp_path / "registry").glob("*.pkl"))
    assert len(pkls) == 3
