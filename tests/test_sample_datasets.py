from src.sample_datasets import (
    BUILDERS,
    SAMPLE_DEFINITIONS,
    build_all_samples,
    find_definition,
    get_definitions_with_paths,
)


def test_definitions_have_unique_slugs():
    slugs = [d.slug for d in SAMPLE_DEFINITIONS]
    assert len(slugs) == len(set(slugs))


def test_all_builders_registered():
    for d in SAMPLE_DEFINITIONS:
        assert d.slug in BUILDERS


def test_build_all_samples(tmp_path, monkeypatch):
    from src import sample_datasets as mod

    monkeypatch.setattr(mod, "SAMPLES_DIR", tmp_path)
    for d in SAMPLE_DEFINITIONS:
        monkeypatch.setattr(d.__class__, "path", property(lambda self: tmp_path / self.file_name))
    outputs = build_all_samples()
    assert len(outputs) == len(SAMPLE_DEFINITIONS)
    for slug, path in outputs.items():
        assert path.exists()
        assert path.stat().st_size > 0


def test_find_definition():
    d = find_definition("quickstart")
    assert d is not None
    assert d.slug == "quickstart"
    assert find_definition("nonexistent") is None


def test_get_definitions_with_paths_shape():
    items = get_definitions_with_paths()
    assert len(items) == len(SAMPLE_DEFINITIONS)
    for item in items:
        assert {"slug", "title", "description", "exists", "path"}.issubset(item.keys())
