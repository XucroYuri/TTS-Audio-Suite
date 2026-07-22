from pathlib import Path

import pytest

from api_bridge.resource_registry import ResourceRegistry


def test_registry_resolves_registered_index_resource(tmp_path: Path):
    source = tmp_path / "index"
    model = source / "checkpoints"
    model.mkdir(parents=True)
    (model / "config.yaml").write_text("model: index\n", encoding="utf-8")
    config = tmp_path / "resources.yaml"
    config.write_text(
        "version: 1\nresources:\n  local-index:\n"
        f"    engine: index_tts\n    source_root: '{source.as_posix()}'\n"
        f"    model_dir: '{model.as_posix()}'\n",
        encoding="utf-8",
    )

    resource = ResourceRegistry.load(config).require("local-index", "index_tts")

    assert resource.model_dir == model.resolve()
    assert resource.source_root == source.resolve()


def test_registry_capabilities_are_redacted_and_sorted(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    for source in (first, second):
        (source / "models").mkdir(parents=True)
    config = tmp_path / "resources.yaml"
    config.write_text(
        "version: 1\nresources:\n"
        f"  zulu:\n    engine: cosyvoice\n    source_root: '{second.as_posix()}'\n"
        f"    model_dir: '{(second / 'models').as_posix()}'\n"
        f"  alpha:\n    engine: index_tts\n    source_root: '{first.as_posix()}'\n"
        f"    model_dir: '{(first / 'models').as_posix()}'\n",
        encoding="utf-8",
    )

    capabilities = ResourceRegistry.load(config).capabilities()

    assert capabilities == [
        {"resource_id": "alpha", "engine": "index_tts", "ready": True},
        {"resource_id": "zulu", "engine": "cosyvoice", "ready": True},
    ]
    private_keys = {
        "source_root",
        "model_dir",
        "gpt_weight",
        "sovits_weight",
        "bert_path",
        "cnhubert_path",
    }
    assert all(private_keys.isdisjoint(capability) for capability in capabilities)


def test_registry_rejects_correct_resource_id_with_wrong_engine(tmp_path: Path):
    source = tmp_path / "index"
    model = source / "checkpoints"
    model.mkdir(parents=True)
    config = tmp_path / "resources.yaml"
    config.write_text(
        "version: 1\nresources:\n  local-index:\n"
        f"    engine: index_tts\n    source_root: '{source.as_posix()}'\n"
        f"    model_dir: '{model.as_posix()}'\n",
        encoding="utf-8",
    )

    registry = ResourceRegistry.load(config)

    with pytest.raises(ValueError, match="belongs to index_tts, not cosyvoice"):
        registry.require("local-index", "cosyvoice")


def test_registry_rejects_missing_gpt_weights(tmp_path: Path):
    source = tmp_path / "gpt"
    source.mkdir()
    config = tmp_path / "resources.yaml"
    config.write_text(
        "version: 1\nresources:\n  voice-a:\n"
        f"    engine: gpt_sovits\n    source_root: '{source.as_posix()}'\n"
        f"    gpt_weight: '{(source / 'a.ckpt').as_posix()}'\n"
        f"    sovits_weight: '{(source / 'a.pth').as_posix()}'\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing file"):
        ResourceRegistry.load(config)


def test_registry_resources_are_immutable_and_paths_are_resolved(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "index"
    model = source / "checkpoints"
    model.mkdir(parents=True)
    config = tmp_path / "resources.yaml"
    config.write_text(
        "version: 1\nresources:\n  local-index:\n"
        "    engine: index_tts\n    source_root: './index'\n"
        "    model_dir: './index/checkpoints'\n",
        encoding="utf-8",
    )

    resource = ResourceRegistry.load(config).require("local-index", "index_tts")

    assert resource.source_root == source.resolve()
    assert resource.model_dir == model.resolve()
    with pytest.raises(AttributeError):
        resource.model_dir = source
