import importlib.util
import io
from pathlib import Path
import sys
import types
import wave

import pytest
import soundfile
import torch

from api_bridge.assets import AudioAssetStore, get_audio_asset_store, reset_audio_asset_store_for_tests


REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_NODE_PATH = REPO_ROOT / "nodes" / "api_bridge" / "audio_asset_node.py"


def wav_bytes() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 160)
    return output.getvalue()


def _load_audio_node(monkeypatch, store: AudioAssetStore):
    module_name = "api_bridge_audio_asset_node_test"
    comfy_audio = types.ModuleType("comfy_extras.nodes_audio")

    def load(path: str):
        samples, sample_rate = soundfile.read(path, dtype="float32", always_2d=True)
        return torch.from_numpy(samples.T), sample_rate

    comfy_audio.load = load
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_audio", comfy_audio)
    spec = importlib.util.spec_from_file_location(module_name, ASSET_NODE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "get_audio_asset_store", lambda: store)
    return module


@pytest.mark.unit
def test_asset_store_uses_generated_name_and_deletes(tmp_path: Path):
    store = AudioAssetStore(tmp_path, max_bytes=1024 * 1024)

    asset = store.create("../voice.wav", wav_bytes())

    assert asset.path.parent == tmp_path.resolve()
    assert asset.path.name != "voice.wav"
    assert asset.path.suffix == ".wav"
    assert asset.path.is_file()
    assert store.require(asset.asset_id).sha256 == asset.sha256

    store.delete(asset.asset_id)

    assert not asset.path.exists()
    with pytest.raises(ValueError, match="unknown asset_id"):
        store.require(asset.asset_id)


@pytest.mark.unit
def test_asset_store_rejects_oversize_before_writing(tmp_path: Path):
    store = AudioAssetStore(tmp_path, max_bytes=len(wav_bytes()) - 1)

    with pytest.raises(ValueError, match="maximum"):
        store.create("voice.wav", wav_bytes())

    assert list(tmp_path.iterdir()) == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("filename", "content", "message"),
    [
        ("voice.txt", wav_bytes(), "extension"),
        ("voice.wav", b"not audio", "audio"),
        ("../../outside.wav", b"not audio", "audio"),
    ],
)
def test_asset_store_rejects_invalid_inputs_and_cleans_partial_files(
    tmp_path: Path, filename: str, content: bytes, message: str
):
    store = AudioAssetStore(tmp_path)

    with pytest.raises(ValueError, match=message):
        store.create(filename, content)

    assert list(tmp_path.iterdir()) == []


@pytest.mark.unit
def test_asset_store_delete_accepts_only_known_asset_ids(tmp_path: Path):
    store = AudioAssetStore(tmp_path)

    with pytest.raises(ValueError, match="unknown asset_id"):
        store.delete("../../voice.wav")


@pytest.mark.unit
def test_default_store_is_bounded_to_the_comfy_input_directory(tmp_path: Path, monkeypatch):
    import folder_paths

    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(tmp_path))
    reset_audio_asset_store_for_tests()
    try:
        store = get_audio_asset_store()
        assert store.root == (tmp_path / "tts-audio-suite").resolve()
    finally:
        reset_audio_asset_store_for_tests()


@pytest.mark.unit
def test_external_audio_asset_node_returns_a_decodable_narrator_voice(tmp_path: Path, monkeypatch):
    store = AudioAssetStore(tmp_path)
    asset = store.create("reference.wav", wav_bytes())
    module = _load_audio_node(monkeypatch, store)

    (voice,) = module.ExternalAudioAssetNode().load_asset(asset.asset_id, "参考文本")

    assert module.ExternalAudioAssetNode.RETURN_TYPES == ("NARRATOR_VOICE",)
    assert voice["reference_text"] == "参考文本"
    assert voice["audio_path"] == str(asset.path)
    assert voice["character_name"] == "external"
    assert voice["audio"]["sample_rate"] == 16000
    assert tuple(voice["audio"]["waveform"].shape) == (1, 1, 160)
    assert soundfile.info(asset.path).samplerate == voice["audio"]["sample_rate"]


@pytest.mark.unit
def test_audio_asset_node_registration_is_additive_and_preserves_existing_mappings(monkeypatch):
    store = AudioAssetStore(Path(monkeypatch.tempdir) if hasattr(monkeypatch, "tempdir") else Path.cwd())
    _load_audio_node(monkeypatch, store)
    spec = importlib.util.spec_from_file_location("api_bridge_asset_nodes_mapping_test", REPO_ROOT / "nodes.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.NODE_CLASS_MAPPINGS["TTSExternalAudioAsset"] is module.ExternalAudioAssetNode
    assert module.NODE_CLASS_MAPPINGS["TTSExternalGPTSovitsEngine"] is module.ExternalGPTSovitsEngineNode
    assert module.NODE_CLASS_MAPPINGS["CharacterVoicesNode"] is module.CharacterVoicesNode
