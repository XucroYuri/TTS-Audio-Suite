import importlib.util
import io
from pathlib import Path
import sys
import threading
import time
import types
import wave

import pytest
import soundfile
import torch

from api_bridge.assets import AudioAssetStore, get_audio_asset_store, reset_audio_asset_store_for_tests


REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_NODE_PATH = REPO_ROOT / "nodes" / "api_bridge" / "audio_asset_node.py"


def wav_bytes(sample: bytes = b"\x00\x00") -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(sample * 160)
    return output.getvalue()


def _load_audio_node(monkeypatch, store: AudioAssetStore):
    module_name = "api_bridge_audio_asset_node_test"
    _install_comfy_audio_stub(monkeypatch)
    spec = importlib.util.spec_from_file_location(module_name, ASSET_NODE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "get_audio_asset_store", lambda: store)
    return module


def _install_comfy_audio_stub(monkeypatch):
    comfy_audio = types.ModuleType("comfy_extras.nodes_audio")

    def load(path: str):
        samples, sample_rate = soundfile.read(path, dtype="float32", always_2d=True)
        return torch.from_numpy(samples.T), sample_rate

    comfy_audio.load = load
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_audio", comfy_audio)


def _load_nodes_with_forced_failure(monkeypatch, failed_suffix: str):
    _install_comfy_audio_stub(monkeypatch)
    nodes_path = REPO_ROOT / "nodes.py"
    original_spec_from_file_location = importlib.util.spec_from_file_location
    spec = original_spec_from_file_location("api_bridge_failure_mapping_test", nodes_path)
    module = importlib.util.module_from_spec(spec)

    def controlled_spec_from_file_location(module_name, location, *args, **kwargs):
        if str(location).replace("\\", "/").endswith(failed_suffix):
            raise ImportError(f"forced failure for {failed_suffix}")
        return original_spec_from_file_location(module_name, location, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "spec_from_file_location", controlled_spec_from_file_location)
    spec.loader.exec_module(module)
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
def test_asset_store_rejects_missing_and_replaced_content_before_a_node_can_load_it(tmp_path: Path, monkeypatch):
    store = AudioAssetStore(tmp_path)
    missing = store.create("missing.wav", wav_bytes())
    missing.path.unlink()

    with pytest.raises(ValueError, match="missing or tampered"):
        store.require(missing.asset_id)

    asset = store.create("reference.wav", wav_bytes())
    module = _load_audio_node(monkeypatch, store)
    monkeypatch.setattr(module, "load", lambda _: pytest.fail("tampered asset reached the audio loader"))
    asset.path.write_bytes(wav_bytes(b"\x01\x00"))

    with pytest.raises(ValueError, match="missing or tampered"):
        module.ExternalAudioAssetNode().load_asset(asset.asset_id, "参考文本")


@pytest.mark.unit
def test_asset_store_rejects_a_registered_path_replaced_with_a_symlink(tmp_path: Path):
    store = AudioAssetStore(tmp_path)
    asset = store.create("reference.wav", wav_bytes())
    replacement = tmp_path / "replacement.wav"
    replacement.write_bytes(wav_bytes(b"\x01\x00"))
    asset.path.unlink()
    try:
        asset.path.symlink_to(replacement)
    except OSError as exc:
        pytest.skip(f"symbolic links unavailable on this platform: {exc}")

    with pytest.raises(ValueError, match="missing or tampered"):
        store.require(asset.asset_id)


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
def test_default_store_rejects_a_managed_directory_linked_outside_comfy_input(tmp_path: Path, monkeypatch):
    import folder_paths

    input_root = tmp_path / "input"
    external_root = tmp_path / "external"
    input_root.mkdir()
    external_root.mkdir()
    managed_root = input_root / "tts-audio-suite"
    try:
        managed_root.symlink_to(external_root, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symbolic links unavailable on this platform: {exc}")
    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(input_root))
    reset_audio_asset_store_for_tests()

    with pytest.raises(ValueError, match="outside the ComfyUI input directory"):
        get_audio_asset_store()

    reset_audio_asset_store_for_tests()


@pytest.mark.unit
def test_default_store_concurrent_initialization_returns_one_instance(tmp_path: Path, monkeypatch):
    import api_bridge.assets as assets
    import folder_paths

    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(tmp_path))
    original_store = assets.AudioAssetStore
    construction_barrier = threading.Barrier(4)

    class BarrierStore(original_store):
        def __init__(self, *args, **kwargs):
            construction_barrier.wait(timeout=5)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(assets, "AudioAssetStore", BarrierStore)
    reset_audio_asset_store_for_tests()
    start_barrier = threading.Barrier(4)
    results = []
    results_lock = threading.Lock()

    def get_store():
        start_barrier.wait(timeout=5)
        result = get_audio_asset_store()
        with results_lock:
            results.append(result)

    threads = [threading.Thread(target=get_store) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()

    assert len(results) == 4
    assert all(item is results[0] for item in results)
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
def test_audio_asset_node_rejects_content_replaced_during_audio_load(tmp_path: Path, monkeypatch):
    store = AudioAssetStore(tmp_path)
    asset = store.create("reference.wav", wav_bytes())
    module = _load_audio_node(monkeypatch, store)

    def replace_during_load(path: str):
        samples, sample_rate = soundfile.read(path, dtype="float32", always_2d=True)
        Path(path).write_bytes(wav_bytes(b"\x01\x00"))
        return torch.from_numpy(samples.T), sample_rate

    monkeypatch.setattr(module, "load", replace_during_load)

    with pytest.raises(ValueError, match="missing or tampered"):
        module.ExternalAudioAssetNode().load_asset(asset.asset_id, "参考文本")


@pytest.mark.unit
def test_audio_asset_delete_waits_for_an_active_node_load(tmp_path: Path, monkeypatch):
    store = AudioAssetStore(tmp_path)
    asset = store.create("reference.wav", wav_bytes())
    module = _load_audio_node(monkeypatch, store)
    entered_load = threading.Event()
    release_load = threading.Event()
    loaded_voice = []
    delete_done = threading.Event()

    def slow_load(path: str):
        entered_load.set()
        assert release_load.wait(timeout=5)
        samples, sample_rate = soundfile.read(path, dtype="float32", always_2d=True)
        return torch.from_numpy(samples.T), sample_rate

    monkeypatch.setattr(module, "load", slow_load)
    node_thread = threading.Thread(
        target=lambda: loaded_voice.append(module.ExternalAudioAssetNode().load_asset(asset.asset_id, "参考文本"))
    )
    delete_thread = threading.Thread(target=lambda: (store.delete(asset.asset_id), delete_done.set()))
    node_thread.start()
    assert entered_load.wait(timeout=5)
    delete_thread.start()
    time.sleep(0.1)
    assert not delete_done.is_set()
    release_load.set()
    node_thread.join(timeout=5)
    delete_thread.join(timeout=5)

    assert not node_thread.is_alive()
    assert not delete_thread.is_alive()
    assert loaded_voice[0][0]["audio_path"] == str(asset.path)
    assert delete_done.is_set()


@pytest.mark.unit
def test_audio_asset_node_registration_is_additive_and_preserves_existing_mappings(tmp_path: Path, monkeypatch):
    store = AudioAssetStore(tmp_path)
    _load_audio_node(monkeypatch, store)
    spec = importlib.util.spec_from_file_location("api_bridge_asset_nodes_mapping_test", REPO_ROOT / "nodes.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.NODE_CLASS_MAPPINGS["TTSExternalAudioAsset"] is module.ExternalAudioAssetNode
    assert module.NODE_CLASS_MAPPINGS["TTSExternalGPTSovitsEngine"] is module.ExternalGPTSovitsEngineNode
    assert module.NODE_CLASS_MAPPINGS["CharacterVoicesNode"] is module.CharacterVoicesNode


@pytest.mark.unit
@pytest.mark.parametrize(
    ("failed_suffix", "present_mapping", "upstream_mapping"),
    [
        ("api_bridge/audio_asset_node.py", "TTSExternalGPTSovitsEngine", "CharacterVoicesNode"),
        ("api_bridge/resource_engine_nodes.py", "TTSExternalAudioAsset", "CharacterVoicesNode"),
        ("shared/character_voices_node.py", "TTSExternalAudioAsset", "TTSExternalGPTSovitsEngine"),
    ],
)
def test_bridge_and_upstream_node_registration_failures_are_isolated(
    monkeypatch, failed_suffix, present_mapping, upstream_mapping
):
    module = _load_nodes_with_forced_failure(monkeypatch, failed_suffix)

    assert present_mapping in module.NODE_CLASS_MAPPINGS
    assert upstream_mapping in module.NODE_CLASS_MAPPINGS
