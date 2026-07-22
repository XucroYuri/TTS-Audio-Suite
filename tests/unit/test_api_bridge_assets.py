import importlib.util
import io
from contextlib import contextmanager
from pathlib import Path
import sys
import threading
import time
import types
import wave

import pytest
import soundfile
import torch

from api_bridge.assets import (
    AssetInUseError,
    AssetQuotaError,
    AudioAssetStore,
    get_audio_asset_store,
    pin_voice_asset,
    reset_audio_asset_store_for_tests,
)


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


def test_asset_store_pin_prevents_delete_until_generation_releases_it(tmp_path: Path):
    store = AudioAssetStore(tmp_path)
    asset = store.create("reference.wav", wav_bytes())

    with store.pin(asset.asset_id):
        with pytest.raises(AssetInUseError, match="asset_in_use"):
            store.delete(asset.asset_id)

    store.delete(asset.asset_id)
    assert not asset.path.exists()


def test_voice_asset_pin_uses_only_the_external_asset_identity(tmp_path: Path):
    store = AudioAssetStore(tmp_path)
    asset = store.create("reference.wav", wav_bytes())
    external_voice = {"asset_id": asset.asset_id, "audio_path": str(asset.path)}

    with pin_voice_asset(external_voice, store=store):
        with pytest.raises(AssetInUseError):
            store.delete(asset.asset_id)

    with pin_voice_asset({"audio_path": str(asset.path)}, store=store):
        pass
    store.delete(asset.asset_id)


def test_voice_asset_pin_normalizes_only_single_wrappers_and_rejects_cycles(tmp_path: Path):
    store = AudioAssetStore(tmp_path)
    asset = store.create("reference.wav", wav_bytes())
    wrapped = [{"asset_id": asset.asset_id}]
    with pin_voice_asset(wrapped, store=store):
        with pytest.raises(AssetInUseError):
            store.delete(asset.asset_id)

    with pin_voice_asset([{"asset_id": asset.asset_id}, {"asset_id": asset.asset_id}], store=store):
        pass
    cyclic = []
    cyclic.append(cyclic)
    with pin_voice_asset(cyclic, store=store):
        pass
    store.delete(asset.asset_id)


def test_asset_store_enforces_total_bytes_and_count_without_deleting_existing_assets(tmp_path: Path):
    content = wav_bytes()
    store = AudioAssetStore(tmp_path, max_total_bytes=len(content) * 2, max_assets=1)
    first = store.create("first.wav", content)

    with pytest.raises(AssetQuotaError, match="asset_quota_exceeded"):
        store.create("second.wav", content)

    assert store.require(first.asset_id).path.exists()
    assert len(list(tmp_path.iterdir())) == 1


def test_asset_store_serializes_concurrent_quota_checks(tmp_path: Path):
    content = wav_bytes()
    store = AudioAssetStore(tmp_path, max_total_bytes=len(content), max_assets=1)
    barrier = threading.Barrier(3)
    outcomes = []

    def create(index: int):
        barrier.wait()
        try:
            outcomes.append(("created", store.create(f"voice-{index}.wav", content).asset_id))
        except AssetQuotaError:
            outcomes.append(("quota", index))

    workers = [threading.Thread(target=create, args=(index,)) for index in range(2)]
    for worker in workers:
        worker.start()
    barrier.wait()
    for worker in workers:
        worker.join(timeout=5)
        assert not worker.is_alive()

    assert sorted(kind for kind, _ in outcomes) == ["created", "quota"]


def test_asset_store_rebuilds_managed_assets_after_restart_and_keeps_invalid_residual_deletable(tmp_path: Path):
    store = AudioAssetStore(tmp_path)
    created = store.create("reference.wav", wav_bytes())
    invalid_id = "a" * 32
    invalid_path = tmp_path / f"{invalid_id}.wav"
    invalid_path.write_bytes(b"not audio")

    rebuilt = AudioAssetStore(tmp_path)

    assert rebuilt.require(created.asset_id).sha256 == created.sha256
    with pytest.raises(ValueError, match="invalid audio"):
        rebuilt.require(invalid_id)
    rebuilt.delete(invalid_id)
    assert not invalid_path.exists()


def test_rebuild_conflicting_uuid_files_are_counted_rejected_and_deleted_together(tmp_path: Path):
    asset_id = "b" * 32
    wav = tmp_path / f"{asset_id}.wav"
    flac = tmp_path / f"{asset_id}.flac"
    wav.write_bytes(wav_bytes())
    flac.write_bytes(wav_bytes())
    store = AudioAssetStore(tmp_path, max_total_bytes=len(wav_bytes()) * 2, max_assets=2)

    with pytest.raises(ValueError, match="conflict"):
        store.require(asset_id)
    assert store._total_bytes == wav.stat().st_size + flac.stat().st_size
    assert store._managed_paths[asset_id] == (flac.resolve(), wav.resolve())
    store.delete(asset_id)
    assert not wav.exists() and not flac.exists()
    assert store._total_bytes == 0


def test_conflict_delete_keeps_retryable_state_when_a_later_unlink_fails(tmp_path: Path, monkeypatch):
    asset_id = "c" * 32
    flac = tmp_path / f"{asset_id}.flac"
    wav = tmp_path / f"{asset_id}.wav"
    flac.write_bytes(wav_bytes())
    wav.write_bytes(wav_bytes())
    store = AudioAssetStore(tmp_path)
    original_unlink = Path.unlink
    failed = False

    def fail_second(self, *args, **kwargs):
        nonlocal failed
        if self == wav and not failed:
            failed = True
            raise PermissionError("locked")
        return original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_second)
    with pytest.raises(PermissionError):
        store.delete(asset_id)
    assert store._managed_paths[asset_id] == (wav.resolve(),)
    assert store._total_bytes == wav.stat().st_size
    assert store._file_count == 1
    store.delete(asset_id)
    assert store._total_bytes == 0 and store._file_count == 0


@pytest.mark.unit
def test_asset_store_rejects_missing_and_replaced_content_before_a_node_can_load_it(tmp_path: Path, monkeypatch):
    store = AudioAssetStore(tmp_path)
    missing = store.create("missing.wav", wav_bytes())
    missing.path.unlink()

    with pytest.raises(ValueError, match="missing or tampered"):
        store.require(missing.asset_id)

    asset = store.create("reference.wav", wav_bytes())
    module = _load_audio_node(monkeypatch, store)
    monkeypatch.setattr(module, "_load_snapshot", lambda _: pytest.fail("tampered asset reached the audio loader"))
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
    construction_entered = threading.Event()
    release_construction = threading.Event()
    construction_count = 0
    construction_count_lock = threading.Lock()

    class BarrierStore(original_store):
        def __init__(self, *args, **kwargs):
            nonlocal construction_count
            with construction_count_lock:
                construction_count += 1
            construction_entered.set()
            assert release_construction.wait(timeout=5)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(assets, "AudioAssetStore", BarrierStore)
    reset_audio_asset_store_for_tests()
    results = []
    results_lock = threading.Lock()

    def get_store():
        result = get_audio_asset_store()
        with results_lock:
            results.append(result)

    first_thread = threading.Thread(target=get_store)
    first_thread.start()
    assert construction_entered.wait(timeout=5)
    threads = [first_thread, *(threading.Thread(target=get_store) for _ in range(3))]
    for thread in threads[1:]:
        thread.start()
    time.sleep(0.1)
    observed_construction_count = construction_count
    release_construction.set()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()

    assert len(results) == 4
    assert observed_construction_count == 1
    assert all(item is results[0] for item in results)
    reset_audio_asset_store_for_tests()


@pytest.mark.unit
def test_reset_cannot_be_undone_by_a_getter_already_constructing_a_store(tmp_path: Path, monkeypatch):
    import api_bridge.assets as assets
    import folder_paths

    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(tmp_path))
    original_store = assets.AudioAssetStore
    construction_entered = threading.Event()
    release_construction = threading.Event()
    reset_done = threading.Event()

    class BlockingStore(original_store):
        def __init__(self, *args, **kwargs):
            construction_entered.set()
            assert release_construction.wait(timeout=5)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(assets, "AudioAssetStore", BlockingStore)
    reset_audio_asset_store_for_tests()
    getter_thread = threading.Thread(target=get_audio_asset_store)
    reset_thread = threading.Thread(target=lambda: (reset_audio_asset_store_for_tests(), reset_done.set()))
    getter_thread.start()
    assert construction_entered.wait(timeout=5)
    reset_thread.start()
    time.sleep(0.1)
    reset_finished_while_constructing = reset_done.is_set()
    release_construction.set()
    getter_thread.join(timeout=5)
    reset_thread.join(timeout=5)

    assert not getter_thread.is_alive()
    assert not reset_thread.is_alive()
    assert not reset_finished_while_constructing
    assert reset_done.is_set()
    assert assets._audio_asset_store is None


@pytest.mark.unit
def test_external_audio_asset_node_returns_a_decodable_narrator_voice(tmp_path: Path, monkeypatch):
    store = AudioAssetStore(tmp_path)
    asset = store.create("reference.wav", wav_bytes())
    module = _load_audio_node(monkeypatch, store)

    (voice,) = module.ExternalAudioAssetNode().load_asset(asset.asset_id, "参考文本")

    assert module.ExternalAudioAssetNode.RETURN_TYPES == ("NARRATOR_VOICE",)
    assert voice["reference_text"] == "参考文本"
    assert voice["asset_id"] == asset.asset_id
    assert voice["audio_path"] == str(asset.path)
    assert voice["character_name"] == "external"
    assert voice["audio"]["sample_rate"] == 16000
    assert tuple(voice["audio"]["waveform"].shape) == (1, 1, 160)
    assert soundfile.info(asset.path).samplerate == voice["audio"]["sample_rate"]


@pytest.mark.unit
def test_audio_asset_node_decodes_the_verified_snapshot_when_source_is_replaced_and_restored(
    tmp_path: Path, monkeypatch
):
    store = AudioAssetStore(tmp_path)
    asset = store.create("reference.wav", wav_bytes())
    module = _load_audio_node(monkeypatch, store)
    original_content = asset.path.read_bytes()
    original_lease = store.lease

    @contextmanager
    def replace_after_snapshot(asset_id: str):
        with original_lease(asset_id) as snapshot:
            asset.path.write_bytes(wav_bytes(b"\x01\x00"))
            try:
                yield snapshot
            finally:
                asset.path.write_bytes(original_content)

    monkeypatch.setattr(store, "lease", replace_after_snapshot)

    (voice,) = module.ExternalAudioAssetNode().load_asset(asset.asset_id, "参考文本")

    assert torch.count_nonzero(voice["audio"]["waveform"]) == 0


@pytest.mark.unit
def test_audio_asset_delete_waits_for_an_active_node_load(tmp_path: Path, monkeypatch):
    store = AudioAssetStore(tmp_path)
    asset = store.create("reference.wav", wav_bytes())
    module = _load_audio_node(monkeypatch, store)
    entered_load = threading.Event()
    release_load = threading.Event()
    loaded_voice = []
    delete_done = threading.Event()

    def slow_load(content: bytes):
        entered_load.set()
        assert release_load.wait(timeout=5)
        samples, sample_rate = soundfile.read(io.BytesIO(content), dtype="float32", always_2d=True)
        return torch.from_numpy(samples.T), sample_rate

    monkeypatch.setattr(module, "_load_snapshot", slow_load)
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
