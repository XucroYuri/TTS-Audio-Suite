"""Regression coverage for older portable IndexTTS2 public signatures."""
import importlib.util
from pathlib import Path

import pytest
import sys
import types

pytestmark = pytest.mark.unit


def runner():
    path = Path(__file__).resolve().parents[2] / 'engines/index_tts/external_subprocess_runner.py'
    spec = importlib.util.spec_from_file_location('portable_index_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_legacy_constructor_maps_precision_and_rejects_enabled_unsupported_features():
    def legacy(cfg_path, model_dir, is_fp16=False, device=None, use_cuda_kernel=None):
        pass

    module = runner()
    supplied = dict(use_fp16=True, device='cuda', use_deepspeed=False, use_torch_compile=False, use_accel=False)
    assert module._constructor_kwargs(legacy, supplied) == {'is_fp16': True, 'device': 'cuda'}
    assert supplied['use_fp16'] is True
    with pytest.raises(ValueError, match='use_deepspeed'):
        module._constructor_kwargs(legacy, {**supplied, 'use_deepspeed': True})


def test_legacy_inference_does_not_forward_wrapper_controls_to_generate():
    captured = {}

    def legacy(text, max_text_tokens_per_sentence=120, **generation_kwargs):
        captured.update(generation_kwargs)
        return max_text_tokens_per_sentence

    options = runner()._inference_kwargs(legacy, dict(text='test', max_text_tokens_per_segment=64, stream_return=False, more_segment_before=0, temperature=0.7))
    assert legacy(**options) == 64
    assert captured == {'temperature': 0.7}


def test_modern_inference_preserves_supported_wrapper_controls():
    def modern(text, max_text_tokens_per_segment=120, stream_return=False, more_segment_before=0, **kwargs):
        pass

    supplied = dict(text='test', max_text_tokens_per_segment=80, stream_return=False, more_segment_before=2)
    assert runner()._inference_kwargs(modern, supplied) == supplied


def test_legacy_inference_rejects_nondefault_streaming_control():
    def legacy(text, **generation_kwargs):
        pass

    with pytest.raises(ValueError, match='stream_return'):
        runner()._inference_kwargs(legacy, {'text':'test', 'stream_return':True})


def test_existing_portable_hub_cache_is_selected_without_forcing_offline(tmp_path, monkeypatch):
    (tmp_path/'hub').mkdir()
    for key in ('HF_HOME', 'HF_HUB_CACHE', 'HF_HUB_OFFLINE'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setitem(sys.modules, 'huggingface_hub', types.SimpleNamespace(constants=object()))
    runner()._prepare_local_hub_cache(tmp_path)
    import os
    assert os.environ['HF_HOME'] == str(tmp_path)
    assert os.environ['HF_HUB_CACHE'] == str(tmp_path/'hub')
    assert 'HF_HUB_OFFLINE' not in os.environ


def test_operator_hub_cache_settings_take_precedence(tmp_path, monkeypatch):
    (tmp_path/'hub').mkdir()
    monkeypatch.setenv('HF_HOME', 'operator-home')
    monkeypatch.setenv('HF_HUB_CACHE', 'operator-cache')
    monkeypatch.setitem(sys.modules, 'huggingface_hub', types.SimpleNamespace(constants=object()))
    runner()._prepare_local_hub_cache(tmp_path)
    import os
    assert os.environ['HF_HOME'] == 'operator-home'
    assert os.environ['HF_HUB_CACHE'] == 'operator-cache'
