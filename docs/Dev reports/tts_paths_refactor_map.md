# TTS model path refactor map

## Summary

`folder_paths.models_dir` returns only the **primary** model root, ignoring any
`--extra-model-paths-config` or `extra_model_paths.yaml` entries for the `TTS`
category. The proper API is `folder_paths.get_folder_paths("TTS")`, which returns
all registered TTS directories.

**Pattern to replace:**

```python
# BEFORE (only searches primary models directory)
os.path.join(folder_paths.models_dir, "TTS", "F5-TTS")
os.path.join(folder_paths.models_dir, "TTS", "ChatterBox")
# etc.

# AFTER (searches all registered TTS roots)
from utils.models.tts_paths import get_tts_root_dirs
for root in get_tts_root_dirs():
    candidate = os.path.join(root, "F5-TTS")
```

## Completed (Wave 1 — F5-TTS)

| File | Call sites patched |
|------|--------------------|
| `utils/models/tts_paths.py` | **New file** — centralized `get_tts_root_dirs()`, `find_tts_model_subdir()`, `find_tts_model_file()` |
| `utils/models/f5tts_manager.py` | `find_f5tts_models()` search_paths + `load_f5tts_model()` f5tts_search_paths |

## Pending (Wave 2 — remaining engines)

These 39 files still use `os.path.join(folder_paths.models_dir, ...)` for TTS
paths. Each should be migrated to `get_tts_root_dirs()` + per-engine subdir join.
The refactor is mechanical: swap the fallback root, preserve the subdirectory
structure and legacy compatibility.

### ChatterBox / ChatterBox 23-Lang (8 files)

| File | Path pattern |
|------|-------------|
| `engines/chatterbox/language_models.py` | `models_dir/TTS/chatterbox`, legacy `models_dir/chatterbox` |
| `engines/chatterbox/tts.py` | `models_dir/chatterbox` |
| `engines/chatterbox/vc.py` | `models_dir/chatterbox` |
| `engines/chatterbox_official_23lang/language_models.py` | `models_dir/TTS/chatterbox_official_23lang` |
| `engines/chatterbox_official_23lang/tts.py` | `models_dir/chatterbox_official_23lang` |
| `engines/chatterbox_official_23lang/vc.py` | `models_dir/chatterbox_official_23lang` |
| `nodes.py` | `models_dir/TTS/chatterbox`, legacy fallbacks |
| `utils/models/manager.py` | `models_dir/TTS/chatterbox`, legacy paths |

### F5-TTS remaining (2 files)

| File | Path pattern |
|------|-------------|
| `nodes/base/f5tts_base_node.py` | `models_dir/TTS`, legacy `models_dir/F5-TTS` |
| `utils/models/unified_model_interface.py` | F5-TTS fallback search paths |

### IndexTTS (4 files)

| File | Path pattern |
|------|-------------|
| `engines/index_tts/index_tts.py` | `models_dir/TTS/IndexTTS` |
| `engines/index_tts/index_tts_downloader.py` | `models_dir/TTS/IndexTTS` |
| `nodes/engines/index_tts_engine_node.py` | `models_dir/TTS/IndexTTS` |
| `nodes/index_tts/qwen_emotion_node.py` | `models_dir/TTS/IndexTTS`, emotion resources |
| `utils/text/index_tts_emotion.py` | `models_dir/TTS/IndexTTS` |

### RVC (3 files)

| File | Path pattern |
|------|-------------|
| `nodes/engines/rvc_engine_node.py` | `models_dir/TTS/RVC` |
| `nodes/training/rvc_training_config_node.py` | `models_dir/TTS/RVC` |
| `engines/rvc/training/trainer.py` | `models_dir/TTS/RVC/training` |

### CosyVoice (2 files)

| File | Path pattern |
|------|-------------|
| `engines/cosyvoice/cosyvoice.py` | `models_dir/TTS/CosyVoice` |
| `engines/cosyvoice/cosyvoice_downloader.py` | `models_dir/TTS/CosyVoice` |

### Step Audio EditX (3 files)

| File | Path pattern |
|------|-------------|
| `engines/step_audio_editx/step_audio_editx.py` | `models_dir/TTS/step_audio_editx` |
| `engines/step_audio_editx/step_audio_editx_downloader.py` | `models_dir/TTS/step_audio_editx` |
| `nodes/engines/step_audio_editx_engine_node.py` | `models_dir/TTS/step_audio_editx` |
| `nodes/step_audio_editx_special/step_audio_editx_audio_editor_node.py` | `models_dir/TTS/step_audio_editx` |

### Qwen3-TTS / ASR (3 files)

| File | Path pattern |
|------|-------------|
| `engines/qwen3_tts/qwen3_tts.py` | `models_dir/TTS/qwen3_tts` |
| `engines/qwen3_tts/qwen3_tts_downloader.py` | `models_dir/TTS/qwen3_tts` |
| `engines/qwen3_tts/qwen3_asr_downloader.py` | `models_dir/TTS/qwen3_tts` (ASR under same root) |

### Higgs Audio (2 files)

| File | Path pattern |
|------|-------------|
| `engines/higgs_audio/higgs_audio_downloader.py` | `models_dir/TTS/HiggsAudio`, legacy `models_dir/HiggsAudio` |
| `engines/higgs_audio_v3/higgs_audio_v3_downloader.py` | `models_dir/TTS/higgs_audio_v3` |

### MOSS / OmniVoice / Granite / Dots / Fish / VibeVoice (8 files)

| File | Path pattern |
|------|-------------|
| `engines/moss_tts/moss_tts_downloader.py` | `models_dir/TTS/moss_tts` |
| `engines/moss_tts/training/common.py` | `models_dir/TTS/moss_tts/loras` |
| `engines/moss_soundeffect_v2/downloader.py` | `models_dir/TTS/moss_soundeffect_v2` |
| `engines/omnivoice/omnivoice_downloader.py` | `models_dir/TTS/omnivoice` |
| `engines/granite_asr/granite_asr_downloader.py` | `models_dir/TTS/granite_asr` |
| `engines/dots_tts/dots_tts_downloader.py` | `models_dir/TTS/dots_tts` |
| `engines/fish_audio_s2/downloader.py` | `models_dir/TTS/fish_audio_s2` |
| `nodes/engines/chatterbox_engine_node.py` | `models_dir/TTS/chatterbox` |
| `nodes/engines/chatterbox_official_23lang_engine_node.py` | `models_dir/TTS/chatterbox_official_23lang` |
| `nodes/engines/cosyvoice_engine_node.py` | `models_dir/TTS/CosyVoice` |

## Migration recipe

For each file above, the transformation is:

```python
# 1. Remove the direct models_dir join:
#    OLD: os.path.join(folder_paths.models_dir, "TTS", "MyEngine")
# 2. Import the helper (if not already):
#    from utils.models.tts_paths import get_tts_root_dirs
# 3. Replace with root iteration:
#    for root in get_tts_root_dirs():
#        candidate = os.path.join(root, "MyEngine")
#        if os.path.isdir(candidate):
#            ...

# If legacy (non-TTS) paths must also be preserved, add them after the TTS roots:
#    legacy_paths = [os.path.join(folder_paths.models_dir, "MyEngine")]  # pre-TTS era
```

## Verification after each engine's migration

1. Launch ComfyUI through the ordinary launcher (which may not pass --models-directory)
2. Verify the engine node lists local models via `/object_info/<EngineNode>`
3. Submit one minimal synthesis and check decoded audio is non-silent
