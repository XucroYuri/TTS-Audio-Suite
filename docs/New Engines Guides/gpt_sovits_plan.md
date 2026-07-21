# GPT-SoVITS Engine Integration Plan

> **Status**: Implementation Phase
> **Branch**: `gpt-sovits-integration`
> **Scope**: v2 / v2Pro / v2ProPlus (mandatory), v1 / v3 / v4 (optional future)

---

## 1. Overview

Add GPT-SoVITS as a new TTS engine to TTS Audio Suite, enabling ComfyUI workflows to use GPT-SoVITS voice cloning models trained through the official WebUI's training pipeline.

### Key Design Principles

1. **Full compatibility with official WebUI training artifacts**: Users should be able to drop their `GPT_weights_v*/` and `SoVITS_weights_v*/` directories directly into `ComfyUI/models/TTS/GPT-SoVITS/` and have them discovered automatically.
2. **Character = exp_name binding**: Each "character" maps to a specific `exp_name` → a matching pair of GPT + SoVITS weights + reference audio from `logs/{exp_name}/`.
3. **v2 ecosystem priority**: v2, v2Pro, and v2ProPlus share the same `SynthesizerTrn` code path (no CFM, no external vocoder), making them the ideal first milestone.

---

## 2. Official Model Capabilities (v2/v2Pro/v2ProPlus)

| Capability | v2 | v2Pro | v2ProPlus |
|---|---|---|---|
| **Zero-shot cloning** | ✅ (3-10s ref audio) | ✅ | ✅ |
| **Few-shot (fine-tuned)** | ✅ | ✅ | ✅ |
| **Languages** | zh, en, ja, yue, ko, auto | same | same |
| **SoVITS model class** | `SynthesizerTrn` | `SynthesizerTrn` | `SynthesizerTrn` |
| **GPT model** | `Text2SemanticLightningModule` | same (uses s1v3.ckpt) | same |
| **Speaker Verification** | ❌ | ✅ `sv_emb` | ✅ `sv_emb` |
| **External Vocoder** | ❌ | ❌ | ❌ |
| **LoRA** | ❌ | ❌ | ❌ |
| **Sample Rate** | 32000 | 32000 | 32000 |

### Inference Pipeline (shared across v2/v2Pro/v2ProPlus)

```
Reference Audio Processing:
  ref_wav → librosa 16kHz → CNHubert SSL → vq_model.extract_latent() → prompt_semantic
  ref_wav → get_spepc() → refer_spec (mel spectrogram)

Text Processing:
  prompt_text + target_text → phones + BERT features
  all_phoneme_ids = [ref_phones] + [tgt_phones]

GPT Inference:
  t2s_model.infer_panel(phonemes, prompt, bert, top_k, top_p, temperature)
  → pred_semantic

SoVITS Decode:
  vq_model.decode(pred_semantic, phones2, refers, [sv_emb if v2Pro])
  → waveform (32000 Hz)
```

### Version Detection (from official `process_ckpt.py`)

Three-layer detection when loading SoVITS weights:
1. **MD5 hash** (first 8192 bytes) → pretrained model fingerprint table
2. **Binary header** (first 2 bytes): `b"01"=v2`, `b"05"=v2Pro`, `b"06"=v2ProPlus`
3. **File size** (fallback): <700MB → v2

GPT weights carry config in `dict_s1["config"]`, no version detection needed there.

---

## 3. Architecture Design

### File Layout

```
engines/gpt_sovits/
├── __init__.py                    # Package init
├── model_loader.py                # Dual-model loading + version detection
├── inference.py                   # Core TTS inference pipeline
└── weight_scanner.py              # Multi-directory weight discovery

engines/adapters/
└── gpt_sovits_adapter.py          # Suite adapter: generate(), character switching

nodes/gpt_sovits/
├── __init__.py
├── gpt_sovits_processor.py        # TTS processor: orchestration, chunking, tags
└── gpt_sovits_srt_processor.py    # SRT processor: subtitle-based generation

nodes/engines/
└── gpt_sovits_engine_node.py      # ComfyUI engine configuration node
```

### Adapter Interface (critical contract)

```python
class GPTSovitsAdapter:
    def initialize_engine(
        self,
        gpt_weight: str,           # path to .ckpt
        sovits_weight: str,        # path to .pth
        bert_path: str,            # Chinese BERT base path
        cnhubert_path: str,        # Chinese HuBERT base path
        device: str = "auto",
        use_fp16: bool = True,
    ): ...

    def generate(
        self,
        text: str,
        text_lang: str,
        ref_audio_path: str,       # 3-10s reference wav
        ref_text: str,             # transcript of reference audio
        ref_lang: str,
        speed: float = 1.0,
        top_k: int = 15,
        top_p: float = 1.0,
        temperature: float = 1.0,
        text_split_method: str = "cut5",
        **kwargs,
    ) -> tuple[torch.Tensor, int]:  # (waveform [1, samples], sample_rate)

    def unload(self): ...
    def to(self, device): ...       # Required for Clear VRAM
```

### Character Binding Design

```python
# Character profile structure
character_profile = {
    "name": "雷电将军",
    "exp_name": "raiden_shogun",
    "version": "v2Pro",            # auto-detected
    "gpt_weight": "GPT_weights_v2Pro/raiden_shogun-e15.ckpt",
    "sovits_weight": "SoVITS_weights_v2Pro/raiden_shogun_e8_s200.pth",
    "ref_audio": "logs/raiden_shogun/5-wav32k/sample_01.wav",
    "ref_text": "此刻，寂灭之时",   # from 2-name2text.txt
    "ref_lang": "zh",
}

# When adapter encounters [雷电将军] in text:
# 1. Check if current loaded weights match character's weights
# 2. If not → hot-switch change_gpt_weights() + change_sovits_weights()
# 3. Set ref_audio, ref_text from character profile
# 4. Proceed with normal TTS generation
```

---

## 4. Weight Directory Scanning

### Scan Strategy (`weight_scanner.py`)

Scan six pairs of directories for user-trained checkpoints:

| Version | GPT Directory | SoVITS Directory |
|---|---|---|
| v1 | `GPT_weights/` | `SoVITS_weights/` |
| v2 | `GPT_weights_v2/` | `SoVITS_weights_v2/` |
| v3 | `GPT_weights_v3/` | `SoVITS_weights_v3/` |
| v4 | `GPT_weights_v4/` | `SoVITS_weights_v4/` |
| v2Pro | `GPT_weights_v2Pro/` | `SoVITS_weights_v2Pro/` |
| v2ProPlus | `GPT_weights_v2ProPlus/` | `SoVITS_weights_v2ProPlus/` |

All directories relative to `ComfyUI/models/TTS/GPT-SoVITS/`.

**Matching logic**: GPT weights named `{exp_name}-e{epoch}.ckpt`, SoVITS weights named `{exp_name}_e{epoch}_s{step}.pth`. Pair them by `exp_name`.

**Pretrained models** (for zero-shot, no training):
- v1: `pretrained_models/s2G488k.pth` + `pretrained_models/s1bert25hz-...ckpt`
- v2: `pretrained_models/gsv-v2final-pretrained/s2G2333k.pth` + `...s1bert25hz-5kh...ckpt`
- v2Pro: `pretrained_models/v2Pro/s2Gv2Pro.pth` + `pretrained_models/s1v3.ckpt`
- v2ProPlus: `pretrained_models/v2Pro/s2Gv2ProPlus.pth` + `pretrained_models/s1v3.ckpt`

---

## 5. Model Download Layout

```
ComfyUI/models/TTS/GPT-SoVITS/
├── pretrained_models/
│   ├── chinese-roberta-wwm-ext-large/    # BERT (~1.2GB)
│   ├── chinese-hubert-base/              # CNHubert (~400MB)
│   ├── s2G488k.pth                       # v1 pretrained SoVITS
│   ├── s1bert25hz-...ckpt               # v1 pretrained GPT
│   ├── gsv-v2final-pretrained/           # v2 pretrained
│   │   ├── s2G2333k.pth
│   │   └── s1bert25hz-5kh-...ckpt
│   ├── s1v3.ckpt                        # v3+ GPT base (shared)
│   ├── v2Pro/
│   │   ├── s2Gv2Pro.pth
│   │   ├── s2Dv2Pro.pth
│   │   ├── s2Gv2ProPlus.pth
│   │   └── s2Dv2ProPlus.pth
│   └── sv/                              # Speaker verification model
│       └── pretrained_eres2netv2w24s4ep4.ckpt
├── GPT_weights_v2/                       # User trained checkpoints
├── GPT_weights_v2Pro/
├── GPT_weights_v2ProPlus/
├── SoVITS_weights_v2/
├── SoVITS_weights_v2Pro/
└── SoVITS_weights_v2ProPlus/
```

---

## 6. Implementation Sequence

### Phase 1: Core Engine (this PR)

| # | File | Lines | Description |
|---|---|---|---|
| 1 | `engines/gpt_sovits/__init__.py` | ~10 | Package init |
| 2 | `engines/gpt_sovits/weight_scanner.py` | ~100 | Scan weight directories, discover available models |
| 3 | `engines/gpt_sovits/model_loader.py` | ~300 | Version detection + dual-model loading + SV |
| 4 | `engines/gpt_sovits/inference.py` | ~250 | TTS inference: ref audio → SSL → GPT → SoVITS |
| 5 | `engines/adapters/gpt_sovits_adapter.py` | ~400 | Suite adapter: generate(), character switching, cache |
| 6 | `nodes/gpt_sovits/__init__.py` | ~5 | Package init |
| 7 | `nodes/gpt_sovits/gpt_sovits_processor.py` | ~200 | TTS processor: orchestration layer |
| 8 | `nodes/gpt_sovits/gpt_sovits_srt_processor.py` | ~150 | SRT processor |
| 9 | `nodes/engines/gpt_sovits_engine_node.py` | ~200 | ComfyUI node UI |
| 10 | Registration files (5 files) | ~50 total | nodes.py, adapters/__init__.py, unified_model_interface.py, engine_registry.py, segment_parameters.py |

**Total**: ~1665 lines across 13 files (8 new + 5 modified)

### Phase 2: Future Enhancements (separate PRs)

- v1/v3/v4 support (requires `SynthesizerTrnV3` + CFM + external vocoder)
- Fine-tuning node (expose official training within ComfyUI)
- Streaming mode
- Multi-reference audio fusion (average timbre from multiple references)

---

## 7. Key Technical Decisions

### 7.1 BERT/CNHubert as Shared Globals

GPT-SoVITS's BERT and CNHubert models are shared across all weight pairs. They should be loaded once as module-level singletons, not per-adapter-instance. This saves ~1.6GB VRAM.

### 7.2 Weight Hot-Switching

When character tags cause a switch between different weight pairs:
- GPT model: `torch.load()` new ckpt → create new `Text2SemanticLightningModule` → replace
- SoVITS model: load new pth → detect version → create `SynthesizerTrn` → replace
- Reference audio cache: invalidated on character switch

### 7.3 Reference Audio from logs/

The `logs/{exp_name}/` directory contains paired `(5-wav32k/audio.wav, 2-name2text.txt transcript)` data. The engine node should expose an option to auto-discover these pairs, presenting them as selectable reference audio inputs per character.

### 7.4 ComfyUI VRAM Management

All engine classes must implement `.to(device)` for Clear VRAM support. The GPT-SoVITS engine has multiple components (t2s_model, vq_model, ssl_model, bert_model), all of which must move together.

### 7.5 Isolated Runtime

GPT-SoVITS uses torch 2.x, transformers, and librosa. It should work in the **Main Environment** (not require runtime isolation) since it has no known conflicts with Transformers 5.

---

## 8. Registration Checklist

After implementation, update these files:

- [ ] `nodes.py` — Add `try/except` import block for `gpt_sovits_engine_node`
- [ ] `engines/adapters/__init__.py` — Export `GPTSovitsAdapter`
- [ ] `utils/models/unified_model_interface.py` — Register `GPTSovitsFactory`
- [ ] `utils/models/engine_registry.py` — Add GPT-SoVITS engine definition
- [ ] `utils/text/segment_parameters.py` — Add `"gpt_sovits"` to `PARAMETER_ENGINES`
- [ ] `docs/Dev reports/tts_audio_suite_engines.yaml` — Add engine metadata
- [ ] `PROJECT_INDEX.md` — Add to engines table

---

## 9. Testing Strategy

### Unit Tests (where feasible)
- `weight_scanner`: test directory scanning returns correct exp_name → (gpt_path, sovits_path) pairs
- `model_loader`: test version detection for v2/v2Pro/v2ProPlus checkpoint files

### Integration Tests (in ComfyUI)
1. Load pretrained v2 base model → generate speech from text + reference audio
2. Load user-trained v2Pro checkpoint pair → verify `sv_emb` path works
3. Character switching: `[角色A]text1 [角色B]text2` → verify weight hot-switch
4. SRT processing: multi-line subtitle → timed audio output
5. Clear VRAM → reload → generate again (no device mismatch errors)

---

## 10. References

- Official GPT-SoVITS: <https://github.com/RVC-Boss/GPT-SoVITS>
- Key source files:
  - `GPT_SoVITS/TTS_infer_pack/TTS.py` — Core TTS class (v2+ unified)
  - `GPT_SoVITS/inference_webui.py` — Legacy inference with `get_tts_wav()`
  - `GPT_SoVITS/process_ckpt.py` — Version detection + checkpoint loading
  - `config.py` — Version-to-directory mapping
- Suite references:
  - `engines/adapters/cosyvoice_adapter.py` — Reference adapter pattern
  - `engines/step_audio_editx/` — Wrapper/model lifecycle reference
  - `docs/New Engines Guides/NEW_ENGINE_IMPLEMENTATION_GUIDE.md`
  - `docs/New Engines Guides/fails_to_avoid_TTS_Engine_Implementation.md`
