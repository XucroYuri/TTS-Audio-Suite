# DramaBox and Chatterbox Multilingual V3 Capability and Scope

Research date: 2026-07-25

## Official references

- DramaBox code: `resemble-ai/DramaBox` at
  `a70a5818e103c1c9fef22409c1e0c707ebf4f8a7`
- DramaBox weights: `ResembleAI/Dramabox` at
  `404f967f653fa1170dc15a9d1ddd3fdb9a0a842d`
- Chatterbox code: `resemble-ai/chatterbox` at
  `5de7a54aa4e5e2baadb0182dde554908b48b85c2`
- Chatterbox weights: `ResembleAI/chatterbox` at
  `5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18`
- ComfyUI reference only: `kat3ri/ComfyUI-DramaBox` at
  `715fcb11cc14d8c185438e2319b52fc00163941c`

The repositories were cloned under
`IgnoredForGitHubDocs/For_reference/`.

## DramaBox capability report

### Native scope

- Task: English text-to-speech with optional zero-shot voice cloning.
- Expressive control: prompt-driven speaker description, delivery, emotion,
  pauses, laughs, sighs, and transitions.
- Voice input: optional reference audio; upstream uses up to 10 seconds.
- No native voice conversion, ASR, or audio editing API.
- No language control. The official model is English-only.
- No extra special node is required. Its structured scene prompt fits the
  existing unified text and SRT nodes.

### Native generation parameters

- `cfg_scale` (official warm-server default `2.5`)
- `stg_scale` (default `1.5`)
- `duration_multiplier` (default `1.1`)
- `seed` (default `42`)
- `ref_duration` (default `10.0` seconds)
- `rescale_scale` (`auto` by default)
- `gen_duration` (`0` means automatic)
- Official long-form chunk limits and crossfade parameters

The initial suite UI exposes `cfg_scale`, `stg_scale`, and
`duration_multiplier`. Seed remains owned by the unified TTS nodes.
Reference duration, rescale, steps, modality guidance, and explicit output
duration stay on official defaults because exposing them would add expert
controls without a demonstrated suite use case. The official duration-aware
long-form path is used automatically instead of adding duplicate chunk UI.

### Audio and generation behavior

- The LTX audio decoder returns stereo audio at 48 kHz.
- The base model was trained on clips around 20 seconds. Current upstream
  supports longer clips with a silence-prior correction and automatically
  chunks prompts targeting about 37 seconds with a 45-second cap.
- The official long-form chunker preserves the scene/speaker prefix and quote
  groups, then joins chunks with a 50 ms equal-power crossfade.
- Upstream applies the Perth watermark only in `generate_to_file()`, not in
  the in-memory `generate()` method. The suite wrapper must therefore apply
  the watermark to in-memory output explicitly.

### Model layout

Organized destination: `ComfyUI/models/TTS/dramabox/DramaBox/`

- `dramabox-dit-v1.safetensors` — 6,575,225,528 bytes
- `dramabox-audio-components.safetensors` — 1,942,831,020 bytes
- `assets/silence_latent_frame.pt` — 1,501 bytes
- `gemma-3-12b-it-bnb-4bit/`
  - two safetensor shards plus tokenizer/config files from
    `unsloth/gemma-3-12b-it-bnb-4bit`

The implementation must use the suite downloader with `local_dir`-style
organized downloads and disable Transformers/Hugging Face fallback downloads.

### Dependencies and runtime

The official requirements include Torch/Torchaudio 2.8, Transformers 4.45+,
bitsandbytes 0.45+, Accelerate, PEFT, PyAV, Einops, SentencePiece,
Safetensors, PyYAML, and Perth. The official source imports successfully in
the configured suite validation environment with Torch 2.10 and Transformers
5.10, so DramaBox belongs in the main Transformers 5 environment.

The optional NVIDIA RE-USE reference denoiser is intentionally excluded:
its Mamba dependencies have no practical Windows installation path and its
NSCLv1 non-commercial license is a poor default for the suite.

### License

DramaBox code and weights are under the LTX-2 Community License, not MIT.
The license requires attribution, use restrictions, modified-file notices,
and a separate paid license for entities with at least USD 10 million in
annual revenue. The upstream license must ship beside any bundled inference
code, and the engine UI/docs must disclose the restriction.

## Existing ComfyUI reference notes

`kat3ri/ComfyUI-DramaBox` confirms useful ComfyUI audio-shape handling,
organized model paths, the warm `TTSServer` API, and practical UI ranges.
It must not be copied as architecture:

- It auto-clones source code at runtime.
- It has no unified model lifecycle, cache, character/pause integration, SRT
  processor, interrupt handling, or generation report integration.
- It directly calls the engine from a standalone node.
- It patches partially imported bitsandbytes modules globally.
- Its README says output is watermarked, but its node calls the unwatermarked
  in-memory upstream path.

## Chatterbox Multilingual V3 capability report

V3 is not a new engine. Official upstream loads it as an opt-in checkpoint through
`ChatterboxMultilingualTTS.from_pretrained(..., t3_model="v3")`; the only
model-family change is selecting `t3_mtl23ls_v3.safetensors` instead of the
V2 T3 checkpoint. Its official generation path also skips the legacy
alignment analyzer, uses repetition penalty `1.2`, and removes the final
degraded pre-EOS speech-token artifact. It keeps
the same 500M architecture, 23-language list, 24 kHz output, voice-reference
mode, tokenizer, voice encoder, S3Gen decoder, and generation parameters:

- `language_id`
- `exaggeration`
- `cfg_weight`
- `temperature`
- `repetition_penalty`
- `min_p`
- `top_p`

The suite forwards V3 `exaggeration` using the upstream/native scale. Manual
testing found little or no audible response across values, so this remains a
current checkpoint limitation rather than a suite-side scaling issue.

The existing `chatterbox_official_23lang` engine already implements Unified
TTS Text, Unified SRT TTS, voice references, caching, character switching,
pause tags, parameter switching, and lifecycle handling. V3 therefore
extends its `model_version` choices and downloader requirements. It must not
create a second engine node or duplicate processors.

## Integration scope

### DramaBox

- Unified TTS Text: yes
- Unified SRT TTS: yes
- Character tags and narrator fallback: yes
- Pause tags: yes
- Segment switching: `seed`, `cfg_scale`, `stg_scale`, and
  `duration_multiplier`
- Generated audio cache: yes
- Long-form strategy: official duration-aware chunker; ignore suite
  character-count chunking inside each already separated character/pause
  segment
- Clear VRAM: full TTSServer teardown and lazy reload because the quantized
  Gemma stack should not be copied to system RAM
- Runtime: main environment
- Voice Changer / ASR / editing / special node: no

### Chatterbox Multilingual V3

- Extend existing Official 23-Lang model version control with V3.
- Keep V2 available for backward compatibility.
- Make V3 the suite default for new configurations so the newly requested
  version is immediately selected. Upstream still defaults to V2 and exposes
  V3 as opt-in.
- Existing saved workflows with V1/V2 values continue to load unchanged.

## Validation matrix

- Static import and registration checks
- Chatterbox V1/V2/V3 file-resolution tests without downloading weights
- DramaBox downloader layout checks without downloading the 16+ GB models
- DramaBox TTS processor tests with a fake adapter for character, pause,
  cache-facing parameter, audio-shape, and combination behavior
- SRT processor interrupt and timing-path tests with fake generation
- Live FL-MCP checks after restarting ComfyUI:
  - engine and unified nodes register
  - smallest DramaBox text workflow loads
  - smallest DramaBox SRT workflow loads
  - full generation is attempted only if all 16+ GB weights and adequate
    VRAM are available
- Human assessment remains required for subjective audio quality.
