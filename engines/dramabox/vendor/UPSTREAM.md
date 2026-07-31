# Bundled DramaBox inference source

This directory contains the inference-critical source copied unchanged from:

- Repository: `https://github.com/resemble-ai/DramaBox`
- Commit: `a70a5818e103c1c9fef22409c1e0c707ebf4f8a7`
- License: LTX-2 Community License Agreement in `LICENSE`

The bundled-code changes are marked inline:

- `ltx2/ltx_pipelines/utils/blocks.py`: local-only Gemma loading prevents
  Transformers from silently downloading outside TTS Audio Suite's organized
  ComfyUI model directory; staged modes can defer and release the warm prompt
  encoder between generation stages.
- `src/inference_server.py`: ComfyUI cancellation exceptions are allowed to
  propagate from progress callbacks instead of being swallowed; the official
  negative-prompt, FP8-cast, compile, and staged-memory controls are exposed to
  the suite wrapper.
