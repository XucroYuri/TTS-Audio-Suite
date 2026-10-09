# API Bridge external-runner interruption

The API Bridge observes a targeted ComfyUI interruption within 250 ms while an
external TTS runner is active. IndexTTS, GPT-SoVITS, and CosyVoice share the
same bounded wait and process-tree cleanup path. If cleanup cannot verify that
the runner tree exited, the Bridge reports cleanup failure instead of reporting
a false interruption success.

## Portable IndexTTS2 checkouts

Set `python_executable` in the private resource registry to use the checkout's
own environment. The Windows fallback recognises both `.venv/Scripts/python.exe`
and `env/python.exe`. Model and source files remain in the upstream checkout.

Older IndexTTS2 releases use `is_fp16` and `max_text_tokens_per_sentence`.
The runner translates those public signatures and keeps wrapper-only controls
out of Transformers generation arguments. Unsupported enabled acceleration or
streaming options fail explicitly; disabled defaults remain compatible.

When `model_dir/hub` exists, the runner initialises Hugging Face's cache constants
before importing the portable checkout. Existing operator cache environment
variables take precedence, and offline mode is not forced.

## GPT-SoVITS checkpoint selection

`TTSExternalGPTSovitsEngine` accepts optional `gpt_checkpoint` and
`sovits_checkpoint` filenames. Empty values use the registered pair. A selection
must be a regular `.ckpt` or `.pth` file in the corresponding registered weight
directory. Absolute paths, directory traversal, links, junctions, and other
checkpoint directories are rejected. Source/runtime and pretrained components
continue to come from `resource_id`; upstream code remains unchanged.
