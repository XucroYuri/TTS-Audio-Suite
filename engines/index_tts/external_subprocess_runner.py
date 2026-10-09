"""Child entrypoint for the plugin-owned official IndexTTS subprocess adapter."""

from __future__ import annotations

import json
import inspect
import os
from pathlib import Path
import sys
import traceback


def _constructor_kwargs(model_class, supplied):
    options = dict(supplied)
    parameters = inspect.signature(model_class).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return options
    if 'is_fp16' in parameters and 'use_fp16' not in parameters:
        options['is_fp16'] = options.pop('use_fp16', False)
    for key in ('use_deepspeed', 'use_torch_compile', 'use_accel'):
        if key not in parameters and key in options:
            if options[key] not in (False, None):
                raise ValueError(f'This IndexTTS checkout does not support {key}')
            del options[key]
    return options


def _inference_kwargs(infer, supplied):
    options = dict(supplied)
    parameters = inspect.signature(infer).parameters
    if 'max_text_tokens_per_sentence' in parameters and 'max_text_tokens_per_segment' not in parameters:
        options['max_text_tokens_per_sentence'] = options.pop('max_text_tokens_per_segment', 120)
    # These are wrapper controls, not Hugging Face generation arguments. Older
    # checkouts forward unknown kwargs to generate(), where even defaults fail.
    for key in ('stream_return', 'more_segment_before'):
        if key not in parameters and key in options:
            if options[key] not in (False, 0, None):
                raise ValueError(f'This IndexTTS checkout does not support {key}')
            del options[key]
    return options


def _prepare_local_hub_cache(model_dir):
    if (model_dir / 'hub').is_dir():
        os.environ.setdefault('HF_HOME', str(model_dir))
        os.environ.setdefault('HF_HUB_CACHE', str(model_dir / 'hub'))
        # Some portable checkouts rewrite HF_HUB_CACHE on import. Initialize
        # Hub constants first so their already-populated cache remains usable.
        from huggingface_hub import constants  # noqa: F401


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 1:
        print("usage: external_subprocess_runner.py <request.json>", file=sys.stderr)
        return 2

    try:
        payload = json.loads(Path(arguments[0]).read_text(encoding="utf-8"))
        source_root = Path(payload["source_root"]).resolve()
        model_dir = Path(payload["model_dir"]).resolve()
        output_path = Path(payload["output_path"]).resolve()
        sys.path.insert(0, str(source_root))

        _prepare_local_hub_cache(model_dir)
        from indextts.infer_v2 import IndexTTS2

        imported_source = Path(sys.modules[IndexTTS2.__module__].__file__).resolve()
        if not imported_source.is_relative_to(source_root):
            raise RuntimeError(f"IndexTTS imported outside registered source_root: {imported_source}")

        constructor = _constructor_kwargs(IndexTTS2, payload['constructor'])
        tts = IndexTTS2(
            cfg_path=str(model_dir / "config.yaml"),
            model_dir=str(model_dir),
            **constructor,
        )
        inference = _inference_kwargs(tts.infer, payload['inference'])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tts.infer(output_path=str(output_path), **inference)
        if not output_path.is_file():
            raise RuntimeError("IndexTTS2.infer returned without writing the output WAV")
        return 0
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
