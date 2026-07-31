"""Child entrypoint for the plugin-owned official IndexTTS subprocess adapter."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import traceback


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

        from indextts.infer_v2 import IndexTTS2

        imported_source = Path(sys.modules[IndexTTS2.__module__].__file__).resolve()
        if not imported_source.is_relative_to(source_root):
            raise RuntimeError(f"IndexTTS imported outside registered source_root: {imported_source}")

        constructor = dict(payload["constructor"])
        tts = IndexTTS2(
            cfg_path=str(model_dir / "config.yaml"),
            model_dir=str(model_dir),
            **constructor,
        )
        inference = dict(payload["inference"])
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
