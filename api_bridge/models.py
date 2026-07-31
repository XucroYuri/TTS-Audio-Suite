from dataclasses import dataclass
from pathlib import Path
from typing import Literal

EngineId = Literal["gpt_sovits", "index_tts", "cosyvoice"]


@dataclass(frozen=True)
class TTSResource:
    resource_id: str
    engine: EngineId
    source_root: Path
    model_dir: Path | None = None
    gpt_weight: Path | None = None
    sovits_weight: Path | None = None
    bert_path: Path | None = None
    cnhubert_path: Path | None = None
    python_executable: Path | None = None
    version: str = "v2"
