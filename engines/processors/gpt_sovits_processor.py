"""Unified text-node processor for local GPT-SoVITS inference."""

from __future__ import annotations

from typing import Any

from engines.adapters.gpt_sovits_adapter import GPTSovitsAdapter


class GPTSovitsProcessor:
    """Own one GPT-SoVITS adapter for a stable unified-engine configuration."""

    def __init__(self, config: dict[str, Any], adapter: GPTSovitsAdapter | None = None) -> None:
        self.config = dict(config)
        self.adapter = adapter or GPTSovitsAdapter()
        self._cleaned = False
        initialize_kwargs = {
            "gpt_weight": self.config["gpt_weight"],
            "sovits_weight": self.config["sovits_weight"],
            "bert_path": self.config.get("bert_path", ""),
            "cnhubert_path": self.config.get("cnhubert_path", ""),
            "device": self.config.get("device", "cuda"),
            "use_fp16": bool(self.config.get("use_fp16", True)),
        }
        if self.config.get("gpt_sovits_home"):
            initialize_kwargs["gpt_sovits_home"] = self.config["gpt_sovits_home"]
        if self.config.get("version"):
            initialize_kwargs["version"] = self.config["version"]
        if self.config.get("python_executable"):
            initialize_kwargs["python_executable"] = self.config["python_executable"]
        self.adapter.initialize_engine(**initialize_kwargs)

    def update_config(self, new_config: dict[str, Any]) -> None:
        self.config.update(new_config)

    def process_text(
        self,
        *,
        text: str,
        speaker_audio: dict[str, Any] | str | None,
        reference_text: str | None,
        seed: int,
        return_info: bool = False,
    ):
        reference_path = speaker_audio.get("audio_path") if isinstance(speaker_audio, dict) else speaker_audio
        waveform, sample_rate = self.adapter.generate(
            text=text,
            text_lang=self.config.get("text_language", "zh"),
            ref_audio_path=reference_path,
            ref_text=reference_text,
            ref_lang=self.config.get("ref_language", "zh"),
            speed=float(self.config.get("speed", 1.0)),
            top_k=int(self.config.get("top_k", 15)),
            top_p=float(self.config.get("top_p", 1.0)),
            temperature=float(self.config.get("temperature", 1.0)),
            how_to_cut=self.config.get("how_to_cut", "凑四句一切"),
            seed=seed,
        )
        self.sample_rate = sample_rate
        duration = waveform.shape[-1] / sample_rate if sample_rate else 0.0
        info = f"GPT-SoVITS generated {duration:.2f}s at {sample_rate}Hz"
        return (waveform, info) if return_info else waveform

    def cleanup(self) -> None:
        if not self._cleaned:
            self.adapter.unload()
            self._cleaned = True

    unload = cleanup
