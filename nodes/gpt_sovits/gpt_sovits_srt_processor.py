"""
GPT-SoVITS SRT Processor

Handles subtitle-based TTS generation using GPT-SoVITS.
Follows the suite pattern: SRT processor imports the main TTS processor
and reuses its generation path for each subtitle segment.
"""

import os
import sys
import torch
import importlib.util

current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import folder_paths
import comfy.model_management as model_management


class GPTSovitsSRTProcessor:
    """Processes SRT subtitle content for GPT-SoVITS TTS."""

    def __init__(self):
        self.tts_processor = None  # Lazy import to avoid circular deps

    def _get_tts_processor(self):
        if self.tts_processor is None:
            spec = importlib.util.spec_from_file_location(
                "gpt_sovits_processor_local",
                os.path.join(current_dir, "gpt_sovits_processor.py")
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.tts_processor = module.GPTSovitsProcessor()
        return self.tts_processor

    def process_srt(
        self,
        adapter,
        srt_content: str,
        character_voices: dict = None,
        narrator_voice: dict = None,
        timing_mode: str = "concatenate",
        **kwargs,
    ) -> tuple:
        """Process SRT content and generate timed audio.

        Args:
            adapter: Initialized GPTSovitsAdapter
            srt_content: SRT subtitle text
            character_voices: Character voice mapping
            narrator_voice: Default narrator voice
            timing_mode: Assembly timing mode
            **kwargs: Additional parameters

        Returns:
            (waveform, sample_rate, timing_info)
        """
        from utils.timing.parser import parse_srt
        from utils.timing.engine import TimingEngine
        from utils.timing.assembly import AudioAssemblyEngine
        from utils.timing.reporting import SRTReportGenerator, generate_adjusted_srt_string

        processor = self._get_tts_processor()

        # Parse SRT
        subtitles = parse_srt(srt_content)
        if not subtitles:
            raise ValueError("No subtitles found in SRT content")

        # Generate audio for each subtitle
        audio_segments = []
        timing_info = []

        for i, sub in enumerate(subtitles):
            if model_management.interrupt_processing:
                raise InterruptedError(f"GPT-SoVITS SRT interrupted at segment {i+1}/{len(subtitles)}")

            text = sub.get("text", "").strip()
            if not text:
                continue

            waveform, sr = processor.process_text(
                adapter,
                text,
                character_voices=character_voices,
                narrator_voice=narrator_voice,
                engine_config=kwargs.get("engine_config", {}),
            )

            audio_segments.append(waveform)
            timing_info.append({
                "index": i,
                "start_time": sub.get("start", 0),
                "end_time": sub.get("end", 0),
                "text": text,
                "duration": waveform.shape[-1] / sr,
            })

        if not audio_segments:
            raise RuntimeError("No audio generated from SRT")

        # Assemble
        engine = AudioAssemblyEngine()
        combined, adjusted_srt = engine.assemble_with_overlaps(
            audio_segments, subtitles, sr, mode=timing_mode
        )

        return combined, sr, {"segments": timing_info, "adjusted_srt": adjusted_srt}
