"""
GPT-SoVITS TTS Processor

Orchestration layer for GPT-SoVITS text-to-speech generation.
Handles reference audio resolution, character tag processing,
and delegates to the adapter for actual inference.
"""

import os
import sys
import torch
import importlib.util

# Add project root to path
current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import folder_paths
import comfy.model_management as model_management

from engines.gpt_sovits.weight_scanner import scan_reference_audio
from utils.audio.cache import get_audio_cache
from utils.text.character_parser import character_parser
from utils.voice.discovery import get_character_mapping, get_available_characters


class GPTSovitsProcessor:
    """Processes text-to-speech requests for the GPT-SoVITS engine."""

    def __init__(self):
        self.audio_cache = get_audio_cache()

    def process_text(
        self,
        adapter,
        text: str,
        character_voices: dict = None,
        narrator_voice: dict = None,
        **kwargs,
    ) -> tuple:
        """Process text and generate audio.

        Args:
            adapter: Initialized GPTSovitsAdapter instance
            text: Input text (may contain [CharacterName] tags)
            character_voices: Character voice mapping dict
            narrator_voice: Default narrator voice config
            **kwargs: Additional generation parameters

        Returns:
            (waveform_tensor, sample_rate)
        """
        engine_config = kwargs.get("engine_config", {})
        text_lang = engine_config.get("text_language", "中文")
        ref_lang = engine_config.get("ref_language", "中文")
        how_to_cut = engine_config.get("how_to_cut", "凑四句一切")
        speed = engine_config.get("speed", 1.0)
        top_k = engine_config.get("top_k", 15)
        top_p = engine_config.get("top_p", 1.0)
        temperature = engine_config.get("temperature", 1.0)

        # Determine reference audio
        ref_audio_path = engine_config.get("ref_audio_override", "")
        ref_text = engine_config.get("ref_text_override", "")

        # Fallback to narrator voice reference
        if not ref_audio_path and narrator_voice:
            ref_audio_path = narrator_voice.get("audio", "")
            ref_text = narrator_voice.get("text", "")

        # Fallback to logs/ directory auto-discovery (from gpt_sovits_home or ComfyUI models)
        if not ref_audio_path:
            logs_dir = engine_config.get("logs_dir", "")
            exp_name = engine_config.get("exp_name", "")
            if logs_dir and exp_name and os.path.isdir(logs_dir):
                refs = scan_reference_audio(logs_dir, exp_name)
                if refs:
                    ref_audio_path = refs[0]["audio"]
                    ref_text = refs[0].get("text", "")
                    print(f"   📁 Auto-discovered ref audio: {os.path.basename(ref_audio_path)}")

        # Process character tags to build character profiles for adapter
        if character_parser.CHARACTER_TAG_PATTERN.search(text):
            self._update_adapter_characters(adapter, character_voices)

        # Check interrupt
        if model_management.interrupt_processing:
            raise InterruptedError("GPT-SoVITS TTS interrupted")

        # Generate
        waveform, sample_rate = adapter.generate(
            text=text,
            text_lang=text_lang,
            ref_audio_path=ref_audio_path,
            ref_text=ref_text,
            ref_lang=ref_lang,
            speed=speed,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            how_to_cut=how_to_cut,
        )

        return waveform, sample_rate

    def _update_adapter_characters(self, adapter, character_voices: dict):
        """Update adapter's character profiles from Character Voices node."""
        if not character_voices:
            return

        profiles = {}
        for char_name, voice_data in character_voices.items():
            if isinstance(voice_data, dict):
                profiles[char_name] = {
                    "ref_audio": voice_data.get("audio", ""),
                    "ref_text": voice_data.get("text", ""),
                }
            elif isinstance(voice_data, (list, tuple)) and len(voice_data) >= 1:
                profiles[char_name] = {
                    "ref_audio": voice_data[0] if voice_data[0] else "",
                    "ref_text": voice_data[1] if len(voice_data) > 1 else "",
                }

        if profiles:
            adapter._character_profiles.update(profiles)
