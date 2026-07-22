"""
GPT-SoVITS Engine Adapter

Bridges the TTS Audio Suite unified interface to the GPT-SoVITS dual-model
architecture. Handles character switching (different weight pairs per character),
reference audio processing, caching, and parameter mapping.

Key difference from other adapters: GPT-SoVITS requires loading TWO model weights
(GPT + SoVITS) per character/voice, not just a single model.
"""

import os
from typing import TYPE_CHECKING, Dict, Any, Optional, List, Tuple

import torch

from engines.gpt_sovits.runtime import configure_gpt_sovits_source
from utils.audio.cache import get_audio_cache
from utils.text.character_parser import character_parser
from utils.voice.discovery import get_character_mapping
from utils.voice.character_logging import resolved_character_label

if TYPE_CHECKING:
    from engines.gpt_sovits.model_loader import GPTSovitsModelLoader


class GPTSovitsAdapter:
    """Adapter for GPT-SoVITS engine providing unified interface compatibility."""

    def __init__(self):
        self.loader: Optional["GPTSovitsModelLoader"] = None
        self.audio_cache = get_audio_cache()

        # Current loaded weight pair (for hot-switch detection)
        self._current_gpt_path: Optional[str] = None
        self._current_sovits_path: Optional[str] = None

        # Character profiles: character_name → {gpt_path, sovits_path, ref_audio, ref_text}
        self._character_profiles: Dict[str, Dict] = {}

        # Default paths
        self._bert_path: Optional[str] = None
        self._cnhubert_path: Optional[str] = None

    def initialize_engine(
        self,
        gpt_weight: str,
        sovits_weight: str,
        bert_path: str,
        cnhubert_path: str,
        device: str = "cuda",
        use_fp16: bool = True,
        character_profiles: Optional[Dict[str, Dict]] = None,
        gpt_sovits_home: Optional[str] = None,
    ):
        """Initialize the GPT-SoVITS engine with a weight pair.

        Args:
            gpt_weight: Path to GPT .ckpt file
            sovits_weight: Path to SoVITS .pth file
            bert_path: Path to Chinese BERT model directory
            cnhubert_path: Path to Chinese HuBERT model directory
            device: Target device
            use_fp16: Use FP16 precision
            character_profiles: Dict of character_name → {gpt_weight, sovits_weight, ref_audio, ref_text}
        """
        configure_gpt_sovits_source(gpt_sovits_home)
        self._bert_path = bert_path
        self._cnhubert_path = cnhubert_path

        if character_profiles:
            self._character_profiles = character_profiles

        self._load_weights(gpt_weight, sovits_weight, device, use_fp16)

    def _load_weights(
        self,
        gpt_weight: str,
        sovits_weight: str,
        device: str = "cuda",
        use_fp16: bool = True,
    ):
        """Load or hot-switch model weights."""
        if (
            self._current_gpt_path == gpt_weight
            and self._current_sovits_path == sovits_weight
            and self.loader is not None
        ):
            return  # Already loaded

        # Unload previous
        if self.loader is not None:
            self.loader.unload()

        print(f"🔄 Loading GPT-SoVITS weights:")
        print(f"   GPT: {os.path.basename(gpt_weight)}")
        print(f"   SoVITS: {os.path.basename(sovits_weight)}")

        from engines.gpt_sovits.model_loader import GPTSovitsModelLoader

        self.loader = GPTSovitsModelLoader(
            gpt_path=gpt_weight,
            sovits_path=sovits_weight,
            bert_path=self._bert_path,
            cnhubert_path=self._cnhubert_path,
            device=device,
            is_half=use_fp16,
        )
        self.loader.load_all()

        self._current_gpt_path = gpt_weight
        self._current_sovits_path = sovits_weight

    def generate(
        self,
        text: str,
        text_lang: str = "zh",
        ref_audio_path: Optional[str] = None,
        ref_text: Optional[str] = None,
        ref_lang: str = "zh",
        speed: float = 1.0,
        top_k: int = 15,
        top_p: float = 1.0,
        temperature: float = 1.0,
        how_to_cut: str = "凑四句一切",
        model_version_override: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, int]:
        """Generate speech from text.

        Args:
            text: Target text (supports [CharacterName] tags)
            text_lang: Language of target text
            ref_audio_path: Reference audio file path
            ref_text: Transcript of reference audio
            ref_lang: Language of reference audio
            speed: Speech speed
            top_k, top_p, temperature: GPT sampling params
            how_to_cut: Text splitting method

        Returns:
            (waveform, sample_rate) where waveform is [1, samples]
        """
        if self.loader is None:
            raise RuntimeError("Engine not initialized. Call initialize_engine() first.")

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # Process character tags
        has_character_tags = character_parser.CHARACTER_TAG_PATTERN.search(text) is not None

        if has_character_tags:
            return self._generate_with_characters(
                text, text_lang, ref_audio_path, ref_text, ref_lang,
                speed, top_k, top_p, temperature, how_to_cut, seed
            )

        # Single segment generation
        if not ref_audio_path:
            raise ValueError("ref_audio_path is required for GPT-SoVITS")

        return self._generate_single(
            text, text_lang, ref_audio_path, ref_text, ref_lang,
            speed, top_k, top_p, temperature, how_to_cut, seed
        )

    def _generate_single(
        self,
        text: str,
        text_lang: str,
        ref_audio_path: str,
        ref_text: Optional[str],
        ref_lang: str,
        speed: float,
        top_k: int,
        top_p: float,
        temperature: float,
        how_to_cut: str,
        seed: Optional[int],
    ) -> Tuple[torch.Tensor, int]:
        """Generate audio for a single text segment."""
        # Cache key
        cache_key = self.audio_cache.generate_cache_key(
            "gpt_sovits",
            text=text,
            ref_audio=ref_audio_path,
            gpt=self._current_gpt_path,
            sovits=self._current_sovits_path,
            speed=speed,
            temperature=temperature,
            seed=seed,
        )

        cached = self.audio_cache.get_cached_audio(cache_key)
        if cached:
            return cached[0], self.loader.hps.data.sampling_rate

        from engines.gpt_sovits.inference import get_tts_wav

        sr, waveform = get_tts_wav(
            self.loader,
            ref_wav_path=ref_audio_path,
            prompt_text=ref_text or "",
            prompt_language=ref_lang,
            text=text,
            text_language=text_lang,
            how_to_cut=how_to_cut,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            speed=speed,
        )

        # Cache result
        duration = waveform.shape[-1] / sr
        self.audio_cache.cache_audio(cache_key, waveform, duration)

        return waveform, sr

    def _generate_with_characters(
        self,
        text: str,
        text_lang: str,
        default_ref_audio: Optional[str],
        default_ref_text: Optional[str],
        ref_lang: str,
        speed: float,
        top_k: int,
        top_p: float,
        temperature: float,
        how_to_cut: str,
        seed: Optional[int],
    ) -> Tuple[torch.Tensor, int]:
        """Generate audio with character tag processing."""
        segments = character_parser.split_by_character(text, include_language=False)

        all_waveforms = []
        sample_rate = self.loader.hps.data.sampling_rate

        for character, seg_text, _ in segments:
            if not seg_text.strip():
                continue

            # Determine character-specific config
            ref_audio = default_ref_audio
            ref_text = default_ref_text

            if character and character in self._character_profiles:
                profile = self._character_profiles[character]
                # Hot-switch weights if needed
                if "gpt_weight" in profile and "sovits_weight" in profile:
                    self._load_weights(
                        profile["gpt_weight"],
                        profile["sovits_weight"],
                        str(self.loader.device),
                        self.loader.is_half,
                    )
                if "ref_audio" in profile:
                    ref_audio = profile["ref_audio"]
                if "ref_text" in profile:
                    ref_text = profile["ref_text"]

            if not ref_audio:
                print(f"⚠️ No reference audio for character '{character}', skipping")
                continue

            wf, sr = self._generate_single(
                seg_text.strip(), text_lang, ref_audio, ref_text, ref_lang,
                speed, top_k, top_p, temperature, how_to_cut, seed
            )
            all_waveforms.append(wf)

        if not all_waveforms:
            raise RuntimeError("No audio generated for any character segment")

        combined = torch.cat(all_waveforms, dim=-1)
        return combined, sample_rate

    def unload(self):
        """Unload engine to free memory."""
        if self.loader:
            self.loader.unload()
            self.loader = None
        self._current_gpt_path = None
        self._current_sovits_path = None

    def to(self, device):
        """Move models to device (for ComfyUI Clear VRAM)."""
        if self.loader:
            self.loader.to(device)
        return self

    def get_sample_rate(self) -> int:
        """Get native sample rate."""
        return self.loader.hps.data.sampling_rate if self.loader else 32000

    def get_supported_languages(self) -> List[str]:
        """Get supported languages."""
        return ["zh", "en", "ja", "yue", "ko", "auto"]
