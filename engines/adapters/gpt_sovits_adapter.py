"""Official GPT-SoVITS ``TTS_Config`` / ``TTS.run`` adapter."""

from __future__ import annotations

import gc
import importlib
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from engines.gpt_sovits.runtime import GPTSovitsRuntimeContext, checkout_cwd, configure_gpt_sovits_source
from engines.gpt_sovits.external_subprocess import ExternalGPTSovitsSubprocessProxy
from utils.audio.cache import AudioCache, get_audio_cache
from utils.device import resolve_torch_device
from utils.text.character_parser import character_parser


_CUT_METHODS = {
    "不切": "cut0",
    "凑四句一切": "cut1",
    "凑50字一切": "cut2",
    "按中文句号。切": "cut3",
    "按英文句号.切": "cut4",
    "按标点符号切": "cut5",
}


class GPTSovitsAdapter:
    """Run the official local GPT-SoVITS inference library exactly once."""

    def __init__(self, audio_cache: Optional[AudioCache] = None) -> None:
        self.audio_cache = audio_cache or get_audio_cache()
        self.runtime = None
        self.runtime_config = None
        self._current_gpt_path: Optional[str] = None
        self._current_sovits_path: Optional[str] = None
        self._character_profiles: Dict[str, Dict[str, Any]] = {}
        self._device = "cpu"
        self._use_fp16 = False
        self._version = "v2"
        self.sample_rate = 32000
        self._runtime_context: Optional[GPTSovitsRuntimeContext] = None

    def _import_official_runtime(self):
        """Import only the stable official library API, never the WebUI entrypoint."""
        if self._runtime_context is None:
            raise RuntimeError("GPT-SoVITS checkout root is required for official runtime imports")
        tts_module = importlib.import_module("TTS_infer_pack.TTS")
        sv_module = importlib.import_module("sv")
        importlib.import_module("ERes2NetV2")
        tts_module.now_dir = str(self._runtime_context.checkout_root)
        sv_module.sv_path = str(self._runtime_context.sv_path)
        return tts_module.TTS_Config, tts_module.TTS

    @staticmethod
    def _normalize_default_configs(runtime_config, context: GPTSovitsRuntimeContext) -> None:
        for config in getattr(runtime_config, "default_configs", {}).values():
            for key in (
                "t2s_weights_path", "vits_weights_path", "bert_base_path", "cnhuhbert_base_path",
            ):
                path = config.get(key)
                if path and not os.path.isabs(path):
                    config[key] = str(context.checkout_root / path)

    def initialize_engine(
        self,
        gpt_weight: str,
        sovits_weight: str,
        bert_path: str,
        cnhubert_path: str,
        device: str = "auto",
        use_fp16: bool = True,
        character_profiles: Optional[Dict[str, Dict[str, Any]]] = None,
        gpt_sovits_home: Optional[str] = None,
        python_executable: Optional[str] = None,
        sv_path: Optional[str] = None,
        runtime_root: Optional[str] = None,
        version: str = "v2",
    ) -> None:
        if not gpt_sovits_home:
            context = configure_gpt_sovits_source()
            gpt_sovits_home = str(context.checkout_root) if context is not None else None
        if not gpt_sovits_home:
            raise RuntimeError("gpt_sovits_home or GPT_SOVITS_PATH must point to an official GPT-SoVITS checkout")
        resolved_device = resolve_torch_device(device)
        normalized_fp16 = bool(use_fp16) and resolved_device != "cpu"
        default_python = os.path.join(
            gpt_sovits_home,
            ".venv",
            "Scripts" if os.name == "nt" else "bin",
            "python.exe" if os.name == "nt" else "python",
        )
        requested_python = python_executable or default_python

        if character_profiles:
            self._character_profiles = dict(character_profiles)

        unchanged = (
            self.runtime is not None
            and self._current_gpt_path == gpt_weight
            and self._current_sovits_path == sovits_weight
            and self._device == resolved_device
            and self._use_fp16 == normalized_fp16
            and self._version == version
            and os.path.normcase(os.path.realpath(str(self.runtime.source_root)))
            == os.path.normcase(os.path.realpath(gpt_sovits_home))
            and os.path.normcase(os.path.realpath(str(self.runtime.python_executable)))
            == os.path.normcase(os.path.realpath(requested_python))
            and os.path.normcase(os.path.realpath(str(getattr(self.runtime, "sv_path", "") or "")))
            == os.path.normcase(os.path.realpath(str(sv_path or "")))
            and os.path.normcase(os.path.realpath(str(getattr(self.runtime, "runtime_root", "") or "")))
            == os.path.normcase(os.path.realpath(str(runtime_root or "")))
        )
        if unchanged:
            return

        self.unload()
        proxy_kwargs = {
            "source_root": gpt_sovits_home,
            "gpt_weight": gpt_weight,
            "sovits_weight": sovits_weight,
            "bert_path": bert_path,
            "cnhubert_path": cnhubert_path,
            "python_executable": python_executable,
            "device": resolved_device,
            "use_fp16": normalized_fp16,
            "version": version,
        }
        if sv_path:
            proxy_kwargs["sv_path"] = sv_path
        if runtime_root:
            proxy_kwargs["runtime_root"] = runtime_root
        self.runtime = ExternalGPTSovitsSubprocessProxy(
            **proxy_kwargs,
        )
        self._current_gpt_path = gpt_weight
        self._current_sovits_path = sovits_weight
        self._device = resolved_device
        self._use_fp16 = normalized_fp16
        self._version = version
        self.sample_rate = int(getattr(self.runtime_config, "sampling_rate", self.sample_rate))

    def _audio_cache_enabled(self) -> bool:
        return not bool(getattr(self.runtime, "source_root", None))

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
        seed: Optional[int] = None,
        **_: Any,
    ) -> Tuple[torch.Tensor, int]:
        if self.runtime is None:
            raise RuntimeError("Engine not initialized. Call initialize_engine() first.")
        if not ref_audio_path:
            raise ValueError("ref_audio_path is required for GPT-SoVITS")

        if character_parser.CHARACTER_TAG_PATTERN.search(text):
            return self._generate_with_characters(
                text, text_lang, ref_audio_path, ref_text, ref_lang,
                speed, top_k, top_p, temperature, how_to_cut, seed,
            )
        return self._generate_single(
            text, text_lang, ref_audio_path, ref_text, ref_lang,
            speed, top_k, top_p, temperature, how_to_cut, seed,
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
        cache_key = self.audio_cache.generate_cache_key(
            "gpt_sovits",
            gpt_weight=self._current_gpt_path,
            sovits_weight=self._current_sovits_path,
            ref_audio_path=ref_audio_path,
            ref_text=ref_text or "",
            ref_lang=ref_lang,
            text_lang=text_lang,
            text=text,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            speed=speed,
            how_to_cut=how_to_cut,
            seed=seed,
        )
        cache_enabled = self._audio_cache_enabled()
        cached = self.audio_cache.get_cached_audio(cache_key) if cache_enabled else None
        if cached:
            return cached[0], self.sample_rate

        inputs = {
            "text": text,
            "text_lang": text_lang,
            "ref_audio_path": ref_audio_path,
            "prompt_text": ref_text or "",
            "prompt_lang": ref_lang,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "text_split_method": _CUT_METHODS.get(how_to_cut, how_to_cut),
            "speed_factor": speed,
            "seed": seed,
        }
        result = self.runtime.run(inputs)
        fragments = [result] if isinstance(result, tuple) else list(result)
        if not fragments:
            raise RuntimeError("Official GPT-SoVITS runtime returned no audio")
        sample_rate = int(fragments[0][0])
        if any(int(fragment_rate) != sample_rate for fragment_rate, _ in fragments):
            raise RuntimeError("Official GPT-SoVITS runtime returned inconsistent sample rates")
        waveform = torch.cat([self._as_waveform(audio) for _, audio in fragments], dim=-1)
        self.sample_rate = sample_rate
        if cache_enabled:
            self.audio_cache.cache_audio(cache_key, waveform, waveform.shape[-1] / sample_rate)
        return waveform, sample_rate

    @staticmethod
    def _as_waveform(audio: Any) -> torch.Tensor:
        waveform = torch.as_tensor(np.asarray(audio)).reshape(-1)
        if not waveform.is_floating_point():
            waveform = waveform.to(torch.float32) / 32768.0
        else:
            waveform = waveform.to(torch.float32)
        return waveform.unsqueeze(0)

    def _generate_with_characters(self, text, text_lang, ref_audio, ref_text, ref_lang,
                                  speed, top_k, top_p, temperature, how_to_cut, seed):
        outputs = []
        sample_rate = self.sample_rate
        for character, segment, _ in character_parser.split_by_character(text, include_language=False):
            if not segment.strip():
                continue
            profile = self._character_profiles.get(character or "", {})
            if "gpt_weight" in profile and "sovits_weight" in profile:
                active_source_root = str(self.runtime.source_root)
                active_python_executable = str(self.runtime.python_executable)
                self.initialize_engine(
                    profile["gpt_weight"], profile["sovits_weight"],
                    profile.get("bert_path", ""), profile.get("cnhubert_path", ""),
                    self._device, self._use_fp16, version=profile.get("version", self._version),
                    gpt_sovits_home=active_source_root,
                    python_executable=active_python_executable,
                )
            waveform, sample_rate = self._generate_single(
                segment.strip(), text_lang, profile.get("ref_audio", ref_audio),
                profile.get("ref_text", ref_text), ref_lang, speed, top_k, top_p,
                temperature, how_to_cut, seed,
            )
            outputs.append(waveform)
        if not outputs:
            raise RuntimeError("No audio generated for any character segment")
        return torch.cat(outputs, dim=-1), sample_rate

    def unload(self) -> None:
        if self.runtime is not None and hasattr(self.runtime, "cleanup"):
            self.runtime.cleanup()
        self.runtime = None
        self.runtime_config = None
        self._current_gpt_path = None
        self._current_sovits_path = None
        self._runtime_context = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def to(self, device: str):
        self._device = resolve_torch_device(device)
        return self

    def get_sample_rate(self) -> int:
        return self.sample_rate

    def get_supported_languages(self) -> List[str]:
        return ["zh", "en", "ja", "yue", "ko", "auto"]
