"""Engine nodes that resolve private local resources by public IDs only."""

from __future__ import annotations

from typing import Any

from api_bridge.resource_registry import get_resource_registry


def _engine_data(engine: str, adapter_class: str, resource_id: str, config: dict[str, Any]):
    return ({"engine_type": engine, "adapter_class": adapter_class, "config": {"resource_id": resource_id, **config}},)


class ExternalGPTSovitsEngineNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"resource_id": ("STRING", {"default": ""})},
            "optional": {
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "use_fp16": ("BOOLEAN", {"default": True}),
                "text_language": ("STRING", {"default": "zh"}),
                "ref_language": ("STRING", {"default": "zh"}),
                "how_to_cut": (["凑四句一切", "凑50字一切", "按中文句号。切", "按英文句号.切", "按标点符号切", "不切"], {"default": "凑四句一切"}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.6, "max": 1.65, "step": 0.05}),
                "top_k": ("INT", {"default": 15, "min": 1, "max": 100, "step": 1}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("TTS_ENGINE",)
    RETURN_NAMES = ("TTS_engine",)
    FUNCTION = "create_engine"
    CATEGORY = "TTS Audio Suite/API Bridge"

    def create_engine(
        self,
        resource_id: str,
        device: str = "auto",
        use_fp16: bool = True,
        text_language: str = "zh",
        ref_language: str = "zh",
        how_to_cut: str = "凑四句一切",
        speed: float = 1.0,
        top_k: int = 15,
        top_p: float = 1.0,
        temperature: float = 1.0,
    ):
        resource = get_resource_registry().require(resource_id, "gpt_sovits")
        return _engine_data("gpt_sovits", "GPTSovitsAdapter", resource_id, {
            "gpt_weight": str(resource.gpt_weight),
            "sovits_weight": str(resource.sovits_weight),
            "bert_path": str(resource.bert_path or ""),
            "cnhubert_path": str(resource.cnhubert_path or ""),
            "gpt_sovits_home": str(resource.source_root),
            "version": resource.version,
            "device": device,
            "use_fp16": use_fp16,
            "text_language": text_language,
            "ref_language": ref_language,
            "how_to_cut": how_to_cut,
            "speed": speed,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
        })


class ExternalIndexTTSEngineNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"resource_id": ("STRING", {"default": ""})},
            "optional": {
                "device": (["auto", "cuda", "xpu", "cpu", "mps"], {"default": "auto"}),
                "use_fp16": ("BOOLEAN", {"default": True}),
                "emotion_alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "use_random": ("BOOLEAN", {"default": False}),
                "max_text_tokens_per_segment": ("INT", {"default": 120, "min": 50, "max": 300, "step": 10}),
                "interval_silence": ("INT", {"default": 200, "min": 0, "max": 1000, "step": 50}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.05}),
                "top_k": ("INT", {"default": 30, "min": 1, "max": 100, "step": 5}),
                "do_sample": ("BOOLEAN", {"default": True}),
                "length_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.1}),
                "num_beams": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
                "repetition_penalty": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 20.0, "step": 0.1}),
                "max_mel_tokens": ("INT", {"default": 1500, "min": 100, "max": 3000, "step": 100}),
                "use_cuda_kernel": (["auto", "true", "false"], {"default": "auto"}),
                "use_deepspeed": ("BOOLEAN", {"default": False}),
                "use_torch_compile": ("BOOLEAN", {"default": False}),
                "use_accel": ("BOOLEAN", {"default": False}),
                "low_vram": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("TTS_ENGINE",)
    RETURN_NAMES = ("TTS_engine",)
    FUNCTION = "create_engine"
    CATEGORY = "TTS Audio Suite/API Bridge"

    def create_engine(self, resource_id: str, device: str = "auto", use_fp16: bool = True, emotion_alpha: float = 1.0,
                      use_random: bool = False, max_text_tokens_per_segment: int = 120, interval_silence: int = 200,
                      temperature: float = 0.8, top_p: float = 0.8, top_k: int = 30, do_sample: bool = True,
                      length_penalty: float = 0.0, num_beams: int = 3, repetition_penalty: float = 10.0,
                      max_mel_tokens: int = 1500, use_cuda_kernel: str = "auto", use_deepspeed: bool = False,
                      use_torch_compile: bool = False, use_accel: bool = False, low_vram: bool = False):
        resource = get_resource_registry().require(resource_id, "index_tts")
        return _engine_data("index_tts", "IndexTTSAdapter", resource_id, {
            "model_path": str(resource.model_dir), "index_tts_home": str(resource.source_root),
            "device": device, "use_fp16": use_fp16, "emotion_alpha": emotion_alpha, "use_random": use_random,
            "max_text_tokens_per_segment": max_text_tokens_per_segment, "interval_silence": interval_silence,
            "temperature": temperature, "top_p": top_p, "top_k": top_k, "do_sample": do_sample,
            "length_penalty": length_penalty, "num_beams": num_beams, "repetition_penalty": repetition_penalty,
            "max_mel_tokens": max_mel_tokens, "use_cuda_kernel": use_cuda_kernel,
            "use_deepspeed": use_deepspeed, "use_torch_compile": use_torch_compile,
            "use_accel": use_accel, "low_vram": low_vram,
        })


class ExternalCosyVoiceEngineNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"resource_id": ("STRING", {"default": ""})},
            "optional": {
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "use_fp16": ("BOOLEAN", {"default": True}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
                "instruct_text": ("STRING", {"multiline": True, "default": ""}),
                "load_trt": ("BOOLEAN", {"default": False}),
                "load_vllm": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("TTS_ENGINE",)
    RETURN_NAMES = ("TTS_engine",)
    FUNCTION = "create_engine"
    CATEGORY = "TTS Audio Suite/API Bridge"

    def create_engine(self, resource_id: str, device: str = "auto", use_fp16: bool = True, speed: float = 1.0,
                      instruct_text: str = "", load_trt: bool = False, load_vllm: bool = False):
        resource = get_resource_registry().require(resource_id, "cosyvoice")
        return _engine_data("cosyvoice", "CosyVoiceAdapter", resource_id, {
            "model_path": str(resource.model_dir), "cosyvoice_home": str(resource.source_root),
            "device": device, "use_fp16": use_fp16, "speed": speed,
            "instruct_text": instruct_text.strip(), "load_trt": load_trt, "load_vllm": load_vllm,
        })
