"""
GPT-SoVITS Engine Configuration Node

Provides comprehensive configuration for GPT-SoVITS voice cloning TTS.
Features dual-weight selection (GPT + SoVITS), reference audio discovery
from logs/ directory, and character profile binding.
"""

import os
import sys
import importlib.util
from typing import List, Dict, Any

# Add project root to path
current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(nodes_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load base node
base_node_path = os.path.join(nodes_dir, "base", "base_node.py")
base_spec = importlib.util.spec_from_file_location("base_node_module", base_node_path)
base_module = importlib.util.module_from_spec(base_spec)
sys.modules["base_node_module"] = base_module
base_spec.loader.exec_module(base_module)
BaseTTSNode = base_module.BaseTTSNode

import folder_paths
from engines.gpt_sovits.weight_scanner import resolve_paths, scan_weights, scan_reference_audio
from utils.device import resolve_torch_device


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


any_typ = AnyType("*")


class GPTSovitsEngineNode(BaseTTSNode):
    """GPT-SoVITS TTS Engine configuration node.

    Select GPT + SoVITS model weights and configure inference parameters.
    """

    @classmethod
    def NAME(cls):
        return "⚙️ GPT-SoVITS Engine"

    @classmethod
    def INPUT_TYPES(cls):
        # Text splitting methods (from official WebUI)
        cut_methods = ["凑四句一切", "凑50字一切", "按中文句号。切", "按英文句号.切", "按标点符号切", "不切"]

        # Languages
        languages = ["中文", "英文", "日文", "粤语", "韩语", "auto"]

        return {
            "required": {
                "weight_pair": ("STRING", {
                    "default": "auto",
                    "tooltip": (
                        "Use auto when the selected source has one weight pair. "
                        "For multiple pairs, use [version] experiment_name or pair:version:experiment. "
                        "This stays usable when gpt_sovits_home points to a local checkout not visible to the static ComfyUI model scan."
                    )
                }),
                "text_language": (languages, {
                    "default": "中文",
                    "tooltip": "Language of the target text to synthesize."
                }),
                "ref_language": (languages, {
                    "default": "中文",
                    "tooltip": "Language of the reference audio transcript."
                }),
                "how_to_cut": (cut_methods, {
                    "default": "凑四句一切",
                    "tooltip": "How to split long text into sentences for processing."
                }),
                "speed": ("FLOAT", {
                    "default": 1.0, "min": 0.6, "max": 1.65, "step": 0.05,
                    "tooltip": "Speech speed factor."
                }),
                "top_k": ("INT", {
                    "default": 15, "min": 1, "max": 100, "step": 1,
                    "tooltip": "GPT top-k sampling parameter."
                }),
                "top_p": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "GPT top-p (nucleus) sampling."
                }),
                "temperature": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "GPT temperature for randomness control."
                }),
            },
            "optional": {
                "gpt_sovits_home": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": (
                        "Path to existing GPT-SoVITS WebUI project root.\n"
                        "When set, reads weights/logs/pretrained models DIRECTLY\n"
                        "from the WebUI installation — no migration needed.\n"
                        "Example: D:\\GPT-SoVITS\\\n\n"
                        "Leave empty to use ComfyUI models/TTS/GPT-SoVITS/"
                    ),
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Device to run inference on."
                }),
                "use_fp16": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use FP16 for faster inference."
                }),
                "ref_audio_override": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Override reference audio path (leave empty to auto-detect from character/logs)."
                }),
                "ref_text_override": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Override reference audio transcript."
                }),
            }
        }

    RETURN_TYPES = ("TTS_ENGINE",)
    RETURN_NAMES = ("TTS_engine",)
    FUNCTION = "create_engine_adapter"
    CATEGORY = "TTS Audio Suite/⚙️ Engines"

    def _resolve_weight_paths(self, selection: str, gpt_sovits_home: str = "") -> Dict:
        """Resolve weight pair selection to actual file paths.

        Uses resolve_paths() to determine the correct source:
        - gpt_sovits_home set → reads from WebUI project root
        - default → ComfyUI models/TTS/GPT-SoVITS/
        """
        comfyui_models = folder_paths.models_dir
        paths = resolve_paths(
            gpt_sovits_home=gpt_sovits_home.strip() or None,
            comfyui_models_dir=comfyui_models,
        )
        scan_result = scan_weights(paths.get("base_dir", ""), paths.get("pretrained_dir"))

        candidates = [
            ({**pret, **paths}, f"[Pretrained] {pret['label']}", f"pretrained:{pret['version']}")
            for pret in scan_result.get("pretrained", [])
        ]
        candidates.extend(
            ({**pair, **paths}, f"[{pair['version']}] {pair['exp_name']}", f"pair:{pair['version']}:{pair['exp_name']}")
            for pair in scan_result.get("pairs", [])
        )
        if selection.strip().lower() == "auto":
            if len(candidates) == 1:
                return candidates[0][0]
            if len(candidates) > 1:
                choices = ", ".join(stable_id for _, _, stable_id in candidates)
                raise ValueError(f"Multiple GPT-SoVITS weight pairs found; set weight_pair to one of: {choices}")
            return None

        for resolved, legacy_label, stable_id in candidates:
            if selection in {legacy_label, stable_id}:
                return resolved

        return None

    def create_engine_adapter(
        self,
        weight_pair: str,
        text_language: str = "中文",
        ref_language: str = "中文",
        how_to_cut: str = "凑四句一切",
        speed: float = 1.0,
        top_k: int = 15,
        top_p: float = 1.0,
        temperature: float = 1.0,
        gpt_sovits_home: str = "",
        device: str = "auto",
        use_fp16: bool = True,
        ref_audio_override: str = "",
        ref_text_override: str = "",
    ):
        """Create GPT-SoVITS engine adapter with configuration.

        When gpt_sovits_home is set, ALL paths (weights, BERT, CNHubert,
        pretrained models) are resolved from the existing WebUI project root.
        No data migration needed.
        """
        try:
            resolved = self._resolve_weight_paths(weight_pair, gpt_sovits_home)
            if resolved is None:
                raise ValueError(f"Could not resolve weight pair: {weight_pair}")

            device = resolve_torch_device(device)
            if device == "cpu":
                use_fp16 = False

            # Validate paths
            for key, label in [
                ("gpt_path", "GPT weight"),
                ("sovits_path", "SoVITS weight"),
                ("bert_path", "BERT model"),
                ("cnhubert_path", "CNHubert model"),
            ]:
                p = resolved.get(key, "")
                if p and not os.path.exists(p):
                    print(f"⚠️ {label} not found: {p}")

            config = {
                "engine_type": "gpt_sovits",
                "gpt_weight": resolved.get("gpt_path", ""),
                "sovits_weight": resolved.get("sovits_path", ""),
                "version": resolved.get("version", "v2"),
                "exp_name": resolved.get("exp_name", ""),
                "is_pretrained": resolved.get("is_pretrained", False),
                "bert_path": resolved.get("bert_path", ""),
                "cnhubert_path": resolved.get("cnhubert_path", ""),
                "logs_dir": resolved.get("logs_dir", ""),
                "gpt_sovits_home": gpt_sovits_home.strip(),
                "source": resolved.get("source", "comfyui"),
                "device": device,
                "use_fp16": use_fp16,
                "text_language": text_language,
                "ref_language": ref_language,
                "how_to_cut": how_to_cut,
                "speed": speed,
                "top_k": top_k,
                "top_p": top_p,
                "temperature": temperature,
                "ref_audio_override": ref_audio_override.strip(),
                "ref_text_override": ref_text_override.strip(),
            }

            print(f"⚙️ GPT-SoVITS: source={resolved.get('source', '?')}, device={device}")
            print(f"   GPT: {os.path.basename(resolved.get('gpt_path', '?'))}")
            print(f"   SoVITS: {os.path.basename(resolved.get('sovits_path', '?'))}")

            engine_data = {
                "engine_type": "gpt_sovits",
                "config": config,
                "adapter_class": "GPTSovitsAdapter",
            }
            return (engine_data,)

        except Exception as e:
            print(f"❌ GPT-SoVITS Engine error: {e}")
            import traceback
            traceback.print_exc()

            return ({
                "engine_type": "gpt_sovits",
                "config": {"error": str(e)},
                "adapter_class": "GPTSovitsAdapter",
            },)


# Register
NODE_CLASS_MAPPINGS = {
    "GPT-SoVITS Engine": GPTSovitsEngineNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GPT-SoVITS Engine": "⚙️ GPT-SoVITS Engine"
}
