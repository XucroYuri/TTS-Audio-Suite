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
from engines.gpt_sovits.weight_scanner import scan_weights, scan_reference_audio


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
        # Scan for available weight pairs
        base_dir = os.path.join(folder_paths.models_dir, "TTS", "GPT-SoVITS")
        scan_result = scan_weights(base_dir) if os.path.isdir(base_dir) else {"pairs": [], "pretrained": []}

        # Build weight pair options
        pair_options = []
        for pair in scan_result["pretrained"]:
            pair_options.append(f"[Pretrained] {pair['label']}")
        for pair in scan_result["pairs"]:
            pair_options.append(f"[{pair['version']}] {pair['exp_name']}")

        if not pair_options:
            pair_options = ["(No models found in models/TTS/GPT-SoVITS/)"]

        # Text splitting methods (from official WebUI)
        cut_methods = ["凑四句一切", "凑50字一切", "按中文句号。切", "按英文句号.切", "按标点符号切", "不切"]

        # Languages
        languages = ["中文", "英文", "日文", "粤语", "韩语", "auto"]

        return {
            "required": {
                "weight_pair": (pair_options, {
                    "default": pair_options[0],
                    "tooltip": "Select a matched GPT + SoVITS weight pair.\nPretrained options use base models for zero-shot cloning.\nUser-trained pairs show [version] experiment_name."
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

    @classmethod
    def _resolve_weight_paths(cls, selection: str) -> Dict[str, str]:
        """Resolve weight pair selection to actual file paths."""
        base_dir = os.path.join(folder_paths.models_dir, "TTS", "GPT-SoVITS")
        scan_result = scan_weights(base_dir) if os.path.isdir(base_dir) else {"pairs": [], "pretrained": []}

        # Check pretrained options
        for pret in scan_result.get("pretrained", []):
            label = f"[Pretrained] {pret['label']}"
            if selection == label:
                return {
                    "gpt_path": pret["gpt_path"],
                    "sovits_path": pret["sovits_path"],
                    "version": pret["version"],
                    "is_pretrained": True,
                }

        # Check user-trained pairs
        for pair in scan_result.get("pairs", []):
            label = f"[{pair['version']}] {pair['exp_name']}"
            if selection == label:
                return {
                    "gpt_path": pair["gpt_path"],
                    "sovits_path": pair["sovits_path"],
                    "version": pair["version"],
                    "exp_name": pair["exp_name"],
                    "is_pretrained": False,
                }

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
        device: str = "auto",
        use_fp16: bool = True,
        ref_audio_override: str = "",
        ref_text_override: str = "",
    ):
        """Create GPT-SoVITS engine adapter with configuration."""
        try:
            # Resolve weights
            resolved = self._resolve_weight_paths(weight_pair)
            if resolved is None:
                raise ValueError(f"Could not resolve weight pair: {weight_pair}")

            # Resolve device
            if device == "auto":
                device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

            # BERT and CNHubert paths
            base_dir = os.path.join(folder_paths.models_dir, "TTS", "GPT-SoVITS")
            bert_path = os.path.join(base_dir, "pretrained_models", "chinese-roberta-wwm-ext-large")
            cnhubert_path = os.path.join(base_dir, "pretrained_models", "chinese-hubert-base")

            # Validate
            for path, name in [
                (resolved["gpt_path"], "GPT weight"),
                (resolved["sovits_path"], "SoVITS weight"),
                (bert_path, "BERT model"),
                (cnhubert_path, "CNHubert model"),
            ]:
                if not os.path.exists(path):
                    print(f"⚠️ {name} not found: {path}")

            config = {
                "engine_type": "gpt_sovits",
                "gpt_weight": resolved["gpt_path"],
                "sovits_weight": resolved["sovits_path"],
                "version": resolved.get("version", "v2"),
                "exp_name": resolved.get("exp_name", ""),
                "is_pretrained": resolved.get("is_pretrained", False),
                "bert_path": bert_path,
                "cnhubert_path": cnhubert_path,
                "device": device,
                "use_fp16": use_fp16,
                "text_language": text_language,
                "ref_language": ref_language,
                "how_to_cut": how_to_cut,
                "speed": speed,
                "top_k": top_k,
                "top_p": top_p,
                "temperature": temperature,
                "ref_audio_override": ref_audio_override,
                "ref_text_override": ref_text_override,
            }

            print(f"⚙️ GPT-SoVITS: Configured on {device}")
            print(f"   GPT: {os.path.basename(resolved['gpt_path'])}")
            print(f"   SoVITS: {os.path.basename(resolved['sovits_path'])}")
            print(f"   Version: {resolved.get('version', 'v2')}")

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

            error_config = {
                "engine_type": "gpt_sovits",
                "config": {
                    "error": str(e),
                },
                "adapter_class": "GPTSovitsAdapter",
            }
            return (error_config,)


# Register
NODE_CLASS_MAPPINGS = {
    "GPT-SoVITS Engine": GPTSovitsEngineNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GPT-SoVITS Engine": "⚙️ GPT-SoVITS Engine"
}
