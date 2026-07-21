"""
CosyVoice3 Engine Configuration Node

Provides comprehensive configuration interface for CosyVoice3 TTS engine with
9-language zero-shot voice cloning, instruct mode for emotions/dialects, and
cross-lingual synthesis.
"""

# FIX: PyYAML 6.0+ compatibility patch - MUST be before any yaml imports
# Issue: 'Loader' object has no attribute 'max_depth' error (GitHub #220)
import yaml
if not hasattr(yaml.Loader, 'max_depth'):
    yaml.Loader.max_depth = 100
if not hasattr(yaml.FullLoader, 'max_depth'):
    yaml.FullLoader.max_depth = 100
if not hasattr(yaml.SafeLoader, 'max_depth'):
    yaml.SafeLoader.max_depth = 100

import os
import sys
import importlib.util
from typing import Dict, Any, List

# Add project root directory to path for imports
current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)  # nodes/
project_root = os.path.dirname(nodes_dir)  # project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load base_node module directly
base_node_path = os.path.join(nodes_dir, "base", "base_node.py")
base_spec = importlib.util.spec_from_file_location("base_node_module", base_node_path)
base_module = importlib.util.module_from_spec(base_spec)
sys.modules["base_node_module"] = base_module
base_spec.loader.exec_module(base_module)

# Import the base class
BaseTTSNode = base_module.BaseTTSNode

import folder_paths
from utils.models.extra_paths import get_all_tts_model_paths

# AnyType for flexible input types (accepts any data type)
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_typ = AnyType("*")


class CosyVoiceEngineNode(BaseTTSNode):
    """
    CosyVoice3 TTS Engine configuration node.
    Provides CosyVoice3 parameters and creates engine adapter for unified nodes.
    """
    
    @classmethod
    def NAME(cls):
        return "⚙️ CosyVoice3 Engine"
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get available model paths
        model_paths = cls._get_model_paths()
        
        return {
            "required": {
                # Model Configuration
                "model_path": (model_paths, {
                    "default": model_paths[0] if model_paths else "Fun-CosyVoice3-0.5B-RL",
                    "tooltip": "CosyVoice3 model variant:\n• Fun-CosyVoice3-0.5B-RL: Reinforcement learning trained variant with improved quality (0.81 CER, 77.4 speaker similarity - recommended)\n• Fun-CosyVoice3-0.5B: Base model\n• local:ModelName: Use locally installed model"
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Device to run CosyVoice3 model on:\n• auto: Best available (CUDA > CPU)\n• cuda: NVIDIA GPU\n• cpu: CPU-only processing (slower)"
                }),

                # Speed Control
                "speed": ("FLOAT", {
                    "default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1,
                    "tooltip": "Speech speed multiplier (0.5=slow, 1.0=normal, 2.0=fast).\n\nIMPORTANT: CosyVoice's official implementation uses mel-spectrogram interpolation for speed control - this is post-processing time-stretching applied AFTER generation, NOT natural prosody/rhythm changes during synthesis. This is a limitation of the official model architecture. Result will sound like artificially sped up or slowed down audio (like playing a recording faster/slower)."
                }),

                # Model Options
                "use_fp16": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use FP16 for faster inference. Disable if you encounter numerical issues."
                }),
            },
            "optional": {
                "cosyvoice_home": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": (
                        "Path to local CosyVoice project root directory.\n"
                        "When set, scans {home}/pretrained_models/ for model files\n"
                        "so you don't need to re-download models already present\n"
                        "in a local CosyVoice installation.\n"
                        "Example: D:\\CosyVoice\\\n\n"
                        "Leave empty to use ComfyUI models/TTS/CosyVoice/"
                    ),
                }),
                # Optional instruction text (like Higgs Audio's system_prompt)
                "instruct_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": """Optional instruction for dialect/emotion/speed control.

⚠️ IMPORTANT: Instruction mode is mutually exclusive with transcript-based voice cloning.
• When instruction is provided: Uses instruction mode (ignores transcripts)
• When empty: Uses zero-shot cloning with transcripts (.txt files) for BEST QUALITY

Examples:
• 请用广东话表达。 (Use Cantonese dialect)
• 请用四川话说。 (Use Sichuan dialect)
• 请用尽可能快地语速说一句话。 (Speak as fast as possible)
• 请用温柔的语气说。 (Use gentle tone)
• 请用生气的语气说。 (Use angry tone)

For best voice cloning quality: Leave this empty and provide .txt transcripts via Character Voices"""
                }),

                # Advanced optimizations (usually not needed)
                "load_trt": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "TensorRT for flow decoder (FP32 only, <1% speedup). Included for compatibility with official repo but provides negligible benefit. Use load_vllm for real LLM acceleration."
                }),
                "load_vllm": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "vLLM for LLM acceleration (~3x faster token generation, ~2.5x total speedup). Linux only - does not work on Windows. Requires vLLM installation and model conversion."
                }),
            }
        }
    
    RETURN_TYPES = ("TTS_ENGINE",)
    RETURN_NAMES = ("TTS_engine",)
    FUNCTION = "create_engine_adapter"
    CATEGORY = "TTS Audio Suite/⚙️ Engines"
    
    @classmethod
    def _get_model_paths(cls, cosyvoice_home: str = "") -> List[str]:
        """Get available CosyVoice3 model paths.

        Scans in priority order:
        1. cosyvoice_home/pretrained_models/ (when home is set)
        2. ComfyUI models/TTS/ directories
        """
        paths = [
            "Fun-CosyVoice3-0.5B-RL",  # RL-enhanced (default, better quality)
            "Fun-CosyVoice3-0.5B"      # Standard
        ]

        # Scan cosyvoice_home if set
        home = cosyvoice_home.strip() if cosyvoice_home else ""
        if home and os.path.isdir(home):
            # Official CosyVoice project: models are in pretrained_models/
            pretrained = os.path.join(home, "pretrained_models")
            for scan_dir in [home, pretrained]:
                if not os.path.isdir(scan_dir):
                    continue
                for entry in os.listdir(scan_dir):
                    entry_path = os.path.join(scan_dir, entry)
                    if not os.path.isdir(entry_path):
                        continue
                    # Check for CosyVoice model marker
                    marker = os.path.join(entry_path, "cosyvoice3.yaml")
                    if os.path.isfile(marker):
                        label = f"local:{entry} (from home)"
                        if label not in paths:
                            paths.insert(0, label)

        try:
            # Check all configured TTS model paths
            all_tts_paths = get_all_tts_model_paths('TTS')

            for base_path in all_tts_paths:
                # Check direct path (models/TTS/Fun-CosyVoice3-0.5B - shared folder)
                cosy_direct = os.path.join(base_path, "Fun-CosyVoice3-0.5B")
                if os.path.exists(os.path.join(cosy_direct, "cosyvoice3.yaml")):
                    if "local:Fun-CosyVoice3-0.5B-RL" not in paths:
                        paths.insert(0, "local:Fun-CosyVoice3-0.5B-RL")
                    if "local:Fun-CosyVoice3-0.5B" not in paths:
                        paths.insert(0, "local:Fun-CosyVoice3-0.5B")

                # Check organized path (models/TTS/CosyVoice/Fun-CosyVoice3-0.5B)
                cosy_organized = os.path.join(base_path, "CosyVoice", "Fun-CosyVoice3-0.5B")
                if os.path.exists(os.path.join(cosy_organized, "cosyvoice3.yaml")):
                    if "local:Fun-CosyVoice3-0.5B-RL" not in paths:
                        paths.insert(0, "local:Fun-CosyVoice3-0.5B-RL")
                    if "local:Fun-CosyVoice3-0.5B" not in paths:
                        paths.insert(0, "local:Fun-CosyVoice3-0.5B")
        except Exception:
            # Fallback to default (shared folder)
            base_dir = os.path.join(folder_paths.models_dir, "TTS", "CosyVoice", "Fun-CosyVoice3-0.5B")
            if os.path.exists(os.path.join(base_dir, "cosyvoice3.yaml")):
                if "local:Fun-CosyVoice3-0.5B-RL" not in paths:
                    paths.insert(0, "local:Fun-CosyVoice3-0.5B-RL")
                if "local:Fun-CosyVoice3-0.5B" not in paths:
                    paths.insert(0, "local:Fun-CosyVoice3-0.5B")

        return paths
    
    def create_engine_adapter(
        self,
        model_path: str,
        device: str,
        speed: float,
        use_fp16: bool,
        instruct_text: str = "",
        cosyvoice_home: str = "",
        load_trt: bool = False,
        load_vllm: bool = False,
    ):
        """
        Create CosyVoice3 engine adapter with configuration.

        When cosyvoice_home is set and model_path starts with "local:",
        the actual model directory is resolved from the home path.

        Mode is auto-detected in adapter:
        - If instruct_text provided → instruct mode
        - If reference_text from Character Voices → zero_shot mode
        - Otherwise → cross_lingual mode

        Returns:
            Tuple containing CosyVoice3 engine configuration data
        """
        try:
            # Resolve model_path from cosyvoice_home if applicable
            resolved_path = model_path
            home = cosyvoice_home.strip() if cosyvoice_home else ""
            if home and model_path.startswith("local:"):
                model_name = model_path.replace("local:", "").replace(" (from home)", "").strip()
                # Try pretrained_models/ subdirectory first (official CosyVoice convention)
                for subdir in ["pretrained_models", ""]:
                    candidate = os.path.join(home, subdir, model_name)
                    if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "cosyvoice3.yaml")):
                        resolved_path = candidate
                        print(f"   Resolved from cosyvoice_home: {resolved_path}")
                        break

            # Create configuration dictionary
            config = {
                "model_path": resolved_path,
                "device": device,
                "speed": speed,
                "use_fp16": use_fp16,
                "instruct_text": instruct_text.strip() if instruct_text else "",
                "cosyvoice_home": home,
                "load_trt": load_trt,
                "load_vllm": load_vllm,
                "engine_type": "cosyvoice",
            }

            print(f"⚙️ CosyVoice3: Configured on {device}")
            print(f"   Model: {model_path}")
            if home:
                print(f"   Home: {home}")
            print(f"   Speed: {speed}x")
            if instruct_text:
                print(f"   Instruction: {instruct_text[:50]}...")

            # Return engine data for consumption by unified TTS nodes
            engine_data = {
                "engine_type": "cosyvoice",
                "config": config,
                "adapter_class": "CosyVoiceAdapter"
            }

            return (engine_data,)

        except Exception as e:
            print(f"❌ CosyVoice3 Engine error: {e}")
            import traceback
            traceback.print_exc()

            # Return error config
            error_config = {
                "engine_type": "cosyvoice",
                "config": {
                    "model_path": model_path,
                    "device": "cpu",  # Fallback to CPU
                    "speed": 1.0,
                    "error": str(e)
                },
                "adapter_class": "CosyVoiceAdapter"
            }
            return (error_config,)


# Register the node
NODE_CLASS_MAPPINGS = {
    "CosyVoice Engine": CosyVoiceEngineNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CosyVoice Engine": "⚙️ CosyVoice3 Engine"
}
