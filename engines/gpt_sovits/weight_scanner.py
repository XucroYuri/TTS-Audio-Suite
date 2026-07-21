"""
GPT-SoVITS Weight Scanner

Discovers available model checkpoints from TWO sources:

1. ComfyUI Model Directory (default):
   ComfyUI/models/TTS/GPT-SoVITS/{GPT_weights_v2, SoVITS_weights_v2, ...}

2. Official GPT-SoVITS WebUI Project (gpt_sovits_home):
   {project_root}/{GPT_weights_v2, SoVITS_weights_v2, logs/, GPT_SoVITS/pretrained_models/}

When gpt_sovits_home is set, ALL paths (weights, pretrained, BERT, CNHubert,
reference audio) are resolved relative to the existing WebUI project root.
No data migration needed.
"""

import os
import re
from typing import Dict, List, Optional, Tuple


# Version-to-directory mapping (from official config.py)
WEIGHT_DIRS = {
    "v1":         {"gpt": "GPT_weights",        "sovits": "SoVITS_weights"},
    "v2":         {"gpt": "GPT_weights_v2",     "sovits": "SoVITS_weights_v2"},
    "v3":         {"gpt": "GPT_weights_v3",     "sovits": "SoVITS_weights_v3"},
    "v4":         {"gpt": "GPT_weights_v4",     "sovits": "SoVITS_weights_v4"},
    "v2Pro":      {"gpt": "GPT_weights_v2Pro",  "sovits": "SoVITS_weights_v2Pro"},
    "v2ProPlus":  {"gpt": "GPT_weights_v2ProPlus", "sovits": "SoVITS_weights_v2ProPlus"},
}

SUPPORTED_VERSIONS = ["v2", "v2Pro", "v2ProPlus"]


def _extract_exp_name_gpt(filename: str) -> Optional[str]:
    """Extract exp_name from GPT checkpoint: {exp_name}-e{epoch}.ckpt"""
    match = re.match(r"^(.+)-e\d+\.ckpt$", filename)
    return match.group(1) if match else None


def _extract_exp_name_sovits(filename: str) -> Optional[str]:
    """Extract exp_name from SoVITS checkpoint: {exp_name}_e{epoch}_s{step}.pth"""
    match = re.match(r"^(.+?)_(?:lora_)?e\d+_s\d+\.pth$", filename)
    return match.group(1) if match else None


def resolve_paths(
    gpt_sovits_home: Optional[str] = None,
    comfyui_models_dir: Optional[str] = None,
) -> Dict[str, str]:
    """Resolve all GPT-SoVITS paths based on configuration.

    Priority:
      1. gpt_sovits_home (official WebUI project root) — zero migration
      2. comfyui_models_dir/TTS/GPT-SoVITS/ — ComfyUI standard location

    Returns:
        Dict with keys:
        - base_dir: root for weight directories
        - pretrained_dir: root for pretrained models
        - bert_path: Chinese BERT model directory
        - cnhubert_path: Chinese HuBERT model directory
        - logs_dir: logs/ directory for reference audio discovery
        - source: "webui" or "comfyui"
    """
    if gpt_sovits_home and os.path.isdir(gpt_sovits_home):
        home = os.path.abspath(gpt_sovits_home)
        pretrained = os.path.join(home, "GPT_SoVITS", "pretrained_models")
        return {
            "base_dir": home,
            "pretrained_dir": pretrained,
            "bert_path": os.path.join(pretrained, "chinese-roberta-wwm-ext-large"),
            "cnhubert_path": os.path.join(pretrained, "chinese-hubert-base"),
            "logs_dir": os.path.join(home, "logs"),
            "source": "webui",
        }

    # Fallback: ComfyUI models directory
    if comfyui_models_dir:
        base = os.path.join(comfyui_models_dir, "TTS", "GPT-SoVITS")
        pretrained = os.path.join(base, "pretrained_models")
        logs = os.path.join(base, "..", "..", "logs")  # unlikely but kept
        return {
            "base_dir": base,
            "pretrained_dir": pretrained,
            "bert_path": os.path.join(pretrained, "chinese-roberta-wwm-ext-large"),
            "cnhubert_path": os.path.join(pretrained, "chinese-hubert-base"),
            "logs_dir": logs,
            "source": "comfyui",
        }

    return {}


def scan_weights(
    base_dir: str,
    pretrained_dir: Optional[str] = None,
) -> Dict[str, List[Dict]]:
    """Scan for all discoverable GPT-SoVITS weight pairs.

    Args:
        base_dir: Directory containing GPT_weights_v* / SoVITS_weights_v*
        pretrained_dir: Directory containing pretrained models (optional)

    Returns:
        Dict with keys: pairs, gpt_only, sovits_only, pretrained
    """
    result = {
        "pairs": [],
        "gpt_only": [],
        "sovits_only": [],
        "pretrained": [],
    }

    if not os.path.isdir(base_dir):
        return result

    # Phase 1: Scan all weight directories
    all_gpt: Dict[str, Tuple[str, str]] = {}
    all_sovits: Dict[str, Tuple[str, str]] = {}

    for version in SUPPORTED_VERSIONS:
        dirs = WEIGHT_DIRS[version]
        gpt_dir = os.path.join(base_dir, dirs["gpt"])
        sovits_dir = os.path.join(base_dir, dirs["sovits"])

        if os.path.isdir(gpt_dir):
            for fname in os.listdir(gpt_dir):
                if not fname.endswith(".ckpt"):
                    continue
                exp_name = _extract_exp_name_gpt(fname)
                if exp_name:
                    all_gpt[exp_name] = (os.path.join(gpt_dir, fname), version)

        if os.path.isdir(sovits_dir):
            for fname in os.listdir(sovits_dir):
                if not fname.endswith(".pth"):
                    continue
                exp_name = _extract_exp_name_sovits(fname)
                if exp_name:
                    all_sovits[exp_name] = (os.path.join(sovits_dir, fname), version)

    # Phase 2: Match pairs by exp_name
    for exp_name in sorted(set(all_gpt.keys()) | set(all_sovits.keys())):
        has_gpt = exp_name in all_gpt
        has_sovits = exp_name in all_sovits

        if has_gpt and has_sovits:
            gpt_path, gpt_ver = all_gpt[exp_name]
            sovits_path, sovits_ver = all_sovits[exp_name]
            version = gpt_ver if gpt_ver == sovits_ver else f"{gpt_ver}/{sovits_ver}"
            result["pairs"].append({
                "exp_name": exp_name,
                "version": version,
                "gpt_path": gpt_path,
                "sovits_path": sovits_path,
            })
        elif has_gpt:
            result["gpt_only"].append({
                "exp_name": exp_name,
                "version": all_gpt[exp_name][1],
                "gpt_path": all_gpt[exp_name][0],
            })
        else:
            result["sovits_only"].append({
                "exp_name": exp_name,
                "version": all_sovits[exp_name][1],
                "sovits_path": all_sovits[exp_name][0],
            })

    # Phase 3: Pretrained models
    if pretrained_dir and os.path.isdir(pretrained_dir):
        result["pretrained"] = _scan_pretrained(pretrained_dir)

    return result


def _scan_pretrained(pretrained_dir: str) -> List[Dict]:
    """Scan pretrained models directory."""
    pretrained = []

    v2_dir = os.path.join(pretrained_dir, "gsv-v2final-pretrained")
    if os.path.isdir(v2_dir):
        gpt_files = [f for f in os.listdir(v2_dir) if f.endswith(".ckpt")]
        sovits_files = [f for f in os.listdir(v2_dir) if f.endswith(".pth")]
        if gpt_files and sovits_files:
            pretrained.append({
                "label": "v2 (Zero-shot)",
                "version": "v2",
                "gpt_path": os.path.join(v2_dir, gpt_files[0]),
                "sovits_path": os.path.join(v2_dir, sovits_files[0]),
                "is_pretrained": True,
            })

    v2pro_dir = os.path.join(pretrained_dir, "v2Pro")
    s1v3 = os.path.join(pretrained_dir, "s1v3.ckpt")
    if os.path.isdir(v2pro_dir) and os.path.isfile(s1v3):
        for key, label in [("s2Gv2Pro.pth", "v2Pro"), ("s2Gv2ProPlus.pth", "v2ProPlus")]:
            sovits_files = [f for f in os.listdir(v2pro_dir) if key in f]
            if sovits_files:
                pretrained.append({
                    "label": f"{label} (Zero-shot)",
                    "version": label,
                    "gpt_path": s1v3,
                    "sovits_path": os.path.join(v2pro_dir, sovits_files[0]),
                    "is_pretrained": True,
                })

    return pretrained


def scan_reference_audio(logs_dir: str, exp_name: str) -> List[Dict]:
    """Scan logs/{exp_name}/ for paired reference audio and text.

    Reads 2-name2text.txt → maps audio names to transcripts,
    then matches against 5-wav32k/ audio files.
    """
    refs = []
    if not logs_dir or not os.path.isdir(logs_dir):
        return refs

    audio_dir = os.path.join(logs_dir, exp_name, "5-wav32k")
    text_file = os.path.join(logs_dir, exp_name, "2-name2text.txt")

    if not os.path.isdir(audio_dir):
        return refs

    text_map = {}
    if os.path.isfile(text_file):
        with open(text_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) >= 3:
                    text_map[parts[0].strip()] = parts[2].strip()

    for fname in sorted(os.listdir(audio_dir)):
        if fname.endswith((".wav", ".mp3", ".flac")):
            audio_path = os.path.join(audio_dir, fname)
            base_name = os.path.splitext(fname)[0]
            ref_text = text_map.get(base_name, text_map.get(fname, ""))
            refs.append({
                "audio": audio_path,
                "text": ref_text,
                "lang": "zh",
            })

    return refs
