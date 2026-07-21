"""
GPT-SoVITS Weight Scanner

Discovers available model checkpoints across all version directories
(v1-v4, v2Pro, v2ProPlus) and pairs GPT + SoVITS weights by exp_name.

Matches the official config.py directory layout.
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

# Supported version priorities (exclude v1/v3/v4 for now)
SUPPORTED_VERSIONS = ["v2", "v2Pro", "v2ProPlus"]


def _extract_exp_name_gpt(filename: str) -> Optional[str]:
    """Extract exp_name from GPT checkpoint filename.

    Format: {exp_name}-e{epoch}.ckpt
    Example: raiden_shogun-e15.ckpt → "raiden_shogun"
    """
    match = re.match(r"^(.+)-e\d+\.ckpt$", filename)
    return match.group(1) if match else None


def _extract_exp_name_sovits(filename: str) -> Optional[str]:
    """Extract exp_name from SoVITS checkpoint filename.

    Format: {exp_name}_e{epoch}_s{step}.pth  or  {exp_name}_lora_e{epoch}_s{step}.pth
    Example: raiden_shogun_e8_s200.pth → "raiden_shogun"
    """
    match = re.match(r"^(.+?)_(?:lora_)?e\d+_s\d+\.pth$", filename)
    return match.group(1) if match else None


def scan_weights(base_dir: str) -> Dict[str, List[Dict]]:
    """Scan base_dir for all discoverable GPT-SoVITS weight pairs.

    Args:
        base_dir: Root directory (e.g., ComfyUI/models/TTS/GPT-SoVITS/)

    Returns:
        Dict with keys:
        - "pairs": List of matched (gpt_path, sovits_path, exp_name, version) dicts
        - "gpt_only": List of GPT weights without matching SoVITS
        - "sovits_only": List of SoVITS weights without matching GPT
        - "pretrained": List of pretrained model options
    """
    result = {
        "pairs": [],
        "gpt_only": [],
        "sovits_only": [],
        "pretrained": [],
    }

    # Phase 1: Scan all weight directories
    all_gpt: Dict[str, Tuple[str, str]] = {}   # exp_name → (path, version)
    all_sovits: Dict[str, Tuple[str, str]] = {} # exp_name → (path, version)

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
                    full_path = os.path.join(gpt_dir, fname)
                    all_gpt[exp_name] = (full_path, version)

        if os.path.isdir(sovits_dir):
            for fname in os.listdir(sovits_dir):
                if not fname.endswith(".pth"):
                    continue
                exp_name = _extract_exp_name_sovits(fname)
                if exp_name:
                    full_path = os.path.join(sovits_dir, fname)
                    all_sovits[exp_name] = (full_path, version)

    # Phase 2: Match pairs by exp_name (same version preferred)
    all_exp_names = set(all_gpt.keys()) | set(all_sovits.keys())
    for exp_name in sorted(all_exp_names):
        has_gpt = exp_name in all_gpt
        has_sovits = exp_name in all_sovits

        if has_gpt and has_sovits:
            gpt_path, gpt_ver = all_gpt[exp_name]
            sovits_path, sovits_ver = all_sovits[exp_name]
            # Use the matched version, preferring the one both agree on
            version = gpt_ver if gpt_ver == sovits_ver else f"{gpt_ver}/{sovits_ver}"
            result["pairs"].append({
                "exp_name": exp_name,
                "version": version,
                "gpt_path": gpt_path,
                "sovits_path": sovits_path,
            })
        elif has_gpt:
            gpt_path, gpt_ver = all_gpt[exp_name]
            result["gpt_only"].append({
                "exp_name": exp_name,
                "version": gpt_ver,
                "gpt_path": gpt_path,
            })
        else:
            sovits_path, sovits_ver = all_sovits[exp_name]
            result["sovits_only"].append({
                "exp_name": exp_name,
                "version": sovits_ver,
                "sovits_path": sovits_path,
            })

    # Phase 3: Discover pretrained (zero-shot) models
    pretrained_dir = os.path.join(base_dir, "pretrained_models")
    if os.path.isdir(pretrained_dir):
        result["pretrained"] = _scan_pretrained(pretrained_dir)

    return result


def _scan_pretrained(pretrained_dir: str) -> List[Dict]:
    """Scan pretrained model directory for zero-shot options."""
    pretrained = []

    # v2 pretrained
    v2_dir = os.path.join(pretrained_dir, "gsv-v2final-pretrained")
    if os.path.isdir(v2_dir):
        gpt_files = [f for f in os.listdir(v2_dir) if f.endswith(".ckpt")]
        sovits_files = [f for f in os.listdir(v2_dir) if f.endswith(".pth")]
        if gpt_files and sovits_files:
            pretrained.append({
                "label": "v2 (Zero-shot, no training)",
                "version": "v2",
                "gpt_path": os.path.join(v2_dir, gpt_files[0]),
                "sovits_path": os.path.join(v2_dir, sovits_files[0]),
                "is_pretrained": True,
            })

    # v2Pro pretrained
    v2pro_dir = os.path.join(pretrained_dir, "v2Pro")
    s1v3 = os.path.join(pretrained_dir, "s1v3.ckpt")
    if os.path.isdir(v2pro_dir) and os.path.isfile(s1v3):
        sovits_files = [f for f in os.listdir(v2pro_dir) if f.endswith(".pth") and "s2Gv2Pro.pth" in f]
        if sovits_files:
            pretrained.append({
                "label": "v2Pro (Zero-shot, no training)",
                "version": "v2Pro",
                "gpt_path": s1v3,
                "sovits_path": os.path.join(v2pro_dir, sovits_files[0]),
                "is_pretrained": True,
            })

    # v2ProPlus pretrained
    if os.path.isdir(v2pro_dir) and os.path.isfile(s1v3):
        sovits_files = [f for f in os.listdir(v2pro_dir) if f.endswith(".pth") and "s2Gv2ProPlus.pth" in f]
        if sovits_files:
            pretrained.append({
                "label": "v2ProPlus (Zero-shot, no training)",
                "version": "v2ProPlus",
                "gpt_path": s1v3,
                "sovits_path": os.path.join(v2pro_dir, sovits_files[0]),
                "is_pretrained": True,
            })

    return pretrained


def scan_reference_audio(logs_dir: str, exp_name: str) -> List[Dict]:
    """Scan logs/{exp_name}/ for paired reference audio and text.

    Args:
        logs_dir: Path to logs/ directory (or None to skip)
        exp_name: Experiment name to look up

    Returns:
        List of {"audio": path, "text": str, "lang": str} dicts
    """
    refs = []
    if not logs_dir or not os.path.isdir(logs_dir):
        return refs

    audio_dir = os.path.join(logs_dir, exp_name, "5-wav32k")
    text_file = os.path.join(logs_dir, exp_name, "2-name2text.txt")

    if not os.path.isdir(audio_dir):
        return refs

    # Parse 2-name2text.txt: format is "audio_name|phonemes|text"
    text_map = {}
    if os.path.isfile(text_file):
        with open(text_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) >= 3:
                    audio_name = parts[0].strip()
                    text_content = parts[2].strip()
                    text_map[audio_name] = text_content

    # Find matching audio files
    for fname in sorted(os.listdir(audio_dir)):
        if fname.endswith((".wav", ".mp3", ".flac")):
            audio_path = os.path.join(audio_dir, fname)
            base_name = os.path.splitext(fname)[0]
            ref_text = text_map.get(base_name, text_map.get(fname, ""))
            refs.append({
                "audio": audio_path,
                "text": ref_text,
                "lang": "zh",  # default, could be inferred
            })

    return refs
