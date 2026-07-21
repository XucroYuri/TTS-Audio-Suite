"""
GPT-SoVITS Model Downloader with Multi-Source Resilience

Download endpoints (priority order):
  1. HuggingFace direct (huggingface.co)
  2. HF Mirror (hf-mirror.com)

Downloads required pretrained models for GPT-SoVITS inference:
  BERT (~1.2GB), CNHubert (~400MB), v2 base (~1.6GB), v2Pro base
"""

import os
import sys
import time
from typing import Optional, List

HF_ENDPOINTS = [
    "https://huggingface.co",
    "https://hf-mirror.com",
]

MODEL_SOURCES = {
    "bert": {
        "repo": "hfl/chinese-roberta-wwm-ext-large",
        "desc": "Chinese BERT (~1.2GB)",
    },
    "cnhubert": {
        "repo": "TencentGameMate/chinese-hubert-base",
        "desc": "Chinese HuBERT (~400MB)",
    },
    "v2_base": {
        "repo": "lj1995/GPT-SoVITS",
        "subfolder": "gsv-v2final-pretrained",
        "desc": "GPT-SoVITS v2 base (~1.6GB)",
        "files": [
            "s2G2333k.pth",
            "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
        ],
    },
    "v2pro_base": {
        "repo": "lj1995/GPT-SoVITS",
        "desc": "GPT-SoVITS v2Pro base",
        "files": [
            "s1v3.ckpt",
            "v2Pro/s2Gv2Pro.pth",
            "v2Pro/s2Dv2Pro.pth",
            "v2Pro/s2Gv2ProPlus.pth",
            "v2Pro/s2Dv2ProPlus.pth",
            "sv/pretrained_eres2netv2w24s4ep4.ckpt",
        ],
    },
}


class MultiSourceDownloader:
    """Downloads models with automatic endpoint fallback and retry."""

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    def download_snapshot(self, repo_id: str, local_dir: str,
                          allow_patterns: Optional[List[str]] = None) -> bool:
        from huggingface_hub import snapshot_download
        for ep in HF_ENDPOINTS:
            os.environ["HF_ENDPOINT"] = ep
            for attempt in range(self.max_retries):
                try:
                    print(f"  [{ep}] {repo_id} (attempt {attempt+1}/{self.max_retries})")
                    snapshot_download(
                        repo_id, local_dir=local_dir,
                        local_dir_use_symlinks=False,
                        allow_patterns=allow_patterns, max_workers=2,
                    )
                    print(f"  [OK] Done")
                    return True
                except Exception as e:
                    wait = (attempt + 1) * 5
                    print(f"  [WARN] Failed: {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(wait)
            print(f"  [FAIL] Endpoint {ep} exhausted")
        return False

    def download_file(self, repo_id: str, filename: str, local_dir: str) -> bool:
        from huggingface_hub import hf_hub_download
        for ep in HF_ENDPOINTS:
            os.environ["HF_ENDPOINT"] = ep
            for attempt in range(self.max_retries):
                try:
                    print(f"  [{ep}] {filename} (attempt {attempt+1}/{self.max_retries})")
                    hf_hub_download(repo_id, filename, local_dir=local_dir,
                                    local_dir_use_symlinks=False)
                    print(f"  [OK] Done")
                    return True
                except Exception as e:
                    wait = (attempt + 1) * 5
                    print(f"  [WARN] Failed: {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(wait)
        return False

    def download_all_pretrained(self, target_dir: str) -> bool:
        all_ok = True
        pretrained = os.path.join(target_dir, "pretrained_models")
        os.makedirs(pretrained, exist_ok=True)

        # BERT
        bert_dir = os.path.join(pretrained, "chinese-roberta-wwm-ext-large")
        if not os.path.isfile(os.path.join(bert_dir, "pytorch_model.bin")):
            print(f"\n[DL] BERT ({MODEL_SOURCES['bert']['desc']})")
            if not self.download_snapshot(MODEL_SOURCES["bert"]["repo"], bert_dir):
                all_ok = False
        else:
            print(f"  [OK] BERT exists")

        # CNHubert
        cn_dir = os.path.join(pretrained, "chinese-hubert-base")
        if not os.path.isfile(os.path.join(cn_dir, "pytorch_model.bin")):
            print(f"\n[DL] CNHubert ({MODEL_SOURCES['cnhubert']['desc']})")
            if not self.download_snapshot(MODEL_SOURCES["cnhubert"]["repo"], cn_dir):
                all_ok = False
        else:
            print(f"  [OK] CNHubert exists")

        # v2 base
        v2_dir = os.path.join(pretrained, "gsv-v2final-pretrained")
        if not os.path.isfile(os.path.join(v2_dir, "s2G2333k.pth")):
            print(f"\n[DL] v2 base ({MODEL_SOURCES['v2_base']['desc']})")
            os.makedirs(v2_dir, exist_ok=True)
            for fname in MODEL_SOURCES["v2_base"]["files"]:
                self.download_file(
                    MODEL_SOURCES["v2_base"]["repo"],
                    f"{MODEL_SOURCES['v2_base']['subfolder']}/{fname}", v2_dir)
        else:
            print(f"  [OK] v2 base exists")

        # v2Pro base
        print(f"\n[DL] v2Pro base ({MODEL_SOURCES['v2pro_base']['desc']})")
        for fname in MODEL_SOURCES["v2pro_base"]["files"]:
            local_path = os.path.join(pretrained, fname)
            if os.path.isfile(local_path):
                print(f"  [OK] {fname} exists")
                continue
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            self.download_file(MODEL_SOURCES["v2pro_base"]["repo"], fname,
                               os.path.dirname(local_path))

        return all_ok


def download_models_cli():
    import argparse
    parser = argparse.ArgumentParser(description="Download GPT-SoVITS pretrained models")
    parser.add_argument("--target", default=None, help="Target directory")
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    if args.target:
        target = args.target
    else:
        try:
            import folder_paths
            target = os.path.join(folder_paths.models_dir, "TTS", "GPT-SoVITS")
        except ImportError:
            target = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                  "models", "TTS", "GPT-SoVITS")

    print(f"Target: {target}")
    downloader = MultiSourceDownloader(max_retries=args.retries)
    ok = downloader.download_all_pretrained(target)
    if ok:
        print("\n[OK] All critical models downloaded")
    else:
        print("\n[WARN] Some downloads failed. Retry or download manually.")
        sys.exit(1)


if __name__ == "__main__":
    download_models_cli()
