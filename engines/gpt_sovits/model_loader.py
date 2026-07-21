"""
GPT-SoVITS Model Loader

Handles version detection from checkpoint binary headers and loads
the dual-model architecture (GPT Text2Semantic + SoVITS acoustic)
with optional Speaker Verification for v2Pro/v2ProPlus.

Version detection follows the official process_ckpt.py three-layer strategy:
  1. MD5 hash of first 8192 bytes → pretrained fingerprint table
  2. Binary header (first 2 bytes) → version encoding
  3. File size (<700MB → v2, else v3)
"""

import os
import sys
import hashlib
import traceback
from io import BytesIO
from typing import Dict, Optional, Tuple

# Ensure GPT-SoVITS source is importable
_GPT_SOVITS_SRC = os.environ.get(
    "GPT_SOVITS_PATH",
    os.path.join(os.path.dirname(__file__), "GPT_SoVITS_src")
)
if os.path.isdir(_GPT_SOVITS_SRC) and _GPT_SOVITS_SRC not in sys.path:
    sys.path.insert(0, _GPT_SOVITS_SRC)

import torch


# ============================================================
# Version Detection (from process_ckpt.py)
# ============================================================

# Binary header → version mapping
HEAD2VERSION = {
    b"\x00\x00": ("v1", "v1", False),
    b"\x00\x01": ("v2", "v2", False),
    b"\x00\x02": ("v2", "v3", False),
    b"\x00\x03": ("v2", "v3", True),   # v3 LoRA
    b"\x00\x04": ("v2", "v4", True),   # v4 LoRA
    b"\x00\x05": ("v2", "v2Pro", False),
    b"\x00\x06": ("v2", "v2ProPlus", False),
}

# MD5 hash → version for known pretrained models
HASH_PRETRAINED = {
    "dc3c97e17592963677a4a1681f30c653": ("v2", "v2", False),       # s2G488k.pth
    "43797be674a37c1c83ee81081941ed0f": ("v2", "v3", False),       # s2Gv3.pth
    "6642b37f3dbb1f76882b69937c95a5f3": ("v2", "v2", False),       # s2G2333k.pth
    "4f26b9476d0c5033e04162c486074374": ("v2", "v4", False),       # s2Gv4.pth
    "c7e9fce2223f3db685cdfa1e6368728a": ("v2", "v2Pro", False),    # s2Gv2Pro.pth
    "66b313e39455b57ab1b0bc0b239c9d0a": ("v2", "v2ProPlus", False),# s2Gv2ProPlus.pth
}


def _get_hash_from_file(path: str) -> str:
    """Compute MD5 hash of first 8192 bytes."""
    with open(path, "rb") as f:
        data = f.read(8192)
    return hashlib.md5(data).hexdigest()


def detect_sovits_version(sovits_path: str) -> Tuple[str, str, bool]:
    """Detect SoVITS model version from checkpoint file.

    Args:
        sovits_path: Path to .pth checkpoint

    Returns:
        (version, model_version, if_lora) tuple
        - version: "v1" or "v2" (symbol set)
        - model_version: "v1"/"v2"/"v3"/"v4"/"v2Pro"/"v2ProPlus"
        - if_lora: True if LoRA adapter weights
    """
    # Layer 1: MD5 hash (pretrained fingerprint)
    file_hash = _get_hash_from_file(sovits_path)
    if file_hash in HASH_PRETRAINED:
        return HASH_PRETRAINED[file_hash]

    # Layer 2: Binary header
    with open(sovits_path, "rb") as f:
        header = f.read(2)
    if header != b"PK":  # PK = standard zip header for old format
        if header in HEAD2VERSION:
            return HEAD2VERSION[header]

    # Layer 3: File size fallback
    if_lora = False
    size = os.path.getsize(sovits_path)
    if size < 82978 * 1024:        # < ~81MB → v1
        return ("v1", "v1", False)
    elif size < 700 * 1024 * 1024: # < 700MB → v2
        return ("v2", "v2", False)
    else:                           # > 700MB → v3
        return ("v2", "v3", False)


def load_sovits_checkpoint(sovits_path: str) -> Dict:
    """Load SoVITS checkpoint, handling custom binary headers."""
    with open(sovits_path, "rb") as f:
        meta = f.read(2)
    if meta != b"PK":
        # Custom header: prepend PK signature for torch.load compatibility
        with open(sovits_path, "rb") as f:
            f.read(2)  # skip custom header
            data = b"PK" + f.read()
        bio = BytesIO(data)
        bio.seek(0)
        return torch.load(bio, map_location="cpu", weights_only=False)
    return torch.load(sovits_path, map_location="cpu", weights_only=False)


# ============================================================
# Model Loading
# ============================================================

class GPTSovitsModelLoader:
    """Loads GPT-SoVITS dual-model architecture for inference."""

    def __init__(
        self,
        gpt_path: str,
        sovits_path: str,
        bert_path: str,
        cnhubert_path: str,
        device: str = "cuda",
        is_half: bool = True,
    ):
        self.gpt_path = gpt_path
        self.sovits_path = sovits_path
        self.bert_path = bert_path
        self.cnhubert_path = cnhubert_path
        self.device = torch.device(device)
        self.is_half = is_half

        # Detected version info
        self.version: str = "v2"
        self.model_version: str = "v2"
        self.if_lora: bool = False

        # Models
        self.t2s_model = None       # Text2Semantic (GPT)
        self.vq_model = None        # SynthesizerTrn (SoVITS)
        self.ssl_model = None       # CNHubert
        self.bert_tokenizer = None
        self.bert_model = None
        self.sv_model = None        # Speaker Verification (v2Pro only)
        self.hps = None             # SoVITS hyperparameters

        self._loaded = False

    def load_all(self):
        """Load all models required for inference."""
        if self._loaded:
            return

        self._load_ssl_model()
        self._load_bert_model()
        self._load_sovits_model()
        self._load_gpt_model()
        self._loaded = True
        print(f"✅ GPT-SoVITS loaded: version={self.model_version}, device={self.device}")

    def _load_ssl_model(self):
        """Load CNHubert SSL feature extractor."""
        from GPT_SoVITS.AR.models.t2s_lightning_module import Text2SemanticLightningModule
        import cnhubert

        cnhubert.cnhubert_base_path = self.cnhubert_path
        self.ssl_model = cnhubert.get_model()
        if self.is_half and str(self.device) != "cpu":
            self.ssl_model = self.ssl_model.half()
        self.ssl_model = self.ssl_model.to(self.device)
        self.ssl_model.eval()

    def _load_bert_model(self):
        """Load Chinese BERT for text feature extraction."""
        from transformers import AutoTokenizer, AutoModelForMaskedLM

        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.bert_path)
        self.bert_model = AutoModelForMaskedLM.from_pretrained(self.bert_path)
        if self.is_half and str(self.device) != "cpu":
            self.bert_model = self.bert_model.half()
        self.bert_model = self.bert_model.to(self.device)
        self.bert_model.eval()

    def _load_sovits_model(self):
        """Load SoVITS acoustic model with version detection."""
        from GPT_SoVITS.TTS_infer_pack.TTS import (
            SynthesizerTrn,
            DictToAttrRecursive,
        )
        from GPT_SoVITS.process_ckpt import get_sovits_version_from_path_fast
        import GPT_SoVITS.process_ckpt

        # Version detection
        self.version, self.model_version, self.if_lora = get_sovits_version_from_path_fast(
            self.sovits_path
        )
        print(f"   SoVITS version: {self.version}/{self.model_version}, LoRA={self.if_lora}")

        # Load checkpoint
        dict_s2 = load_sovits_checkpoint(self.sovits_path)
        hps = dict_s2["config"]
        hps = DictToAttrRecursive(hps)
        hps.model.semantic_frame_rate = "25hz"

        # Infer symbol version from embedding shape
        if "enc_p.text_embedding.weight" not in dict_s2["weight"]:
            hps.model.version = "v2"
        elif dict_s2["weight"]["enc_p.text_embedding.weight"].shape[0] == 322:
            hps.model.version = "v1"
        else:
            hps.model.version = "v2"

        # Override with detected model_version if applicable
        if self.model_version not in {"v3", "v4"}:
            if "Pro" not in self.model_version:
                self.model_version = hps.model.version
            else:
                hps.model.version = self.model_version
        else:
            hps.model.version = self.model_version

        # Initialize model
        self.vq_model = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        )

        if "pretrained" not in self.sovits_path:
            try:
                del self.vq_model.enc_q
            except Exception:
                pass

        if self.is_half and str(self.device) != "cpu":
            self.vq_model = self.vq_model.half()
        self.vq_model = self.vq_model.to(self.device)

        if not self.if_lora:
            res = self.vq_model.load_state_dict(dict_s2["weight"], strict=False)
            print(f"   SoVITS state dict: {res}")
        else:
            # LoRA requires loading base model first, then LoRA adapter
            # For v3/v4 only; skip for now since we target v2 ecosystem
            print(f"   ⚠️ LoRA not yet supported; loading base weights only")
            self.vq_model.load_state_dict(dict_s2["weight"], strict=False)

        self.vq_model.eval()
        self.hps = hps

        # Load Speaker Verification model for v2Pro/v2ProPlus
        self.is_v2pro = self.model_version in {"v2Pro", "v2ProPlus"}
        if self.is_v2pro:
            self._load_sv_model()

    def _load_sv_model(self):
        """Load Speaker Verification model for v2Pro/v2ProPlus."""
        try:
            from GPT_SoVITS.TTS_infer_pack.TTS import SpeakerVerification

            sv_path = os.path.join(
                os.path.dirname(self.sovits_path), "..", "pretrained_models",
                "sv", "pretrained_eres2netv2w24s4ep4.ckpt"
            )
            if not os.path.isfile(sv_path):
                print(f"   ⚠️ SV model not found at {sv_path}, skipping")
                self.is_v2pro = False
                return
            self.sv_model = SpeakerVerification(sv_path, self.device)
            self.sv_model.model.eval()
            print(f"   ✅ Speaker Verification loaded")
        except Exception as e:
            print(f"   ⚠️ SV model load failed: {e}")
            self.is_v2pro = False

    def _load_gpt_model(self):
        """Load GPT Text2Semantic model."""
        from GPT_SoVITS.AR.models.t2s_lightning_module import Text2SemanticLightningModule

        dict_s1 = torch.load(self.gpt_path, map_location="cpu", weights_only=False)
        config = dict_s1["config"]

        self.t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
        self.t2s_model.load_state_dict(dict_s1["weight"])

        if self.is_half and str(self.device) != "cpu":
            self.t2s_model = self.t2s_model.half()
        self.t2s_model = self.t2s_model.to(self.device)
        self.t2s_model.eval()

        self.max_sec = config["data"]["max_sec"]
        self.hz = 50
        print(f"   GPT max_sec={self.max_sec}, hz={self.hz}")

    def unload(self):
        """Unload all models to free memory."""
        self.t2s_model = None
        self.vq_model = None
        self.ssl_model = None
        self.bert_model = None
        self.bert_tokenizer = None
        self.sv_model = None
        self.hps = None
        self._loaded = False
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def to(self, device):
        """Move all models to specified device (for ComfyUI Clear VRAM)."""
        self.device = torch.device(device) if isinstance(device, str) else device
        for attr in ["t2s_model", "vq_model", "ssl_model", "bert_model", "sv_model"]:
            model = getattr(self, attr, None)
            if model is not None and hasattr(model, "to"):
                setattr(self, attr, model.to(self.device))
        return self
