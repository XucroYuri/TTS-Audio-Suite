"""ComfyUI node for an API-registered reference-audio asset."""

import io

import soundfile
import torch

from api_bridge.assets import get_audio_asset_store


class ExternalAudioAssetNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "asset_id": ("STRING", {"default": ""}),
                "reference_text": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("NARRATOR_VOICE",)
    RETURN_NAMES = ("voice",)
    FUNCTION = "load_asset"
    CATEGORY = "TTS Audio Suite/API Bridge"

    def load_asset(self, asset_id: str, reference_text: str):
        with get_audio_asset_store().lease(asset_id) as snapshot:
            waveform, sample_rate = _load_snapshot(snapshot.content)
            audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
            voice = {
                "asset_id": snapshot.asset.asset_id,
                "audio": audio,
                "audio_path": str(snapshot.asset.path),
                "reference_text": reference_text,
                "character_name": "external",
            }
        return (
            voice,
        )


def _load_snapshot(content: bytes) -> tuple[torch.Tensor, int]:
    """Decode the exact bytes authenticated by ``AudioAssetStore.lease``."""
    try:
        samples, sample_rate = soundfile.read(io.BytesIO(content), dtype="float32", always_2d=True)
    except Exception as exc:
        raise ValueError("invalid audio snapshot") from exc
    return torch.from_numpy(samples.T).contiguous(), sample_rate
