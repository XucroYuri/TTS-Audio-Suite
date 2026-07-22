"""ComfyUI node for an API-registered reference-audio asset."""

from comfy_extras.nodes_audio import load

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
        with get_audio_asset_store().lease(asset_id) as asset:
            waveform, sample_rate = load(str(asset.path))
            if not hasattr(waveform, "unsqueeze"):
                raise ValueError("ComfyUI audio loader returned an invalid waveform")
            audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
            voice = {
                "audio": audio,
                "audio_path": str(asset.path),
                "reference_text": reference_text,
                "character_name": "external",
            }
        return (
            voice,
        )
