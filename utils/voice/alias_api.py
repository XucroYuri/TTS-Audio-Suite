"""HTTP routes used by the Character Alias Manager frontend."""

from __future__ import annotations

import os
from typing import Any, Dict

from utils.voice.alias_store import (
    clear_user_aliases,
    get_user_alias_file,
    parse_alias_document,
    write_user_aliases,
)
from utils.voice.discovery import (
    get_available_characters,
    get_character_alias_records,
    refresh_character_aliases,
    voice_discovery,
)


def _payload(force_refresh: bool = False) -> Dict[str, Any]:
    layers = get_character_alias_records(force_refresh=force_refresh)
    try:
        from utils.models.language_mapper import LANGUAGE_ALIASES
        languages = sorted(set(LANGUAGE_ALIASES.values()))
    except Exception:
        languages = ["de", "en", "es", "fr", "it", "ja", "no", "pt", "th"]

    characters = sorted(get_available_characters(force_refresh=False))
    character_details = {}
    for character in characters:
        info = voice_discovery.get_character_voice_info(character, engine_type="audio_only") or {}
        character_details[character] = {
            "hasAudio": bool(info.get("audio_path")),
            "hasReferenceText": bool(str(info.get("text_content") or "").strip()),
        }

    records = layers["records"]
    user_count = sum(record.get("source") == "user" for record in records)
    alias_file = get_user_alias_file()
    user_groups = parse_alias_document(alias_file) if alias_file and os.path.exists(alias_file) else []
    return {
        "aliases": records,
        "inheritedAliases": layers["inherited"],
        "characters": characters,
        "characterDetails": character_details,
        "languages": languages,
        "counts": {
            "all": len(records),
            "user": user_count,
            "inherited": len(records) - user_count,
        },
        "userFile": alias_file or "",
        "userGroups": user_groups,
        "hasUserFile": bool(alias_file and os.path.exists(alias_file)),
    }


def register_character_alias_routes(routes, web) -> None:
    """Register alias CRUD routes on ComfyUI's shared aiohttp router."""

    def json_response(payload: Dict[str, Any], status: int = 200):
        response = web.json_response(payload, status=status)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        return response

    @routes.get("/api/tts-audio-suite/character-aliases")
    async def get_character_aliases_endpoint(request):
        try:
            force_refresh = request.query.get("refresh", "0").strip().lower() in {"1", "true", "yes"}
            return json_response(_payload(force_refresh=force_refresh))
        except Exception as error:
            print(f"⚠️ Error retrieving character aliases: {error}")
            return json_response({"error": str(error)}, status=500)

    @routes.post("/api/tts-audio-suite/character-aliases")
    async def save_character_aliases_endpoint(request):
        try:
            data = await request.json()
            alias_file = write_user_aliases(data.get("aliases"), data.get("groups"))
            refresh_character_aliases()
            payload = _payload()
            payload.update({"status": "success", "userFile": alias_file})
            return json_response(payload)
        except ValueError as error:
            return json_response({"error": str(error)}, status=400)
        except Exception as error:
            print(f"⚠️ Error saving character aliases: {error}")
            return json_response({"error": str(error)}, status=500)

    @routes.post("/api/tts-audio-suite/character-aliases/reset")
    async def reset_character_aliases_endpoint(request):
        try:
            clear_user_aliases()
            refresh_character_aliases()
            payload = _payload()
            payload["status"] = "success"
            return json_response(payload)
        except Exception as error:
            print(f"⚠️ Error resetting character aliases: {error}")
            return json_response({"error": str(error)}, status=500)

    @routes.get("/api/tts-audio-suite/character-preview")
    async def get_character_preview_endpoint(request):
        """Stream a canonical character voice using runtime alias resolution."""
        try:
            character_name = request.query.get("character_name", "").strip()
            if not character_name:
                return json_response({"error": "character_name is required"}, status=400)
            audio_path, _ = voice_discovery.load_character_voice(character_name, engine_type="audio_only")
            if not audio_path or not os.path.exists(audio_path):
                return json_response({"error": f"Character voice not found: {character_name}"}, status=404)
            response = web.FileResponse(path=audio_path)
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
            return response
        except Exception as error:
            print(f"⚠️ Error serving character preview audio: {error}")
            return json_response({"error": str(error)}, status=500)
