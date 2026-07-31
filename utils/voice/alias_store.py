"""Per-user character alias persistence for the in-ComfyUI alias manager."""

from __future__ import annotations

import os
import re
import tempfile
from typing import Any, Dict, List, Optional, Tuple

try:
    import folder_paths
except ImportError:
    folder_paths = None


ALIAS_FILENAME = "#character_alias_map.txt"
MAX_USER_ALIASES = 1000
MAX_USER_GROUPS = 100
_LANGUAGE_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_GROUP_RE = re.compile(r"^##\s*group\s*:\s*(.+?)\s*$", re.IGNORECASE)
_MANAGED_COMMENTS = {
    "Character Alias Map - managed by TTS Audio Suite",
    "User aliases in this file override example and model-folder aliases.",
    "Format: Alias = Character_Name, optional_language",
}


def get_user_alias_file() -> Optional[str]:
    """Return the current ComfyUI profile's highest-priority alias file."""
    if folder_paths is None:
        return None
    base_dir = folder_paths.get_system_user_directory("tts_audio_suite")
    return os.path.join(base_dir, "voices", ALIAS_FILENAME)


def parse_alias_line(line: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Parse the legacy equals or tab-separated alias formats."""
    alias = target = language = None
    if "=" in line:
        alias, right_side = (part.strip() for part in line.split("=", 1))
        if "," in right_side:
            target, language = (part.strip() for part in right_side.split(",", 1))
        else:
            target = right_side
    elif "\t" in line:
        parts = [part.strip() for part in line.split("\t") if part.strip()]
        if len(parts) >= 2:
            alias, target = parts[:2]
            language = parts[2] if len(parts) >= 3 else None

    if not alias or not target:
        return None, None, None
    return alias, target, language or None


def parse_alias_document(path: str, infer_legacy_groups: bool = False) -> List[Dict[str, Any]]:
    """Read aliases as ordered groups while retaining ordinary comments as notes."""
    groups: List[Dict[str, Any]] = [{"name": "Ungrouped", "notes": [], "aliases": [], "_explicit": False}]
    current = groups[0]
    with open(path, "r", encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if not line:
                continue
            group_match = _GROUP_RE.match(line)
            if group_match:
                current = {"name": group_match.group(1).strip(), "notes": [], "aliases": [], "_explicit": True}
                groups.append(current)
                continue
            if line.startswith("#"):
                note = line.lstrip("#").strip()
                if infer_legacy_groups and note.endswith(":"):
                    current = {"name": note[:-1].strip(), "notes": [], "aliases": [], "_explicit": False}
                    groups.append(current)
                    continue
                if note and note not in _MANAGED_COMMENTS:
                    current["notes"].append(note)
                continue
            alias, target, language = parse_alias_line(line)
            if alias and target:
                current["aliases"].append({
                    "alias": alias, "target": target, "language": language or "",
                })
    result = []
    for group in groups:
        if group["aliases"] or group["notes"] or group["_explicit"]:
            result.append({key: value for key, value in group.items() if key != "_explicit"})
    return result


def normalize_user_aliases(entries: Any) -> List[Dict[str, str]]:
    """Validate API input while preserving the user's display casing."""
    if not isinstance(entries, list):
        raise ValueError("aliases must be a list")
    if len(entries) > MAX_USER_ALIASES:
        raise ValueError(f"aliases cannot contain more than {MAX_USER_ALIASES} entries")

    normalized: List[Dict[str, str]] = []
    seen = set()
    for index, entry in enumerate(entries, 1):
        if not isinstance(entry, dict):
            raise ValueError(f"alias entry {index} must be an object")

        alias = str(entry.get("alias", "")).strip()
        target = str(entry.get("target", "")).strip()
        language = str(entry.get("language", "")).strip()
        if not alias or not target:
            raise ValueError(f"alias entry {index} requires both alias and character voice")
        if any(character in alias for character in "=[]\r\n\t"):
            raise ValueError(f"alias '{alias}' contains a reserved character")
        if any(character in target for character in ",\r\n\t"):
            raise ValueError(f"character voice '{target}' contains a reserved character")
        if language and not _LANGUAGE_RE.fullmatch(language):
            raise ValueError(f"language '{language}' must be a language code")

        key = alias.casefold()
        if key in seen:
            raise ValueError(f"duplicate alias '{alias}'")
        seen.add(key)
        normalized.append({"alias": alias, "target": target, "language": language.lower()})

    return normalized


def normalize_user_groups(groups: Any) -> List[Dict[str, Any]]:
    """Validate ordered UI groups and their aliases."""
    if not isinstance(groups, list):
        raise ValueError("groups must be a list")
    if len(groups) > MAX_USER_GROUPS:
        raise ValueError(f"groups cannot contain more than {MAX_USER_GROUPS} entries")

    normalized_groups = []
    all_aliases = []
    seen_names = set()
    for index, group in enumerate(groups, 1):
        if not isinstance(group, dict):
            raise ValueError(f"group {index} must be an object")
        name = str(group.get("name", "")).strip() or "Ungrouped"
        if any(character in name for character in "\r\n"):
            raise ValueError(f"group '{name}' contains a reserved character")
        name_key = name.casefold()
        if name_key in seen_names:
            raise ValueError(f"duplicate group '{name}'")
        seen_names.add(name_key)
        raw_notes = [str(note) for note in group.get("notes", [])]
        if any("\n" in note or "\r" in note for note in raw_notes):
            raise ValueError(f"group '{name}' contains a multiline note")
        notes = [note.strip().lstrip("#").strip() for note in raw_notes]
        notes = [note for note in notes if note]
        aliases = normalize_user_aliases(group.get("aliases", []))
        all_aliases.extend(aliases)
        normalized_groups.append({"name": name, "notes": notes, "aliases": aliases})

    # Validate aliases across groups, not only inside each group.
    normalize_user_aliases(all_aliases)
    return normalized_groups


def write_user_aliases(entries: Any = None, groups: Any = None) -> str:
    """Atomically replace the per-user alias override file."""
    document = normalize_user_groups(groups) if groups is not None else [
        {"name": "Ungrouped", "notes": [], "aliases": normalize_user_aliases(entries)}
    ]
    alias_file = get_user_alias_file()
    if not alias_file:
        raise RuntimeError("ComfyUI user storage is unavailable")

    os.makedirs(os.path.dirname(alias_file), exist_ok=True)
    lines = [
        "# Character Alias Map - managed by TTS Audio Suite",
        "# User aliases in this file override example and model-folder aliases.",
        "# Format: Alias = Character_Name, optional_language",
        "",
    ]
    for group_index, group in enumerate(document):
        if group_index or group["name"] != "Ungrouped":
            if lines[-1]:
                lines.append("")
            lines.append(f"## group: {group['name']}")
        for note in group["notes"]:
            lines.append(f"# {note}")
        for entry in group["aliases"]:
            suffix = f", {entry['language']}" if entry["language"] else ""
            lines.append(f"{entry['alias']} = {entry['target']}{suffix}")
    content = "\n".join(lines).rstrip() + "\n"

    handle, temp_path = tempfile.mkstemp(
        prefix=f".{ALIAS_FILENAME}.", suffix=".tmp", dir=os.path.dirname(alias_file)
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, alias_file)
    except Exception:
        try:
            os.remove(temp_path)
        except OSError:
            pass
        raise
    return alias_file


def clear_user_aliases() -> Optional[str]:
    """Remove only UI-owned aliases; inherited files are never touched."""
    alias_file = get_user_alias_file()
    if alias_file and os.path.exists(alias_file):
        os.remove(alias_file)
    return alias_file
