"""DramaBox text orchestration for character, pause, and segment parameters."""

import os
import re
import sys
from collections import deque
from typing import Any, Dict, List, Tuple, Union

import comfy.model_management as model_management
import torch

current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(nodes_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.audio.chunk_timing import ChunkTimingHelper
from utils.text.character_parser import character_parser
from utils.text.pause_processor import PauseTagProcessor
from utils.text.segment_parameters import ParameterValidator, apply_segment_parameters
from utils.voice.character_logging import (
    format_resolved_character_block,
    resolved_character_label,
)
from utils.voice.discovery import get_available_characters, get_character_mapping, voice_discovery
from utils.voice.reference import effective_voice_audio


class DramaBoxProcessor:
    """Keep suite orchestration outside the official DramaBox model wrapper."""

    SAMPLE_RATE = 48000
    DIALOGUE_QUOTE_MARKS = frozenset('"“”„‟«»')
    DIALOGUE_QUOTE_PAIRS = (
        ('"', '"'),
        ("“", "”"),
        ("„", "“"),
        ("‟", "‟"),
        ("«", "»"),
    )

    def __init__(self, adapter, engine_config: Dict[str, Any]):
        self.adapter = adapter
        self.config = engine_config.copy() if engine_config else {}
        self._warned_missing_seg_templates = set()

    def update_config(self, new_config: Dict[str, Any]):
        self.config = new_config.copy() if new_config else {}
        self.adapter.update_config(self.config)

    @staticmethod
    def _check_interrupt():
        if model_management.interrupt_processing:
            raise InterruptedError("DramaBox text generation interrupted by user")

    def _setup_character_parser(self, text: str):
        character_tags = re.findall(r"\[([^\]]+)\]", text or "")
        characters_from_tags = [
            tag.split("|")[0].strip()
            for tag in character_tags
            if not tag.lower().startswith("pause:")
        ]

        all_available = set(get_available_characters() or [])
        for alias, target in voice_discovery.get_character_aliases().items():
            all_available.add(alias.lower())
            all_available.add(target.lower())
        all_available.update(char.lower() for char in characters_from_tags)
        all_available.add("narrator")
        character_parser.set_available_characters(list(all_available))

        for character, language in voice_discovery.get_character_language_defaults().items():
            character_parser.set_character_language_default(character, language)
        character_parser.reset_session_cache()

    @staticmethod
    def _voice_note(voice_ref: Dict[str, Any], ref_duration: float) -> str:
        if not isinstance(voice_ref, dict) or effective_voice_audio(voice_ref) is None:
            return " [no voice reference]"
        return f" [{float(ref_duration):g}s reference window]"

    def _resolve_voice(
        self,
        character: str,
        voice_mapping: Dict[str, Any],
        discovered_mapping: Dict[str, Any],
    ) -> Dict[str, Any]:
        if character in voice_mapping:
            value = voice_mapping[character]
            return value.copy() if isinstance(value, dict) else {"audio": value}
        if character != "narrator":
            audio_path, reference_text = discovered_mapping.get(character, (None, None))
            if audio_path:
                return {
                    "audio_path": audio_path,
                    "reference_text": reference_text or "",
                }
        narrator = voice_mapping.get("narrator", {})
        return narrator.copy() if isinstance(narrator, dict) else {"audio": narrator}

    @staticmethod
    def _parameter_log(
        filtered_params: Dict[str, Any],
        current_config: Dict[str, Any],
        current_seed: int,
    ) -> str:
        if not filtered_params:
            return ""
        parts = []
        if "seed" in filtered_params:
            parts.append(f"seed={current_seed}")
        for key in (
            "cfg_scale",
            "stg_scale",
            "duration_multiplier",
            "gen_duration",
            "ref_duration",
            "rescale_scale",
            "prompt_template",
            "negative_prompt",
        ):
            if key in filtered_params:
                value = current_config.get(key)
                if key in {"negative_prompt", "prompt_template"}:
                    value = repr(str(value))
                parts.append(f"{key}={value}")
        return ", ".join(parts)

    @staticmethod
    def _text_weight(text: str) -> int:
        return max(1, len(re.sub(r"\s+", "", str(text or ""))))

    def _prepare_prompt(
        self,
        text: str,
        prompt_template: str = '"{seg}"',
    ) -> str:
        """Prepare literal dialogue while preserving complete scene prompts."""
        prompt = str(text or "").strip()
        if not prompt:
            return prompt

        has_quotes = any(mark in prompt for mark in self.DIALOGUE_QUOTE_MARKS)
        dialogue_only = any(
            prompt.startswith(opening)
            and prompt.endswith(closing)
            and not any(
                mark in prompt[len(opening):-len(closing)]
                for mark in self.DIALOGUE_QUOTE_MARKS
            )
            for opening, closing in self.DIALOGUE_QUOTE_PAIRS
        )
        if has_quotes and not dialogue_only:
            return prompt

        template = str(prompt_template or "")
        if not template:
            return prompt

        if "{seg}" not in template:
            if template not in self._warned_missing_seg_templates:
                print(
                    "⚠️ DramaBox prompt template did not contain {seg}; "
                    'appended spoken segment as "{seg}" automatically.'
                )
                self._warned_missing_seg_templates.add(template)
            separator = "" if template[-1].isspace() else " "
            template = f'{template}{separator}"{{seg}}"'
        segment_value = prompt[1:-1] if dialogue_only else prompt
        return template.replace("{seg}", segment_value)

    def _plan_duration_targets(
        self,
        segments,
        total_duration: float,
    ) -> deque:
        """Allocate one native generation target per spoken fragment."""
        items = []
        pause_total = 0.0
        for segment in segments:
            text = (segment.text or "").strip()
            if not text:
                continue
            if PauseTagProcessor.has_pause_tags(text):
                fragments, _ = PauseTagProcessor.parse_pause_tags(text)
            else:
                fragments = [("text", text)]
            explicit = (segment.parameters or {}).get("gen_duration")
            explicit = float(explicit) if explicit is not None and float(explicit) > 0 else None
            for kind, content in fragments:
                if kind == "pause":
                    pause_total += float(content)
                elif str(content).strip():
                    items.append({
                        "explicit": explicit,
                        "weight": self._text_weight(str(content)),
                    })

        variable = [item for item in items if item["explicit"] is None]
        explicit_total = sum(float(item["explicit"]) for item in items if item["explicit"] is not None)
        remaining = float(total_duration) - pause_total - explicit_total
        if variable and remaining <= 0:
            remaining = max(0.05, float(total_duration) / len(variable)) * len(variable)
            print(
                "⚠️ DramaBox SRT: Subtitle timing leaves no room after pause tags "
                "and explicit durations; using minimal native speech targets."
            )

        total_weight = sum(item["weight"] for item in variable)
        allocated = 0.0
        variable_index = 0
        targets = deque()
        for item in items:
            if item["explicit"] is not None:
                targets.append(float(item["explicit"]))
                continue
            if variable_index == len(variable) - 1:
                target = max(0.05, remaining - allocated)
            else:
                target = max(0.05, remaining * item["weight"] / total_weight)
                allocated += target
            variable_index += 1
            targets.append(float(target))
        return targets

    def process_text(
        self,
        text: str,
        voice_mapping: Dict[str, Any],
        seed: int,
        enable_chunking: bool = True,
        max_chars_per_chunk: int = 400,
        chunk_combination_method: str = "auto",
        silence_between_chunks_ms: int = 100,
        enable_audio_cache: bool = True,
        show_text_logging: bool = True,
        duration_budget: float = None,
        **_unused,
    ) -> List[Dict[str, Any]]:
        """Generate suite-level character/pause segments.

        DramaBox owns chunking inside each segment because its official
        duration-aware chunker preserves scene prefixes and quote groups.
        """
        del enable_chunking, max_chars_per_chunk
        del chunk_combination_method, silence_between_chunks_ms

        self._check_interrupt()
        self._setup_character_parser(text)
        base_config = self.config.copy()
        segments = character_parser.parse_text_segments(text)
        if not segments:
            segments = character_parser.parse_text_segments("narrator " + text)
        duration_targets = (
            self._plan_duration_targets(segments, float(duration_budget))
            if duration_budget is not None and float(duration_budget) > 0
            else deque()
        )

        characters = list({segment.character for segment in segments if segment.character})
        discovered_mapping = get_character_mapping(characters, engine_type="audio_only")
        records: List[Dict[str, Any]] = []

        for segment in segments:
            self._check_interrupt()
            segment_text = (segment.text or "").strip()
            if not segment_text:
                continue

            segment_params = segment.parameters or {}
            filtered = ParameterValidator.filter_parameters_for_engine(
                segment_params, "dramabox"
            )
            current_config = apply_segment_parameters(
                base_config, segment_params, "dramabox"
            ) if segment_params else base_config.copy()
            current_seed = int(current_config.get("seed", seed))
            self.adapter.update_config(current_config)

            character = segment.character or "narrator"
            voice_ref = self._resolve_voice(character, voice_mapping, discovered_mapping)
            display_name = resolved_character_label(character, voice_ref)
            parameter_log = self._parameter_log(filtered, current_config, current_seed)
            print(
                f"🎭 DramaBox - Generating for '{display_name}'"
                f"{self._voice_note(voice_ref, current_config.get('ref_duration', 10.0))}"
            )
            print(
                "   Settings: "
                f"cfg_scale={current_config.get('cfg_scale', 2.5)}, "
                f"stg_scale={current_config.get('stg_scale', 1.5)}, "
                f"duration_multiplier={current_config.get('duration_multiplier', 1.1)}, "
                f"gen_duration={current_config.get('gen_duration', 0.0)}, "
                f"ref_duration={current_config.get('ref_duration', 10.0)}, "
                f"rescale_scale={current_config.get('rescale_scale', 'auto')}, "
                f"seed={current_seed}"
            )
            if parameter_log:
                print(f"🎛️ DramaBox segment params: {parameter_log}")

            def generate_fragment(fragment: str):
                self._check_interrupt()
                fragment_config = current_config.copy()
                prompt = self._prepare_prompt(
                    fragment,
                    str(
                        fragment_config.get(
                            "prompt_template", '"{seg}"'
                        )
                    ),
                )
                if show_text_logging:
                    print(
                        format_resolved_character_block(
                            character, prompt, voice_ref
                        )
                    )
                if duration_targets:
                    fragment_config["gen_duration"] = duration_targets.popleft()
                    print(
                        f"   🎯 Native target: "
                        f"{float(fragment_config['gen_duration']):.3f}s"
                    )
                self.adapter.update_config(fragment_config)
                audio = self.adapter.generate_single(
                    text=prompt,
                    voice_ref=voice_ref,
                    seed=current_seed,
                    enable_audio_cache=enable_audio_cache,
                    character_name=character,
                )
                if audio.dim() == 1:
                    audio = audio.unsqueeze(0)
                records.append({
                    "waveform": audio.cpu(),
                    "sample_rate": self.SAMPLE_RATE,
                    "text": fragment,
                    "character": character,
                    "generation_status": dict(
                        getattr(
                            self.adapter,
                            "last_generation_status",
                            {"near_silent": False},
                        )
                    ),
                })

            if PauseTagProcessor.has_pause_tags(segment_text):
                pause_segments, _ = PauseTagProcessor.parse_pause_tags(segment_text)
                for fragment_type, content in pause_segments:
                    self._check_interrupt()
                    if fragment_type == "text" and str(content).strip():
                        generate_fragment(str(content).strip())
                    elif fragment_type == "pause":
                        silence = PauseTagProcessor.create_silence_segment(
                            content,
                            self.SAMPLE_RATE,
                            torch.device("cpu"),
                            torch.float32,
                        )
                        if silence.dim() == 1:
                            silence = silence.unsqueeze(0)
                        if silence.shape[0] == 1:
                            silence = silence.repeat(2, 1)
                        records.append({
                            "waveform": silence,
                            "sample_rate": self.SAMPLE_RATE,
                            "text": f"[pause:{content}s]",
                        })
            else:
                generate_fragment(segment_text)

        self.adapter.update_config(base_config)
        return records

    def combine_audio_segments(
        self,
        segments: List[Dict[str, Any]],
        method: str = "auto",
        silence_ms: int = 100,
        original_text: str = "",
        return_info: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        if not segments:
            empty = torch.zeros(0, dtype=torch.float32)
            return (empty, {}) if return_info else empty

        audio, chunk_info = ChunkTimingHelper.combine_audio_with_timing(
            audio_segments=[segment["waveform"] for segment in segments],
            combination_method=method,
            silence_ms=silence_ms,
            crossfade_duration=0.1,
            sample_rate=self.SAMPLE_RATE,
            text_length=len(original_text),
            original_text=original_text,
            text_chunks=[segment.get("text", "") for segment in segments],
        )
        return (audio, chunk_info) if return_info else audio
