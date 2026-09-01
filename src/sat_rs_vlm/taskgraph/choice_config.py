"""Configuration frozen for cached reasoning-to-choice execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ChoiceSystemConfig:
    backend: str = "kv_cached_logits"
    final_suffix: str = "\n\nFinal choice:"
    legacy_regex_fallback: bool = False
    multi_select_threshold: float = 0.0
    multi_empty_policy: str = "EMPTY"
    preserve_reasoning_text: bool = True

    def __post_init__(self) -> None:
        if self.backend != "kv_cached_logits":
            raise ValueError("choice.backend must be kv_cached_logits")
        if not self.final_suffix:
            raise ValueError("choice.final_suffix must not be empty")
        if self.multi_empty_policy not in {"EMPTY", "UNRESOLVED"}:
            raise ValueError("choice.multi_empty_policy must be EMPTY or UNRESOLVED")

    @classmethod
    def from_mapping(cls, value: Any) -> ChoiceSystemConfig:
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise TypeError("choice config must be a mapping")
        return cls(
            backend=str(value.get("backend", "kv_cached_logits")),
            final_suffix=str(value.get("final_suffix", "\n\nFinal choice:")),
            legacy_regex_fallback=bool(value.get("legacy_regex_fallback", False)),
            multi_select_threshold=float(value.get("multi_select_threshold", 0.0)),
            multi_empty_policy=str(value.get("multi_empty_policy", "EMPTY")),
            preserve_reasoning_text=bool(value.get("preserve_reasoning_text", True)),
        )
