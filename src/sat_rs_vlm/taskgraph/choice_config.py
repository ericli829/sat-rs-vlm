"""Configuration frozen for cached reasoning-to-choice execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_SINGLE_CHOICE_SUFFIX = "\n\nFinal choice:"
DEFAULT_MULTI_VERIFY_TEMPLATE = (
    "Verify the following option independently.\n\n"
    "Candidate option {choice_id}: {option_text}\n"
    "Is this option a correct answer to the original question?\n"
    "Answer YES or NO:"
)


@dataclass(frozen=True)
class ChoiceSystemConfig:
    backend: str = "kv_cached_logits"
    single_choice_suffix: str = DEFAULT_SINGLE_CHOICE_SUFFIX
    multi_verify_template: str = DEFAULT_MULTI_VERIFY_TEMPLATE
    legacy_regex_fallback: bool = False
    multi_select_threshold: float = 0.0
    multi_empty_policy: str = "EMPTY"
    preserve_reasoning_text: bool = True
    # MME "This image doesn't feature the ..." options are selected by the
    # semantic provider far more often than the reference ground truth uses
    # them (E is the correct answer ~2 of 3538 samples).  When enabled the
    # resolver removes such options before scoring.
    forbid_non_feature_options: bool = False

    def __post_init__(self) -> None:
        if self.backend != "kv_cached_logits":
            raise ValueError("choice.backend must be kv_cached_logits")
        if not self.single_choice_suffix:
            raise ValueError("choice.single_choice_suffix must not be empty")
        if (
            "{choice_id}" not in self.multi_verify_template
            or "{option_text}" not in self.multi_verify_template
        ):
            raise ValueError(
                "choice.multi_verify_template must include {choice_id} and {option_text}"
            )
        try:
            rendered = self.multi_verify_template.format(choice_id="A", option_text="example")
        except (AttributeError, IndexError, KeyError, ValueError) as exc:
            raise ValueError(
                "choice.multi_verify_template must support choice_id and option_text"
            ) from exc
        if not rendered:
            raise ValueError("choice.multi_verify_template must not be empty")
        if self.multi_empty_policy not in {"EMPTY", "UNRESOLVED"}:
            raise ValueError("choice.multi_empty_policy must be EMPTY or UNRESOLVED")

    @classmethod
    def from_mapping(cls, value: Any) -> ChoiceSystemConfig:
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise TypeError("choice config must be a mapping")
        single_choice_suffix = value.get(
            "single_choice_suffix",
            value.get("final_suffix", DEFAULT_SINGLE_CHOICE_SUFFIX),
        )
        return cls(
            backend=str(value.get("backend", "kv_cached_logits")),
            single_choice_suffix=str(single_choice_suffix),
            multi_verify_template=str(
                value.get("multi_verify_template", DEFAULT_MULTI_VERIFY_TEMPLATE)
            ),
            legacy_regex_fallback=bool(value.get("legacy_regex_fallback", False)),
            multi_select_threshold=float(value.get("multi_select_threshold", 0.0)),
            multi_empty_policy=str(value.get("multi_empty_policy", "EMPTY")),
            preserve_reasoning_text=bool(value.get("preserve_reasoning_text", True)),
            forbid_non_feature_options=bool(value.get("forbid_non_feature_options", False)),
        )
