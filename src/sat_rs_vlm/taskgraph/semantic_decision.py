"""Canonical finite semantic decisions backed by the shared VLM KV cache."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from .choice_config import DEFAULT_MULTI_VERIFY_TEMPLATE
from .providers import (
    CachedChoiceUnavailableError,
    FiniteDecisionRequest,
    FiniteDecisionResult,
    ModelInput,
    SemanticVLMProvider,
)

DEFAULT_SINGLE_DECISION_SUFFIX = "\n\nCanonical value:"


class SemanticDecisionUnresolvedError(RuntimeError):
    """A finite semantic decision could not produce one defensible value."""


@dataclass(frozen=True)
class SemanticDecisionConfig:
    enabled: bool = True
    legacy_text_fallback: bool = False
    preserve_reasoning_text: bool = True
    single_decision_suffix: str = DEFAULT_SINGLE_DECISION_SUFFIX
    multi_verify_template: str = DEFAULT_MULTI_VERIFY_TEMPLATE
    multi_select_threshold: float = 0.0
    uncertainty_epsilon: float = 0.0
    attributes: dict[str, tuple[str, ...]] | None = None

    def __post_init__(self) -> None:
        if not self.enabled:
            raise ValueError(
                "semantic_decision.enabled=false is not supported for finite operators"
            )
        if self.legacy_text_fallback:
            raise ValueError("semantic_decision legacy text fallback is intentionally unsupported")
        if not self.single_decision_suffix:
            raise ValueError("semantic decision suffix must not be empty")
        if (
            "{choice_id}" not in self.multi_verify_template
            or "{option_text}" not in self.multi_verify_template
        ):
            raise ValueError("semantic multi verify template must include candidate placeholders")
        if self.uncertainty_epsilon < 0.0:
            raise ValueError("semantic uncertainty epsilon must be non-negative")
        for attribute, values in (self.attributes or {}).items():
            if not attribute.strip() or not values or len(values) != len(set(values)):
                raise ValueError("semantic attribute value spaces must be non-empty and unique")

    @classmethod
    def from_mapping(cls, value: Any) -> SemanticDecisionConfig:
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise TypeError("semantic_decision config must be a mapping")
        raw_attributes = value.get("attributes", {})
        if not isinstance(raw_attributes, dict):
            raise TypeError("semantic_decision.attributes must be a mapping")
        attributes: dict[str, tuple[str, ...]] = {}
        for name, raw_space in raw_attributes.items():
            values = raw_space.get("values") if isinstance(raw_space, dict) else raw_space
            if not isinstance(values, (list, tuple)):
                raise TypeError(f"semantic attribute {name!r} values must be a sequence")
            attributes[str(name).casefold()] = tuple(str(item) for item in values)
        return cls(
            enabled=bool(value.get("enabled", True)),
            legacy_text_fallback=bool(value.get("legacy_text_fallback", False)),
            preserve_reasoning_text=bool(value.get("preserve_reasoning_text", True)),
            single_decision_suffix=str(
                value.get("single_decision_suffix", DEFAULT_SINGLE_DECISION_SUFFIX)
            ),
            multi_verify_template=str(
                value.get("multi_verify_template", DEFAULT_MULTI_VERIFY_TEMPLATE)
            ),
            multi_select_threshold=float(value.get("multi_select_threshold", 0.0)),
            uncertainty_epsilon=float(value.get("uncertainty_epsilon", 0.0)),
            attributes=attributes,
        )

    def attribute_values(self, attribute: str) -> tuple[str, ...] | None:
        return (self.attributes or {}).get(attribute.casefold())


@dataclass(frozen=True)
class SemanticDecision:
    values: tuple[str, ...]
    provenance: dict[str, Any]


def _safe_metadata(value: Any) -> Any:
    """Keep trace/store provenance JSON-safe and exclude model/cache objects."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {
            str(key): safe
            for key, item in value.items()
            if (safe := _safe_metadata(item)) is not None
        }
    if isinstance(value, (list, tuple)):
        return [safe for item in value if (safe := _safe_metadata(item)) is not None]
    return None


class SemanticDecisionLayer:
    """Convert cached finite scores to canonical semantic values."""

    def __init__(
        self,
        provider: SemanticVLMProvider,
        config: SemanticDecisionConfig | None = None,
    ) -> None:
        self.provider = provider
        self.config = config or SemanticDecisionConfig()

    def _decide(
        self,
        model_input: ModelInput,
        candidates: tuple[str, ...],
        *,
        mode: str,
        purpose: str,
        reasoning_instruction: str,
        semantic_method: str,
    ) -> SemanticDecision:
        scorer = getattr(self.provider, "reason_and_decide", None)
        if not callable(scorer):
            raise CachedChoiceUnavailableError(
                "semantic provider does not implement cached finite decisions"
            )
        result = cast(
            FiniteDecisionResult,
            scorer(
                FiniteDecisionRequest(
                    model_input=model_input,
                    decision_mode=mode,
                    candidate_ids=candidates,
                    candidate_texts=candidates,
                    single_decision_suffix=self.config.single_decision_suffix,
                    multi_verify_template=self.config.multi_verify_template,
                    select_threshold=self.config.multi_select_threshold,
                    purpose=purpose,
                    reasoning_instruction=reasoning_instruction,
                )
            ),
        )
        if not result.cache_reused:
            raise SemanticDecisionUnresolvedError(f"{purpose} did not reuse the reasoning KV cache")
        if any(value not in candidates for value in result.selected_ids):
            raise SemanticDecisionUnresolvedError(f"{purpose} selected a non-canonical value")
        if mode in {"SINGLE", "BINARY"} and len(result.selected_ids) != 1:
            raise SemanticDecisionUnresolvedError(f"{purpose} did not select exactly one value")
        safe_metadata = _safe_metadata(result.metadata)
        provenance: dict[str, Any] = {
            "provider": result.provider,
            "model_id": result.model_id,
            "method": semantic_method,
            "provider_method": result.method,
            "canonical": True,
            "scores": dict(result.scores),
            "selected": list(result.selected_ids),
            "cache_reused": result.cache_reused,
            "latency_ms": dict(result.latency_ms),
            "semantic_decision_total_ms": result.latency_ms.get("total_ms"),
            "execution_mode": "intermediate_semantic",
            "semantic_method": semantic_method,
            "final_choice_fusion": False,
            "fusion_reason": "not_final_source",
        }
        if isinstance(safe_metadata, dict):
            provenance["decision_metadata"] = safe_metadata
        if self.config.preserve_reasoning_text:
            provenance["reasoning_text"] = result.reasoning_text
        return SemanticDecision(tuple(result.selected_ids), provenance)

    def choose_one(
        self,
        model_input: ModelInput,
        candidates: tuple[str, ...],
        *,
        purpose: str,
        reasoning_instruction: str,
    ) -> SemanticDecision:
        return self._decide(
            model_input,
            candidates,
            mode="SINGLE",
            purpose=purpose,
            reasoning_instruction=reasoning_instruction,
            semantic_method="kv_cached_categorical",
        )

    def choose_many(
        self,
        model_input: ModelInput,
        candidates: tuple[str, ...],
        *,
        purpose: str,
        reasoning_instruction: str,
    ) -> SemanticDecision:
        return self._decide(
            model_input,
            candidates,
            mode="MULTI",
            purpose=purpose,
            reasoning_instruction=reasoning_instruction,
            semantic_method="kv_cached_multi",
        )

    def verify(
        self,
        model_input: ModelInput,
        *,
        purpose: str,
        reasoning_instruction: str,
    ) -> tuple[bool, dict[str, Any]]:
        decision = self._decide(
            model_input,
            ("YES", "NO"),
            mode="BINARY",
            purpose=purpose,
            reasoning_instruction=reasoning_instruction,
            semantic_method="kv_cached_binary",
        )
        delta = float(decision.provenance["scores"]["YES"]) - float(
            decision.provenance["scores"]["NO"]
        )
        if abs(delta) <= self.config.uncertainty_epsilon:
            raise SemanticDecisionUnresolvedError(
                f"{purpose} score margin {delta} is within uncertainty epsilon"
            )
        return delta > 0.0, {**decision.provenance, "score_margin": delta}
