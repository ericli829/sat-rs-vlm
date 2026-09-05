"""Auxiliary evidence-sufficiency assessment using the existing semantic VLM cache path."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

from .input_composer import InputComposer
from .providers import (
    EvidenceSufficiencyRequest,
    EvidenceSufficiencyResult,
    EvidenceSufficiencyStatus,
    FiniteDecisionRequest,
    SemanticVLMProvider,
)
from .runtime_types import (
    Entity,
    EntitySet,
    Evidence,
    EvidenceSet,
    ImageRef,
    Region,
    RegionSet,
    RouteContext,
    RuntimeObject,
    runtime_summary,
)


@dataclass(frozen=True)
class AnswerabilityConfig:
    enabled: bool = True
    return_error_result: bool = True
    prompt_version: str = "answerability-v1"
    single_decision_suffix: str = "\n\nEvidence sufficiency status:"
    multi_verify_template: str = (
        "\n\nCandidate status {choice_id}: {option_text}\nIs this the correct status? YES or NO:"
    )

    def __post_init__(self) -> None:
        if not self.prompt_version.strip():
            raise ValueError("answerability prompt_version must not be empty")
        if not self.single_decision_suffix:
            raise ValueError("answerability decision suffix must not be empty")
        if (
            "{choice_id}" not in self.multi_verify_template
            or "{option_text}" not in self.multi_verify_template
        ):
            raise ValueError("answerability verify template requires choice_id and option_text")

    @classmethod
    def from_mapping(cls, value: Any) -> AnswerabilityConfig:
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise TypeError("answerability config must be a mapping")
        return cls(
            enabled=bool(value.get("enabled", True)),
            return_error_result=bool(value.get("return_error_result", True)),
            prompt_version=str(value.get("prompt_version", "answerability-v1")),
            single_decision_suffix=str(
                value.get("single_decision_suffix", "\n\nEvidence sufficiency status:")
            ),
            multi_verify_template=str(
                value.get("multi_verify_template", cls.multi_verify_template)
            ),
        )


class EvidenceSufficiencyExecutor:
    """Assess evidence without entering the graph or changing exhaustive-count control flow."""

    provider_name = "answerability"
    _STATUSES = (
        EvidenceSufficiencyStatus.SUFFICIENT,
        EvidenceSufficiencyStatus.NEED_MORE_EVIDENCE,
        EvidenceSufficiencyStatus.UNRESOLVED,
    )

    def __init__(
        self,
        provider: SemanticVLMProvider,
        composer: InputComposer,
        config: AnswerabilityConfig | None = None,
    ) -> None:
        self.provider = provider
        self.composer = composer
        self.config = config or AnswerabilityConfig()

    def _fingerprint(self, request: EvidenceSufficiencyRequest) -> str:
        def source_identity(source: RuntimeObject) -> dict[str, object]:
            summary = runtime_summary(source)
            image_key: str | None
            if isinstance(source, ImageRef):
                image_key = source.uri_or_key
            elif isinstance(source, Region):
                image_key = source.image.uri_or_key
            elif isinstance(source, RegionSet):
                image_key = source.regions[0].image.uri_or_key if source.regions else None
            elif isinstance(source, Entity):
                image_key = source.region.image.uri_or_key
            elif isinstance(source, EntitySet):
                image_key = source.entities[0].region.image.uri_or_key if source.entities else None
            elif isinstance(source, Evidence):
                return source_identity(source.value)
            elif isinstance(source, EvidenceSet):
                return {
                    "summary": summary,
                    "evidence": [source_identity(item.value) for item in source.evidence],
                }
            elif isinstance(source, RouteContext):
                image_key = source.image.uri_or_key
            else:
                image_key = None
            return {"summary": summary, "image_key": image_key}

        payload = {
            "sample_id": request.sample_id,
            "question": request.question,
            "task_hint": request.task_hint,
            "evidence_version": request.evidence_version,
            "prompt_version": self.config.prompt_version,
            "sources": [source_identity(source) for source in request.sources],
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _confidence(scores: dict[str, float], selected: str) -> float | None:
        if selected not in scores:
            return None
        competitors = [score for key, score in scores.items() if key != selected]
        if not competitors:
            return 1.0
        margin = scores[selected] - max(competitors)
        return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, margin))))

    def assess(self, request: EvidenceSufficiencyRequest) -> EvidenceSufficiencyResult:
        fingerprint = self._fingerprint(request)
        if not self.config.enabled:
            return EvidenceSufficiencyResult(
                EvidenceSufficiencyStatus.UNRESOLVED,
                reason_code="answerability_disabled",
                provider=self.provider_name,
                method="disabled",
                metadata={"evidence_fingerprint": fingerprint},
            )
        named = {
            f"evidence_{index}": source for index, source in enumerate(request.sources, start=1)
        }
        task_hint = f"\nTask hint: {request.task_hint}" if request.task_hint else ""
        question = (
            "Assess only whether the supplied evidence is sufficient to answer the task reliably. "
            "Do not answer the task and do not emit reasoning. Choose SUFFICIENT when the evidence "
            "supports a reliable answer, NEED_MORE_EVIDENCE when another crop/view is required, "
            "or UNRESOLVED when sufficiency itself cannot be determined.\n"
            f"Task question: {request.question}{task_hint}\n"
            f"Prompt version: {self.config.prompt_version}"
        )
        model_input = self.composer.compose_named(named, question=question)
        try:
            decided = self.provider.reason_and_decide(
                FiniteDecisionRequest(
                    model_input=model_input,
                    decision_mode="SINGLE",
                    candidate_ids=tuple(status.value for status in self._STATUSES),
                    candidate_texts=tuple(status.value for status in self._STATUSES),
                    single_decision_suffix=self.config.single_decision_suffix,
                    multi_verify_template=self.config.multi_verify_template,
                    purpose="answerability",
                    reasoning_instruction=(
                        "Judge evidence sufficiency only. Do not solve the original task. "
                        "A separate "
                        "constrained continuation selects the structured status."
                    ),
                )
            )
        except Exception as exc:
            if not self.config.return_error_result:
                raise
            return EvidenceSufficiencyResult(
                EvidenceSufficiencyStatus.ERROR,
                reason_code=type(exc).__name__,
                provider=str(getattr(self.provider, "provider_name", "unknown")),
                method="provider_error",
                metadata={
                    "evidence_fingerprint": fingerprint,
                    "cache_scope": "request",
                    "error_type": type(exc).__name__,
                },
            )
        selected = decided.selected_ids[0]
        safe_metadata = {
            key: value
            for key, value in decided.metadata.items()
            if key not in {"reasoning_text", "session_id"}
        }
        return EvidenceSufficiencyResult(
            EvidenceSufficiencyStatus(selected),
            self._confidence(decided.scores, selected),
            reason_code="finite_cached_decision",
            provider=decided.provider,
            model_id=decided.model_id,
            method=decided.method,
            cache_reused=decided.cache_reused,
            latency_ms=dict(decided.latency_ms),
            metadata={
                **safe_metadata,
                "evidence_fingerprint": fingerprint,
                "evidence_version": request.evidence_version,
                "prompt_version": self.config.prompt_version,
                "cache_scope": "request",
                "reasoning_exposed": False,
            },
        )
