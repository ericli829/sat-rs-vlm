"""Deterministic and KV-cached final benchmark choice resolution."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import cast

from .choice_config import ChoiceSystemConfig
from .input_composer import InputComposer
from .providers import (
    CachedChoiceUnavailableError,
    ChoiceScoringRequest,
    ModelInput,
    SemanticVLMProvider,
    VLMRequest,
)
from .runtime_types import (
    Boolean,
    ChoiceResult,
    ChoiceScoreResult,
    EntitySet,
    Label,
    LabelSet,
    RuntimeObject,
    ScalarFloat,
    ScalarInt,
)
from .schema import AnswerType
from .spatial_choice import SpatialPositionChoiceResolver


@dataclass(frozen=True)
class ChoiceRequest:
    sources: tuple[RuntimeObject, ...]
    question: str | None
    options: tuple[str, ...]
    answer_type: AnswerType = AnswerType.CHOICE_SINGLE

    def __post_init__(self) -> None:
        normalized = AnswerType(self.answer_type)
        if normalized not in {AnswerType.CHOICE_SINGLE, AnswerType.CHOICE_MULTI}:
            raise ValueError("ChoiceRequest answer_type must be CHOICE_SINGLE or CHOICE_MULTI")
        if not self.options:
            raise ValueError("ChoiceRequest requires original dataset options")
        object.__setattr__(self, "answer_type", normalized)


class ChoiceResolver:
    """Resolve choices without parsing free reasoning text on the production path."""

    _OPTION_PREFIX = re.compile(r"^\s*[\(\[]?[A-Z][\)\].:]?\s*")

    def __init__(
        self,
        provider: SemanticVLMProvider,
        composer: InputComposer,
        config: ChoiceSystemConfig | None = None,
    ) -> None:
        self.provider = provider
        self.composer = composer
        self.config = config or ChoiceSystemConfig()
        self.last_model_input: ModelInput | None = None
        self.last_score_result: ChoiceScoreResult | None = None
        self.spatial_resolver = SpatialPositionChoiceResolver()

    @staticmethod
    def _choice_ids(options: tuple[str, ...]) -> tuple[str, ...]:
        if len(options) > 26:
            raise ValueError("final benchmark choice currently supports at most 26 options")
        return tuple(chr(ord("A") + index) for index in range(len(options)))

    @classmethod
    def _option_value(cls, option: str) -> str:
        return cls._OPTION_PREFIX.sub("", option).strip().casefold()

    @staticmethod
    def _source_values(source: RuntimeObject) -> tuple[str, ...] | None:
        if isinstance(source, ScalarInt):
            return (str(source.value),)
        if isinstance(source, ScalarFloat):
            values = [str(source.value)]
            if source.value.is_integer():
                values.append(str(int(source.value)))
            return tuple(values)
        if isinstance(source, Boolean):
            return ("true", "yes", "1") if source.value else ("false", "no", "0")
        if isinstance(source, Label):
            return (source.value.strip().casefold(),)
        if isinstance(source, LabelSet):
            return tuple(value.strip().casefold() for value in source.values)
        return None

    _NUMBER_WORDS = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "sixteen": 16,
    }

    @classmethod
    def _option_number(cls, option: str) -> int | None:
        text = cls._option_value(option).rstrip(".")
        numeric = re.fullmatch(r"(\d+)", text)
        if numeric:
            return int(numeric.group(1))
        return cls._NUMBER_WORDS.get(text)

    @classmethod
    def _closest_numeric_score(cls, request: ChoiceRequest) -> ChoiceScoreResult | None:
        """Map a single ScalarInt count to the closest numeric option.

        Non-numeric options (typically the 'does not feature the count'
        fallback) are excluded: when the graph resolved a numeric count, the
        answer must be a number.  Ties stay ambiguous and fall through to the
        semantic provider.
        """
        numeric = [source for source in request.sources if isinstance(source, ScalarInt)]
        if (
            request.answer_type is not AnswerType.CHOICE_SINGLE
            or len(numeric) != 1
            or len(numeric) != len(request.sources)
        ):
            return None
        target = int(numeric[0].value)
        choice_ids = cls._choice_ids(request.options)
        scored: list[tuple[int, str]] = []
        for choice_id, option in zip(choice_ids, request.options, strict=True):
            option_number = cls._option_number(option)
            if option_number is not None:
                scored.append((abs(target - option_number), choice_id))
        if not scored:
            return None
        best = min(distance for distance, _ in scored)
        winners = [choice_id for distance, choice_id in scored if distance == best]
        if len(winners) != 1:
            return None
        selected = winners[0]
        return ChoiceScoreResult(
            selected_ids=(selected,),
            scores={
                choice_id: (1.0 if choice_id == selected else 0.0) for choice_id in choice_ids
            },
            answer_type=request.answer_type.value,
            reasoning_text=(
                f"Resolved numeric evidence {target}; "
                "selected the closest numeric option."
            ),
            provider="structured_deterministic",
            model_id="none",
            method="structured_closest_numeric_mapping",
            cache_reused=False,
            latency_ms={"total_ms": 0.0},
            metadata={"model_called": False, "numeric_evidence": target},
        )

    def _structured_score(self, request: ChoiceRequest) -> ChoiceScoreResult | None:
        source_values: list[str] = []
        for source in request.sources:
            values = self._source_values(source)
            if values is None:
                return None
            source_values.extend(values)
        normalized_values = set(source_values)
        choice_ids = self._choice_ids(request.options)
        matches = tuple(
            choice_id
            for choice_id, option in zip(choice_ids, request.options, strict=True)
            if self._option_value(option) in normalized_values
        )
        if not matches and (answer_id := self._closest_numeric_score(request)):
            return answer_id
        if request.answer_type is AnswerType.CHOICE_SINGLE and len(matches) != 1:
            return None
        if request.answer_type is AnswerType.CHOICE_MULTI and not matches:
            return None
        return ChoiceScoreResult(
            selected_ids=matches,
            scores={choice_id: (1.0 if choice_id in matches else 0.0) for choice_id in choice_ids},
            answer_type=request.answer_type.value,
            reasoning_text=None,
            provider="structured_deterministic",
            model_id="none",
            method="structured_exact_option_mapping",
            cache_reused=False,
            latency_ms={"total_ms": 0.0},
            metadata={"model_called": False},
        )

    def _precomputed_score(self, request: ChoiceRequest) -> ChoiceScoreResult | None:
        scores = [source for source in request.sources if isinstance(source, ChoiceScoreResult)]
        if not scores:
            return None
        if len(scores) != 1 or len(request.sources) != 1:
            raise ValueError("final choice accepts exactly one precomputed ChoiceScoreResult")
        score = scores[0]
        if score.answer_type != request.answer_type.value:
            raise ValueError("precomputed choice answer_type differs from final answer_type")
        legal = self._choice_ids(request.options)
        if any(choice_id not in legal for choice_id in score.scores):
            raise ValueError("precomputed choice contains ids outside original options")
        canonical = tuple(choice_id for choice_id in legal if choice_id in score.selected_ids)
        return replace(score, selected_ids=canonical)

    @staticmethod
    def _reasoning_question(request: ChoiceRequest) -> str:
        return (request.question or "Match the resolved evidence to the original options.").strip()

    @staticmethod
    def _legacy_choice_id(text: str, options: tuple[str, ...]) -> str:
        """Explicit compatibility-only parser; never enabled by default."""

        valid = tuple(chr(ord("A") + index) for index in range(len(options)))
        match = re.fullmatch(r"\s*([A-Z])\s*", text.upper())
        if match and match.group(1) in valid:
            return match.group(1)
        normalized = text.strip().casefold()
        for choice_id, option in zip(valid, options, strict=True):
            option_value = ChoiceResolver._OPTION_PREFIX.sub("", option).strip().casefold()
            if normalized == option_value:
                return choice_id
        raise ValueError(f"legacy choice provider did not return one exact legal id: {text!r}")

    def _legacy_score(self, request: ChoiceRequest) -> ChoiceScoreResult:
        if request.answer_type is not AnswerType.CHOICE_SINGLE:
            raise RuntimeError("legacy text fallback does not support CHOICE_MULTI")
        if self.last_model_input is None:
            raise RuntimeError("legacy choice fallback requires a composed model input")
        result = self.provider.infer(VLMRequest(self.last_model_input, output_contract="choice"))
        selected = self._legacy_choice_id(result.text, request.options)
        choice_ids = self._choice_ids(request.options)
        return ChoiceScoreResult(
            selected_ids=(selected,),
            scores={choice_id: (1.0 if choice_id == selected else 0.0) for choice_id in choice_ids},
            answer_type=request.answer_type.value,
            reasoning_text=result.text,
            provider=result.provider,
            model_id=str(result.metadata.get("model_id", "legacy")),
            method="legacy_exact_text_parser",
            cache_reused=False,
            latency_ms={},
            metadata={"legacy_fallback": True, **result.metadata},
        )

    def _cached_score(self, request: ChoiceRequest) -> ChoiceScoreResult:
        if self.last_model_input is None:
            raise RuntimeError("cached choice requires a composed model input")
        scorer = getattr(self.provider, "reason_and_choose", None)
        if scorer is None:
            if self.config.legacy_regex_fallback:
                return self._legacy_score(request)
            raise CachedChoiceUnavailableError(
                "choice provider does not implement cached choice scoring"
            )
        return cast(
            ChoiceScoreResult,
            scorer(
                ChoiceScoringRequest(
                    model_input=self.last_model_input,
                    answer_type=request.answer_type.value,
                    choice_ids=self._choice_ids(request.options),
                    option_texts=request.options,
                    single_choice_suffix=self.config.single_choice_suffix,
                    multi_verify_template=self.config.multi_verify_template,
                    multi_select_threshold=self.config.multi_select_threshold,
                    purpose="final_choice",
                )
            ),
        )

    @staticmethod
    def _needs_semantic_fallback(source: RuntimeObject) -> bool:
        return isinstance(source, EntitySet) and bool(
            source.provenance.get("fallback_required")
            or source.provenance.get("resolution_status") == "UNRESOLVED"
            or source.provenance.get("resolution_status") == "SEMANTIC_FALLBACK_RESOLVED"
        )

    def _semantic_fallback_score(self, request: ChoiceRequest) -> ChoiceScoreResult | None:
        if len(request.sources) != 1 or not self._needs_semantic_fallback(request.sources[0]):
            return None
        model_input = self.composer.compose_named(
            {"candidates": request.sources[0]},
            question=(
                "Use only the supplied bounded visual candidates to resolve the remaining "
                "referring expression and answer the residual question. Do not search outside "
                "the supplied candidates and do not infer a candidate that is not shown.\n\n"
                f"Original question: {request.question or ''}\n"
                "The candidate resolver could not establish a reliable singleton. "
                "Choose the answer option supported by the supplied candidates."
            ),
            options=request.options,
        )
        self.last_model_input = model_input
        scorer = getattr(self.provider, "reason_and_choose", None)
        if not callable(scorer):
            raise CachedChoiceUnavailableError(
                "semantic fallback provider does not implement cached choice scoring"
            )
        scored = cast(
            ChoiceScoreResult,
            scorer(
                ChoiceScoringRequest(
                    model_input=model_input,
                    answer_type=request.answer_type.value,
                    choice_ids=self._choice_ids(request.options),
                    option_texts=request.options,
                    single_choice_suffix=self.config.single_choice_suffix,
                    multi_verify_template=self.config.multi_verify_template,
                    multi_select_threshold=self.config.multi_select_threshold,
                    purpose="semantic_candidate_fallback",
                )
            ),
        )
        return replace(
            scored,
            metadata={
                **scored.metadata,
                "fallback_triggered": True,
                "fallback_reason": "UNRESOLVED_REFERENT",
                "fallback_provider": scored.provider,
                "fallback_scope": "bounded_candidate_set",
                "final_resolution_status": "SEMANTIC_FALLBACK_RESOLVED",
                "fallback_input_metadata": dict(model_input.metadata),
            },
        )

    def _to_result(self, score: ChoiceScoreResult) -> ChoiceResult:
        empty_status = None
        if score.answer_type == "CHOICE_MULTI" and not score.selected_ids:
            empty_status = self.config.multi_empty_policy
        provenance = {
            "provider": score.provider,
            "model_id": score.model_id,
            "answer_type": score.answer_type,
            "scores": dict(score.scores),
            "method": score.method,
            "cache_reused": score.cache_reused,
            "latency_ms": dict(score.latency_ms),
            "empty_multi_status": empty_status,
            **score.metadata,
        }
        return ChoiceResult(
            selected_ids=score.selected_ids,
            answer_type=score.answer_type,
            raw_response=score.reasoning_text or "",
            confidence=None,
            provenance=provenance,
        )

    def resolve(self, request: ChoiceRequest) -> ChoiceResult:
        fallback_score = self._semantic_fallback_score(request)
        if fallback_score is not None:
            self.last_score_result = fallback_score
            return self._to_result(fallback_score)
        spatial_score = self.spatial_resolver.resolve(
            request.sources,
            request.options,
            question=request.question,
        )
        if spatial_score is not None:
            self.last_model_input = ModelInput(
                visual_inputs=(),
                structured_context="",
                question=self._reasoning_question(request),
                options=request.options,
                metadata=dict(spatial_score.metadata),
            )
            self.last_score_result = spatial_score
            return self._to_result(spatial_score)
        score = self._structured_score(request) or self._precomputed_score(request)
        if score is None:
            self.last_model_input = self.composer.compose(
                list(request.sources),
                question=self._reasoning_question(request),
                options=request.options,
            )
            score = self._cached_score(request)
        else:
            self.last_model_input = self.composer.compose(
                list(request.sources),
                question=self._reasoning_question(request),
                options=request.options,
            )
        self.last_score_result = score
        return self._to_result(score)
