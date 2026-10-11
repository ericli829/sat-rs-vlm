"""Resolve typed evidence once, then validate and serialize the wire answer."""

from __future__ import annotations

import json
import math
from dataclasses import replace

from sat_rs_vlm.taskgraph.choice import ChoiceRequest, ChoiceResolver
from sat_rs_vlm.taskgraph.input_composer import InputComposer
from sat_rs_vlm.taskgraph.providers import SemanticVLMProvider, VLMRequest
from sat_rs_vlm.taskgraph.runtime_types import (
    Answer,
    Boolean,
    CandidateIntegrityError,
    Entity,
    ImageRef,
    Label,
    LabelSet,
    Region,
    ScalarInt,
    SelectResult,
    SelectResultConsumptionError,
    unique_bbox_evidence,
    unwrap_select_result,
)
from sat_rs_vlm.taskgraph.semantic_decision import SemanticDecisionLayer

from .coordinates import declared_space, to_pixel_region
from .errors import AnswerContractViolation, BBoxValidationError
from .models import AnswerContract, PredictRequest


def _sources(evidence: object) -> tuple:
    values = evidence if isinstance(evidence, tuple) else (evidence,)
    return tuple(
        unwrap_select_result(x, allow_empty=False, consumer="competition_final")
        if isinstance(x, SelectResult)
        else x
        for x in values
    )


def _text(source: object) -> str | None:
    if isinstance(source, (Label, ScalarInt)):
        return str(source.value)
    if isinstance(source, Boolean):
        return "Yes" if source.value else "No"
    if isinstance(source, LabelSet):
        return ", ".join(source.values)
    if isinstance(source, Answer):
        return source.text.strip()
    return None


def validate_bbox(value: object, contract: AnswerContract) -> list[float]:
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise BBoxValidationError()
    if any(
        isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x)
        for x in value
    ):
        raise BBoxValidationError()
    x1, y1, x2, y2 = value
    if (
        contract.image_width is None
        or contract.image_height is None
        or not (0 <= x1 < x2 <= contract.image_width and 0 <= y1 < y2 <= contract.image_height)
    ):
        raise BBoxValidationError()
    return [float(x) for x in value]


class AnswerResolver:
    def __init__(
        self,
        provider: SemanticVLMProvider,
        composer: InputComposer,
        choice_resolver: ChoiceResolver,
        decision_layer: SemanticDecisionLayer,
        short_text_instruction: str = "Return only the shortest direct answer. No explanation.",
        bbox_model_coordinate_format: str | None = None,
    ):
        self.provider = provider
        self.composer = composer
        self.choice_resolver = choice_resolver
        self.decision_layer = decision_layer
        self.short_text_instruction = short_text_instruction
        if bbox_model_coordinate_format is not None:
            declared_space({"coordinate_format": bbox_model_coordinate_format})
        self.bbox_model_coordinate_format = bbox_model_coordinate_format
        self.last_bbox_conversion: dict | None = None

    def resolve(
        self,
        evidence: object,
        question: str,
        contract: AnswerContract,
        original_images: tuple[ImageRef, ...],
    ) -> object:
        self.last_bbox_conversion = None
        # BBOX must inspect every wrapper's state before materializing its one
        # target; the other contracts retain their existing SELECT policy.
        sources = (
            (evidence if isinstance(evidence, tuple) else (evidence,))
            if contract.kind == "bbox"
            else _sources(evidence)
        )
        if not sources:
            raise AnswerContractViolation()
        if contract.kind == "single_choice":
            result = self.choice_resolver.resolve(
                ChoiceRequest(
                    sources,
                    question,
                    contract.option_texts,
                    choice_ids=contract.values,
                    strip_option_prefix=False,
                )
            )
            if result.choice_id not in contract.values:
                raise AnswerContractViolation()
            return result.choice_id
        if contract.kind == "enum":
            if len(sources) == 1 and (value := _text(sources[0])) is not None:
                matches = [
                    key
                    for key in contract.values
                    if (
                        key.casefold() == value.casefold()
                        if contract.case_insensitive
                        else key == value
                    )
                ]
                if len(matches) == 1:
                    return matches[0]
            model_input = replace(self._compose(sources, question), options=contract.values)
            decision = self.decision_layer.choose_one(
                model_input,
                contract.values,
                purpose="competition_enum",
                reasoning_instruction=(
                    "Resolve the evidence and select exactly one permitted value."
                ),
            )
            return decision.values[0]
        if contract.kind == "integer":
            if len(sources) != 1 or not isinstance(sources[0], ScalarInt):
                raise AnswerContractViolation("Integer requires authoritative ScalarInt evidence")
            value = sources[0].value
            if type(value) is not int or (
                contract.minimum is not None and value < contract.minimum
            ):
                raise AnswerContractViolation()
            return value
        if contract.kind == "bbox":
            if len(sources) != 1:
                raise BBoxValidationError()
            source = sources[0]
            if not isinstance(source, Answer):
                try:
                    source = unique_bbox_evidence(source)
                except (CandidateIntegrityError, SelectResultConsumptionError) as exc:
                    raise BBoxValidationError("Bbox requires one resolved target") from exc
            if getattr(source, "provenance", {}).get("resolution_status") in {
                "UNRESOLVED",
                "AMBIGUOUS",
                "ERROR",
            }:
                raise BBoxValidationError("Unresolved bbox evidence")
            provenance = dict(getattr(source, "provenance", {}))
            if isinstance(source, Entity):
                provenance.update(source.provenance)
                source = source.region
            if isinstance(source, Answer):
                try:
                    payload = json.loads(source.text)
                except (TypeError, ValueError) as exc:
                    raise BBoxValidationError("Bbox output must be strict JSON") from exc
                if isinstance(payload, dict):
                    if set(payload) - {"label", "bbox"} or "bbox" not in payload:
                        raise BBoxValidationError("Expected one model bbox")
                    box = payload["bbox"]
                else:
                    box = payload
                index = provenance.get("image_index", 0 if len(original_images) == 1 else None)
                if type(index) is not int or not 0 <= index < len(original_images):
                    raise BBoxValidationError("Model bbox image is ambiguous")
                image = original_images[index]
                space = declared_space(provenance, default=self.bbox_model_coordinate_format)
            elif isinstance(source, Region):
                provenance.update(source.provenance)
                matching = [
                    image
                    for image in original_images
                    if image.path.resolve() == source.image.path.resolve()
                ]
                if not matching:
                    raise BBoxValidationError("Bbox belongs to a different image")
                image = matching[0]
                if (source.image.width is not None and source.image.width != image.width) or (
                    source.image.height is not None and source.image.height != image.height
                ):
                    raise BBoxValidationError("Bbox source dimensions disagree with the original")
                box = source.bbox_xyxy_global
                space = declared_space(provenance, default="pixel_xyxy")
            else:
                raise BBoxValidationError("Bbox requires global Region evidence")
            if (image.width, image.height) != (
                contract.image_width,
                contract.image_height,
            ):
                raise BBoxValidationError("Bbox belongs to a different image")
            conversion = to_pixel_region(box, image, coordinate_format=space, provenance=provenance)
            self.last_bbox_conversion = conversion.metadata
            return validate_bbox(conversion.region.bbox_xyxy_global, contract)
        if contract.kind == "short_text":
            if len(sources) == 1 and (value := _text(sources[0])) is not None:
                return value
            model_input = self._compose(sources, question + "\n" + self.short_text_instruction)
            return self.provider.infer(VLMRequest(model_input, "short_text")).text.strip()
        raise AnswerContractViolation()

    def _compose(self, sources: tuple, question: str):
        if len(sources) == 2 and all(isinstance(x, ImageRef) for x in sources):
            return self.composer.compose_named(
                {"before": sources[0], "after": sources[1]}, question=question
            )
        return self.composer.compose(list(sources), question=question)


def serialize_response(request: PredictRequest, contract: AnswerContract, value: object) -> dict:
    if contract.kind == "bbox":
        answer = validate_bbox(value, contract)
    elif contract.kind == "integer":
        if type(value) is not int or (contract.minimum is not None and value < contract.minimum):
            raise AnswerContractViolation()
        answer = str(value)
    else:
        if not isinstance(value, str):
            raise AnswerContractViolation()
        answer = value
        if contract.kind in {"enum", "single_choice"} and answer not in contract.values:
            raise AnswerContractViolation()
        if contract.kind == "short_text" and len(answer) < contract.min_length:
            raise AnswerContractViolation()
        if contract.kind == "short_text":
            lowered = answer.casefold().strip()
            if any(
                tag in lowered for tag in ("<think", "</think>", "<analysis", "</analysis>")
            ) or (lowered.startswith(("analysis:", "reasoning:", "chain of thought:"))):
                # Do not expose a provider's explicitly marked reasoning as a short answer.
                # Reject it instead of regex-extracting or inventing a final answer.
                raise AnswerContractViolation()
    if isinstance(answer, str) and (not answer.strip() or len(answer) > 4096):
        raise AnswerContractViolation()
    response = {
        "protocol_version": "2.0",
        "request_id": request.request_id,
        "item_id": request.item_id,
        "answer": answer,
    }
    if len(json.dumps(response, ensure_ascii=False, allow_nan=False).encode("utf-8")) > 1024 * 1024:
        raise AnswerContractViolation()
    return response
