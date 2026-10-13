"""Protocol routing with ordered temporal and original-resolution decisions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from sat_rs_vlm.taskgraph.routing import ExecutionMode
from sat_rs_vlm.taskgraph.runtime import RuntimeRequest
from sat_rs_vlm.taskgraph.schema import AnswerType, QuestionType

from .image_package import ResolvedImage
from .models import AnswerContract, PredictRequest

_COMPLEX = re.compile(
    r"\b(?:left|right|above|below|between|near|within|inside|outside|largest|smallest|"
    r"closest|nearest|farthest|before|after|changed|change|route|path|top|bottom|upper|"
    r"lower|topmost|bottommost|uppermost|lowermost|center|centre|middle)\b|"
    r"左|右|上|下|顶部|底部|上方|下方|左侧|右侧|中心|中间|附近|之间|最大|最小|最近|最远|变化|路径",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CompetitionRoutingConfig:
    # The production 2B semantic processor is explicitly capped at 409,600
    # pixels. Routing compares original dimensions against this separate
    # business threshold before any processor resizing occurs.
    direct_vlm_max_pixels: int = 409_600

    def __post_init__(self) -> None:
        if self.direct_vlm_max_pixels < 1:
            raise ValueError("direct_vlm_max_pixels must be positive")


def _taskgraph_answer_type(contract: AnswerContract) -> AnswerType:
    mapping = {
        "single_choice": AnswerType.CHOICE_SINGLE,
        "enum": AnswerType.LABEL,
        "short_text": AnswerType.TEXT,
        "integer": AnswerType.INTEGER,
        "bbox": AnswerType.BBOX,
    }
    try:
        return mapping[contract.kind]
    except KeyError as exc:
        raise ValueError(f"unsupported competition answer contract: {contract.kind}") from exc


def _contract_payload(contract: AnswerContract) -> dict[str, Any]:
    return {
        "kind": contract.kind,
        "values": list(contract.values),
        "case_insensitive": contract.case_insensitive,
        "minimum": contract.minimum,
        "min_length": contract.min_length,
        "image_width": contract.image_width,
        "image_height": contract.image_height,
    }


class CompetitionRequestAdapter:
    def __init__(self, config: CompetitionRoutingConfig | None = None) -> None:
        self.config = config or CompetitionRoutingConfig()

    def adapt(
        self, request: PredictRequest, images: tuple[ResolvedImage, ...], contract: AnswerContract
    ) -> RuntimeRequest:
        question = request.question
        relational = bool(_COMPLEX.search(question))
        pixels = images[0].width * images[0].height if len(images) == 1 else None
        memory_route_eligible = False

        # Temporal identity and order outrank all single-image rules.
        if len(images) == 2:
            mode = ExecutionMode.DIRECT_VLM
            task = "temporal_change"
            reason = "dual_image_temporal_priority"
            question += "\nThe first image is BEFORE; the second image is AFTER."
        elif pixels is not None and pixels <= self.config.direct_vlm_max_pixels:
            mode = ExecutionMode.DIRECT_VLM
            task = "count" if contract.kind == "integer" else (
                "localization" if contract.kind == "bbox" else "semantic"
            )
            reason = "single_image_within_direct_vlm_resolution"
        else:
            mode = ExecutionMode.TASKGRAPH_UHR
            task = "count" if contract.kind == "integer" else (
                "localization" if contract.kind == "bbox" else "semantic"
            )
            memory_route_eligible = True
            if contract.kind == "bbox":
                reason = "high_resolution_bbox_taskgraph"
                question += (
                    "\nResolve one target and provide final evidence in original global pixel "
                    "coordinates."
                )
            elif relational:
                reason = "high_resolution_complex_taskgraph"
            else:
                reason = "high_resolution_default_taskgraph"
            if contract.kind == "integer":
                question += "\nThe final evidence must be an authoritative integer count."

        return RuntimeRequest(
            sample_id=request.item_id,
            dataset="competition_protocol_v2",
            task_category=task,
            question=question,
            image_paths=tuple(str(image.path) for image in images),
            options=contract.option_texts,
            choice_ids=contract.values if contract.kind == "single_choice" else (),
            question_type=(
                QuestionType.MULTIPLE_CHOICE_SINGLE
                if contract.kind == "single_choice"
                else QuestionType.INTEGER
                if contract.kind == "integer"
                else QuestionType.FREE_FORM
            ),
            execution_mode=mode,
            defer_final_answer=True,
            memory_question=request.question,
            verified_image_sha256s=tuple(image.descriptor.sha256 for image in images),
            allow_hash_memory=memory_route_eligible,
            memory_route_eligible=memory_route_eligible,
            routing_reason=reason,
            expected_final_answer_type=_taskgraph_answer_type(contract).value,
            answer_contract=_contract_payload(contract),
        )
