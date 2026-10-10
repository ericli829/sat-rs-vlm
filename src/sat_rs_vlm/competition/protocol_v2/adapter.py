"""Protocol routing without benchmark names or size-only TaskGraph selection."""

from __future__ import annotations

import re
from dataclasses import dataclass

from sat_rs_vlm.taskgraph.routing import ExecutionMode
from sat_rs_vlm.taskgraph.runtime import RuntimeRequest
from sat_rs_vlm.taskgraph.schema import QuestionType

from .image_package import ResolvedImage
from .models import AnswerContract, PredictRequest

_SIMPLE_COUNT = re.compile(
    r"^\s*how many\s+([\w -]+?)\s+(?:are there|are (?:visible|present)|"
    r"can (?:you|be) (?:see|seen))(?:\s+(?:in|on) (?:this|the) (?:image|picture))?[?.]?\s*$",
    re.IGNORECASE,
)
_COMPLEX = re.compile(
    r"\b(?:left|right|above|below|between|near|within|inside|outside|largest|smallest|"
    r"closest|before|after|changed|change|route|path)\b|左|右|附近|之间|最大|最小|变化|路径",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CompetitionRoutingConfig:
    simple_count_detection: bool = True
    graph_for_relational_questions: bool = True


class CompetitionRequestAdapter:
    def __init__(self, config: CompetitionRoutingConfig | None = None) -> None:
        self.config = config or CompetitionRoutingConfig()

    def adapt(
        self, request: PredictRequest, images: tuple[ResolvedImage, ...], contract: AnswerContract
    ) -> RuntimeRequest:
        mode = ExecutionMode.DIRECT_VLM
        task = "semantic"
        target = None
        question = request.question
        count_match = _SIMPLE_COUNT.fullmatch(question)
        relational = bool(_COMPLEX.search(question))
        # Dimensions are package-validated context, never a trigger by themselves.
        if contract.kind == "bbox":
            mode, task = ExecutionMode.TASKGRAPH_UHR, "localization"
            question += (
                "\nResolve the referred target using localization and selection. "
                "Final evidence must be one Entity or Region in original global pixel coordinates."
            )
        elif contract.kind == "integer":
            task = "count"
            if (
                len(images) == 1
                and count_match
                and not relational
                and self.config.simple_count_detection
            ):
                mode = ExecutionMode.DIRECT_DETECTION
                target = count_match.group(1).strip()
            else:
                mode = ExecutionMode.TASKGRAPH_UHR
                question += (
                    "\nReturn authoritative integer evidence from COUNT or integer geometry."
                )
        elif len(images) == 1 and relational and self.config.graph_for_relational_questions:
            mode = ExecutionMode.TASKGRAPH_UHR
        elif len(images) == 2:
            task = "temporal_change"
        if len(images) == 2:
            question += "\nThe first image is BEFORE; the second image is AFTER."
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
            target_category=target,
            memory_question=request.question,
            verified_image_sha256s=tuple(image.descriptor.sha256 for image in images),
            allow_hash_memory=True,
        )
