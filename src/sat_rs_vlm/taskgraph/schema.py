"""Canonical production copy of the frozen TaskGraph v1.1 contract.

This module deliberately has no runtime dependency on ``taskgraph_lab``.  The
lab is a planner-data workspace; this schema is the production parsing boundary.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .contracts import DEPRECATED_OPERATORS, OPERATOR_INPUT_CONTRACTS


class StringEnum(str, Enum):
    pass


class QuestionType(StringEnum):
    MULTIPLE_CHOICE_SINGLE = "MULTIPLE_CHOICE_SINGLE"
    MULTIPLE_CHOICE_MULTI = "MULTIPLE_CHOICE_MULTI"
    FREE_FORM = "FREE_FORM"
    BOOLEAN = "BOOLEAN"
    INTEGER = "INTEGER"


class IntentLabel(StringEnum):
    SIMPLE_COUNT = "SIMPLE_COUNT"
    RELATIONAL_COUNT = "RELATIONAL_COUNT"
    ATTRIBUTE_QUERY = "ATTRIBUTE_QUERY"
    OBJECT_RELATION = "OBJECT_RELATION"
    OBJECT_CLASSIFICATION = "OBJECT_CLASSIFICATION"
    REGIONAL_CLASSIFICATION = "REGIONAL_CLASSIFICATION"
    MULTILABEL_CLASSIFICATION = "MULTILABEL_CLASSIFICATION"
    MOTION_QUERY = "MOTION_QUERY"
    CHANGE_COUNT = "CHANGE_COUNT"
    ROUTE_PLANNING = "ROUTE_PLANNING"
    COMPLEX_REASONING = "COMPLEX_REASONING"
    OTHER = "OTHER"


class AnswerType(StringEnum):
    CHOICE_SINGLE = "CHOICE_SINGLE"
    CHOICE_MULTI = "CHOICE_MULTI"
    INTEGER = "INTEGER"
    BOOLEAN = "BOOLEAN"
    LABEL = "LABEL"
    LABEL_SET = "LABEL_SET"
    TEXT = "TEXT"


class OperatorName(StringEnum):
    REGION = "REGION"
    REGION_FROM_BBOX = "REGION_FROM_BBOX"
    FIND_MARKER = "FIND_MARKER"
    LOCATE = "LOCATE"
    SELECT = "SELECT"
    GROUP = "GROUP"
    COUNT = "COUNT"
    ATTRIBUTE = "ATTRIBUTE"
    CLASSIFY = "CLASSIFY"
    MULTILABEL_CLASSIFY = "MULTILABEL_CLASSIFY"
    MOTION = "MOTION"
    RELATION = "RELATION"
    ABS_DIFF = "ABS_DIFF"
    VLM_REASON = "VLM_REASON"
    BUILD_ROUTE_CONTEXT = "BUILD_ROUTE_CONTEXT"
    ROUTE_REASON = "ROUTE_REASON"
    MATCH_CHOICE = "MATCH_CHOICE"


class RegionPosition(StringEnum):
    TOP = "TOP"
    BOTTOM = "BOTTOM"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    CENTER = "CENTER"
    TOP_LEFT = "TOP_LEFT"
    TOP_RIGHT = "TOP_RIGHT"
    BOTTOM_LEFT = "BOTTOM_LEFT"
    BOTTOM_RIGHT = "BOTTOM_RIGHT"
    TOP_CENTER = "TOP_CENTER"
    BOTTOM_CENTER = "BOTTOM_CENTER"
    CENTER_LEFT = "CENTER_LEFT"
    CENTER_RIGHT = "CENTER_RIGHT"


class SelectMode(StringEnum):
    RELATION = "RELATION"
    RANK = "RANK"
    ORDINAL = "ORDINAL"
    EXTREME = "EXTREME"
    SUBREGION = "SUBREGION"


class SelectionCardinality(StringEnum):
    SINGLE = "SINGLE"
    MULTI = "MULTI"


class RankCriterion(StringEnum):
    BBOX_AREA = "bbox_area"
    SCORE = "score"


class SpatialRelation(StringEnum):
    LEFT_OF = "LEFT_OF"
    RIGHT_OF = "RIGHT_OF"
    ABOVE = "ABOVE"
    BELOW = "BELOW"
    UPPER_LEFT_OF = "UPPER_LEFT_OF"
    UPPER_RIGHT_OF = "UPPER_RIGHT_OF"
    LOWER_LEFT_OF = "LOWER_LEFT_OF"
    LOWER_RIGHT_OF = "LOWER_RIGHT_OF"
    NEAR = "NEAR"
    NEXT_TO = "NEXT_TO"
    INSIDE = "INSIDE"
    OVERLAP = "OVERLAP"
    OUTSIDE = "OUTSIDE"
    BETWEEN = "BETWEEN"
    AROUND = "AROUND"
    IN_FRONT_OF = "IN_FRONT_OF"
    BEHIND = "BEHIND"


class SortOrder(StringEnum):
    ASCENDING = "ASCENDING"
    DESCENDING = "DESCENDING"
    TOP_TO_BOTTOM = "TOP_TO_BOTTOM"
    BOTTOM_TO_TOP = "BOTTOM_TO_TOP"
    LEFT_TO_RIGHT = "LEFT_TO_RIGHT"
    RIGHT_TO_LEFT = "RIGHT_TO_LEFT"


class ExtremeDirection(StringEnum):
    LEFTMOST = "LEFTMOST"
    RIGHTMOST = "RIGHTMOST"
    TOPMOST = "TOPMOST"
    BOTTOMMOST = "BOTTOMMOST"


class SubregionType(StringEnum):
    LEFT_SIDE = "LEFT_SIDE"
    RIGHT_SIDE = "RIGHT_SIDE"
    ABOVE = "ABOVE"
    BELOW = "BELOW"
    INSIDE = "INSIDE"
    OUTSIDE = "OUTSIDE"
    BOTH_SIDES = "BOTH_SIDES"
    AROUND = "AROUND"


class GroupMode(StringEnum):
    ROW = "ROW"
    COLUMN = "COLUMN"
    CLUSTER = "CLUSTER"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=False)


Scalar = str | int | float | bool
INTRINSIC_ATTRIBUTES = {"color", "shape", "size", "state", "pattern", "has_part"}
RUNTIME_RESULT_HINT = re.compile(
    r"\b(?:(?:detected|predicted|computed|resolved)\s+)?"
    r"(?:count|number|result|value|label)\s+(?:is|equals|=)\s+"
    r"(?:-?\d+(?:\.\d+)?|true|false|[A-Z][\w-]*)\b",
    re.IGNORECASE,
)
NUMERIC_VALUE_HINT = re.compile(r"(?<![\w$])-?\d+(?:\.\d+)?(?!\w)")


class InputSpec(StrictModel):
    type: Literal["image"] = "image"
    uri_or_key: str = Field(min_length=1)


class AttributeSpec(StrictModel):
    name: str = Field(min_length=1)
    value: Scalar | None = None
    part: str | None = None


class TargetSpec(StrictModel):
    category: str = Field(min_length=1)
    attributes: dict[str, Scalar] = Field(default_factory=dict)

    @field_validator("attributes")
    @classmethod
    def intrinsic_attributes_only(cls, value: dict[str, Scalar]) -> dict[str, Scalar]:
        invalid = sorted(set(value) - INTRINSIC_ATTRIBUTES)
        if invalid:
            raise ValueError("unsupported intrinsic attributes: " + ", ".join(invalid))
        return value

    def phrase(self) -> str:
        prefix = " ".join(str(value) for value in self.attributes.values())
        return f"{prefix} {self.category}".strip()


class MarkerSpec(StrictModel):
    color: str | None = None
    shape: str = Field(min_length=1)


class RegionParams(StrictModel):
    position: RegionPosition


class RegionFromBBoxParams(StrictModel):
    bbox: tuple[float, float, float, float]
    image_size: tuple[int, int] | None = None

    @field_validator("image_size")
    @classmethod
    def positive_image_size(cls, value: tuple[int, int] | None) -> tuple[int, int] | None:
        if value is not None and any(item <= 0 for item in value):
            raise ValueError("image_size values must be positive")
        return value


class FindMarkerParams(StrictModel):
    marker: MarkerSpec


class LocateParams(StrictModel):
    target: TargetSpec


class SelectParams(StrictModel):
    mode: SelectMode
    relation: SpatialRelation | None = None
    criterion: RankCriterion | None = None
    rank: int | None = Field(default=None, ge=1)
    order: SortOrder | None = None
    index: int | None = Field(default=None, ge=1)
    direction: ExtremeDirection | None = None
    subregion: SubregionType | None = None
    # All SELECT geometry uses original-image pixels.  A missing margin is
    # resolved deterministically from the current scope and recorded in output.
    margin: float | None = Field(default=None, ge=0.0)
    overlap_iou_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    selection_type: SelectionCardinality | None = None

    @model_validator(mode="after")
    def validate_mode_shape(self) -> SelectParams:
        required = {
            SelectMode.RELATION: {"relation"},
            SelectMode.RANK: {"criterion", "rank", "order"},
            SelectMode.ORDINAL: {"index", "order"},
            SelectMode.EXTREME: {"direction"},
            SelectMode.SUBREGION: {"subregion"},
        }[self.mode]
        fields = {
            "relation", "criterion", "rank", "order", "index", "direction", "subregion",
            "margin", "overlap_iou_threshold", "selection_type",
        }
        present = {name for name in fields if getattr(self, name) is not None}
        allowed_optional = {
            SelectMode.RELATION: {"margin", "overlap_iou_threshold", "selection_type"},
            SelectMode.SUBREGION: {"margin"},
        }[self.mode] if self.mode in {SelectMode.RELATION, SelectMode.SUBREGION} else set()
        if not required.issubset(present) or present - required - allowed_optional:
            raise ValueError(
                f"SELECT {self.mode.value} params mismatch; "
                f"missing={sorted(required - present)}, unexpected={sorted(present - required)}"
            )
        if self.mode is SelectMode.RANK and self.order not in {
            SortOrder.ASCENDING,
            SortOrder.DESCENDING,
        }:
            raise ValueError("SELECT RANK order must be ASCENDING or DESCENDING")
        if self.mode is SelectMode.ORDINAL and self.order not in {
            SortOrder.TOP_TO_BOTTOM,
            SortOrder.BOTTOM_TO_TOP,
            SortOrder.LEFT_TO_RIGHT,
            SortOrder.RIGHT_TO_LEFT,
        }:
            raise ValueError("SELECT ORDINAL requires a spatial order")
        return self


class GroupParams(StrictModel):
    mode: GroupMode


class CountParams(StrictModel):
    target: TargetSpec
    entire: bool


class AttributeParams(StrictModel):
    attribute: str = Field(min_length=1)
    part: str | None = None


class ClassifyParams(StrictModel):
    label_space: list[str] | None = None


class MultiLabelClassifyParams(StrictModel):
    label_space: list[str] = Field(min_length=1)


class EmptyParams(StrictModel):
    pass


class VLMReasonParams(StrictModel):
    question: str = Field(min_length=1)
    choices: str | list[str] | None = None

    @field_validator("choices")
    @classmethod
    def choices_reference(cls, value: str | list[str] | None) -> str | list[str] | None:
        if isinstance(value, str) and value != "$choices":
            raise ValueError("string choices must be $choices")
        return value


class RouteReasonParams(VLMReasonParams):
    choices: str | list[str]


class MatchChoiceParams(StrictModel):
    choices: str | list[str]

    @field_validator("choices")
    @classmethod
    def choices_reference(cls, value: str | list[str]) -> str | list[str]:
        if isinstance(value, str) and value != "$choices":
            raise ValueError("string choices must be $choices")
        return value


PARAM_MODELS: dict[OperatorName, type[StrictModel]] = {
    OperatorName.REGION: RegionParams,
    OperatorName.REGION_FROM_BBOX: RegionFromBBoxParams,
    OperatorName.FIND_MARKER: FindMarkerParams,
    OperatorName.LOCATE: LocateParams,
    OperatorName.SELECT: SelectParams,
    OperatorName.GROUP: GroupParams,
    OperatorName.COUNT: CountParams,
    OperatorName.ATTRIBUTE: AttributeParams,
    OperatorName.CLASSIFY: ClassifyParams,
    OperatorName.MULTILABEL_CLASSIFY: MultiLabelClassifyParams,
    OperatorName.MOTION: EmptyParams,
    OperatorName.RELATION: EmptyParams,
    OperatorName.ABS_DIFF: EmptyParams,
    OperatorName.VLM_REASON: VLMReasonParams,
    OperatorName.BUILD_ROUTE_CONTEXT: EmptyParams,
    OperatorName.ROUTE_REASON: RouteReasonParams,
    OperatorName.MATCH_CHOICE: MatchChoiceParams,
}


class GraphNode(StrictModel):
    id: str = Field(pattern=r"^n[1-9][0-9]*$")
    op: OperatorName
    inputs: dict[str, str | list[str]]
    params: dict[str, Any]

    @field_validator("inputs")
    @classmethod
    def refs_only(cls, value: dict[str, str | list[str]]) -> dict[str, str | list[str]]:
        refs = [
            item for raw in value.values() for item in (raw if isinstance(raw, list) else [raw])
        ]
        if any(not ref.startswith("$") for ref in refs):
            raise ValueError("node inputs must be references")
        return value

    @model_validator(mode="after")
    def validate_operator(self) -> GraphNode:
        OPERATOR_INPUT_CONTRACTS[self.op.value].validate_keys(self.inputs, operator=self.op.value)
        parsed = PARAM_MODELS[self.op].model_validate(self.params)
        self.params = parsed.model_dump(mode="json", exclude_none=False)
        return self

    @property
    def deprecated(self) -> bool:
        return self.op.value in DEPRECATED_OPERATORS


class FinalSpec(StrictModel):
    sources: list[str] = Field(min_length=1)
    question: str = Field(min_length=1)
    answer_type: AnswerType

    @field_validator("sources")
    @classmethod
    def node_refs_only(cls, value: list[str]) -> list[str]:
        if any(re.fullmatch(r"\$n[1-9][0-9]*", ref) is None for ref in value):
            raise ValueError("final.sources must contain only $nX references")
        if len(value) != len(set(value)):
            raise ValueError("final.sources must not contain duplicate references")
        return value

    @field_validator("question")
    @classmethod
    def static_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("final.question must not be empty")
        return value


class PlannerTarget(StrictModel):
    intent: IntentLabel | None = None
    nodes: list[GraphNode] = Field(min_length=1)
    final: FinalSpec


class TaskGraph(StrictModel):
    version: Literal["taskgraph-v1.1"] = "taskgraph-v1.1"
    question: str = Field(min_length=1)
    question_type: QuestionType
    choices: list[str] | None = None
    inputs: dict[str, InputSpec] = Field(min_length=1)
    intent: IntentLabel | None = None
    nodes: list[GraphNode] = Field(min_length=1)
    final: FinalSpec

    @model_validator(mode="after")
    def validate_graph(self) -> TaskGraph:
        if (
            self.question_type
            in {
                QuestionType.MULTIPLE_CHOICE_SINGLE,
                QuestionType.MULTIPLE_CHOICE_MULTI,
            }
            and not self.choices
        ):
            raise ValueError("multiple-choice samples require choices")
        known = set(self.inputs)
        ids: set[str] = set()
        for node in self.nodes:
            if node.id in ids:
                raise ValueError(f"duplicate node id: {node.id}")
            refs = [
                item
                for raw in node.inputs.values()
                for item in (raw if isinstance(raw, list) else [raw])
            ]
            missing = [ref for ref in refs if ref[1:] not in known and ref not in known]
            if missing:
                raise ValueError(f"{node.id} contains forward or missing refs: {missing}")
            ids.add(node.id)
            known.add(node.id)
        missing_final = [ref for ref in self.final.sources if ref[1:] not in ids]
        if missing_final:
            raise ValueError(f"unknown final refs: {missing_final}")
        count_final = any(
            node.id in {ref[1:] for ref in self.final.sources} and node.op is OperatorName.COUNT
            for node in self.nodes
        )
        if RUNTIME_RESULT_HINT.search(self.final.question) or (
            count_final and NUMERIC_VALUE_HINT.search(self.final.question)
        ):
            raise ValueError("final.question must be static and contain no runtime prediction")
        return self


def parse_taskgraph(value: str | bytes | dict[str, Any]) -> TaskGraph:
    """Parse a full graph while keeping the original dataset choices unchanged."""

    if isinstance(value, (str, bytes)):
        return TaskGraph.model_validate_json(value)
    return TaskGraph.model_validate(value)
