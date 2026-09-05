from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .enums import (
    AnswerType,
    ExtremeDirection,
    GroupMode,
    IntentLabel,
    OperatorName,
    QuestionType,
    RegionPosition,
    SelectMode,
    SortOrder,
    SpatialRelation,
    SubregionType,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=False)


Scalar = str | int | float | bool
INTRINSIC_ATTRIBUTES = {"color", "shape", "size", "state", "pattern", "has_part"}


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
            raise ValueError(
                "TargetSpec attributes must be intrinsic; unsupported keys: " + ", ".join(invalid)
            )
        return value


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
    criterion: str | None = None
    rank: int | None = Field(default=None, ge=1)
    order: SortOrder | None = None
    index: int | None = Field(default=None, ge=1)
    direction: ExtremeDirection | None = None
    subregion: SubregionType | None = None

    @model_validator(mode="after")
    def validate_mode_shape(self) -> SelectParams:
        required = {
            SelectMode.RELATION: {"relation"},
            SelectMode.RANK: {"criterion", "rank", "order"},
            SelectMode.ORDINAL: {"index", "order"},
            SelectMode.EXTREME: {"direction"},
            SelectMode.SUBREGION: {"subregion"},
        }[self.mode]
        fields = {"relation", "criterion", "rank", "order", "index", "direction", "subregion"}
        present = {name for name in fields if getattr(self, name) is not None}
        missing = required - present
        unexpected = present - required
        if missing or unexpected:
            raise ValueError(
                f"SELECT {self.mode.value} params mismatch; "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
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
            raise ValueError("string choices must be the system reference $choices")
        return value


class RouteReasonParams(VLMReasonParams):
    choices: str | list[str]


class MatchChoiceParams(StrictModel):
    choices: str | list[str]

    @field_validator("choices")
    @classmethod
    def choices_reference(cls, value: str | list[str]) -> str | list[str]:
        if isinstance(value, str) and value != "$choices":
            raise ValueError("string choices must be the system reference $choices")
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
        invalid = [ref for ref in refs if not ref.startswith("$")]
        if invalid:
            raise ValueError(f"node inputs must be references, got: {invalid}")
        return value

    @model_validator(mode="after")
    def validate_operator_params(self) -> GraphNode:
        for name, value in self.inputs.items():
            if not isinstance(value, list):
                continue
            if not value:
                raise ValueError(f"{self.op.value}.{name} reference list must not be empty")
            if not (self.op is OperatorName.VLM_REASON and name == "evidence"):
                raise ValueError(
                    f"list references are only allowed for VLM_REASON.evidence, got "
                    f"{self.op.value}.{name}"
                )
        parsed = PARAM_MODELS[self.op].model_validate(self.params)
        self.params = parsed.model_dump(mode="json", exclude_none=False)
        return self


class FinalSpec(StrictModel):
    sources: list[str] = Field(min_length=1)
    question: str | None = None
    answer_type: AnswerType

    @field_validator("sources")
    @classmethod
    def node_refs_only(cls, value: list[str]) -> list[str]:
        invalid = [ref for ref in value if re.fullmatch(r"\$n[1-9][0-9]*", ref) is None]
        if invalid:
            raise ValueError(f"final.sources must contain only $nX references, got: {invalid}")
        if len(value) != len(set(value)):
            raise ValueError("final.sources must not contain duplicate references")
        return value

    @field_validator("question")
    @classmethod
    def nonblank_static_question(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("final.question must not be empty")
        return stripped


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
    def question_choice_consistency(self) -> TaskGraph:
        if (
            self.question_type
            in {
                QuestionType.MULTIPLE_CHOICE,
                QuestionType.MULTIPLE_CHOICE_SINGLE,
                QuestionType.MULTIPLE_CHOICE_MULTI,
            }
            and not self.choices
        ):
            raise ValueError("multiple-choice samples require choices")
        return self
