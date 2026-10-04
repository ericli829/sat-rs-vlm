"""Wire models and an answer-only contract; no execution/model settings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .errors import ProtocolValidationError, UnsupportedConstraintError

MIME_TYPES = Literal["image/png", "image/jpeg", "image/webp"]
ASSET_PATTERN = r"^[0-9a-f]{32}$"
HASH_PATTERN = r"^[0-9a-f]{64}$"


class WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ImageDescriptor(WireModel):
    asset_id: str = Field(pattern=ASSET_PATTERN)
    sha256: str = Field(pattern=HASH_PATTERN)
    mime_type: MIME_TYPES


class ResponseConstraint(WireModel):
    type: Literal["enum", "single_choice", "short_text", "integer", "bbox"]
    values: list[str] | None = None
    case_insensitive: bool = False
    question_form: str | None = None
    min_length: int | None = Field(default=None, ge=0, le=4096)
    minimum: int | None = None
    coordinate_format: Literal["pixel_xyxy"] | None = None
    length: Literal[4] | None = None

    @model_validator(mode="after")
    def check_shape(self) -> ResponseConstraint:
        if self.type in {"enum", "single_choice"}:
            if not self.values or any(not v for v in self.values):
                raise ValueError("finite values must be non-empty")
            keys = [v.casefold() if self.case_insensitive else v for v in self.values]
            if len(set(keys)) != len(keys):
                raise ValueError("finite values must be unambiguous")
        elif self.values is not None:
            raise ValueError("values only apply to finite constraints")
        if self.type == "bbox" and (self.coordinate_format != "pixel_xyxy" or self.length != 4):
            raise ValueError("bbox requires pixel_xyxy and length=4")
        return self


class PredictRequest(WireModel):
    protocol_version: Literal["2.0"]
    request_id: str = Field(min_length=1)
    item_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    images: list[ImageDescriptor] = Field(min_length=1, max_length=2)
    response_constraint: ResponseConstraint
    answer_type: str = Field(min_length=1)
    choices: dict[str, str] | None = None
    image_width: int | None = Field(default=None, gt=0)
    image_height: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def check_contract(self) -> PredictRequest:
        c = self.response_constraint
        # The supplied protocol defines bbox and change_region labels explicitly;
        # it does not enumerate all other answer_type spellings. Constraint.type
        # controls serialization, so do not invent an additional allowlist.
        if (c.type == "bbox") != (self.answer_type == "bbox"):
            raise ValueError("bbox answer_type disagrees with response_constraint")
        if (
            c.type == "enum"
            and c.question_form == "change_region"
            and self.answer_type != "region_label"
        ):
            raise ValueError("change_region requires region_label")
        if c.type == "single_choice":
            if not self.choices or set(self.choices) != set(c.values or ()):
                raise ValueError("choices must exactly cover the legal identifiers")
            if any(not text.strip() for text in self.choices.values()):
                raise ValueError("choice text must not be empty")
        elif self.choices is not None:
            raise ValueError("choices only apply to single_choice")
        if c.type == "bbox":
            if self.image_width is None or self.image_height is None:
                raise ValueError("bbox requires original image dimensions")
        if not self.question.strip():
            raise ValueError("empty question")
        return self


@dataclass(frozen=True)
class AnswerContract:
    kind: str
    values: tuple[str, ...] = ()
    option_texts: tuple[str, ...] = ()
    case_insensitive: bool = False
    minimum: int | None = None
    min_length: int = 0
    image_width: int | None = None
    image_height: int | None = None

    @classmethod
    def from_request(cls, request: PredictRequest) -> AnswerContract:
        c = request.response_constraint
        values = tuple(c.values or ())
        return cls(
            c.type,
            values,
            tuple(request.choices[key] for key in values) if request.choices else (),
            c.case_insensitive,
            c.minimum,
            c.min_length or 0,
            request.image_width,
            request.image_height,
        )


def parse_request(value: object) -> PredictRequest:
    if isinstance(value, dict):
        c = value.get("response_constraint")
        if isinstance(c, dict) and c.get("type") not in {
            "enum",
            "single_choice",
            "short_text",
            "integer",
            "bbox",
        }:
            raise UnsupportedConstraintError()
    try:
        return PredictRequest.model_validate(value)
    except ValidationError as exc:
        # Keep Pydantic input values and paths out of public responses.
        raise ProtocolValidationError() from exc
