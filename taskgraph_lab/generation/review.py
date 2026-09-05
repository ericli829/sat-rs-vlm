from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from taskgraph_lab.datasets.base import NormalizedSample
from taskgraph_lab.taskgraph.enums import GraphQuality


class ReviewIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    node: str | None = None
    type: str
    message: str


class ReviewResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdict: GraphQuality
    issues: list[ReviewIssue] = Field(default_factory=list)


def render_review_prompt(template: str, sample: NormalizedSample, target: dict[str, Any]) -> str:
    payload = {
        "sample": {
            "question": sample.question,
            "question_type": sample.question_type.value,
            "choices": sample.choices,
            "inputs": {key: value.model_dump(mode="json") for key, value in sample.inputs.items()},
            "metadata": sample.metadata,
        },
        "taskgraph": target,
    }
    return "Review this sample and candidate:\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def parse_review(text: str) -> ReviewResult:
    return ReviewResult.model_validate(json.loads(text))
