from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

BATCH_VERSION = "taskgraph-batch-v1"


class BatchTransportIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    sample_id: str | None = None
    result_index: int | None = None


class BatchTransportItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: str
    taskgraph: dict[str, Any]
    raw_item: dict[str, Any]
    request_index: int


class BatchParseResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_version: str | None = None
    valid_results: list[BatchTransportItem] = Field(default_factory=list)
    missing_ids: list[str] = Field(default_factory=list)
    duplicate_ids: list[str] = Field(default_factory=list)
    unknown_ids: list[str] = Field(default_factory=list)
    malformed_items: list[BatchTransportIssue] = Field(default_factory=list)
    out_of_order: bool = False
    actual_ids: list[str] = Field(default_factory=list)
    transport_errors: list[BatchTransportIssue] = Field(default_factory=list)
    catastrophic: bool = False


def _expected_ids(expected_samples: Sequence[Any]) -> list[str]:
    ids: list[str] = []
    for sample in expected_samples:
        if isinstance(sample, str):
            sample_id = sample
        elif isinstance(sample, Mapping):
            sample_id = sample.get("sample_id")
        else:
            sample_id = getattr(sample, "sample_id", None)
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("every expected sample must have a non-empty sample_id")
        ids.append(sample_id)
    if len(ids) != len(set(ids)):
        raise ValueError("expected sample ids must be unique")
    return ids


def _catastrophic(code: str, message: str) -> BatchParseResult:
    issue = BatchTransportIssue(code=code, message=message)
    return BatchParseResult(catastrophic=True, transport_errors=[issue])


def parse_teacher_batch(
    raw_response: str,
    expected_samples: Sequence[Any],
) -> BatchParseResult:
    """Parse only the batch envelope while preserving every independent item."""
    expected_ids = _expected_ids(expected_samples)
    expected_set = set(expected_ids)
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        return _catastrophic("batch_json_parse", str(exc))
    if not isinstance(payload, Mapping):
        return _catastrophic("batch_root_type", "batch response must be a JSON object")
    version = payload.get("batch_version")
    if version != BATCH_VERSION:
        return _catastrophic(
            "batch_version",
            f"batch_version must be {BATCH_VERSION!r}, got {version!r}",
        )
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        return _catastrophic("batch_results_type", "results must be a list")

    malformed: list[BatchTransportIssue] = []
    parsed: list[BatchTransportItem] = []
    actual_ids: list[str] = []
    occurrences: Counter[str] = Counter()
    unknown_ids: list[str] = []
    for index, raw_item in enumerate(raw_results):
        if not isinstance(raw_item, Mapping):
            malformed.append(
                BatchTransportIssue(
                    code="malformed_result",
                    message="result must be an object",
                    result_index=index,
                )
            )
            continue
        sample_id = raw_item.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            malformed.append(
                BatchTransportIssue(
                    code="missing_sample_id",
                    message="result.sample_id must be a non-empty string",
                    result_index=index,
                )
            )
            continue
        actual_ids.append(sample_id)
        occurrences[sample_id] += 1
        if sample_id not in expected_set:
            if sample_id not in unknown_ids:
                unknown_ids.append(sample_id)
            continue
        taskgraph = raw_item.get("taskgraph")
        if not isinstance(taskgraph, Mapping):
            malformed.append(
                BatchTransportIssue(
                    code="malformed_taskgraph",
                    message="result.taskgraph must be an object",
                    sample_id=sample_id,
                    result_index=index,
                )
            )
            continue
        if set(raw_item) != {"sample_id", "taskgraph"}:
            malformed.append(
                BatchTransportIssue(
                    code="unexpected_result_fields",
                    message=(
                        "batch result may contain only sample_id and taskgraph; got "
                        f"{sorted(raw_item)}"
                    ),
                    sample_id=sample_id,
                    result_index=index,
                )
            )
            continue
        parsed.append(
            BatchTransportItem(
                sample_id=sample_id,
                taskgraph=dict(taskgraph),
                raw_item=dict(raw_item),
                request_index=index,
            )
        )

    duplicate_ids = [sample_id for sample_id in expected_ids if occurrences[sample_id] > 1]
    missing_ids = [sample_id for sample_id in expected_ids if occurrences[sample_id] == 0]
    expected_present_order = [sample_id for sample_id in expected_ids if sample_id in actual_ids]
    actual_known_order = [sample_id for sample_id in actual_ids if sample_id in expected_set]
    out_of_order = actual_known_order != expected_present_order

    valid_by_id = {item.sample_id: item for item in parsed if occurrences[item.sample_id] == 1}
    valid_results = [
        valid_by_id[sample_id] for sample_id in expected_ids if sample_id in valid_by_id
    ]
    errors = list(malformed)
    errors.extend(
        BatchTransportIssue(
            code="missing_result",
            message="expected sample is missing from batch results",
            sample_id=sample_id,
        )
        for sample_id in missing_ids
    )
    errors.extend(
        BatchTransportIssue(
            code="duplicate_sample_id",
            message="sample_id appears more than once in batch results",
            sample_id=sample_id,
        )
        for sample_id in duplicate_ids
    )
    errors.extend(
        BatchTransportIssue(
            code="unknown_sample_id",
            message="sample_id was not present in the request",
            sample_id=sample_id,
        )
        for sample_id in unknown_ids
    )
    if out_of_order:
        errors.append(
            BatchTransportIssue(
                code="result_order_mismatch",
                message="result order does not match input sample order",
            )
        )
    return BatchParseResult(
        batch_version=str(version),
        valid_results=valid_results,
        missing_ids=missing_ids,
        duplicate_ids=duplicate_ids,
        unknown_ids=unknown_ids,
        malformed_items=malformed,
        out_of_order=out_of_order,
        actual_ids=actual_ids,
        transport_errors=errors,
        catastrophic=False,
    )
