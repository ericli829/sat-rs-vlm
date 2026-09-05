from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from taskgraph_lab.datasets.base import load_records, normalize_choices

CHOICE_ANSWER_TYPES = {"CHOICE_SINGLE", "CHOICE_MULTI"}
OPTION_LABEL = re.compile(r"^\s*\(([A-Z])\)")
SEPARATORS = re.compile(r"[\s,;/|+&]+")


def _choice_labels(choices: list[str] | None) -> set[str]:
    labels: set[str] = set()
    for choice in choices or []:
        match = OPTION_LABEL.match(choice.upper())
        if match:
            labels.add(match.group(1))
    return labels


def selected_choice_labels(answer: Any, choices: list[str] | None) -> list[str]:
    """Parse source answer labels without exposing answers to the Teacher prompt."""
    allowed = _choice_labels(choices)
    if answer is None or not allowed:
        return []
    if isinstance(answer, (list, tuple, set)):
        raw_parts = [str(item).strip().upper() for item in answer]
    else:
        raw_parts = [str(answer).strip().upper()]

    selected: list[str] = []
    for raw in raw_parts:
        compact = SEPARATORS.sub("", raw)
        compact = compact.replace("(", "").replace(")", "")
        if compact and all(char in allowed for char in compact):
            selected.extend(compact)
            continue
        parenthesized = re.findall(r"\(([A-Z])\)", raw)
        if parenthesized:
            selected.extend(label for label in parenthesized if label in allowed)
            continue
        option = re.search(r"\bOPTION\s+([A-Z])\b", raw)
        if option and option.group(1) in allowed:
            selected.append(option.group(1))
    return list(dict.fromkeys(selected))


def audit_choice_answer(
    *,
    answer: Any,
    choices: list[str] | None,
    final_answer_type: str | None,
) -> dict[str, Any]:
    selected = selected_choice_labels(answer, choices)
    if not selected:
        return {
            "status": "unknown",
            "valid": None,
            "source_answer": answer,
            "selected_choice_labels": [],
            "expected_answer_type": None,
            "actual_answer_type": final_answer_type,
            "reason": "source answer cardinality could not be parsed",
        }
    if len(selected) > 1:
        expected = "CHOICE_MULTI"
        compatible = {"CHOICE_MULTI"}
    else:
        # A multi-select question may legitimately have exactly one correct label
        # for a particular image. One selected label therefore supplies no upper
        # bound on task cardinality; it only proves this is a choice-family answer.
        expected = "CHOICE_SINGLE_OR_CHOICE_MULTI"
        compatible = CHOICE_ANSWER_TYPES
    valid = final_answer_type in compatible
    return {
        "status": "valid" if valid else "invalid",
        "valid": valid,
        "source_answer": answer,
        "selected_choice_labels": selected,
        "expected_answer_type": expected,
        "actual_answer_type": final_answer_type,
        "reason": None
        if valid
        else "Teacher final answer cardinality is incompatible with source answer",
    }


def _add_answer_record(
    index: dict[str, dict[str, Any]],
    *,
    sample_id: str,
    answer: Any,
    choices: Any,
    source: str,
) -> None:
    if sample_id in index:
        raise ValueError(f"duplicate answer source for sample_id: {sample_id}")
    index[sample_id] = {
        "answer": answer,
        "choices": normalize_choices(choices),
        "source": source,
    }


def load_answer_index(
    *,
    xlrs_json: str | Path | None = None,
    mme_json: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Load ground-truth answers only for post-generation dataset auditing."""
    index: dict[str, dict[str, Any]] = {}
    if xlrs_json is not None:
        for row_index, row in load_records(xlrs_json, dataset="XLRS-Bench answers"):
            sample_id = str(row.get("sample_id") or f"xlrs_{row_index:06d}")
            _add_answer_record(
                index,
                sample_id=sample_id,
                answer=row.get("answer"),
                choices=row.get("multi_choice_options", row.get("choices", row.get("options"))),
                source="XLRS_Bench",
            )
    if mme_json is not None:
        for row_index, row in load_records(mme_json, dataset="MME RealWorld answers"):
            subtask = str(row.get("Subtask", row.get("subtask", ""))).strip()
            image_value = row.get("Image", row.get("image"))
            image_text = str(image_value or "").replace("\\", "/").lower()
            if subtask and subtask.lower().replace("_", " ") != "remote sensing":
                continue
            if not subtask and not image_text.startswith("remote_sensing/"):
                continue
            question_id = str(
                row.get("Question_id", row.get("question_id", row.get("sample_id", row_index)))
            )
            safe_id = question_id.strip("/").replace("/", "_").replace(" ", "_")
            _add_answer_record(
                index,
                sample_id=f"mme_rs_{safe_id}",
                answer=row.get("Ground truth", row.get("answer")),
                choices=row.get("Answer choices", row.get("choices", row.get("options"))),
                source="MME_RealWorld_RS",
            )
    return index


def final_answer_type(taskgraph: Mapping[str, Any] | None) -> str | None:
    if not isinstance(taskgraph, Mapping):
        return None
    final = taskgraph.get("final")
    if not isinstance(final, Mapping):
        return None
    value = final.get("answer_type")
    return str(value) if value is not None else None
