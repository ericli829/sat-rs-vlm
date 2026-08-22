"""Formal exact-cardinality eligibility and metrics for counting evaluations."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from sat_rs_vlm.data.object_adapter_v0 import _cardinality_prompt_target, count_bin
from sat_rs_vlm.evaluation.parsers import parse_count

COUNT_BINS = ("0-2", "3-5", "6-10", "11+")
PROTOCOL_NAME = "formal_e2_parse_count_plus_exact_cardinality_eligibility_v1"


def _metric_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    parsed = [row for row in rows if row.get("parsed_prediction") is not None]
    errors = [float(row["signed_error"]) for row in parsed]
    absolute_errors = [abs(value) for value in errors]
    parsed_n = len(parsed)
    return {
        "n": total,
        "parsed_n": parsed_n,
        "prediction_parse_rate": parsed_n / total if total else None,
        # Prediction parse failures remain in the accuracy denominator.
        "acc_exact": (
            sum(bool(row["exact_count_correct"]) for row in rows) / total if total else None
        ),
        "acc_within_1": (
            sum(bool(row["within_1_correct"]) for row in rows) / total if total else None
        ),
        "mae": sum(absolute_errors) / parsed_n if parsed_n else None,
        "rmse": math.sqrt(sum(value * value for value in errors) / parsed_n)
        if parsed_n
        else None,
        "bias": sum(errors) / parsed_n if parsed_n else None,
        "error_metrics_denominator": "parsed_n",
        # Backward-compatible aliases used by merger reports.
        "parse_rate": parsed_n / total if total else None,
        "exact": sum(bool(row["exact_count_correct"]) for row in rows) / total
        if total
        else None,
        "within_1": sum(bool(row["within_1_correct"]) for row in rows) / total
        if total
        else None,
    }


def classify_counting_predictions(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the formal parser and exact-cardinality eligibility to predictions.

    A valid score row requires an exact-cardinality question and a parseable
    reference. Prediction parse failures remain in the accuracy denominator.
    """

    diagnostics = {
        "raw_task_type_counting_rows": 0,
        "excluded_non_cardinality": 0,
        "excluded_missing_question": 0,
        "excluded_invalid_reference": 0,
        "valid_cardinality_rows": 0,
        "prediction_parse_failures": 0,
    }
    valid_rows: list[dict[str, Any]] = []
    parse_failures: list[dict[str, Any]] = []
    for source in rows:
        if str(source.get("task_type", "")).strip().lower() != "counting":
            continue
        diagnostics["raw_task_type_counting_rows"] += 1
        question = str(source.get("question", "")).strip()
        if not question:
            diagnostics["excluded_missing_question"] += 1
            continue
        if _cardinality_prompt_target(question) is None:
            diagnostics["excluded_non_cardinality"] += 1
            continue
        expected = parse_count(source.get("reference"))
        if expected.value is None:
            diagnostics["excluded_invalid_reference"] += 1
            continue
        predicted = parse_count(source.get("prediction"))
        signed_error = (
            int(predicted.value) - int(expected.value) if predicted.value is not None else None
        )
        row = {
            **dict(source),
            "parsed_reference": int(expected.value),
            "parsed_prediction": (
                int(predicted.value) if predicted.value is not None else None
            ),
            "prediction_parse_error": predicted.reason,
            "count_bin": count_bin(int(expected.value)),
            "signed_error": signed_error,
            "exact_count_correct": bool(signed_error == 0) if signed_error is not None else False,
            "within_1_correct": bool(abs(signed_error) <= 1)
            if signed_error is not None
            else False,
        }
        valid_rows.append(row)
        diagnostics["valid_cardinality_rows"] += 1
        if predicted.value is None:
            diagnostics["prediction_parse_failures"] += 1
            parse_failures.append(
                {
                    "id": str(source.get("id", "")),
                    "prediction": source.get("prediction"),
                    "reason": predicted.reason,
                }
            )
    return {
        "valid_rows": valid_rows,
        "parse_failures": parse_failures,
        "diagnostics": diagnostics,
    }


def summarize_exact_cardinality_counting(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize valid exact-cardinality rows using formal E2 parser semantics."""

    classified = classify_counting_predictions(rows)
    valid_rows = classified["valid_rows"]
    binned: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in valid_rows:
        binned[str(row["count_bin"])].append(row)
    return {
        "schema_version": "1.0",
        "metrics_protocol": PROTOCOL_NAME,
        "eligibility": "exact_cardinality_question_and_parseable_reference",
        "diagnostics": classified["diagnostics"],
        "overall": _metric_rows(valid_rows),
        "count_bins": {name: _metric_rows(binned[name]) for name in COUNT_BINS},
    }


def prediction_parse_failures(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return failure diagnostics for every valid cardinality prediction."""

    return list(classify_counting_predictions(rows)["parse_failures"])
