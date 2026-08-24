"""Reference-safe comparison helpers for counting prediction ensembles."""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from sat_rs_vlm.data.task_protocol import parse_count
from sat_rs_vlm.evaluation.records import EvaluationError


class EnsembleComparisonError(EvaluationError):
    """Prediction files cannot be aligned or contain unsafe duplicates."""


def index_counting_rows(
    rows: Iterable[Mapping[str, Any]], *, source: str = "rows"
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("id", "")).strip()
        if not sample_id:
            raise EnsembleComparisonError(f"{source}: row has an empty id")
        if sample_id in indexed:
            raise EnsembleComparisonError(f"{source}: duplicate id {sample_id}")
        if str(row.get("task_type", "counting")).lower() != "counting":
            raise EnsembleComparisonError(f"{source}: non-counting row {sample_id}")
        indexed[sample_id] = dict(row)
    return indexed


def _parsed(row: Mapping[str, Any]) -> int | None:
    return parse_count(row.get("prediction", "")).value


def _reference(row: Mapping[str, Any]) -> int | None:
    return parse_count(row.get("reference", "")).value


def _aligned(
    candidate_rows: Sequence[Iterable[Mapping[str, Any]]],
) -> tuple[list[str], list[dict[str, dict[str, Any]]]]:
    indexes = [
        index_counting_rows(rows, source=f"candidate_{i}") for i, rows in enumerate(candidate_rows)
    ]
    if not indexes:
        raise EnsembleComparisonError("at least one candidate is required")
    ids = set(indexes[0])
    for index in indexes[1:]:
        if set(index) != ids:
            raise EnsembleComparisonError("candidate ids are not pair-compatible")
    ordered = sorted(ids)
    for sample_id in ordered:
        references = {_reference(index[sample_id]) for index in indexes}
        if len(references) != 1:
            raise EnsembleComparisonError(f"reference mismatch for id {sample_id}")
    return ordered, indexes


def pairwise_counting_comparison(
    baseline_rows: Iterable[Mapping[str, Any]],
    candidate_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute agreement, correctness overlap and oracle accuracy by id."""

    ids, indexes = _aligned([baseline_rows, candidate_rows])
    baseline, candidate = indexes
    both_correct = a_only = b_only = neither = agreement = 0
    details: list[dict[str, Any]] = []
    for sample_id in ids:
        a = _parsed(baseline[sample_id])
        b = _parsed(candidate[sample_id])
        reference = _reference(baseline[sample_id])
        a_correct = a is not None and reference is not None and a == reference
        b_correct = b is not None and reference is not None and b == reference
        agreement += int(a == b)
        if a_correct and b_correct:
            both_correct += 1
        elif a_correct:
            a_only += 1
        elif b_correct:
            b_only += 1
        else:
            neither += 1
        details.append(
            {
                "id": sample_id,
                "reference": reference,
                "a_prediction": a,
                "b_prediction": b,
                "a_correct": a_correct,
                "b_correct": b_correct,
            }
        )
    total = len(ids)
    return {
        "n": total,
        "pairwise_prediction_agreement": agreement / total if total else None,
        "correctness_overlap": {
            "both_correct": both_correct,
            "a_only_correct": a_only,
            "b_only_correct": b_only,
            "neither_correct": neither,
        },
        "oracle_accuracy": (both_correct + a_only + b_only) / total if total else None,
        "a_accuracy": (both_correct + a_only) / total if total else None,
        "b_accuracy": (both_correct + b_only) / total if total else None,
        "rows": details,
    }


def majority_vote_counting(
    candidate_rows: Sequence[Iterable[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Majority vote with deterministic median tie-break and no threshold search."""

    ids, indexes = _aligned(candidate_rows)
    votes: list[dict[str, Any]] = []
    for sample_id in ids:
        reference = _reference(indexes[0][sample_id])
        values = [_parsed(index[sample_id]) for index in indexes]
        valid = [value for value in values if value is not None]
        if not valid:
            selected = None
            selection = "unparsed"
        else:
            counts = Counter(valid)
            top = max(counts.values())
            tied = sorted(value for value, count in counts.items() if count == top)
            selected = int(statistics.median(tied))
            selection = "majority" if len(tied) == 1 else "median_of_tied_modes"
        votes.append(
            {
                "id": sample_id,
                "reference": reference,
                "candidate_predictions": values,
                "prediction": selected,
                "correct": selected is not None and reference is not None and selected == reference,
                "selection": selection,
            }
        )
    return {
        "n": len(votes),
        "accuracy": sum(bool(row["correct"]) for row in votes) / len(votes) if votes else None,
        "rows": votes,
        "threshold_search": {"performed": False, "development_only": False},
    }


def median_vote_counting(candidate_rows: Sequence[Iterable[Mapping[str, Any]]]) -> dict[str, Any]:
    """Median vote over parseable integer predictions, preserving missingness."""

    ids, indexes = _aligned(candidate_rows)
    rows: list[dict[str, Any]] = []
    for sample_id in ids:
        values = [_parsed(index[sample_id]) for index in indexes]
        valid = [value for value in values if value is not None]
        reference = _reference(indexes[0][sample_id])
        prediction = int(statistics.median(valid)) if valid else None
        rows.append(
            {
                "id": sample_id,
                "reference": reference,
                "candidate_predictions": values,
                "prediction": prediction,
                "correct": prediction is not None
                and reference is not None
                and prediction == reference,
            }
        )
    return {
        "n": len(rows),
        "accuracy": sum(bool(row["correct"]) for row in rows) / len(rows) if rows else None,
        "rows": rows,
        "threshold_search": {"performed": False, "development_only": False},
    }
