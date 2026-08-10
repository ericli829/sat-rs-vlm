"""参考文本语义事实的逐样本指标。"""

from __future__ import annotations

from typing import Any, TypeVar

from sat_rs_vlm.evaluation.semantic.extractors import SemanticFacts

Fact = TypeVar("Fact", bound=tuple[Any, ...] | str)


def _set_counts(predicted: set[Fact], reference: set[Fact]) -> tuple[int, int, int]:
    return len(predicted & reference), len(predicted - reference), len(reference - predicted)


def _prf(tp: int, fp: int, fn: int) -> tuple[float | None, float | None, float | None]:
    predicted = tp + fp
    expected = tp + fn
    precision = tp / predicted if predicted else (0.0 if expected else None)
    recall = tp / expected if expected else None
    if precision is None or recall is None:
        f1 = None if precision is None and recall is None else 0.0
    elif precision + recall:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return precision, recall, f1


def semantic_sample_metrics(
    prediction: SemanticFacts,
    reference: SemanticFacts,
) -> dict[str, float | int | None]:
    predicted_objects = set(prediction.objects)
    reference_objects = set(reference.objects)
    object_tp, object_fp, object_fn = _set_counts(predicted_objects, reference_objects)
    object_precision, object_recall, object_f1 = _prf(object_tp, object_fp, object_fn)

    predicted_counts = set(prediction.counts)
    reference_counts = set(reference.counts)
    count_correct = len(predicted_counts & reference_counts)

    predicted_relations = set(prediction.relations)
    reference_relations = set(reference.relations)
    relation_tp, relation_fp, relation_fn = _set_counts(predicted_relations, reference_relations)
    relation_precision, relation_recall, relation_f1 = _prf(relation_tp, relation_fp, relation_fn)

    predicted_changes = set(prediction.changes)
    reference_changes = set(reference.changes)
    change_tp, change_fp, change_fn = _set_counts(predicted_changes, reference_changes)
    change_precision, change_recall, change_f1 = _prf(change_tp, change_fp, change_fn)

    return {
        "object_tp": object_tp,
        "object_fp": object_fp,
        "object_fn": object_fn,
        "object_precision": object_precision,
        "object_recall": object_recall,
        "object_f1": object_f1,
        "reference_unsupported_object_rate": (
            object_fp / len(predicted_objects) if predicted_objects else 0.0
        ),
        "object_omission_rate": object_fn / len(reference_objects) if reference_objects else 0.0,
        "count_correct": count_correct,
        "count_reference_facts": len(reference_counts),
        "count_consistency_accuracy": (
            count_correct / len(reference_counts) if reference_counts else None
        ),
        "spatial_relation_tp": relation_tp,
        "spatial_relation_fp": relation_fp,
        "spatial_relation_fn": relation_fn,
        "spatial_relation_precision": relation_precision,
        "spatial_relation_recall": relation_recall,
        "spatial_relation_f1": relation_f1,
        "spatial_relation_accuracy": (
            relation_tp / len(reference_relations) if reference_relations else None
        ),
        "change_event_tp": change_tp,
        "change_event_fp": change_fp,
        "change_event_fn": change_fn,
        "change_event_precision": change_precision,
        "change_event_recall": change_recall,
        "change_event_f1": change_f1,
    }
