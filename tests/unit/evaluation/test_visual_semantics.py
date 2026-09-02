"""Tests for image-grounded LEVIR Caption semantic scoring helpers."""

from __future__ import annotations

import pytest

from sat_rs_vlm.evaluation.visual_semantics import (
    VisualSemanticFacts,
    aggregate_visual_semantic_metrics,
    extract_visual_semantics,
    parse_gold_semantics,
    sample_semantic_metrics,
)


def test_extractor_keeps_opposite_temporal_directions_distinct() -> None:
    addition = extract_visual_semantics("A new building was constructed beside a road.")
    removal = extract_visual_semantics("Two houses were demolished.")

    assert addition.change_label == 1
    assert addition.events == {("building", "appearance_construction")}
    assert removal.events == {("building", "disappearance_demolition")}


def test_no_change_gold_requires_none_fields() -> None:
    gold = parse_gold_semantics(
        {
            "gold_change_label": "0",
            "gold_changed_objects": "none",
            "gold_change_directions": "none",
        }
    )

    assert gold is not None
    assert gold.change_label == 0
    with pytest.raises(ValueError, match="requires"):
        parse_gold_semantics(
            {
                "gold_change_label": "0",
                "gold_changed_objects": "building",
                "gold_change_directions": "none",
            }
        )


def test_gold_uses_explicit_events_instead_of_cartesian_product() -> None:
    gold = parse_gold_semantics(
        {
            "gold_change_label": "1",
            "gold_changed_objects": "building|road",
            "gold_change_directions": "appearance_construction|disappearance_demolition",
            "gold_change_events": (
                "building:appearance_construction|road:disappearance_demolition"
            ),
        }
    )

    assert gold is not None
    assert gold.events == {
        ("building", "appearance_construction"),
        ("road", "disappearance_demolition"),
    }


def test_extractor_does_not_credit_empty_caption_as_no_change() -> None:
    prediction = extract_visual_semantics("")
    gold = VisualSemanticFacts(0, frozenset(), frozenset(), frozenset())

    metrics = sample_semantic_metrics(prediction, gold)

    assert prediction.change_label is None
    assert metrics["binary_parse_success"] is False
    assert metrics["binary_correct"] is False


def test_extractor_retains_objectless_change_claim_for_binary_diagnosis() -> None:
    prediction = extract_visual_semantics("The second image has changed significantly.")

    assert prediction.change_label == 1
    assert prediction.objects == set()
    assert prediction.directions == {"state_change_unspecified"}
    assert prediction.events == set()


def test_event_metrics_expose_opposite_direction_error() -> None:
    gold = VisualSemanticFacts(
        1,
        frozenset({"building"}),
        frozenset({"appearance_construction"}),
        frozenset({("building", "appearance_construction")}),
    )
    prediction = VisualSemanticFacts(
        1,
        frozenset({"building"}),
        frozenset({"disappearance_demolition"}),
        frozenset({("building", "disappearance_demolition")}),
    )

    metrics = sample_semantic_metrics(prediction, gold)

    assert metrics["event_tp"] == 0
    assert metrics["opposite_temporal_error_count"] == 1


def test_aggregate_visual_metrics_uses_image_gold_events() -> None:
    gold = VisualSemanticFacts(
        1,
        frozenset({"road"}),
        frozenset({"appearance_construction"}),
        frozenset({("road", "appearance_construction")}),
    )
    prediction = VisualSemanticFacts(
        1,
        frozenset({"road"}),
        frozenset({"appearance_construction"}),
        frozenset({("road", "appearance_construction")}),
    )
    row = {
        "gold": {
            "change_label": gold.change_label,
            "objects": sorted(gold.objects),
            "directions": sorted(gold.directions),
        },
        "prediction": prediction.to_dict(),
        "sample_metrics": sample_semantic_metrics(prediction, gold),
    }

    summary = aggregate_visual_semantic_metrics([row])

    assert summary["binary"]["accuracy"] == 1.0
    assert summary["object"]["f1"] == 1.0
    assert summary["direction"]["f1"] == 1.0
    assert summary["object_direction_event"]["f1"] == 1.0


def test_no_change_rows_do_not_inflate_positive_event_scores() -> None:
    no_change = VisualSemanticFacts(0, frozenset(), frozenset(), frozenset())
    changed_gold = VisualSemanticFacts(
        1,
        frozenset({"building"}),
        frozenset({"appearance_construction"}),
        frozenset({("building", "appearance_construction")}),
    )
    rows = [
        {
            "gold": {"change_label": 0, "objects": [], "directions": []},
            "prediction": no_change.to_dict(),
            "sample_metrics": sample_semantic_metrics(no_change, no_change),
        },
        {
            "gold": {
                "change_label": 1,
                "objects": ["building"],
                "directions": ["appearance_construction"],
            },
            "prediction": no_change.to_dict(),
            "sample_metrics": sample_semantic_metrics(no_change, changed_gold),
        },
    ]

    summary = aggregate_visual_semantic_metrics(rows)

    assert summary["object"]["num_positive_gold_samples"] == 1
    assert summary["object"]["exact_match_accuracy"] == 0.0
    assert summary["object_direction_event"]["f1"] == 0.0


def test_binary_coverage_keeps_unresolved_rows_out_of_resolved_confusion_matrix() -> None:
    changed_gold = VisualSemanticFacts(
        1,
        frozenset({"building"}),
        frozenset({"appearance_construction"}),
        frozenset({("building", "appearance_construction")}),
    )
    resolved_prediction = VisualSemanticFacts(0, frozenset(), frozenset(), frozenset())
    unresolved_prediction = VisualSemanticFacts(
        None,
        frozenset({"building"}),
        frozenset({"appearance_construction"}),
        frozenset({("building", "appearance_construction")}),
    )
    rows = [
        {
            "gold": {"change_label": 0, "objects": [], "directions": []},
            "prediction": resolved_prediction.to_dict(),
            "sample_metrics": sample_semantic_metrics(
                resolved_prediction,
                VisualSemanticFacts(0, frozenset(), frozenset(), frozenset()),
            ),
        },
        {
            "gold": {
                "change_label": 1,
                "objects": ["building"],
                "directions": ["appearance_construction"],
            },
            "prediction": unresolved_prediction.to_dict(),
            "sample_metrics": sample_semantic_metrics(unresolved_prediction, changed_gold),
        },
    ]

    summary = aggregate_visual_semantic_metrics(rows)

    assert summary["binary"]["decision_coverage"] == 0.5
    assert summary["binary"]["resolved_sample_count"] == 1
    assert summary["binary"]["unresolved_prediction_count"] == 1
    assert summary["binary"]["unresolved_gold_change_count"] == 1
    assert summary["binary"]["accuracy"] == 1.0
    assert summary["binary"]["all_sample_accuracy_with_unresolved_as_incorrect"] == 0.5
    assert summary["binary"]["confusion_matrix"] == {"tp": 0, "tn": 1, "fp": 0, "fn": 0}
