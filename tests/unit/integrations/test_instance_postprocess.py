from __future__ import annotations

from sat_rs_vlm.integrations.detectors.instance_postprocess import (
    box_iou,
    intersection_over_smaller,
    postprocess_instances,
)


def test_nested_repeat_is_removed_without_merging_neighboring_instances() -> None:
    boxes = [
        [0.0, 0.0, 100.0, 100.0],
        [5.0, 5.0, 68.0, 68.0],  # same target, smaller nested proposal; IoU is below 0.4
        [88.0, 0.0, 108.0, 20.0],  # a nearby, separate instance
    ]
    assert box_iou(boxes[0], boxes[1]) < 0.4
    assert intersection_over_smaller(boxes[0], boxes[1]) == 1.0

    result_boxes, result_scores, trace = postprocess_instances(
        boxes,
        [0.90, 0.80, 0.75],
        reference_areas=[100_000.0] * len(boxes),
    )

    assert result_boxes == [boxes[0], boxes[2]]
    assert result_scores == [0.90, 0.75]
    assert trace["same_instance_duplicates_removed"] == 1
    assert trace["candidate_pool_complete"] is True


def test_partial_overlap_between_distinct_instances_is_preserved() -> None:
    boxes = [[0.0, 0.0, 20.0, 20.0], [12.0, 0.0, 32.0, 20.0]]
    assert 0.0 < box_iou(*boxes) < 0.4

    result_boxes, _, trace = postprocess_instances(
        boxes,
        [0.8, 0.7],
        reference_areas=[640_000.0, 640_000.0],
    )

    assert result_boxes == boxes
    assert trace["confirmed_instance_count"] == 2


def test_cross_tile_clipped_duplicate_is_removed_with_low_iou() -> None:
    boxes = [[0.0, 0.0, 200.0, 400.0], [128.0, 0.0, 228.0, 400.0]]
    assert box_iou(*boxes) < 0.4
    assert intersection_over_smaller(*boxes) == 0.72

    result_boxes, _, trace = postprocess_instances(
        boxes,
        [0.82, 0.78],
        source_groups=["tile_a", "tile_b"],
        reference_areas=[640_000.0, 640_000.0],
    )

    assert len(result_boxes) == 1
    assert trace["same_instance_duplicates_removed"] == 1


def test_local_split_is_confirmed_even_when_refiner_also_returns_collection_box() -> None:
    parent = [100.0, 100.0, 500.0, 400.0]
    children = [[140.0, 200.0, 170.0, 230.0], [350.0, 200.0, 380.0, 230.0]]
    refined = [parent, *children]

    result_boxes, _, trace = postprocess_instances(
        [parent],
        [0.8],
        reference_areas=[500_000.0],
        refine=lambda _region: (refined, [0.7, 0.91, 0.89]),
    )

    assert result_boxes == children
    assert trace["local_refinement_confirmed_splits"] == 1


def test_nearby_small_regions_are_not_declared_independent_instances() -> None:
    parent = [0.0, 0.0, 400.0, 300.0]
    close_candidates = [[100.0, 100.0, 140.0, 140.0], [130.0, 100.0, 170.0, 140.0]]

    result_boxes, _, trace = postprocess_instances(
        [parent, *close_candidates],
        [0.8, 0.84, 0.82],
        reference_areas=[640_000.0] * 3,
        refine=lambda _region: (close_candidates, [0.85, 0.83]),
    )

    assert result_boxes == []
    assert trace["candidate_pool_complete"] is False
    assert trace["unresolved_regions"][0]["status"] == "UNRESOLVED"


def test_collection_box_is_replaced_by_locally_confirmed_instances() -> None:
    parent = [100.0, 100.0, 500.0, 400.0]
    children = [[140.0, 200.0, 170.0, 230.0], [350.0, 200.0, 380.0, 230.0]]
    calls: list[list[float]] = []

    def refine(region: list[float]) -> tuple[list[list[float]], list[float]]:
        calls.append(region)
        return children, [0.91, 0.88]

    result_boxes, result_scores, trace = postprocess_instances(
        [parent, *children],
        [0.60, 0.82, 0.79],
        reference_areas=[640_000.0] * 3,
        refine=refine,
    )

    assert calls == [parent]
    assert result_boxes == children
    assert result_scores == [0.91, 0.88]
    assert trace["local_refinement_confirmed_splits"] == 1
    assert trace["unresolved_regions"] == []
    assert trace["aggregate_regions"][0]["instance_count"] == 2


def test_uncertain_nested_region_is_not_counted_as_an_instance() -> None:
    parent = [0.0, 0.0, 400.0, 300.0]
    child = [100.0, 100.0, 130.0, 130.0]

    result_boxes, result_scores, trace = postprocess_instances(
        [parent, child],
        [0.65, 0.70],
        reference_areas=[640_000.0, 640_000.0],
        refine=lambda _region: ([], []),
    )

    assert result_boxes == []
    assert result_scores == []
    assert trace["candidate_pool_complete"] is False
    assert trace["unresolved_regions"][0]["status"] == "UNRESOLVED"


def test_single_large_or_tilted_ship_is_kept_after_local_confirmation() -> None:
    parent = [100.0, 100.0, 500.0, 500.0]
    refined = [90.0, 100.0, 510.0, 495.0]

    result_boxes, result_scores, trace = postprocess_instances(
        [parent],
        [0.8],
        reference_areas=[640_000.0],
        refine=lambda _region: ([refined], [0.86]),
    )

    assert result_boxes == [refined]
    assert result_scores == [0.86]
    assert trace["local_refinement_confirmed_singletons"] == 1
    assert trace["candidate_pool_complete"] is True


def test_collection_without_refinement_stays_unresolved() -> None:
    parent = [100.0, 100.0, 500.0, 400.0]
    children = [[140.0, 200.0, 170.0, 230.0], [350.0, 200.0, 380.0, 230.0]]

    result_boxes, _, trace = postprocess_instances(
        [parent, *children],
        [0.6, 0.8, 0.79],
        reference_areas=[640_000.0] * 3,
    )

    assert result_boxes == []
    assert trace["local_refinement_attempted"] == 0
    assert trace["candidate_pool_complete"] is False
    assert trace["unresolved_regions"][0]["contained_candidate_count"] == 2
