from dataclasses import FrozenInstanceError

import pytest

from sat_rs_vlm.taskgraph.locate_routing import analyze_locate
from sat_rs_vlm.taskgraph.schema import GraphNode


def node(id, op, inputs, **params):
    return GraphNode(id=id, op=op, inputs=inputs, params=params)


def locate(id="n1"):
    return node(id, "LOCATE", {"image": "$image"}, target={"category": "building"})


@pytest.mark.parametrize(
    "params",
    [
        {"mode": "EXTREME", "direction": "TOPMOST"},
        {"mode": "RANK", "criterion": "bbox_area", "rank": 1, "order": "DESCENDING"},
        {"mode": "ORDINAL", "index": 4, "order": "BOTTOM_TO_TOP"},
    ],
)
def test_set_comparison_is_not_singleton(params):
    graph = [
        locate(),
        node("n2", "SELECT", {"candidates": "$n1"}, **params),
        node("n3", "ATTRIBUTE", {"entity": "$n2"}, attribute="color"),
    ]
    hint = analyze_locate("n1", graph)
    assert hint.backend == "LAE_DINO"
    assert "n2:SELECT.candidates" in hint.consumer_paths[0]


def test_reference_does_not_follow_candidate_count_and_shared_set_wins():
    graph = [
        locate(),
        locate("n2"),
        node(
            "n3",
            "SELECT",
            {"candidates": "$n2", "reference": "$n1"},
            mode="RELATION",
            relation="NEAR",
        ),
        node("n4", "COUNT", {"entities": "$n3"}, target={"category": "car"}, entire=True),
    ]
    assert analyze_locate("n1", graph).backend == "VLM_SINGLE"
    assert analyze_locate("n2", graph).backend == "LAE_DINO"
    graph.append(node("n5", "GROUP", {"entities": "$n1"}, mode="CLUSTER"))
    assert analyze_locate("n1", graph).backend == "LAE_DINO"


def test_local_region_count_image_does_not_contaminate_single_route():
    graph = [
        node("n1", "REGION", {"image": "$image"}, position="BOTTOM_RIGHT"),
        node("n2", "SELECT", {"candidates": "$n1"}, mode="SUBREGION", subregion="BELOW"),
        node("n3", "LOCATE", {"image": "$n2"}, target={"category": "building"}),
        node("n4", "COUNT", {"image": "$n2"}, target={"category": "car"}, entire=True),
        node("n5", "ATTRIBUTE", {"entity": "$n3"}, attribute="color"),
    ]
    assert analyze_locate("n3", graph).backend == "VLM_SINGLE"
    hint = analyze_locate("n3", graph)
    with pytest.raises(FrozenInstanceError):
        hint.route_reason = "changed"


def test_unknown_cardinality_is_conservative_and_question_is_not_read():
    graph = [
        locate(),
        node("n2", "SELECT", {"candidates": "$n1"}, mode="SUBREGION", subregion="BELOW"),
    ]
    hint = analyze_locate("n1", graph)
    assert hint.backend == "LAE_DINO"
    assert hint.route_reason == "ambiguous_set_requirement"


def test_semantic_candidates_and_shared_attribute_always_preserve_set():
    graph = [
        locate(),
        node(
            "n2",
            "SELECT",
            {"candidates": "$n1"},
            mode="SEMANTIC",
            query="trees right and grass below",
            selection_type="SINGLE",
        ),
        node("n3", "ATTRIBUTE", {"entity": "$n1"}, attribute="color"),
    ]
    assert analyze_locate("n1", graph).backend == "LAE_DINO"


def test_generic_evidence_cardinality_cannot_be_guessed_from_question():
    graph = [
        locate(),
        node("n2", "VLM_REASON", {"evidence": ["$n1"]}, question="Analyze the evidence"),
    ]
    assert analyze_locate("n1", graph).route_reason == "ambiguous_set_requirement"
