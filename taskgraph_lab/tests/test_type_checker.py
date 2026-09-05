from __future__ import annotations

from taskgraph_lab.taskgraph.validator import validate_candidate


def test_abs_diff_wrong_type() -> None:
    graph = {
        "intent": "CHANGE_COUNT",
        "nodes": [
            {
                "id": "n1",
                "op": "LOCATE",
                "inputs": {"image": "$image0"},
                "params": {"target": {"category": "ship", "attributes": {}}},
                "output": "ships",
            },
            {
                "id": "n2",
                "op": "ABS_DIFF",
                "inputs": {"a": "$n1", "b": "$n1"},
                "params": {},
                "output": "difference",
            },
        ],
        "final": {"source": "$n2", "answer_type": "INTEGER"},
    }
    _, report = validate_candidate(graph, inputs={"image0": {}})
    assert not report.type_valid
    assert sum(item.code == "input_type_mismatch" for item in report.errors) == 2


def test_route_graph() -> None:
    graph = {
        "intent": "ROUTE_PLANNING",
        "nodes": [
            {
                "id": "n1",
                "op": "LOCATE",
                "inputs": {"image": "$image0"},
                "params": {"target": {"category": "roundabout", "attributes": {}}},
                "output": "starts",
            },
            {
                "id": "n2",
                "op": "SELECT",
                "inputs": {"candidates": "$n1"},
                "params": {"mode": "RANK", "criterion": "size", "rank": 1, "order": "DESCENDING"},
                "output": "start",
            },
            {
                "id": "n3",
                "op": "LOCATE",
                "inputs": {"image": "$image0"},
                "params": {"target": {"category": "pond", "attributes": {}}},
                "output": "goals",
            },
            {
                "id": "n4",
                "op": "SELECT",
                "inputs": {"candidates": "$n3"},
                "params": {"mode": "ORDINAL", "index": 1, "order": "LEFT_TO_RIGHT"},
                "output": "goal",
            },
            {
                "id": "n5",
                "op": "BUILD_ROUTE_CONTEXT",
                "inputs": {"image": "$image0", "start": "$n2", "goal": "$n4"},
                "params": {},
                "output": "context",
            },
            {
                "id": "n6",
                "op": "ROUTE_REASON",
                "inputs": {"context": "$n5"},
                "params": {"question": "$question", "choices": "$choices"},
                "output": "answer",
            },
        ],
        "final": {"source": "$n6", "answer_type": "CHOICE_SINGLE"},
    }
    _, report = validate_candidate(
        graph, inputs={"image0": {}}, question_type="MULTIPLE_CHOICE_SINGLE"
    )
    assert report.valid, report.model_dump()


def test_select_result_requires_visual_scope_and_attribute_singleton() -> None:
    visual_scope = {
        "nodes": [
            {
                "id": "n1",
                "op": "LOCATE",
                "inputs": {"image": "$image0"},
                "params": {"target": {"category": "ship", "attributes": {}}},
            },
            {
                "id": "n2",
                "op": "SELECT",
                "inputs": {"candidates": "$n1"},
                "params": {"mode": "RELATION", "relation": "LEFT_OF"},
            },
            {
                "id": "n3",
                "op": "LOCATE",
                "inputs": {"image": "$n2"},
                "params": {"target": {"category": "boat", "attributes": {}}},
            },
        ],
        "final": {
            "sources": ["$n3"],
            "question": "Which boat is selected?",
            "answer_type": "CHOICE_SINGLE",
        },
    }
    _, visual_report = validate_candidate(visual_scope, inputs={"image0": {}})
    assert "select_result_not_visual_scope" in {item.code for item in visual_report.errors}

    attribute_graph = {
        "nodes": [
            visual_scope["nodes"][0],
            visual_scope["nodes"][1],
            {
                "id": "n3",
                "op": "ATTRIBUTE",
                "inputs": {"entity": "$n2"},
                "params": {"attribute": "color"},
            },
        ],
        "final": {"sources": ["$n3"], "answer_type": "LABEL"},
    }
    _, attribute_report = validate_candidate(attribute_graph, inputs={"image0": {}})
    assert "attribute_requires_singleton" in {item.code for item in attribute_report.errors}


def test_subregion_select_is_singleton_for_attribute() -> None:
    graph = {
        "nodes": [
            {
                "id": "n1",
                "op": "LOCATE",
                "inputs": {"image": "$image0"},
                "params": {"target": {"category": "building", "attributes": {}}},
            },
            {
                "id": "n2",
                "op": "SELECT",
                "inputs": {"candidates": "$n1"},
                "params": {
                    "mode": "RANK",
                    "criterion": "bbox_area",
                    "rank": 1,
                    "order": "DESCENDING",
                },
            },
            {
                "id": "n3",
                "op": "SELECT",
                "inputs": {"candidates": "$n2"},
                "params": {"mode": "SUBREGION", "subregion": "ABOVE"},
            },
            {
                "id": "n4",
                "op": "ATTRIBUTE",
                "inputs": {"entity": "$n3"},
                "params": {"attribute": "color", "part": "top"},
            },
        ],
        "final": {"sources": ["$n4"], "answer_type": "LABEL"},
    }
    _, report = validate_candidate(graph, inputs={"image0": {}})
    assert report.valid, report.model_dump()


def test_subregion_select_is_valid_visual_scope_for_locate_and_count() -> None:
    graph = {
        "nodes": [
            {
                "id": "n1",
                "op": "LOCATE",
                "inputs": {"image": "$image0"},
                "params": {"target": {"category": "parking lot", "attributes": {}}},
            },
            {
                "id": "n2",
                "op": "SELECT",
                "inputs": {"candidates": "$n1"},
                "params": {"mode": "SUBREGION", "subregion": "RIGHT_SIDE"},
            },
            {
                "id": "n3",
                "op": "LOCATE",
                "inputs": {"image": "$n2"},
                "params": {"target": {"category": "car", "attributes": {}}},
            },
            {
                "id": "n4",
                "op": "COUNT",
                "inputs": {"entities": "$n3"},
                "params": {"target": {"category": "car", "attributes": {}}, "entire": False},
            },
        ],
        "final": {"sources": ["$n4"], "answer_type": "CHOICE_SINGLE"},
    }
    _, report = validate_candidate(graph, inputs={"image0": {}})
    assert report.valid, report.model_dump()
