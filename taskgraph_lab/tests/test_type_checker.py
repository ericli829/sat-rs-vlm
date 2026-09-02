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
