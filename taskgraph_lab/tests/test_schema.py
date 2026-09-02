from __future__ import annotations

from taskgraph_lab.taskgraph.schema import PlannerTarget
from taskgraph_lab.taskgraph.validator import validate_candidate


def count_graph() -> dict:
    return {
        "intent": "SIMPLE_COUNT",
        "nodes": [
            {
                "id": "n1",
                "op": "REGION",
                "inputs": {"image": "$image0"},
                "params": {"position": "BOTTOM_LEFT"},
            },
            {
                "id": "n2",
                "op": "COUNT",
                "inputs": {"image": "$n1"},
                "params": {"target": {"category": "airplane", "attributes": {}}, "entire": False},
            },
            {
                "id": "n3",
                "op": "MATCH_CHOICE",
                "inputs": {"value": "$n2"},
                "params": {"choices": "$choices"},
            },
        ],
        "final": {
            "sources": ["$n3"],
            "question": "Which option matches the resolved result?",
            "answer_type": "CHOICE_SINGLE",
        },
    }


def test_taskgraph_schema_valid_example() -> None:
    graph = PlannerTarget.model_validate(count_graph())
    assert graph.nodes[1].params == {
        "target": {"category": "airplane", "attributes": {}},
        "entire": False,
    }


def test_invalid_operator_is_schema_error() -> None:
    graph = count_graph()
    graph["nodes"][0]["op"] = "FLY"
    _, report = validate_candidate(graph, inputs={"image0": {"type": "image", "uri_or_key": "x"}})
    assert not report.schema_valid
    assert not report.valid


def test_count_extra_params_are_rejected() -> None:
    graph = count_graph()
    graph["nodes"][1]["params"]["threshold"] = 0.2
    _, report = validate_candidate(graph, inputs={"image0": {"type": "image", "uri_or_key": "x"}})
    assert not report.schema_valid


def test_count_correct_entire_flag_schema() -> None:
    graph = count_graph()
    graph["nodes"] = [graph["nodes"][1]]
    graph["nodes"][0]["id"] = "n1"
    graph["nodes"][0]["inputs"] = {"image": "$image0"}
    graph["nodes"][0]["params"]["entire"] = True
    graph["final"] = {
        "sources": ["$n1"],
        "question": "What is this count?",
        "answer_type": "INTEGER",
    }
    _, report = validate_candidate(
        graph, inputs={"image0": {"type": "image", "uri_or_key": "x"}}, question_type="INTEGER"
    )
    assert report.valid


def test_bbox_graph() -> None:
    graph = {
        "intent": "ATTRIBUTE_QUERY",
        "nodes": [
            {
                "id": "n1",
                "op": "REGION_FROM_BBOX",
                "inputs": {"image": "$image0"},
                "params": {"bbox": [1, 2, 3, 4], "image_size": [100, 100]},
                "output": "bbox",
            },
            {
                "id": "n2",
                "op": "ATTRIBUTE",
                "inputs": {"entity": "$n1"},
                "params": {"attribute": "color"},
                "output": "color",
            },
            {
                "id": "n3",
                "op": "MATCH_CHOICE",
                "inputs": {"value": "$n2"},
                "params": {"choices": "$choices"},
                "output": "answer",
            },
        ],
        "final": {
            "sources": ["$n3"],
            "question": "Which option matches the determined color?",
            "answer_type": "CHOICE_SINGLE",
        },
    }
    _, report = validate_candidate(
        graph,
        inputs={"image0": {}},
        question="color in bounding box",
        question_type="MULTIPLE_CHOICE_SINGLE",
    )
    assert report.valid
    assert not report.warnings
