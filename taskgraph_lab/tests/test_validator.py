from __future__ import annotations

from taskgraph_lab.taskgraph.validator import validate_candidate


def test_missing_ref() -> None:
    graph = {
        "intent": "SIMPLE_COUNT",
        "nodes": [
            {
                "id": "n1",
                "op": "COUNT",
                "inputs": {"image": "$n9"},
                "params": {"target": {"category": "ship", "attributes": {}}, "entire": False},
                "output": "count",
            }
        ],
        "final": {"source": "$n1", "answer_type": "INTEGER"},
    }
    _, report = validate_candidate(graph, inputs={"image0": {}})
    assert any(item.code == "missing_node_ref" for item in report.errors)


def test_forward_and_cyclic_refs() -> None:
    graph = {
        "intent": "COMPLEX_REASONING",
        "nodes": [
            {
                "id": "n1",
                "op": "VLM_REASON",
                "inputs": {"evidence": ["$n2"]},
                "params": {"question": "$question"},
                "output": "a",
            },
            {
                "id": "n2",
                "op": "VLM_REASON",
                "inputs": {"evidence": ["$n1"]},
                "params": {"question": "$question"},
                "output": "b",
            },
        ],
        "final": {"source": "$n2", "answer_type": "TEXT"},
    }
    _, report = validate_candidate(graph, inputs={"image0": {}})
    codes = {item.code for item in report.errors}
    assert "forward_reference" in codes
    assert "dag_cycle" in codes


def test_marker_two_branch_abs_diff_graph() -> None:
    nodes = [
        {
            "id": "n1",
            "op": "FIND_MARKER",
            "inputs": {"image": "$image0"},
            "params": {"marker": {"color": "red", "shape": "circle"}},
            "output": "m0",
        },
        {
            "id": "n2",
            "op": "COUNT",
            "inputs": {"image": "$n1"},
            "params": {"target": {"category": "farm", "attributes": {}}, "entire": False},
            "output": "c0",
        },
        {
            "id": "n3",
            "op": "FIND_MARKER",
            "inputs": {"image": "$image1"},
            "params": {"marker": {"color": "red", "shape": "circle"}},
            "output": "m1",
        },
        {
            "id": "n4",
            "op": "COUNT",
            "inputs": {"image": "$n3"},
            "params": {"target": {"category": "farm", "attributes": {}}, "entire": False},
            "output": "c1",
        },
        {
            "id": "n5",
            "op": "ABS_DIFF",
            "inputs": {"a": "$n2", "b": "$n4"},
            "params": {},
            "output": "difference",
        },
        {
            "id": "n6",
            "op": "MATCH_CHOICE",
            "inputs": {"value": "$n5"},
            "params": {"choices": "$choices"},
            "output": "answer",
        },
    ]
    graph = {
        "intent": "CHANGE_COUNT",
        "nodes": nodes,
        "final": {"source": "$n6", "answer_type": "CHOICE_SINGLE"},
    }
    _, report = validate_candidate(
        graph,
        inputs={"image0": {}, "image1": {}},
        question="difference in red circles",
        question_type="MULTIPLE_CHOICE_SINGLE",
    )
    assert report.valid


def test_marker_warning_does_not_reject() -> None:
    graph = {
        "intent": "COMPLEX_REASONING",
        "nodes": [
            {
                "id": "n1",
                "op": "VLM_REASON",
                "inputs": {"image": "$image0"},
                "params": {"question": "$question", "choices": "$choices"},
                "output": "answer",
            }
        ],
        "final": {"source": "$n1", "answer_type": "CHOICE_SINGLE"},
    }
    _, report = validate_candidate(
        graph,
        inputs={"image0": {}},
        question="What is inside the red circle?",
        question_type="MULTIPLE_CHOICE_SINGLE",
    )
    assert report.valid
    assert any(item.code == "marker_without_find_marker" for item in report.warnings)
