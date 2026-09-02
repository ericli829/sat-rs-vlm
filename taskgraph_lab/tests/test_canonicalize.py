from __future__ import annotations

from taskgraph_lab.taskgraph.canonicalize import canonicalize_target, stable_json_dumps


def test_canonicalization_is_deterministic_and_topological() -> None:
    graph = {
        "intent": "SIMPLE_COUNT",
        "nodes": [
            {
                "id": "n9",
                "op": "MATCH_CHOICE",
                "inputs": {"value": "$n7"},
                "params": {"choices": "$choices"},
                "output": "answer",
            },
            {
                "id": "n7",
                "op": "COUNT",
                "inputs": {"image": "$image0"},
                "params": {"target": {"category": "ship", "attributes": {}}, "entire": True},
                "output": "count",
            },
        ],
        "final": {"source": "$n9", "answer_type": "CHOICE_SINGLE"},
    }
    first = canonicalize_target(graph)
    second = canonicalize_target(graph)
    assert [node["id"] for node in first["nodes"]] == ["n1", "n2"]
    assert first["nodes"][1]["inputs"]["value"] == "$n1"
    assert first["final"]["sources"] == ["$n2"]
    assert "source" not in first["final"]
    assert "question" not in first["final"]
    assert all("output" not in node for node in first["nodes"])
    assert stable_json_dumps(first) == stable_json_dumps(second)
