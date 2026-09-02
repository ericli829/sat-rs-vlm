from __future__ import annotations

import json

from taskgraph_lab.generation.batch_parser import parse_teacher_batch


def _graph(label: str) -> dict:
    return {
        "intent": "OTHER",
        "nodes": [
            {
                "id": "n1",
                "op": "CLASSIFY",
                "inputs": {"input": "$image0"},
                "params": {"label_space": [label]},
            }
        ],
        "final": {"sources": ["$n1"], "answer_type": "CHOICE_SINGLE"},
    }


def _response(ids: list[str]) -> str:
    return json.dumps(
        {
            "batch_version": "taskgraph-batch-v1",
            "results": [
                {"sample_id": sample_id, "taskgraph": _graph(sample_id)} for sample_id in ids
            ],
        }
    )


def test_two_valid_samples_preserve_order() -> None:
    parsed = parse_teacher_batch(_response(["a", "b"]), ["a", "b"])
    assert not parsed.catastrophic
    assert [item.sample_id for item in parsed.valid_results] == ["a", "b"]
    assert not parsed.out_of_order
    assert parsed.transport_errors == []


def test_four_valid_samples() -> None:
    ids = ["a", "b", "c", "d"]
    parsed = parse_teacher_batch(_response(ids), ids)
    assert [item.sample_id for item in parsed.valid_results] == ids
    assert parsed.missing_ids == []
    assert parsed.duplicate_ids == []
