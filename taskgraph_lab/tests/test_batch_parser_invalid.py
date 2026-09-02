from __future__ import annotations

import json

from taskgraph_lab.generation.batch_parser import parse_teacher_batch

GRAPH = {
    "intent": "OTHER",
    "nodes": [],
    "final": {},
}


def _payload(results: list) -> str:
    return json.dumps({"batch_version": "taskgraph-batch-v1", "results": results})


def test_catastrophic_json_and_version_failures() -> None:
    assert parse_teacher_batch("not json", ["a"]).catastrophic
    wrong = json.dumps({"batch_version": "wrong", "results": []})
    assert parse_teacher_batch(wrong, ["a"]).catastrophic


def test_missing_duplicate_unknown_and_malformed_are_reported_independently() -> None:
    parsed = parse_teacher_batch(
        _payload(
            [
                {"sample_id": "a", "taskgraph": GRAPH},
                {"sample_id": "a", "taskgraph": GRAPH},
                {"sample_id": "unknown", "taskgraph": GRAPH},
                {"taskgraph": GRAPH},
                {"sample_id": "c", "taskgraph": "bad"},
            ]
        ),
        ["a", "b", "c"],
    )
    assert not parsed.catastrophic
    assert parsed.missing_ids == ["b"]
    assert parsed.duplicate_ids == ["a"]
    assert parsed.unknown_ids == ["unknown"]
    assert {item.code for item in parsed.malformed_items} == {
        "missing_sample_id",
        "malformed_taskgraph",
    }
    assert parsed.valid_results == []


def test_out_of_order_is_recorded_but_items_are_normalized() -> None:
    parsed = parse_teacher_batch(
        _payload(
            [
                {"sample_id": "b", "taskgraph": GRAPH},
                {"sample_id": "a", "taskgraph": GRAPH},
            ]
        ),
        ["a", "b"],
    )
    assert parsed.out_of_order
    assert [item.sample_id for item in parsed.valid_results] == ["a", "b"]
    assert "result_order_mismatch" in {error.code for error in parsed.transport_errors}


def test_result_transport_metadata_is_not_silently_accepted() -> None:
    parsed = parse_teacher_batch(
        _payload([{"sample_id": "a", "taskgraph": GRAPH, "status": "valid"}]),
        ["a"],
    )
    assert parsed.valid_results == []
    assert parsed.malformed_items[0].code == "unexpected_result_fields"
