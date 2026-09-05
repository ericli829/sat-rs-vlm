from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from sat_rs_vlm.integrations.counting import CountingSystemProvider
from sat_rs_vlm.taskgraph.evaluation_runner import run_taskgraph_evaluation
from sat_rs_vlm.taskgraph.input_composer import InputComposer
from sat_rs_vlm.taskgraph.operators import CountExecutor, OperatorContext
from sat_rs_vlm.taskgraph.providers import CountingRequest
from sat_rs_vlm.taskgraph.routing import ExecutionMode
from sat_rs_vlm.taskgraph.runtime import (
    RuntimeRequest,
    RuntimeResult,
    fake_runtime,
    runtime_from_config,
)
from sat_rs_vlm.taskgraph.runtime_types import (
    Answer,
    Entity,
    EntitySet,
    ImageRef,
    Region,
    ScalarInt,
)
from sat_rs_vlm.taskgraph.schema import GraphNode, TargetSpec
from sat_rs_vlm.taskgraph.tracing import ExecutionTrace


def _image(tmp_path: Path) -> ImageRef:
    path = tmp_path / "counting.png"
    image = Image.new("RGB", (32, 16), "black")
    draw = ImageDraw.Draw(image)
    for box in ((2, 2, 5, 5), (10, 2, 13, 5), (22, 2, 25, 5)):
        draw.rectangle(box, fill="white")
    image.save(path)
    return ImageRef(str(path), width=32, height=16)


def _counting_config() -> dict[str, object]:
    return {
        "detector": {"kind": "fake"},
        "scale": {
            "global": {"enabled": False},
            "native": {"enabled": True, "tile_size": 16, "overlap": 0},
            "fine": {"enabled": False},
        },
        "count": {
            "score_threshold": 0.20,
            "nms_iou": 0.40,
            "cross_scale_iou": 0.50,
            "keep_raw_proposals": True,
        },
        "gate": {"enabled": False},
    }


def _count_node(input_role: str) -> GraphNode:
    return GraphNode.model_validate(
        {
            "id": "n1",
            "op": "COUNT",
            "inputs": {input_role: "$value"},
            "params": {"target": {"category": "ship", "attributes": {}}, "entire": False},
        }
    )


class _NoCallCountingProvider:
    provider_name = "must_not_count"

    def count(self, request: CountingRequest):
        raise AssertionError("COUNT(EntitySet) must not call the counting provider")

    def close(self) -> None:
        return None


def test_count_entity_set_is_deterministic_and_does_not_call_provider(tmp_path: Path) -> None:
    image = ImageRef(str(tmp_path / "memory.png"), width=32, height=16)
    entities = EntitySet(
        tuple(
            Entity(Region(image, (index * 4.0, 0.0, index * 4.0 + 2.0, 2.0)), "ship", 0.9)
            for index in range(3)
        )
    )
    result = CountExecutor(_NoCallCountingProvider()).execute(
        _count_node("entities"),
        {"entities": entities},
        OperatorContext("How many ships?", (), InputComposer()),
    )

    assert result.value == ScalarInt(3, {"provider": "cardinality", "source": "EntitySet"})


def test_count_image_uses_exhaustive_counting_and_global_boxes(tmp_path: Path) -> None:
    provider = CountingSystemProvider.from_config(_counting_config())
    try:
        result = provider.count(
            CountingRequest(_image(tmp_path), TargetSpec(category="ship"), entire=True)
        )
    finally:
        provider.close()

    assert result.count == 3
    assert result.metadata["entire"] is True
    assert result.metadata["tiles_run"] == 2
    assert all(
        0.0 <= box[0] < box[2] <= 32.0
        for box in (item.region.bbox_xyxy_global for item in result.detections.entities)
    )


def test_count_region_is_scoped_but_keeps_global_coordinates(tmp_path: Path) -> None:
    provider = CountingSystemProvider.from_config(_counting_config())
    image = _image(tmp_path)
    try:
        result = provider.count(
            CountingRequest(
                Region(image, (0.0, 0.0, 16.0, 16.0)),
                TargetSpec(category="ship"),
                True,
            )
        )
    finally:
        provider.close()

    assert result.count == 2
    assert result.metadata["entire"] is False
    assert result.metadata["requested_entire"] is True
    assert all(item.region.bbox_xyxy_global[2] <= 16.0 for item in result.detections.entities)


def test_locate_select_count_uses_filtered_entity_set_cardinality() -> None:
    question = "How many ships are in the selected position?"
    graph = {
        "version": "taskgraph-v1.1",
        "question": question,
        "question_type": "FREE_FORM",
        "inputs": {"image0": {"type": "image", "uri_or_key": "fixture"}},
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
                "params": {"mode": "ORDINAL", "index": 1, "order": "LEFT_TO_RIGHT"},
            },
            {
                "id": "n3",
                "op": "COUNT",
                "inputs": {"entities": "$n2"},
                "params": {"target": {"category": "ship", "attributes": {}}, "entire": False},
            },
        ],
        "final": {"sources": ["$n3"], "question": "", "answer_type": "INTEGER"},
    }
    runtime = fake_runtime(detection_boxes=[[0, 0, 1, 1], [2, 0, 3, 1], [4, 0, 5, 1]])
    try:
        result = runtime.run(
            RuntimeRequest(
                "locate-select-count",
                "MME_RealWorld_RS",
                "count",
                question,
                ("tests/fixtures/miniature_dataset/images/counting.ppm",),
                graph=graph,
            )
        )
    finally:
        runtime.close()

    assert result.store.get("$n3") == ScalarInt(
        1, {"provider": "cardinality", "source": "EntitySet"}
    )
    assert runtime.providers.counting.calls == []
    assert len(runtime.providers.detection.calls) == 1


def test_relational_count_uses_select_result_before_counting() -> None:
    question = "How many vehicles are left of the harbor?"
    graph = {
        "version": "taskgraph-v1.1",
        "question": question,
        "question_type": "FREE_FORM",
        "inputs": {"image0": {"type": "image", "uri_or_key": "fixture"}},
        "nodes": [
            {
                "id": "n1",
                "op": "LOCATE",
                "inputs": {"image": "$image0"},
                "params": {"target": {"category": "harbor", "attributes": {}}},
            },
            {
                "id": "n2",
                "op": "LOCATE",
                "inputs": {"image": "$image0"},
                "params": {"target": {"category": "vehicle", "attributes": {}}},
            },
            {
                "id": "n3",
                "op": "SELECT",
                "inputs": {"candidates": "$n2", "reference": "$n1"},
                "params": {
                    "mode": "RELATION",
                    "relation": "LEFT_OF",
                    "selection_type": "MULTI",
                },
            },
            {
                "id": "n4",
                "op": "COUNT",
                "inputs": {"entities": "$n3"},
                "params": {"target": {"category": "vehicle", "attributes": {}}, "entire": False},
            },
        ],
        "final": {"sources": ["$n4"], "question": "", "answer_type": "INTEGER"},
    }
    runtime = fake_runtime(
        detection_boxes=[[0, 0, 4, 4], [40, 0, 44, 4]],
        retrieval_candidates=[([20, 0, 30, 10], 0.9)],
    )
    try:
        result = runtime.run(
            RuntimeRequest(
                "relational-count",
                "XLRS_Bench",
                "count",
                question,
                ("tests/fixtures/miniature_dataset/images/counting.ppm",),
                graph=graph,
            )
        )
    finally:
        runtime.close()

    assert result.store.get("$n4").value == 1
    assert runtime.providers.counting.calls == []


def test_counting_config_is_separate_from_locate_detection() -> None:
    runtime = runtime_from_config(
        {
            "providers": {
                "detection": {"kind": "fake", "boxes": []},
                "counting": {"kind": "counting_system", **_counting_config()},
                "semantic_2b": {"kind": "fake"},
                "route_4b": {"kind": "fake"},
                "choice": {"reuse": "semantic_2b"},
                "region_retriever": {"kind": "fake"},
            }
        }
    )
    try:
        assert runtime.providers.detection.provider_name == "fake_lae"
        assert runtime.providers.counting.provider_name == "counting_system:fake"
    finally:
        runtime.close()


class _CountingFailure(RuntimeError):
    error_type = "counting_failed"
    stage = "counting"


class _EvaluationRuntime:
    def run(self, request: RuntimeRequest) -> RuntimeResult:
        if request.sample_id == "bad":
            raise _CountingFailure("tile inference failed")
        return RuntimeResult(
            ExecutionMode.DIRECT_VLM,
            Answer("ok"),
            ExecutionTrace(request.sample_id, ExecutionMode.DIRECT_VLM.value),
        )


def test_counting_failure_is_recorded_and_evaluation_continues(tmp_path: Path) -> None:
    samples = [
        {"sample_id": "bad", "question": "count", "image_paths": ["missing.png"]},
        {"sample_id": "good", "question": "count", "image_paths": ["missing.png"]},
    ]
    output = tmp_path / "predictions.jsonl"

    summary = run_taskgraph_evaluation(_EvaluationRuntime(), samples, output)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert [row["status"] for row in rows] == ["failure", "success"]
    assert rows[0]["stage"] == "counting"
    assert rows[0]["error_type"] == "counting_failed"
    assert summary["failure_by_stage"] == {"counting": 1}
