from __future__ import annotations

import tempfile
from pathlib import Path

from sat_rs_vlm.integrations.counting.bootstrap import counting_system_src, ensure_counting_system_importable

ensure_counting_system_importable()

from counting_system.detector.fake import FakeDetector
from counting_system.synth import Blob, write_blob_image
from sat_rs_vlm.integrations.counting import (
    CountingSystemDetectionAdapter,
    to_counting_scope,
    to_counting_target,
)
from sat_rs_vlm.taskgraph import RuntimeRequest, TaskGraphRuntime, runtime_from_config
from sat_rs_vlm.taskgraph.input_composer import InputComposer
from sat_rs_vlm.taskgraph.operators import CountExecutor as GraphCountExecutor
from sat_rs_vlm.taskgraph.operators import OperatorContext
from sat_rs_vlm.taskgraph.providers import DetectionRequest, FakeRegionRetriever, FakeSemanticVLMProvider
from sat_rs_vlm.taskgraph.runtime import RuntimeProviders
from sat_rs_vlm.taskgraph.runtime_types import Entity, EntitySet, ImageRef, Region, ScalarInt, SelectResult, SelectStatus
from sat_rs_vlm.taskgraph.schema import GraphNode, TargetSpec


class RecordingFakeDetector(FakeDetector):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[object] = []

    def detect(self, request):  # type: ignore[override]
        self.calls.append(request)
        return super().detect(request)


def _count_graph(question: str, options: list[str]) -> dict:
    return {
        "version": "taskgraph-v1.1",
        "question": question,
        "question_type": "MULTIPLE_CHOICE_SINGLE",
        "choices": options,
        "inputs": {"image0": {"type": "image", "uri_or_key": "fixture"}},
        "intent": "SIMPLE_COUNT",
        "nodes": [
            {
                "id": "n1",
                "op": "COUNT",
                "inputs": {"image": "$image0"},
                "params": {"target": {"category": "ship", "attributes": {}}, "entire": True},
            }
        ],
        "final": {
            "sources": ["$n1"],
            "question": "Which option matches this count?",
            "answer_type": "CHOICE_SINGLE",
        },
    }


def _runtime(adapter: CountingSystemDetectionAdapter) -> TaskGraphRuntime:
    shared = FakeSemanticVLMProvider({}, "A")
    return TaskGraphRuntime(
        RuntimeProviders(
            detection=adapter,
            semantic_2b=shared,
            route_4b=FakeSemanticVLMProvider({}, "A"),
            retriever=FakeRegionRetriever([]),
            choice=shared,
        )
    )


def test_counting_system_package_is_importable() -> None:
    src = ensure_counting_system_importable()
    assert src.is_dir()
    assert (counting_system_src() / "counting_system" / "executor.py").is_file()


def test_bridge_preserves_global_xyxy_and_target_phrase() -> None:
    with tempfile.TemporaryDirectory() as raw:
        tmp_path = Path(raw)
        image_path = tmp_path / "blank.png"
        write_blob_image(str(image_path), width=64, height=64, blobs=[Blob((8, 8, 16, 16))])
        graph_image = ImageRef(str(image_path), 64, 64)
        region = Region(graph_image, (4.0, 4.0, 32.0, 32.0), {"region_id": "TOP_LEFT"})
        cs_region = to_counting_scope(region)
        assert cs_region.bbox_xyxy_global == region.bbox_xyxy_global
        assert Path(cs_region.image.uri_or_key) == Path(graph_image.uri_or_key)
        spec = to_counting_target(TargetSpec(category="ship", attributes={"color": "white"}))
        assert spec.category == "ship"
        assert spec.phrase() == "white ship"


def test_adapter_detect_matches_detection_provider_contract() -> None:
    with tempfile.TemporaryDirectory() as raw:
        tmp_path = Path(raw)
        blobs = [
            Blob((40, 40, 80, 80)),
            Blob((200, 50, 240, 90)),
            Blob((60, 200, 100, 240)),
        ]
        image_path = write_blob_image(
            str(tmp_path / "ships.png"), width=512, height=512, blobs=blobs
        ).uri_or_key
        recorder = RecordingFakeDetector()
        adapter = CountingSystemDetectionAdapter(
            config={
                "scale": {
                    "global": {"enabled": False},
                    "native": {"enabled": True, "tile_size": 256, "overlap": 32},
                    "fine": {"enabled": False},
                },
                "count": {"score_threshold": 0.05, "nms_iou": 0.4},
            },
            detector=recorder,
        )
        request = DetectionRequest(
            ImageRef(image_path, 512, 512),
            TargetSpec(category="ship", attributes={}),
            "COUNT",
        )
        try:
            detected = adapter.detect(request)
            assert detected.provider.startswith("counting_system")
            assert len(detected.detections.entities) == 3
            boxes = [tuple(item.region.bbox_xyxy_global) for item in detected.detections.entities]
            for blob in blobs:
                assert any(
                    abs(box[0] - blob.bbox[0]) <= 3 and abs(box[1] - blob.bbox[1]) <= 3
                    for box in boxes
                )
            assert detected.metadata["detector_calls"] == len(recorder.calls)
            assert detected.metadata["detector_calls"] >= 1
            assert recorder.calls[0].target.category == "ship"
            assert all(call.image.size[0] > 0 for call in recorder.calls)
        finally:
            adapter.close()


def test_taskgraph_count_image_emits_authoritative_scalar() -> None:
    with tempfile.TemporaryDirectory() as raw:
        tmp_path = Path(raw)
        blobs = [Blob((30, 30, 70, 70)), Blob((200, 40, 240, 80)), Blob((40, 200, 80, 240))]
        image_path = write_blob_image(
            str(tmp_path / "count.png"), width=320, height=320, blobs=blobs
        ).uri_or_key
        adapter = CountingSystemDetectionAdapter(
            config={
                "scale": {
                    "global": {"enabled": False},
                    "native": {"enabled": True, "tile_size": 256, "overlap": 32},
                    "fine": {"enabled": False},
                }
            },
            detector=FakeDetector(),
        )
        runtime = _runtime(adapter)
        question = "How many ships are there?"
        options = ["A 1", "B 2", "C 3", "D 4"]
        try:
            result = runtime.run(
                RuntimeRequest(
                    "count-image",
                    "MME_RealWorld_RS",
                    "count",
                    question,
                    (image_path,),
                    tuple(options),
                    graph=_count_graph(question, options),
                )
            )
            scalar = result.store.get("$n1")
            assert isinstance(scalar, ScalarInt)
            assert scalar.value == 3
            assert result.output.choice_id == "C"
            assert adapter.detect_requests[0].task_hint == "COUNT"
            count_trace = next(item for item in result.trace.nodes if item.node_id == "n1")
            assert count_trace.input_refs == {"image": "$image0"}
            assert "counting_system" in count_trace.provider
        finally:
            runtime.close()


def test_taskgraph_count_entities_skips_detector() -> None:
    image = ImageRef("memory://unused", 32, 32)
    entities = EntitySet(
        tuple(
            Entity(
                Region(image, (float(i), 0.0, float(i) + 1.0, 1.0)),
                "umbrella",
                1.0,
            )
            for i in range(2)
        )
    )
    adapter = CountingSystemDetectionAdapter(detector=FakeDetector())
    executor = GraphCountExecutor(adapter)
    node = GraphNode.model_validate(
        {
            "id": "n5",
            "op": "COUNT",
            "inputs": {"entities": "$n4"},
            "params": {"target": {"category": "umbrella", "attributes": {}}, "entire": False},
        }
    )
    context = OperatorContext("How many umbrellas?", (), InputComposer())
    try:
        outcome = executor.execute(node, {"entities": entities}, context)
        assert isinstance(outcome.value, ScalarInt)
        assert outcome.value.value == 2
        assert outcome.provider == "cardinality"
        assert adapter.detect_requests == []
        empty = SelectResult(EntitySet(()), SelectStatus.EMPTY, "geometry")
        empty_out = executor.execute(node, {"entities": empty}, context)
        assert empty_out.value.value == 0
    finally:
        adapter.close()


def test_runtime_from_config_wires_counting_system() -> None:
    config = {
        "providers": {
            "detection": {"kind": "counting_system", "backend": "fake"},
            "semantic_2b": {"kind": "fake", "default": "A"},
            "route_4b": {"kind": "fake", "default": "A"},
            "choice": {"reuse": "semantic_2b"},
            "region_retriever": {"kind": "fake"},
        }
    }
    runtime = runtime_from_config(config)
    try:
        assert runtime.providers.detection.provider_name.startswith("counting_system")
    finally:
        runtime.close()
