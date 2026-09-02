from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from sat_rs_vlm.integrations.counting.bootstrap import (
    counting_system_src,
    ensure_counting_system_importable,
)

ensure_counting_system_importable()

from counting_system.detector.fake import FakeDetector
from counting_system.synth import Blob, write_blob_image
from sat_rs_vlm.integrations.counting import (
    CountingProposalDetectorBridge,
    CountingSystemProvider,
    to_counting_scope,
    to_counting_target,
)
from sat_rs_vlm.integrations.detectors.protocol import ProposalResult
from sat_rs_vlm.taskgraph import RuntimeRequest, TaskGraphRuntime, runtime_from_config
from sat_rs_vlm.taskgraph.input_composer import InputComposer
from sat_rs_vlm.taskgraph.operators import CountExecutor as GraphCountExecutor
from sat_rs_vlm.taskgraph.operators import LocateExecutor, OperatorContext
from sat_rs_vlm.taskgraph.providers import (
    CountingRequest,
    FakeCountingProvider,
    FakeDetectionProvider,
    FakeRegionRetriever,
    FakeSemanticVLMProvider,
)
from sat_rs_vlm.taskgraph.runtime import RuntimeProviders
from sat_rs_vlm.taskgraph.runtime_types import (
    Entity,
    EntitySet,
    ImageRef,
    Region,
    ScalarInt,
    SelectResult,
    SelectStatus,
)
from sat_rs_vlm.taskgraph.schema import GraphNode, TargetSpec


class RecordingFakeDetector(FakeDetector):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[object] = []
        self.closed = False

    def detect(self, request):  # type: ignore[override]
        self.calls.append(request)
        return super().detect(request)

    def close(self) -> None:
        self.closed = True


class RecordingProposalProvider:
    provider_name = "fake_proposal"

    def __init__(self, boxes: list[list[float]], scores: list[float] | None = None) -> None:
        self.boxes = boxes
        self.scores = scores or [0.9] * len(boxes)
        self.predict_calls: list[tuple[Path, str]] = []
        self.closed = False

    def predict(self, image_path: Path, target_phrase: str) -> ProposalResult:
        self.predict_calls.append((Path(image_path), target_phrase))
        return ProposalResult(
            boxes_xyxy=[list(box) for box in self.boxes],
            scores=list(self.scores),
            latency_ms=1.0,
            provider=self.provider_name,
            model_id="mock",
        )

    def close(self) -> None:
        self.closed = True


def _count_graph(question: str, options: list[str], *, entire: bool = True) -> dict:
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
                "params": {"target": {"category": "ship", "attributes": {}}, "entire": entire},
            }
        ],
        "final": {
            "sources": ["$n1"],
            "question": "Which option matches this count?",
            "answer_type": "CHOICE_SINGLE",
        },
    }


def _locate_graph(question: str) -> dict:
    return {
        "version": "taskgraph-v1.1",
        "question": question,
        "question_type": "FREE_FORM",
        "inputs": {"image0": {"type": "image", "uri_or_key": "fixture"}},
        "intent": "OTHER",
        "nodes": [
            {
                "id": "n1",
                "op": "LOCATE",
                "inputs": {"image": "$image0"},
                "params": {"target": {"category": "ship", "attributes": {}}},
            }
        ],
        "final": {
            "sources": ["$n1"],
            "question": question,
            "answer_type": "TEXT",
        },
    }


def _runtime(
    *,
    detection: FakeDetectionProvider | None = None,
    counting: CountingSystemProvider | FakeCountingProvider | None = None,
) -> TaskGraphRuntime:
    shared = FakeSemanticVLMProvider({}, "A")
    return TaskGraphRuntime(
        RuntimeProviders(
            detection=detection or FakeDetectionProvider([]),
            semantic_2b=shared,
            route_4b=FakeSemanticVLMProvider({}, "A"),
            retriever=FakeRegionRetriever([]),
            choice=shared,
            counting=counting or FakeCountingProvider([]),
        )
    )


def _native_config() -> dict:
    return {
        "scale": {
            "global": {"enabled": False},
            "native": {"enabled": True, "tile_size": 256, "overlap": 32},
            "fine": {"enabled": False},
        },
        "count": {
            "score_threshold": 0.05,
            "nms_iou": 0.4,
            "cross_scale_iou": 0.5,
            "keep_raw_proposals": True,
        },
        "gate": {"enabled": False},
    }


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


def test_provider_count_uses_tiling_and_global_xyxy() -> None:
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
        provider = CountingSystemProvider(config=_native_config(), detector=recorder)
        request = CountingRequest(
            ImageRef(image_path, 512, 512),
            TargetSpec(category="ship", attributes={}),
            entire=True,
        )
        try:
            counted = provider.count(request)
            assert counted.provider.startswith("counting_system")
            assert counted.count == 3
            assert len(counted.detections.entities) == 3
            boxes = [tuple(item.region.bbox_xyxy_global) for item in counted.detections.entities]
            for blob in blobs:
                assert any(
                    abs(box[0] - blob.bbox[0]) <= 3 and abs(box[1] - blob.bbox[1]) <= 3
                    for box in boxes
                )
            assert counted.metadata["detector_calls"] == len(recorder.calls)
            assert counted.metadata["detector_calls"] >= 1
            assert counted.metadata["entire"] is True
            assert recorder.calls[0].target.category == "ship"
            assert all(call.image.size[0] > 0 for call in recorder.calls)
        finally:
            provider.close()


def test_taskgraph_count_image_fake_e2e_emits_scalar_and_choice() -> None:
    with tempfile.TemporaryDirectory() as raw:
        tmp_path = Path(raw)
        blobs = [Blob((30, 30, 70, 70)), Blob((200, 40, 240, 80)), Blob((40, 200, 80, 240))]
        image_path = write_blob_image(
            str(tmp_path / "count.png"), width=320, height=320, blobs=blobs
        ).uri_or_key
        recorder = RecordingFakeDetector()
        provider = CountingSystemProvider(config=_native_config(), detector=recorder)
        detection = FakeDetectionProvider([[0.0, 0.0, 1.0, 1.0]])
        runtime = _runtime(detection=detection, counting=provider)
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
                    graph=_count_graph(question, options, entire=True),
                )
            )
            scalar = result.store.get("$n1")
            assert isinstance(scalar, ScalarInt)
            assert scalar.value == 3
            assert result.output.choice_id == "C"
            assert provider.count_requests[0].entire is True
            assert len(recorder.calls) >= 1
            assert detection.calls == []
            count_trace = next(item for item in result.trace.nodes if item.node_id == "n1")
            assert count_trace.input_refs == {"image": "$image0"}
            assert "counting_system" in count_trace.provider
        finally:
            runtime.close()


def test_count_entities_skips_counting_provider() -> None:
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
    counting = FakeCountingProvider([[0.0, 0.0, 1.0, 1.0]])
    executor = GraphCountExecutor(counting)
    node = GraphNode.model_validate(
        {
            "id": "n5",
            "op": "COUNT",
            "inputs": {"entities": "$n4"},
            "params": {"target": {"category": "umbrella", "attributes": {}}, "entire": False},
        }
    )
    context = OperatorContext("How many umbrellas?", (), InputComposer())
    outcome = executor.execute(node, {"entities": entities}, context)
    assert isinstance(outcome.value, ScalarInt)
    assert outcome.value.value == 2
    assert outcome.provider == "cardinality"
    assert counting.calls == []

    empty = SelectResult(EntitySet(()), SelectStatus.EMPTY, "geometry")
    empty_out = executor.execute(node, {"entities": empty}, context)
    assert empty_out.value.value == 0
    assert counting.calls == []


def test_count_ambiguous_select_result_is_refused() -> None:
    image = ImageRef("memory://unused", 32, 32)
    selected = EntitySet((Entity(Region(image, (0.0, 0.0, 1.0, 1.0)), "ship", 1.0),))
    counting = FakeCountingProvider()
    executor = GraphCountExecutor(counting)
    node = GraphNode.model_validate(
        {
            "id": "n5",
            "op": "COUNT",
            "inputs": {"entities": "$n4"},
            "params": {"target": {"category": "ship", "attributes": {}}, "entire": False},
        }
    )
    context = OperatorContext("How many ships?", (), InputComposer())
    ambiguous = SelectResult(selected, SelectStatus.AMBIGUOUS, "geometry")
    with pytest.raises(ValueError, match="AMBIGUOUS"):
        executor.execute(node, {"entities": ambiguous}, context)
    assert counting.calls == []


def test_count_image_propagates_entire_true_and_false() -> None:
    counting = FakeCountingProvider([[0.0, 0.0, 1.0, 1.0]])
    executor = GraphCountExecutor(counting)
    image = ImageRef("memory://img", 8, 8)
    context = OperatorContext("How many ships?", (), InputComposer())
    for entire in (True, False):
        counting.calls.clear()
        node = GraphNode.model_validate(
            {
                "id": "n1",
                "op": "COUNT",
                "inputs": {"image": "$image0"},
                "params": {"target": {"category": "ship", "attributes": {}}, "entire": entire},
            }
        )
        outcome = executor.execute(node, {"image": image}, context)
        assert outcome.value.value == 1
        assert counting.calls[0].entire is entire


def test_count_region_forces_effective_entire_false() -> None:
    counting = FakeCountingProvider([[0.0, 0.0, 4.0, 4.0]])
    executor = GraphCountExecutor(counting)
    image = ImageRef("memory://img", 8, 8)
    region = Region(image, (0.0, 0.0, 4.0, 4.0))
    node = GraphNode.model_validate(
        {
            "id": "n1",
            "op": "COUNT",
            "inputs": {"image": "$n0"},
            "params": {"target": {"category": "ship", "attributes": {}}, "entire": True},
        }
    )
    context = OperatorContext("How many ships in this region?", (), InputComposer())
    outcome = executor.execute(node, {"image": region}, context)
    assert counting.calls[0].entire is True
    assert outcome.value.provenance["entire"] is False
    assert counting.calls[0].scope is region


def test_locate_does_not_call_counting_provider() -> None:
    detection = FakeDetectionProvider([[0.0, 0.0, 2.0, 2.0]])
    counting = FakeCountingProvider([[9.0, 9.0, 10.0, 10.0]])
    locate = LocateExecutor(detection, FakeRegionRetriever([]))
    image = ImageRef("memory://img", 16, 16)
    node = GraphNode.model_validate(
        {
            "id": "n1",
            "op": "LOCATE",
            "inputs": {"image": "$image0"},
            "params": {"target": {"category": "ship", "attributes": {}}},
        }
    )
    context = OperatorContext("Where is the ship?", (), InputComposer())
    locate.execute(node, {"image": image}, context)
    assert len(detection.calls) == 1
    assert counting.calls == []


def test_count_does_not_call_detection_provider() -> None:
    detection = FakeDetectionProvider([[0.0, 0.0, 2.0, 2.0]])
    counting = FakeCountingProvider([[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]])
    runtime = _runtime(detection=detection, counting=counting)
    question = "How many ships are there?"
    options = ["A 1", "B 2", "C 3", "D 4"]
    try:
        result = runtime.run(
            RuntimeRequest(
                "count-only",
                "MME_RealWorld_RS",
                "count",
                question,
                ("tests/fixtures/miniature_dataset/images/counting.ppm",),
                tuple(options),
                graph=_count_graph(question, options),
            )
        )
        assert result.store.get("$n1").value == 2
        assert len(counting.calls) == 1
        assert detection.calls == []
    finally:
        runtime.close()


def test_locate_graph_does_not_call_counting_provider() -> None:
    detection = FakeDetectionProvider([[0.0, 0.0, 2.0, 2.0]])
    counting = FakeCountingProvider([[9.0, 9.0, 10.0, 10.0]])
    runtime = _runtime(detection=detection, counting=counting)
    question = "Locate the ship."
    try:
        result = runtime.run(
            RuntimeRequest(
                "locate-only",
                "MME_RealWorld_RS",
                "grounding",
                question,
                ("tests/fixtures/miniature_dataset/images/counting.ppm",),
                graph=_locate_graph(question),
            )
        )
        located = result.store.get("$n1")
        assert isinstance(located, EntitySet)
        assert len(located.entities) == 1
        assert len(detection.calls) == 1
        assert counting.calls == []
    finally:
        runtime.close()


def test_runtime_from_config_separates_detection_and_counting() -> None:
    config = {
        "providers": {
            "detection": {"kind": "fake", "boxes": [[0.0, 0.0, 1.0, 1.0]]},
            "counting": {
                "kind": "counting_system",
                "detector": {"kind": "fake"},
                "scale": {
                    "global": {"enabled": False},
                    "native": {"enabled": True, "tile_size": 1333, "overlap": 200},
                    "fine": {"enabled": False},
                },
                "count": {
                    "score_threshold": 0.20,
                    "nms_iou": 0.40,
                    "cross_scale_iou": 0.50,
                    "keep_raw_proposals": True,
                },
                "gate": {"enabled": False, "threshold": 0.12},
            },
            "semantic_2b": {"kind": "fake", "default": "A"},
            "route_4b": {"kind": "fake", "default": "A"},
            "choice": {"reuse": "semantic_2b"},
            "region_retriever": {"kind": "fake"},
        }
    }
    runtime = runtime_from_config(config)
    try:
        assert runtime.providers.detection.provider_name == "fake_lae"
        assert runtime.providers.counting.provider_name.startswith("counting_system")
        inner = runtime.providers.counting._executor.config
        assert inner["scale"]["native"]["tile_size"] == 1333
        assert inner["scale"]["native"]["overlap"] == 200
        assert inner["scale"]["global"]["enabled"] is False
        assert inner["scale"]["fine"]["enabled"] is False
        assert inner["count"]["score_threshold"] == 0.20
        assert inner["count"]["nms_iou"] == 0.40
        assert inner["count"]["cross_scale_iou"] == 0.50
        assert inner["count"]["keep_raw_proposals"] is True
        assert inner["gate"]["enabled"] is False
        assert inner["gate"]["threshold"] == 0.12
    finally:
        runtime.close()


def test_detection_kind_counting_system_is_rejected() -> None:
    with pytest.raises(ValueError, match="no longer supported"):
        runtime_from_config({"providers": {"detection": {"kind": "counting_system"}}})


def test_tiled_counting_detector_is_rejected() -> None:
    with pytest.raises(ValueError, match="double-tile"):
        CountingSystemProvider.from_config({"detector": {"kind": "tiled"}})


def test_provider_close_releases_executor_and_detector() -> None:
    recorder = RecordingFakeDetector()
    provider = CountingSystemProvider(config=_native_config(), detector=recorder)
    provider.close()
    assert recorder.closed is True


def test_proposal_bridge_maps_local_boxes_to_original_xyxy() -> None:
    ensure_counting_system_importable()
    from counting_system.tiling import Tile
    from PIL import Image

    from counting_system.detector.base import DetectionRequest
    from counting_system.target import build_target

    proposal = RecordingProposalProvider([[1.0, 2.0, 5.0, 6.0]])
    bridge = CountingProposalDetectorBridge(proposal)
    crop = Image.new("RGB", (20, 20), (0, 0, 0))
    tile = Tile(
        tile_id="native_0",
        scale_id="native",
        image=ImageRef("memory://img", 100, 100),
        crop_xyxy=(10.0, 20.0, 30.0, 40.0),
        core_xyxy=(10.0, 20.0, 30.0, 40.0),
        detector_input=20,
        scope_xyxy=(0.0, 0.0, 100.0, 100.0),
    )
    try:
        response = bridge.detect(
            DetectionRequest(image=crop, target=build_target("ship"), tile=tile)
        )
        assert len(proposal.predict_calls) == 1
        assert proposal.predict_calls[0][1] == "ship"
        box = response.detections[0].bbox_xyxy_global
        assert box == (11.0, 22.0, 15.0, 26.0)
        assert response.detections[0].provenance["coordinate_mode"] == (
            "absolute_original_pixel_xyxy"
        )
    finally:
        bridge.close()
        assert proposal.closed is True
