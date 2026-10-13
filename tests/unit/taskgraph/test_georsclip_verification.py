from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from PIL import Image
from scripts.integrations.lae_dino_worker import _request_options, _without_spatial_resize

from sat_rs_vlm.integrations.counting import CountingSystemProvider
from sat_rs_vlm.integrations.counting.detector_bridge import CountingProposalDetectorBridge
from sat_rs_vlm.integrations.counting.proposals import StrategyProposalProvider
from sat_rs_vlm.integrations.detectors.protocol import ProposalResult
from sat_rs_vlm.integrations.retrievers.candidate_verification import (
    GeoRSCLIPCandidateVerifier,
    VerifiedCountingProvider,
    VerifiedDetectionProvider,
)
from sat_rs_vlm.integrations.retrievers.protocol import PromptGroupRetrievalResult
from sat_rs_vlm.taskgraph.input_composer import InputComposer
from sat_rs_vlm.taskgraph.operators import (
    CountExecutor,
    LocateExecutor,
    OperatorContext,
    SelectExecutor,
)
from sat_rs_vlm.taskgraph.providers import (
    CountingRequest,
    CountingResult,
    DetectionRequest,
    DetectionSet,
    FakeRegionRetriever,
    ProposalDetectionAdapter,
    require_complete_count,
)
from sat_rs_vlm.taskgraph.routing import ExecutionMode
from sat_rs_vlm.taskgraph.runtime import RuntimeRequest, fake_runtime, runtime_from_config
from sat_rs_vlm.taskgraph.runtime_types import Entity, EntitySet, ImageRef, Region
from sat_rs_vlm.taskgraph.schema import GraphNode, TargetSpec

CATALOG = yaml.safe_load(Path("configs/prompts/georsclip_major_categories.yaml").read_text("utf-8"))
STRATEGIES = {
    "category_map": {"car": "dense_small", "ship": "dense_small", "building": "large_multi"},
    "dense_small": {"tile_size": 800, "overlap_ratio": 0.15, "score_threshold": 0.2},
    "large_multi": {"grid_size": 2, "overlap_ratio": 0.0, "score_threshold": 0.2},
}


def image(tmp_path, size=(200, 100)):
    path = tmp_path / "synthetic.png"
    Image.new("RGB", size, "white").save(path)
    return ImageRef(str(path), *size)


def candidates(source, count=3, **provenance):
    return EntitySet(
        tuple(
            Entity(
                Region(source, (i * 3.0, 1.0, i * 3.0 + 2, 3.0)),
                "car",
                0.9,
                {"candidate_id": f"original_{i}", "instance_status": "CONFIRMED_INSTANCE"},
            )
            for i in range(count)
        ),
        {"candidate_pool_complete": True, **provenance},
    )


class FakeCLIP:
    model_id = "fake-georsclip"
    region_crop_policy = "square_letterbox_gray"
    batch_size = 2

    def __init__(self, coarse=None, attributes=None, error=False):
        self.coarse = coarse or {}
        self.attributes = attributes or {}
        self.error = error
        self.calls = []
        self.close_calls = 0

    def score_prompt_groups(self, path, groups, boxes):
        self.calls.append((groups, boxes))
        if self.error:
            raise RuntimeError("model failure")
        if "positive" in groups:
            values = {
                "positive": [0.4 + self.coarse.get(int(b[0]), 0.1) for b in boxes],
                "negative": [0.4] * len(boxes),
            }
        else:
            query = groups["qualified"][0]
            margins = self.attributes.get(query, {})
            values = {
                "qualified": [0.4 + margins.get(int(b[0]), 0.1) for b in boxes],
                "unrestricted": [0.4] * len(boxes),
                "contrast": [0.4] * len(boxes),
            }
        return PromptGroupRetrievalResult(values, 0, "fake", self.model_id)

    def close(self):
        self.close_calls += 1


class FixedProvider:
    provider_name = "fake-dino-fused"

    def __init__(self, pool):
        self.pool = pool
        self.calls = []

    def detect(self, request):
        self.calls.append(request)
        return DetectionSet(self.pool, 0, self.provider_name)

    def count(self, request):
        self.calls.append(request)
        return CountingResult(len(self.pool.entities), self.pool, self.provider_name, 0)

    def close(self):
        pass


def verifier(clip=None, **config):
    return GeoRSCLIPCandidateVerifier(clip or FakeCLIP(), CATALOG, config)


def test_reuse_exact_score_formulas_preserve_original_ids_and_detector_score(tmp_path):
    source = image(tmp_path)
    clip = FakeCLIP(coarse={0: 0.2, 3: -0.1, 6: 0.1}, attributes={"red car": {0: 0.1, 6: -0.2}})
    result = verifier(clip).verify_candidates(
        source, source, TargetSpec(category="car", attributes={"color": "red"}), candidates(source)
    )
    assert [d["status"] for d in result.decisions] == ["MATCH", "REJECT", "REJECT"]
    assert result.decisions[0]["mixed_attribute_score"] == pytest.approx(0.1)
    assert result.decisions[0]["combined_score"] == pytest.approx(0.15)
    assert result.matched[0].provenance["candidate_id"] == "original_0"
    assert result.matched[0].score == 0.9
    assert result.metadata["candidate_pool_complete"]
    assert len(clip.calls[0][0]["positive"]) == 21
    assert len(clip.calls[0][0]["negative"]) == 24


@pytest.mark.parametrize(
    "attributes",
    [{"size": "large"}, {"has_part": "chimney"}, {"state": "moving"}, {"color": "green"}],
)
def test_dynamic_attributes_are_scored_with_binary_output(tmp_path, attributes):
    source = image(tmp_path)
    result = verifier().verify_candidates(
        source, source, TargetSpec(category="car", attributes=attributes), candidates(source)
    )
    assert len(result.detections.entities) == 3
    assert result.metadata["counts"]["MATCH"] == 3
    assert set(result.metadata["counts"]) == {"MATCH", "REJECT"}
    assert result.metadata["candidate_pool_complete"]


def test_multiple_configured_attributes_are_and_not_average_acceptance(tmp_path):
    source = image(tmp_path)
    clip = FakeCLIP(attributes={"red car": {0: 0.4}, "moving car": {0: -0.1}})
    result = verifier(
        clip, supported_attributes={"color": ["red"], "state": ["moving"]}
    ).verify_candidates(
        source,
        source,
        TargetSpec(category="car", attributes={"color": "red", "state": "moving"}),
        candidates(source, 1),
    )
    assert result.decisions[0]["status"] == "REJECT"
    assert result.decisions[0]["mixed_attribute_score"] > 0


def test_zero_margin_model_error_and_invalid_scope_remain_auditable(tmp_path):
    source = image(tmp_path)
    zero = verifier(FakeCLIP(coarse={0: 0})).verify_candidates(
        source, source, TargetSpec(category="car"), candidates(source, 1)
    )
    assert zero.decisions[0]["status"] == "REJECT"
    failed = verifier(FakeCLIP(error=True)).verify_candidates(
        source, source, TargetSpec(category="car"), candidates(source)
    )
    assert failed.metadata["counts"]["MATCH"] == 3
    assert failed.metadata["errors"]
    assert len(failed.detections.entities) == 3
    assert not failed.metadata["candidate_pool_complete"]
    outside = verifier().verify_candidates(
        source, Region(source, (30, 30, 40, 40)), TargetSpec(category="car"), candidates(source)
    )
    assert outside.metadata["counts"]["REJECT"] == 3


def test_more_than_batch_limit_no_candidate_disappears_and_no_attribute_category_only(tmp_path):
    source = image(tmp_path, (1000, 100))
    clip = FakeCLIP()
    result = verifier(clip).verify_candidates(
        source, source, TargetSpec(category="car"), candidates(source, 200)
    )
    assert len(result.matched) == 200
    assert len(clip.calls) == 1 and len(clip.calls[0][1]) == 200


def test_empty_complete_pool_and_unknown_category(tmp_path):
    source = image(tmp_path)
    result = verifier().verify_candidates(
        source, source, TargetSpec(category="car"), candidates(source, 0)
    )
    assert result.metadata["candidate_pool_complete"]
    unknown = verifier().verify_candidates(
        source, source, TargetSpec(category="unmapped object"), candidates(source)
    )
    assert unknown.metadata["counts"]["MATCH"] == 3
    assert unknown.metadata["errors"]
    assert not unknown.metadata["candidate_pool_complete"]


@pytest.mark.parametrize(
    "provenance",
    [
        {"candidate_truncated": True},
        {"candidate_pool_complete": False},
        {"instance_resolution_complete": False},
    ],
)
def test_incomplete_dino_diagnostics_do_not_block_integer_output(tmp_path, provenance):
    source = image(tmp_path)
    provider = VerifiedCountingProvider(FixedProvider(candidates(source, **provenance)), verifier())
    result = provider.count(CountingRequest(source, TargetSpec(category="car"), True))
    assert result.count == require_complete_count(result) == 3
    assert result.metadata["candidate_pool_complete"] is False
    assert result.metadata["count_output_available"] is True


def count_node(role="image"):
    return GraphNode.model_validate(
        {
            "id": "n2",
            "op": "COUNT",
            "inputs": {role: "$input"},
            "params": {
                "target": {"category": "car", "attributes": {"color": "red"}},
                "entire": True,
            },
        }
    )


def test_count_image_uses_filtered_count_and_entity_set_does_not_rescore(tmp_path):
    source = image(tmp_path)
    clip = FakeCLIP(attributes={"red car": {3: -0.1, 6: -0.1}})
    raw = FixedProvider(candidates(source))
    provider = VerifiedCountingProvider(raw, verifier(clip))
    context = OperatorContext("How many red cars?", (), InputComposer())
    outcome = CountExecutor(provider).execute(count_node(), {"image": source}, context)
    assert outcome.value.value == 1
    result = provider.count(
        CountingRequest(source, TargetSpec(category="car", attributes={"color": "red"}), True)
    )
    calls = len(clip.calls), len(raw.calls)
    counted = CountExecutor(provider).execute(
        count_node("entities"), {"entities": result.detections}, context
    )
    assert counted.value.value == 1
    assert (len(clip.calls), len(raw.calls)) == calls


def test_direct_count_matches_graph_count_and_parses_modifier_locally(tmp_path):
    source = image(tmp_path)
    clip = FakeCLIP(attributes={"red car": {3: -0.1, 6: -0.1}})
    raw = FixedProvider(candidates(source))
    wrapped = VerifiedCountingProvider(raw, verifier(clip))
    graph_count = (
        CountExecutor(wrapped)
        .execute(
            count_node(),
            {"image": source},
            OperatorContext("How many red cars?", (), InputComposer()),
        )
        .value.value
    )
    runtime = fake_runtime()
    runtime.providers.counting = wrapped
    try:
        result = runtime.run(
            RuntimeRequest(
                "direct",
                "unit",
                "count",
                "How many red cars?",
                (source.uri_or_key,),
                execution_mode=ExecutionMode.DIRECT_DETECTION,
                target_category="red cars",
            )
        )
        assert result.output.value == graph_count == 1
        assert raw.calls[-1].target == TargetSpec(category="car", attributes={"color": "red"})
    finally:
        runtime.close()
    assert clip.close_calls == 1


def test_locate_clip_is_unique_attribute_path_and_disables_old_rerank(tmp_path):
    source = image(tmp_path)
    raw = FixedProvider(candidates(source))
    provider = VerifiedDetectionProvider(raw, verifier())
    node = GraphNode.model_validate(
        {
            "id": "n1",
            "op": "LOCATE",
            "inputs": {"image": "$input"},
            "params": {"target": {"category": "car", "attributes": {"color": "red"}}},
        }
    )
    count = count_node("entities").model_copy(update={"inputs": {"entities": "$n1"}})
    refiner = SimpleNamespace(config=SimpleNamespace(attribute_policy="vlm"), semantic=object())
    outcome = LocateExecutor(provider, FakeRegionRetriever(), refiner=refiner).execute(
        node,
        {"image": source},
        OperatorContext("How many red cars?", (), InputComposer(), graph_nodes=(node, count)),
    )
    assert len(outcome.value.entities) == 3
    assert outcome.trace_metadata["attributes"]["policy"] == "georsclip"
    assert raw.calls[0].use_clip_rerank is False and raw.calls[0].apply_locate_policy is False


def test_selection_of_partial_clip_pool_still_produces_count(tmp_path):
    source = image(tmp_path)
    result = verifier().verify_candidates(
        source,
        source,
        TargetSpec(category="car", attributes={"size": "large"}),
        candidates(source, candidate_pool_complete=False),
    )
    selected = SelectExecutor._selected_like(
        result.detections, (0,), provenance={"selection": "fixture"}
    )
    output = CountExecutor(FixedProvider(candidates(source))).execute(
        count_node("entities"),
        {"entities": selected},
        OperatorContext("count", (), InputComposer()),
    )
    assert output.value.value == 1


def test_runtime_configuration_shares_one_verifier_and_disabled_config_needs_no_model(
    tmp_path, monkeypatch
):
    clip = FakeCLIP()
    monkeypatch.setattr(
        "sat_rs_vlm.integrations.retrievers.registry.create_retriever_provider", lambda *args: clip
    )
    runtime = runtime_from_config({"candidate_verification": {"enabled": True}})
    try:
        assert runtime.providers.detection._verifier is runtime.providers.counting._verifier
        assert runtime.providers.counting.normalize_target(
            TargetSpec(category="blue cars")
        ) == TargetSpec(category="car", attributes={"color": "blue"})
    finally:
        runtime.close()
    assert clip.close_calls == 1
    plain = runtime_from_config({"candidate_verification": {"enabled": False}})
    plain.close()


@pytest.mark.parametrize("category", ["car", "ship"])
def test_dense_tiles_800_without_spatial_resize(category):
    from counting_system.runtime import ImageRef as CountImage
    from counting_system.target import TargetSpec as CountTarget
    from counting_system.tiling import plan_tiles

    source = CountImage("unused", 1700, 900)
    tiles = plan_tiles(
        source,
        (0, 0, 1700, 900),
        CountTarget(category),
        {"detection_strategies": STRATEGIES},
        entire=True,
    )
    assert len(tiles) == 6
    assert all(
        max(tile.crop_size) == 800
        and tile.detector_input == 0
        and tile.spatial_resize_enabled is False
        for tile in tiles
    )
    assert max(t.crop_xyxy[2] for t in tiles) == 1700
    assert max(t.crop_xyxy[3] for t in tiles) == 900


def test_large_quadrants_preserve_aspect_and_default_processor(tmp_path):
    from counting_system.image_ops import crop_tile
    from counting_system.runtime import ImageRef as CountImage
    from counting_system.target import build_target
    from counting_system.tiling import plan_tiles

    source = CountImage("unused", 2000, 1000)
    tiles = plan_tiles(
        source,
        (0, 0, 2000, 1000),
        build_target("building"),
        {"detection_strategies": STRATEGIES},
        entire=True,
    )
    assert len(tiles) == 4
    assert all(
        tile.crop_size == (1000, 500) and tile.spatial_resize_enabled is True for tile in tiles
    )
    with Image.new("RGB", (2000, 1000)) as pil:
        for tile in tiles:
            with crop_tile(pil, tile) as crop:
                assert crop.size == (1000, 500)


class RawDINO:
    provider_name = "fake-raw-dino"
    model_id = "fake"
    inference_batch_size = 1

    def __init__(self):
        self.calls = []

    def predict(self, *args):
        raise AssertionError("strategy must bypass top-K predict")

    def predict_raw(self, path, query, *, options=None):
        with Image.open(path) as pil:
            self.calls.append((pil.size, query, options))
        return ProposalResult(
            [[10, 10, 20, 20]],
            [0.9],
            0,
            self.provider_name,
            self.model_id,
            {"candidate_pool_complete": True},
        )

    def close(self):
        pass


def test_locate_and_count_reuse_tiler_restore_region_and_raw_options(tmp_path):
    source = image(tmp_path, (2000, 1000))
    region = Region(source, (100, 100, 1100, 600))
    raw = RawDINO()
    proposal = StrategyProposalProvider(raw, STRATEGIES)
    detected = ProposalDetectionAdapter(proposal).detect(
        DetectionRequest(
            region, TargetSpec(category="car"), apply_locate_policy=False, use_clip_rerank=False
        )
    )
    counted = CountingSystemProvider(
        config={"detection_strategies": STRATEGIES}, detector=CountingProposalDetectorBridge(raw)
    ).count(CountingRequest(region, TargetSpec(category="car"), False))
    assert [e.region.bbox_xyxy_global for e in detected.detections.entities] == [
        e.region.bbox_xyxy_global for e in counted.detections.entities
    ]
    assert detected.detections.entities[0].region.bbox_xyxy_global == (110, 110, 120, 120)
    assert all(
        call[2] == {"spatial_resize_enabled": False, "score_threshold": 0.2} for call in raw.calls
    )
    assert detected.metadata["detection_strategy"]["name"] == "dense_small"


def test_worker_resize_option_removes_only_resize_transforms():
    original = [
        {"type": "LoadImageFromFile"},
        {"type": "FixScaleResize", "scale": (800, 1333)},
        {"type": "Normalize"},
        {"type": "PackDetInputs"},
    ]
    assert [item["type"] for item in _without_spatial_resize(original)] == [
        "LoadImageFromFile",
        "Normalize",
        "PackDetInputs",
    ]
    assert len(original) == 4
    options = _request_options(
        SimpleNamespace(score_threshold=0.05, top_k=100, nms_threshold=None),
        {
            "options": {
                "score_threshold": 0.2,
                "raw_proposals": True,
                "spatial_resize_enabled": False,
            }
        },
    )
    assert (
        options.score_threshold == 0.2
        and options.raw_proposals
        and not options.spatial_resize_enabled
    )


def test_duplicate_at_tile_boundary_is_fused_before_clip(tmp_path):
    from counting_system.detector.base import DetectionResponse
    from counting_system.runtime import Detection

    source = image(tmp_path, (1450, 700))

    class TileDetector:
        name = "fake-boundary"

        def detect(self, request):
            assert request.score_threshold == 0.2
            return DetectionResponse(
                [
                    Detection(
                        (690, 100, 710, 120),
                        0.9,
                        "car",
                        request.tile.tile_id,
                        request.tile.scale_id,
                    )
                ],
                1,
                self.name,
            )

    counted = CountingSystemProvider(
        config={"detection_strategies": STRATEGIES}, detector=TileDetector()
    ).count(CountingRequest(source, TargetSpec(category="car"), True))
    assert counted.count == 1
    assert counted.metadata["fusion"]["raw"] == 2
    assert counted.metadata["fusion"]["after_fusion"] == 1


def test_direct_count_dynamic_clause_still_produces_integer(tmp_path):
    source = image(tmp_path)
    runtime = fake_runtime()
    runtime.providers.counting = VerifiedCountingProvider(
        FixedProvider(candidates(source)), verifier()
    )
    try:
        output = runtime.run(
            RuntimeRequest(
                "unresolved",
                "unit",
                "count",
                "How many cars with sunroofs?",
                (source.uri_or_key,),
                execution_mode=ExecutionMode.DIRECT_DETECTION,
                target_category="car",
            )
        )
        assert output.output.value == 3
        assert output.trace.telemetry["counting_metadata"]["count_output_available"] is True
    finally:
        runtime.close()


def test_missing_clip_checkpoint_falls_back_to_dino_and_outputs_count(tmp_path):
    source = image(tmp_path)
    runtime = runtime_from_config(
        {
            "providers": {"counting": {"kind": "fake", "boxes": [[0, 0, 2, 2]]}},
            "candidate_verification": {
                "enabled": True,
                "retriever": {
                    "provider": "georsclip",
                    "config": {"checkpoint": str(tmp_path / "missing.pt")},
                },
            },
        }
    )
    try:
        result = runtime.providers.counting.count(
            CountingRequest(source, TargetSpec(category="car"), True)
        )
        assert result.count == require_complete_count(result) == 1
        assert result.metadata["clip_verification"]["errors"]["model_init"]
    finally:
        runtime.close()


@pytest.mark.parametrize("raw_count", [None, 99])
def test_missing_or_noninstance_scalar_uses_available_candidates_for_output(tmp_path, raw_count):
    source = image(tmp_path)

    class NonInstanceProvider(FixedProvider):
        def count(self, request):
            return CountingResult(raw_count, self.pool, self.provider_name, 0)

    result = VerifiedCountingProvider(NonInstanceProvider(candidates(source)), verifier()).count(
        CountingRequest(source, TargetSpec(category="car"), True)
    )
    assert result.count == require_complete_count(result) == 3
    assert result.metadata["input_count_semantics_complete"] is False
    assert len(result.detections.entities) == 3


def test_gray_letterbox_preserves_whole_nonsquare_crop_without_loading_model(tmp_path):
    from sat_rs_vlm.integrations.retrievers.openclip import OpenCLIPRetrieverProvider

    checkpoint = tmp_path / "unused.pt"
    checkpoint.write_bytes(b"no-model-needed")
    provider = OpenCLIPRetrieverProvider(
        {"checkpoint": str(checkpoint), "region_crop_policy": "square_letterbox_gray"}
    )
    with Image.new("RGB", (10, 4), "red") as source:
        crop, _ = provider._crop(source, (0, 0, 10, 4), 0)
        try:
            assert crop.size == (10, 10)
            assert crop.getpixel((0, 0)) == (127, 127, 127)
            assert crop.getpixel((0, 3)) == (255, 0, 0)
            assert crop.getpixel((9, 6)) == (255, 0, 0)
            assert provider._model is None
        finally:
            crop.close()
            provider.close()


def test_clip_failure_keeps_dino_boxes_and_always_returns_integer(tmp_path):
    source = image(tmp_path)
    checker = verifier(FakeCLIP(error=True))
    raw = FixedProvider(candidates(source))
    detected = VerifiedDetectionProvider(raw, checker).detect(
        DetectionRequest(source, TargetSpec(category="car", attributes={"color": "red"}))
    )
    counted = VerifiedCountingProvider(raw, checker).count(
        CountingRequest(source, TargetSpec(category="car", attributes={"color": "red"}), True)
    )
    assert len(detected.detections.entities) == 3
    assert counted.count == require_complete_count(counted) == 3
    assert detected.detections.provenance["resolution_status"] == "MULTIPLE_VALID"
    assert detected.detections.provenance["allow_partial_output"] is True
    assert detected.metadata["clip_verification"]["errors"]


def test_detector_failure_has_empty_output_and_explicit_fallback_diagnostics(tmp_path):
    source = image(tmp_path)

    class FailedDINO(FixedProvider):
        def detect(self, request):
            raise RuntimeError("DINO unavailable")

        def count(self, request):
            raise RuntimeError("DINO unavailable")

    raw = FailedDINO(candidates(source))
    detected = VerifiedDetectionProvider(raw, verifier()).detect(
        DetectionRequest(source, TargetSpec(category="car"))
    )
    counted = VerifiedCountingProvider(raw, verifier()).count(
        CountingRequest(source, TargetSpec(category="car"), True)
    )
    assert detected.detections.entities == ()
    assert counted.count == require_complete_count(counted) == 0
    assert "DINO unavailable" in counted.metadata["detector_error"]


def test_tile_failure_keeps_previously_detected_candidates_for_count_output(tmp_path):
    from counting_system.detector.base import DetectionResponse
    from counting_system.runtime import Detection

    source = image(tmp_path, (1450, 700))

    class PartialDINO:
        name = "partial-dino"
        calls = 0

        def detect(self, request):
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("second tile failed")
            return DetectionResponse(
                [
                    Detection(
                        (10, 10, 20, 20), 0.9, "car", request.tile.tile_id, request.tile.scale_id
                    )
                ],
                1,
                self.name,
            )

    underlying = CountingSystemProvider(
        config={"detection_strategies": STRATEGIES}, detector=PartialDINO()
    )
    result = VerifiedCountingProvider(underlying, verifier()).count(
        CountingRequest(source, TargetSpec(category="car"), True)
    )
    assert result.count == require_complete_count(result) == 1
    assert "second tile failed" in result.metadata["detector_error"]


def test_missing_dino_and_clip_setup_still_return_count_output(tmp_path):
    source = image(tmp_path)
    runtime = runtime_from_config(
        {
            "providers": {
                "detection": {"kind": "lae_dino_lae1m"},
                "counting": {"kind": "counting_system", "detector": {"kind": "lae_dino_lae1m"}},
            },
            "candidate_verification": {"enabled": True},
        }
    )
    try:
        result = runtime.providers.counting.count(
            CountingRequest(source, TargetSpec(category="car"), True)
        )
        assert result.count == require_complete_count(result) == 0
        assert result.metadata["detector_error"]
        assert result.metadata["clip_verification"]["errors"]["model_init"]
    finally:
        runtime.close()
