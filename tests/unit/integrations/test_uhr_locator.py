import builtins
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts.integrations.visrag_worker import score_request

from sat_rs_vlm.integrations.detectors.protocol import ProposalResult
from sat_rs_vlm.integrations.locators.answer import MultiROIRequest
from sat_rs_vlm.integrations.locators.beam import (
    StopPolicyConfig,
    adaptive_beam_select,
    evaluate_stop,
)
from sat_rs_vlm.integrations.locators.config import load_locator_config
from sat_rs_vlm.integrations.locators.fusion import RegionFusion
from sat_rs_vlm.integrations.locators.geometry import (
    bbox_area,
    clamp_bbox,
    expand_with_halo,
    rectangle_union_area,
    spatial_prior,
    subdivide_core,
)
from sat_rs_vlm.integrations.locators.registry import create_locator
from sat_rs_vlm.integrations.locators.router import TaskRouter
from sat_rs_vlm.integrations.locators.scoring.detector import DetectorRegionScorer
from sat_rs_vlm.integrations.locators.scoring.spatial import SpatialRegionScorer
from sat_rs_vlm.integrations.locators.types import LocatorError, SearchRegion
from sat_rs_vlm.integrations.retrievers.cache import retrieval_cache_key
from sat_rs_vlm.integrations.retrievers.protocol import RetrievalError, RetrievalResult
from sat_rs_vlm.integrations.retrievers.registry import create_retriever_provider
from sat_rs_vlm.integrations.retrievers.visrag import (
    _OFFICIAL_QUERY_INSTRUCTION,
    _representations,
)
from sat_rs_vlm.semantics import RelationSpec, TaskSpec


def _region(
    region_id: str,
    box: tuple[float, float, float, float],
    *,
    score: float = 0.0,
    depth: int = 1,
) -> SearchRegion:
    return SearchRegion(
        region_id=region_id,
        parent_id="root",
        depth=depth,
        core_xyxy=box,
        view_xyxy=box,
        score=score,
    )


def test_core_halo_subdivision_and_global_coordinate_clamp() -> None:
    children = subdivide_core((0.0, 0.0, 900.0, 900.0), 3)
    assert len(children) == 9
    assert children[0] == (0.0, 0.0, 300.0, 300.0)
    assert children[-1] == (600.0, 600.0, 900.0, 900.0)
    assert sum(bbox_area(box) for box in children) == pytest.approx(810000.0)
    assert expand_with_halo(children[0], 0.15, 900, 900) == (
        0.0,
        0.0,
        345.0,
        345.0,
    )
    assert clamp_bbox((-5.0, 10.0, 1000.0, 100.0), 900, 900) == (
        0.0,
        10.0,
        900.0,
        100.0,
    )
    assert rectangle_union_area(children) == pytest.approx(810000.0)
    with pytest.raises(LocatorError, match="outside image"):
        clamp_bbox((1000.0, 0.0, 1100.0, 10.0), 900, 900)


def test_detector_bbox_coverage_contributes_to_both_boundary_regions() -> None:
    task = TaskSpec(raw_question="How many aircraft?", operation="count", targets=("aircraft",))
    regions = [
        _region("left", (0.0, 0.0, 50.0, 100.0)),
        _region("right", (50.0, 0.0, 100.0, 100.0)),
    ]
    proposals = ProposalResult(
        boxes_xyxy=[[40.0, 10.0, 60.0, 30.0]],
        scores=[0.8],
        latency_ms=1.0,
        provider="mock",
        model_id="mock",
    )
    result = DetectorRegionScorer().score(task, regions, proposals)
    assert result.scores == pytest.approx((0.4, 0.4))
    assert [item["nonzero_contributions"] for item in result.metadata] == [1, 1]


def test_detector_scorer_is_unavailable_without_a_real_target() -> None:
    task = TaskSpec(raw_question="Why?", operation="open_reasoning")
    result = DetectorRegionScorer().score(task, [_region("a", (0, 0, 10, 10))], None)
    assert result.available is False
    assert result.reason == "task_has_no_detector_target"


def test_spatial_scorer_and_complex_relation_guard() -> None:
    regions = [
        _region("upper-left", (0.0, 0.0, 100.0, 100.0)),
        _region("lower-right", (900.0, 900.0, 1000.0, 1000.0)),
    ]
    task = TaskSpec(
        raw_question="How many aircraft are in the upper-left?",
        operation="count",
        targets=("aircraft",),
        spatial_scope="upper_left",
    )
    result = SpatialRegionScorer().score(task, regions, 1000, 1000)
    assert result.available is True
    assert result.scores[0] > result.scores[1]
    assert spatial_prior(regions[0].core_xyxy, 1000, 1000, "north") > 0.9

    relation = TaskSpec(
        raw_question="Is the ship north of the harbor?",
        operation="relation",
        targets=("ship", "harbor"),
        relations=(RelationSpec("ship", "north_of", "harbor"),),
    )
    guarded = SpatialRegionScorer().score(relation, regions, 1000, 1000)
    assert guarded.available is False
    assert guarded.reason == "object_relation_requires_anchor_bbox"


def test_adaptive_beam_uses_minimum_cumulative_mass() -> None:
    regions = [
        _region("a", (0, 0, 10, 10)),
        _region("b", (10, 0, 20, 10)),
        _region("c", (20, 0, 30, 10)),
    ]
    result = adaptive_beam_select(
        regions,
        [3.0, 2.0, 1.0],
        temperature=1.0,
        cumulative_mass=0.8,
        max_beam=3,
        redundancy_weight=0.0,
    )
    assert result.selected_indices == (0, 1)
    assert result.cumulative_probability >= 0.8
    assert result.cumulative_probability - result.probabilities[1] < 0.8


def test_adaptive_beam_diversity_penalizes_duplicate_region() -> None:
    regions = [
        _region("a", (0, 0, 20, 20)),
        _region("duplicate", (0, 0, 20, 20)),
        _region("diverse", (20, 0, 40, 20)),
    ]
    result = adaptive_beam_select(
        regions,
        [1.0, 0.99, 0.9],
        temperature=1.0,
        cumulative_mass=1.0,
        max_beam=2,
        redundancy_weight=1.0,
    )
    assert result.selected_indices == (0, 2)
    assert result.redundancy_penalties[1] == pytest.approx(1.0)


def test_stop_conditions_report_each_trigger() -> None:
    region = _region("leaf", (0, 0, 100, 100), score=0.5, depth=3)
    decision = evaluate_stop(
        region,
        StopPolicyConfig(
            target_view_size=128,
            max_depth=3,
            min_score_gain=0.1,
            max_regions=9,
            max_area_ratio=1.0,
            posterior_stop_threshold=0.9,
        ),
        evaluated_regions=9,
        inspected_area_ratio=1.1,
        score_gain=0.01,
        posterior_max=0.95,
    )
    assert decision.stop is True
    assert set(decision.reasons) == {
        "target_view_size",
        "max_depth",
        "max_regions",
        "max_area_ratio",
        "min_score_gain",
        "posterior_concentrated",
    }


def test_region_fusion_suppresses_overlap_and_preserves_global_provenance() -> None:
    regions = [
        _region("best", (0, 0, 100, 100), score=0.9),
        _region("duplicate", (5, 5, 100, 100), score=0.8),
        _region("other", (200, 200, 300, 300), score=0.7),
    ]
    fused = RegionFusion(
        {"overlap_iou_threshold": 0.5, "max_regions": 3, "context_margin": 0.1}
    ).fuse(regions, 400, 400)
    assert [region.region_id for region in fused] == ["best", "other"]
    assert fused[0].view_xyxy == (0.0, 0.0, 110.0, 110.0)
    assert fused[0].metadata["coordinate_mode"] == "absolute_original_pixel_xyxy"


def test_answer_model_boundary_requires_aligned_global_roi_provenance(tmp_path: Path) -> None:
    request = MultiROIRequest(
        image_path=tmp_path / "image.png",
        question="How many aircraft?",
        regions_xyxy=((0.0, 0.0, 10.0, 10.0),),
        region_provenance=({"region_id": "root.0"},),
    )
    assert request.regions_xyxy[0] == (0.0, 0.0, 10.0, 10.0)
    with pytest.raises(LocatorError, match="length mismatch"):
        MultiROIRequest(
            image_path=tmp_path / "image.png",
            question="How many aircraft?",
            regions_xyxy=((0.0, 0.0, 10.0, 10.0),),
            region_provenance=({}, {}),
        )


def test_router_routes_by_task_not_dataset_name() -> None:
    router = TaskRouter()
    count = router.route(
        TaskSpec(
            raw_question="How many aircraft?",
            operation="count",
            targets=("aircraft",),
        )
    )
    assert count.route == "detector_first"
    assert count.use_detector and count.use_retrieval
    direct = router.route(
        TaskSpec(
            raw_question="What is here in bbox [0,0,10,10]?",
            operation="grounding",
            given_bbox=(0.0, 0.0, 10.0, 10.0),
        )
    )
    assert direct.bypass_locator and direct.route == "given_bbox_direct"
    unknown = router.route(TaskSpec(raw_question="Question?", operation="unknown"))
    assert unknown.use_retrieval and not unknown.use_detector


def test_retrieval_result_validation_and_mock_order(tmp_path: Path) -> None:
    with pytest.raises(RetrievalError, match="finite"):
        RetrievalResult(
            scores=[math.nan],
            latency_ms=1.0,
            provider="x",
            model_id="y",
        )
    with pytest.raises(RetrievalError, match="length mismatch"):
        RetrievalResult(
            scores=[1.0],
            latency_ms=1.0,
            provider="x",
            model_id="y",
        ).validate_length(2)
    provider = create_retriever_provider("mock", {"scores": [0.3, 0.8]})
    result = provider.score_regions(
        tmp_path / "not-read-by-mock.png",
        "query",
        [(0, 0, 10, 10), (10, 10, 20, 20)],
    )
    assert result.scores == [0.3, 0.8]


def test_mock_registry_does_not_import_heavy_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "transformers" or name.startswith("transformers."):
            raise AssertionError("mock registry imported transformers")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    provider = create_retriever_provider("mock", {})
    assert provider.provider_name == "mock"


def test_visrag_registry_is_lazy_until_first_score(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "transformers" or name.startswith("transformers."):
            raise AssertionError("VisRAG loaded transformers during construction")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    provider = create_retriever_provider("visrag", {"model_path": str(tmp_path)})
    assert provider.provider_name == "visrag"
    assert provider.is_loaded is False


def test_visrag_uses_official_weighted_pooling_and_normalization() -> None:
    torch = pytest.importorskip("torch")
    hidden = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]])
    attention_mask = torch.tensor([[1, 1]])
    representations = _representations(
        SimpleNamespace(last_hidden_state=hidden, attention_mask=attention_mask),
        torch,
    )
    expected = torch.tensor([[1.0, 4.0]])
    expected = torch.nn.functional.normalize(expected, p=2, dim=1)
    assert torch.allclose(representations, expected)


def test_visrag_default_query_instruction_matches_official_model_card(tmp_path: Path) -> None:
    provider = create_retriever_provider("visrag", {"model_path": str(tmp_path)})
    assert provider.query_instruction == _OFFICIAL_QUERY_INSTRUCTION


def test_visrag_sidecar_is_lazy_and_worker_protocol_is_json_native(tmp_path: Path) -> None:
    provider = create_retriever_provider(
        "visrag",
        {
            "model_path": str(tmp_path),
            "runtime": "sidecar",
            "worker_python": "isolated-python",
        },
    )
    try:
        assert provider.is_loaded is False
        assert provider._delegate._client.command[0] == "isolated-python"
    finally:
        provider.close()

    class FakeProvider:
        def score_regions(self, image_path, query, regions_xyxy):
            assert image_path == Path("image.png")
            assert query == "airport"
            return RetrievalResult(
                scores=[0.25] * len(regions_xyxy),
                latency_ms=2.0,
                provider="visrag",
                model_id="fixture",
                metadata={"raw_scores": [0.25]},
            )

    response = score_request(
        FakeProvider(),
        {
            "id": "request-1",
            "image": "image.png",
            "query": "airport",
            "regions_xyxy": [[0, 0, 10, 10]],
        },
    )
    assert response["result"]["scores"] == [0.25]
    assert json.loads(json.dumps(response))["status"] == "ok"


def test_retrieval_cache_key_covers_query_bbox_model_and_parameters(tmp_path: Path) -> None:
    image = tmp_path / "image.bin"
    image.write_bytes(b"image")
    base = {
        "image_path": image,
        "region_xyxy": (0.0, 0.0, 10.0, 10.0),
        "query": "aircraft",
        "provider": "mock",
        "model_identity": {"id": "one"},
        "parameters": {"temperature": 1.0},
    }
    key = retrieval_cache_key(**base)
    assert key != retrieval_cache_key(**{**base, "query": "ship"})
    assert key != retrieval_cache_key(**{**base, "region_xyxy": (1, 0, 10, 10)})
    assert key != retrieval_cache_key(**{**base, "model_identity": {"id": "two"}})
    assert key != retrieval_cache_key(**{**base, "parameters": {"temperature": 2.0}})


def test_unselected_real_provider_profiles_do_not_require_environment() -> None:
    config = load_locator_config("configs/locator/uhr_hierarchical.yaml")
    locator = create_locator("hierarchical", config)
    try:
        assert locator.detector_provider.provider_name == "mock"
        assert locator.retriever_provider.provider_name == "mock"
    finally:
        locator.close()


def test_selected_visrag_profile_uses_configured_isolated_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VISRAG_MODEL_PATH", str(tmp_path))
    monkeypatch.setenv("VISRAG_PYTHON", sys.executable)
    config = load_locator_config("configs/locator/uhr_hierarchical.yaml")
    config["detector"]["enabled"] = False
    config["scorers"]["detector"]["enabled"] = False
    config["retriever"]["provider"] = "visrag"
    locator = create_locator("hierarchical", config)
    try:
        assert locator.retriever_provider.runtime == "sidecar"
        assert locator.retriever_provider.is_loaded is False
    finally:
        locator.close()
