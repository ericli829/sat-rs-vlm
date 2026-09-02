from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw
from pydantic import ValidationError

from sat_rs_vlm.integrations.locators.config import load_locator_config
from sat_rs_vlm.integrations.locators.registry import create_locator
from sat_rs_vlm.taskgraph.executor import (
    CapabilityRouter,
    ExecutorBinding,
    TaskGraphExecutionError,
)
from sat_rs_vlm.taskgraph.input_composer import InputComposer
from sat_rs_vlm.taskgraph.operators import (
    GeometryExecutor,
    OperatorContext,
    SelectExecutor,
    SemanticExecutor,
)
from sat_rs_vlm.taskgraph.providers import (
    FakeRegionRetriever,
    FakeSemanticVLMProvider,
    LocatorRegionRetrieverAdapter,
    RegionCandidate,
    RegionCandidates,
    RegionRetrievalRequest,
    ScoredGridRegionRetrieverAdapter,
)
from sat_rs_vlm.taskgraph.runtime_types import (
    Entity,
    EntitySet,
    ImageRef,
    Label,
    Region,
    RegionSet,
    RouteContext,
    ScalarInt,
    SelectResult,
    SelectStatus,
)
from sat_rs_vlm.taskgraph.schema import GraphNode, OperatorName


def _node(
    operator: str,
    inputs: dict[str, str | list[str]],
    params: dict[str, object],
) -> GraphNode:
    return GraphNode.model_validate(
        {"id": "n1", "op": operator, "inputs": inputs, "params": params}
    )


def _image(path: Path, size: tuple[int, int] = (200, 120)) -> ImageRef:
    Image.new("RGB", size, "white").save(path)
    return ImageRef(str(path), width=size[0], height=size[1])


def _entity(
    image: ImageRef, box: tuple[float, float, float, float], score: float | None = None
) -> Entity:
    return Entity(Region(image, box), "object", score)


@pytest.mark.parametrize(
    ("operator", "inputs", "params"),
    [
        (
            "LOCATE",
            {"image": "$image0", "region": "$n1"},
            {"target": {"category": "ship", "attributes": {}}},
        ),
        (
            "COUNT",
            {"image": "$image0", "entities": "$n1"},
            {"target": {"category": "ship", "attributes": {}}, "entire": True},
        ),
        (
            "COUNT",
            {},
            {"target": {"category": "ship", "attributes": {}}, "entire": True},
        ),
        ("RELATION", {"subject": "$n1"}, {}),
    ],
)
def test_schema_rejects_operator_input_contract_violations(
    operator: str,
    inputs: dict[str, str | list[str]],
    params: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        _node(operator, inputs, params)


def test_schema_accepts_vlm_reason_evidence_list_and_marks_match_choice_deprecated() -> None:
    node = _node(
        "VLM_REASON",
        {"evidence": ["$n1", "$n2"]},
        {"question": "$question"},
    )
    assert node.inputs["evidence"] == ["$n1", "$n2"]
    legacy = _node(
        "MATCH_CHOICE",
        {"value": "$n1"},
        {"choices": "$choices"},
    )
    assert legacy.deprecated is True


def test_runtime_contract_rejects_abs_diff_wrong_type(tmp_path: Path) -> None:
    composer = InputComposer(tmp_path / "inputs")
    node = _node("ABS_DIFF", {"a": "$n1", "b": "$n2"}, {})
    router = CapabilityRouter({OperatorName.ABS_DIFF: ExecutorBinding(GeometryExecutor())})
    try:
        with pytest.raises(TaskGraphExecutionError) as error:
            router.execute(
                node,
                {"a": Label("not-an-int"), "b": ScalarInt(2)},
                OperatorContext("difference", (), composer),
            )
        assert error.value.details["provider"] == "input_contract"
        assert "expected" in str(error.value.details["exception"])
    finally:
        composer.close()


def test_fake_retriever_clips_candidates_to_region_and_nested_scope(tmp_path: Path) -> None:
    image = _image(tmp_path / "scope.png", (100, 80))
    parent = Region(image, (10, 10, 90, 70))
    child = Region(image, (30, 20, 50, 40))
    retriever = FakeRegionRetriever(
        [
            ((0, 0, 5, 5), 1.0),
            ((25, 15, 40, 30), 0.9),
            ((45, 35, 60, 50), 0.8),
        ]
    )
    result = retriever.retrieve(RegionRetrievalRequest(parent, "target", search_scope=child))
    assert [candidate.region.bbox_xyxy_global for candidate in result.candidates] == [
        (30.0, 20.0, 40.0, 30.0),
        (45.0, 35.0, 50.0, 40.0),
    ]
    assert all(
        child.bbox_xyxy_global[0] <= candidate.region.bbox_xyxy_global[0]
        and child.bbox_xyxy_global[1] <= candidate.region.bbox_xyxy_global[1]
        and candidate.region.bbox_xyxy_global[2] <= child.bbox_xyxy_global[2]
        and candidate.region.bbox_xyxy_global[3] <= child.bbox_xyxy_global[3]
        for candidate in result.candidates
    )
    with pytest.raises(ValueError, match="contained"):
        RegionRetrievalRequest(parent, "target", search_scope=Region(image, (0, 0, 20, 20)))


def test_locator_retriever_maps_local_crop_boxes_to_global(tmp_path: Path) -> None:
    image = _image(tmp_path / "locator.png", (100, 80))

    class Locator:
        provider_name = "recording_locator"

        def __init__(self) -> None:
            self.seen_size: tuple[int, int] | None = None

        def locate(self, image_path: Path, query: str) -> SimpleNamespace:
            with Image.open(image_path) as source:
                self.seen_size = source.size
            return SimpleNamespace(
                regions_xyxy=((1, 2, 5, 6),),
                scores=(0.9,),
                region_details=({},),
                latency_ms={"total": 1.0},
            )

        def close(self) -> None:
            return None

    locator = Locator()
    scope = Region(image, (20, 10, 60, 50))
    result = LocatorRegionRetrieverAdapter(locator).retrieve(
        RegionRetrievalRequest(scope, "marker")
    )
    assert locator.seen_size == (40, 40)
    assert result.candidates[0].region.bbox_xyxy_global == (21.0, 12.0, 25.0, 16.0)
    assert result.candidates[0].provenance["local_bbox_xyxy"] == [1, 2, 5, 6]


def test_hierarchical_locator_adapter_preserves_nested_global_scope(tmp_path: Path) -> None:
    image = _image(tmp_path / "nested-uhr.png", (200, 120))
    parent = Region(image, (100.0, 0.0, 200.0, 60.0))
    config = load_locator_config("configs/locator/uhr_hierarchical.yaml")
    config["search"].update({"target_view_size": 1, "max_depth": 1})
    config["scorers"]["detector"]["enabled"] = False
    config["scorers"]["spatial"]["enabled"] = False
    config["fusion"].update({"max_regions": 4, "score_threshold": -1.0})
    locator = create_locator("hierarchical", config)
    try:
        result = LocatorRegionRetrieverAdapter(locator).retrieve(
            RegionRetrievalRequest(parent, "harbor", search_scope=parent)
        )
    finally:
        locator.close()

    assert result.provider == "uhr_hierarchical"
    assert result.metadata["provider_provenance"]["retriever"]["provider"] == "mock"
    assert result.candidates
    assert all(
        100.0 <= candidate.region.bbox_xyxy_global[0]
        and candidate.region.bbox_xyxy_global[1] >= 0.0
        and candidate.region.bbox_xyxy_global[2] <= 200.0
        and candidate.region.bbox_xyxy_global[3] <= 60.0
        for candidate in result.candidates
    )
    assert all(
        candidate.provenance["global_bbox_xyxy"] == list(candidate.region.bbox_xyxy_global)
        for candidate in result.candidates
    )


def test_region_candidates_reject_non_finite_values(tmp_path: Path) -> None:
    image = _image(tmp_path / "candidate.png", (32, 32))
    region = Region(image, (0, 0, 16, 16))

    with pytest.raises(ValueError, match="relevance_score"):
        RegionCandidate(region, float("nan"))
    with pytest.raises(ValueError, match="latency_ms"):
        RegionCandidates((), "fixture", float("inf"))
    with pytest.raises(ValueError, match="provider"):
        RegionCandidates((), " ", 0.0)


def test_scored_grid_retriever_uses_region_scope(tmp_path: Path) -> None:
    image = _image(tmp_path / "grid.png", (120, 100))
    scope = Region(image, (20, 10, 80, 70))

    class Scorer:
        provider_name = "recording_scorer"

        def __init__(self) -> None:
            self.boxes: list[tuple[float, float, float, float]] = []

        def score_regions(
            self,
            image_path: Path,
            query: str,
            regions_xyxy: list[tuple[float, float, float, float]],
        ) -> SimpleNamespace:
            self.boxes = list(regions_xyxy)
            return SimpleNamespace(
                scores=[float(index) for index in range(len(regions_xyxy))],
                model_id="fake",
                latency_ms=0.0,
            )

        def close(self) -> None:
            return None

    scorer = Scorer()
    result = ScoredGridRegionRetrieverAdapter(scorer, grid_size=2).retrieve(
        RegionRetrievalRequest(scope, "object")
    )
    assert scorer.boxes[0] == (20.0, 10.0, 50.0, 40.0)
    assert all(
        scope.bbox_xyxy_global[0] <= candidate.region.bbox_xyxy_global[0]
        and candidate.region.bbox_xyxy_global[2] <= scope.bbox_xyxy_global[2]
        for candidate in result.candidates
    )


def test_scored_grid_retriever_supports_overlapping_windows(tmp_path: Path) -> None:
    image = _image(tmp_path / "sliding.png", (100, 100))

    class Scorer:
        provider_name = "recording_scorer"

        def __init__(self) -> None:
            self.boxes: list[tuple[float, float, float, float]] = []

        def score_regions(
            self,
            image_path: Path,
            query: str,
            regions_xyxy: list[tuple[float, float, float, float]],
        ) -> SimpleNamespace:
            self.boxes = list(regions_xyxy)
            return SimpleNamespace(
                scores=[float(index) for index in range(len(regions_xyxy))],
                model_id="fake",
                latency_ms=0.0,
                metadata={"generation_used": False},
            )

        def close(self) -> None:
            return None

    scorer = Scorer()
    result = ScoredGridRegionRetrieverAdapter(
        scorer,
        grid_size=3,
        candidate_window_ratio=0.5,
    ).retrieve(RegionRetrievalRequest(image, "harbor"))
    assert scorer.boxes[0] == (0.0, 0.0, 50.0, 50.0)
    assert scorer.boxes[4] == (25.0, 25.0, 75.0, 75.0)
    assert scorer.boxes[-1] == (50.0, 50.0, 100.0, 100.0)
    assert len(result.candidates) == 5
    assert result.candidates[0].provenance["candidate_geometry"] == {
        "layout": "uniform_sliding_grid",
        "window_ratio": 0.5,
        "overlapping": True,
    }


def test_entity_set_composer_keeps_single_and_clustered_evidence_local(tmp_path: Path) -> None:
    image = _image(tmp_path / "uhr_clustered.png", (4096, 4096))
    composer = InputComposer(
        tmp_path / "localized_inputs",
        candidate_halo_ratio=0.2,
        entity_set_union_area_threshold=0.25,
    )
    try:
        single = EntitySet((_entity(image, (1900, 1900, 1940, 1940), 0.9),))
        single_input = composer.compose_named(
            {"evidence": single}, question="Inspect the supplied target."
        )
        with Image.open(single_input.visual_inputs[0]) as rendered:
            assert rendered.width < 4096 and rendered.height < 4096

        clustered = EntitySet(
            (
                _entity(image, (1800, 1800, 1840, 1840), 0.9),
                _entity(image, (1900, 1850, 1940, 1890), 0.8),
                _entity(image, (2000, 1900, 2040, 1940), 0.7),
            )
        )
        clustered_input = composer.compose_named(
            {"evidence": clustered}, question="Compare the supplied candidates."
        )
        metadata = clustered_input.metadata["entity_sets"][0]
        assert metadata["strategy"] == "union_crop"
        assert metadata["whole_image_visual_used"] is False
        assert metadata["crop_count"] == 1
        assert metadata["union_bbox_xyxy_global"] == [1800, 1800, 2040, 1940]
        canvas = metadata["canvases"][0]
        assert canvas["crop_bbox_xyxy_global"] != [0, 0, 4096, 4096]
        assert len(canvas["candidate_boxes"]) == 3
        with Image.open(clustered_input.visual_inputs[0]) as rendered:
            assert rendered.width < 4096 and rendered.height < 4096
    finally:
        composer.close()


def test_entity_set_composer_uses_bounded_multi_crop_for_distributed_candidates(
    tmp_path: Path,
) -> None:
    image = _image(tmp_path / "uhr_distributed.png", (4096, 4096))
    entities = EntitySet(
        (
            _entity(image, (40, 60, 100, 120), 0.9),
            _entity(image, (3950, 3930, 4010, 3990), 0.8),
        )
    )
    composer = InputComposer(
        tmp_path / "distributed_inputs",
        candidate_halo_ratio=0.2,
        entity_set_union_area_threshold=0.25,
        entity_set_max_side=512,
    )
    try:
        model_input = composer.compose_named(
            {"evidence": entities}, question="Compare the supplied candidates."
        )
        metadata = model_input.metadata["entity_sets"][0]
        assert metadata["strategy"] == "bounded_multi_crop"
        assert metadata["whole_image_visual_used"] is False
        assert metadata["crop_count"] == 2
        assert len(model_input.visual_inputs) == 2
        for path, canvas in zip(
            model_input.visual_inputs, metadata["canvases"], strict=True
        ):
            assert canvas["crop_bbox_xyxy_global"] != [0, 0, 4096, 4096]
            with Image.open(path) as rendered:
                assert max(rendered.size) <= 512
    finally:
        composer.close()


def test_entity_set_composer_caps_distributed_crops_with_score_top_k(tmp_path: Path) -> None:
    image = _image(tmp_path / "uhr_many_distributed.png", (4096, 4096))
    entities = EntitySet(
        tuple(
            _entity(
                image,
                (40 + index * 700, 60 + index * 650, 100 + index * 700, 120 + index * 650),
                score,
            )
            for index, score in enumerate((0.2, 0.95, 0.4, 0.8, 0.7))
        )
    )
    composer = InputComposer(
        tmp_path / "many_distributed_inputs",
        entity_set_union_area_threshold=0.05,
        entity_set_max_crops=2,
    )
    try:
        model_input = composer.compose_named(
            {"evidence": entities}, question="Compare the highest-scoring candidates."
        )
        metadata = model_input.metadata["entity_sets"][0]
        assert metadata["strategy"] == "bounded_multi_crop"
        assert metadata["requested_candidate_count"] == 5
        assert metadata["selected_candidate_indices"] == [1, 3]
        assert metadata["omitted_candidate_indices"] == [0, 2, 4]
        assert metadata["selection_policy"] == "score_descending_then_source_index_top_k"
        assert metadata["crop_count"] == 2
        assert len(model_input.visual_inputs) == 2
    finally:
        composer.close()


def test_candidate_canvas_is_local_and_has_stable_global_mapping(tmp_path: Path) -> None:
    image = _image(tmp_path / "canvas.png", (1000, 800))
    candidates = EntitySet(
        (
            _entity(image, (400, 300, 430, 340), 0.8),
            _entity(image, (460, 310, 490, 350), 0.7),
            _entity(image, (520, 320, 550, 360), 0.6),
        )
    )
    reference = _entity(image, (440, 370, 500, 410), 0.9)
    composer = InputComposer(tmp_path / "composed", candidate_halo_ratio=0.2)
    try:
        model_input = composer.compose_named(
            {"candidates": candidates, "reference": reference},
            question="Which candidates are next to REF?",
        )
        assert model_input.visual_roles == ("CANDIDATE_CANVAS",)
        with Image.open(model_input.visual_inputs[0]) as canvas:
            assert canvas.width < 1000 and canvas.height < 800
        mapping = model_input.metadata["candidate_mapping"]
        assert list(mapping) == ["A", "B", "C"]
        assert mapping["B"]["bbox_xyxy_global"] == [460.0, 310.0, 490.0, 350.0]
        assert model_input.metadata["role_mapping"]["reference"][0]["id"] == "REF"
        assert "CANDIDATES: A/B/C" in model_input.structured_context
        assert "REFERENCE: REF" in model_input.structured_context
    finally:
        composer.close()


def test_fuzzy_select_restores_letter_ids_to_entities(tmp_path: Path) -> None:
    image = _image(tmp_path / "select.png", (600, 400))
    candidates = EntitySet(
        tuple(
            _entity(image, (100 + index * 40, 100, 125 + index * 40, 130), 0.9 - index * 0.1)
            for index in range(3)
        )
    )
    reference = _entity(image, (180, 160, 220, 200), 0.95)
    provider = FakeSemanticVLMProvider({"selection": "A, C"})
    composer = InputComposer(tmp_path / "select_inputs")
    node = _node(
        "SELECT",
        {"candidates": "$n1", "reference": "$n2"},
        {"mode": "RELATION", "relation": "NEXT_TO"},
    )
    try:
        result = SelectExecutor(provider).execute(
            node,
            {"candidates": candidates, "reference": reference},
            OperatorContext("select", (), composer),
        )
        assert isinstance(result.value, SelectResult)
        assert result.value.status is SelectStatus.OK
        assert result.value.method == "qwen3_vl_kv_cached_choice"
        assert isinstance(result.value.selected, EntitySet)
        assert [item.region.bbox_xyxy_global for item in result.value.selected.entities] == [
            candidates.entities[0].region.bbox_xyxy_global,
            candidates.entities[2].region.bbox_xyxy_global,
        ]
        assert result.value.provenance["candidate_ids"] == ["candidate_0001", "candidate_0003"]
        assert provider.choice_calls[0].model_input.metadata["canvas_kind"] == "CANDIDATE_CANVAS"
        assert provider.calls == []
    finally:
        composer.close()


def test_relation_canvas_preserves_subject_and_reference_roles(tmp_path: Path) -> None:
    image = _image(tmp_path / "relation.png", (700, 500))
    subject = EntitySet((_entity(image, (200, 180, 240, 220), 0.8),))
    reference = EntitySet((_entity(image, (280, 190, 330, 240), 0.9),))
    provider = FakeSemanticVLMProvider({"relation": "LEFT_OF"})
    composer = InputComposer(tmp_path / "relation_inputs")
    node = _node("RELATION", {"subject": "$n1", "reference": "$n2"}, {})
    try:
        result = SemanticExecutor(provider).execute(
            node,
            {"subject": subject, "reference": reference},
            OperatorContext("relation", (), composer),
        )
        model_input = provider.semantic_calls[0].model_input
        assert isinstance(result.value, Label)
        assert result.value.value == "LEFT_OF"
        assert result.value.provenance["method"] == "kv_cached_categorical"
        assert result.value.provenance["cache_reused"] is True
        assert provider.calls == []
        assert model_input.visual_roles == ("RELATION_CANVAS",)
        roles = model_input.metadata["role_mapping"]
        assert roles["subject"][0]["id"] == "SUBJECT"
        assert roles["reference"][0]["id"] == "REFERENCE"
        assert "SUBJECT: SUBJECT" in model_input.structured_context
        assert "REFERENCE: REFERENCE" in model_input.structured_context
    finally:
        composer.close()


@pytest.mark.parametrize("mode", ["ROW", "COLUMN", "CLUSTER"])
def test_group_assigns_deterministic_geometry_groups(tmp_path: Path, mode: str) -> None:
    image = _image(tmp_path / f"group_{mode}.png")
    if mode == "ROW":
        boxes = ((80, 60, 90, 70), (20, 20, 30, 30), (60, 20, 70, 30), (20, 60, 30, 70))
    elif mode == "COLUMN":
        boxes = ((60, 80, 70, 90), (20, 20, 30, 30), (20, 60, 30, 70), (60, 20, 70, 30))
    else:
        boxes = ((20, 20, 30, 30), (35, 25, 45, 35), (150, 80, 160, 90), (165, 85, 175, 95))
    entities = EntitySet(tuple(_entity(image, box) for box in boxes))
    result = GeometryExecutor._group(entities, mode)
    assert result.provenance["group_count"] == 2
    assert [entity.provenance["group_id"] for entity in result.entities] == [0, 0, 1, 1]
    assert len(result.provenance["groups"]) == 2


@pytest.mark.parametrize("mode", ["ROW", "COLUMN", "CLUSTER"])
def test_group_handles_empty_and_single_entity(tmp_path: Path, mode: str) -> None:
    image = _image(tmp_path / f"group_edge_{mode}.png")
    empty = GeometryExecutor._group(EntitySet(()), mode)
    single = GeometryExecutor._group(
        EntitySet((_entity(image, (20, 20, 40, 40)),)),
        mode,
    )
    assert empty.entities == ()
    assert empty.provenance["group_count"] == 0
    assert single.provenance["group_count"] == 1
    assert single.entities[0].provenance["group_id"] == 0


def test_group_row_boundary_is_deterministic(tmp_path: Path) -> None:
    image = _image(tmp_path / "group_boundary.png")
    # Median height is 20, so the inclusive row tolerance is exactly 15 px.
    entities = EntitySet(
        (
            _entity(image, (10, 10, 30, 30)),
            _entity(image, (50, 25, 70, 45)),
            _entity(image, (90, 41, 110, 61)),
        )
    )
    result = GeometryExecutor._group(entities, "ROW")
    assert [entity.provenance["group_id"] for entity in result.entities] == [0, 0, 1]


@pytest.mark.parametrize("shape", ["rectangle", "circle", "partial_border"])
def test_find_marker_returns_connected_components(tmp_path: Path, shape: str) -> None:
    path = tmp_path / f"marker_{shape}.png"
    canvas = Image.new("RGB", (100, 80), "white")
    draw = ImageDraw.Draw(canvas)
    if shape == "rectangle":
        draw.rectangle((15, 15, 35, 35), outline="red", width=3)
    elif shape == "circle":
        draw.ellipse((15, 15, 35, 35), outline="red", width=3)
    else:
        draw.rectangle((0, 10, 20, 30), outline="red", width=3)
    draw.rectangle((60, 40, 75, 55), outline="red", width=3)
    canvas.save(path)
    result = GeometryExecutor._marker(ImageRef(str(path)), "red")
    assert isinstance(result, RegionSet)
    assert len(result.regions) == 2
    assert result.provenance["component_count"] == 2


def test_find_marker_filters_pixel_noise_and_oversized_color_regions(tmp_path: Path) -> None:
    path = tmp_path / "marker_filter.png"
    canvas = Image.new("RGB", (200, 120), "white")
    draw = ImageDraw.Draw(canvas)
    draw.point((2, 2), fill="red")
    draw.rectangle((0, 50, 199, 119), fill="red")
    draw.ellipse((25, 15, 45, 35), outline="red", width=3)
    draw.ellipse((80, 15, 100, 35), outline="red", width=3)
    canvas.save(path)

    result = GeometryExecutor._marker(ImageRef(str(path)), "red", "circle")
    assert [region.bbox_xyxy_global for region in result.regions] == [
        (25.0, 15.0, 46.0, 36.0),
        (80.0, 15.0, 101.0, 36.0),
    ]
    assert result.provenance["component_count"] == 2
    assert result.provenance["rejected_component_count"] == 2


@pytest.mark.parametrize("count", [1, 2])
def test_find_marker_keeps_one_or_two_nearby_circles(tmp_path: Path, count: int) -> None:
    path = tmp_path / f"marker_nearby_{count}.png"
    canvas = Image.new("RGB", (100, 70), "white")
    draw = ImageDraw.Draw(canvas)
    draw.ellipse((10, 15, 30, 35), outline="red", width=3)
    if count == 2:
        draw.ellipse((33, 15, 53, 35), outline="red", width=3)
    canvas.save(path)
    result = GeometryExecutor._marker(ImageRef(str(path)), "red", "circle")
    assert len(result.regions) == count


def test_region_from_bbox_scales_dataset_coordinates(tmp_path: Path) -> None:
    image = _image(tmp_path / "bbox.png", (200, 100))
    node = _node(
        "REGION_FROM_BBOX",
        {"image": "$image0"},
        {"bbox": [10, 5, 20, 15], "image_size": [100, 50]},
    )
    composer = InputComposer(tmp_path / "bbox_inputs")
    try:
        result = GeometryExecutor().execute(
            node,
            {"image": image},
            OperatorContext("bbox", (), composer),
        )
        assert isinstance(result.value, Region)
        assert result.value.bbox_xyxy_global == (20.0, 10.0, 40.0, 30.0)
        assert result.value.provenance["scale_xy"] == [2.0, 2.0]
    finally:
        composer.close()


def test_region_from_bbox_uses_absolute_coordinates_and_clips_nested_scope(
    tmp_path: Path,
) -> None:
    image = _image(tmp_path / "nested_bbox.png", (300, 200))
    scope = Region(image, (100, 50, 250, 180))
    composer = InputComposer(tmp_path / "nested_bbox_inputs")
    try:
        clipped = (
            GeometryExecutor()
            .execute(
                _node(
                    "REGION_FROM_BBOX",
                    {"image": "$n1"},
                    {"bbox": [80, 40, 180, 100], "image_size": None},
                ),
                {"image": scope},
                OperatorContext("bbox", (), composer),
            )
            .value
        )
        assert isinstance(clipped, Region)
        assert clipped.bbox_xyxy_global == (100.0, 50.0, 180.0, 100.0)
        assert clipped.provenance["coordinates"] == "absolute_xyxy_global"

        with pytest.raises(ValueError, match="positive area before clipping"):
            GeometryExecutor().execute(
                _node(
                    "REGION_FROM_BBOX",
                    {"image": "$n1"},
                    {"bbox": [180, 100, 80, 40], "image_size": None},
                ),
                {"image": scope},
                OperatorContext("bbox", (), composer),
            )
        with pytest.raises(ValueError, match="does not intersect"):
            GeometryExecutor().execute(
                _node(
                    "REGION_FROM_BBOX",
                    {"image": "$n1"},
                    {"bbox": [0, 0, 20, 20], "image_size": None},
                ),
                {"image": scope},
                OperatorContext("bbox", (), composer),
            )
    finally:
        composer.close()


def test_route_entityset_uses_highest_score_and_rejects_unscored_ambiguity(
    tmp_path: Path,
) -> None:
    image = _image(tmp_path / "route.png", (300, 200))
    start = EntitySet(
        (
            _entity(image, (10, 10, 20, 20), 0.2),
            _entity(image, (40, 40, 50, 50), 0.9),
        )
    )
    goal = EntitySet((_entity(image, (200, 150, 220, 170), 0.7),))
    context = GeometryExecutor._route_context(image, start, goal)
    assert isinstance(context, RouteContext)
    assert context.start == start.entities[1]
    assert context.provenance["start_selection"]["selected_index"] == 1

    ambiguous = EntitySet(
        (
            _entity(image, (10, 10, 20, 20)),
            _entity(image, (40, 40, 50, 50)),
        )
    )
    with pytest.raises(ValueError, match="ambiguous"):
        GeometryExecutor._route_context(image, ambiguous, goal)

    partially_scored = EntitySet(
        (
            _entity(image, (10, 10, 20, 20), 0.9),
            _entity(image, (40, 40, 50, 50)),
        )
    )
    with pytest.raises(ValueError, match="every candidate needs a score"):
        GeometryExecutor._route_context(image, partially_scored, goal)

    tied = EntitySet(
        (
            _entity(image, (10, 10, 20, 20), 0.9),
            _entity(image, (40, 40, 50, 50), 0.9),
        )
    )
    with pytest.raises(ValueError, match="highest score is tied"):
        GeometryExecutor._route_context(image, tied, goal)

    assert context.provenance["context_size"] == [256.0, 200.0]


def test_route_visual_resize_preserves_endpoint_coordinate_mapping(tmp_path: Path) -> None:
    image = _image(tmp_path / "large_route.png", (1200, 800))
    start = _entity(image, (30, 40, 80, 90), 0.9)
    goal = _entity(image, (1050, 680, 1120, 750), 0.8)
    context = GeometryExecutor._route_context(image, start, goal)
    composer = InputComposer(tmp_path / "route_render", route_max_side=256)
    try:
        model_input = composer.compose_named(
            {"context": context},
            question="Choose the route.",
            options=("A", "B"),
        )
    finally:
        composer.close()
    metadata = model_input.metadata["route_context"]
    assert max(metadata["render_size"]) == 256
    assert metadata["resize_scale"] < 1.0
    origin_x, origin_y = metadata["coordinate_transform"]["origin_global"]
    scale_x, scale_y = metadata["coordinate_transform"]["scale_xy"]
    rendered = metadata["endpoints"]["start"]["bbox_xyxy_render"]
    restored = [
        rendered[0] / scale_x + origin_x,
        rendered[1] / scale_y + origin_y,
        rendered[2] / scale_x + origin_x,
        rendered[3] / scale_y + origin_y,
    ]
    assert restored == pytest.approx(list(start.region.bbox_xyxy_global))
