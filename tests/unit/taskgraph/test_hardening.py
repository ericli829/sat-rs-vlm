from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw
from pydantic import ValidationError

from sat_rs_vlm.taskgraph.executor import (
    CapabilityRouter,
    ExecutorBinding,
    TaskGraphExecutionError,
)
from sat_rs_vlm.taskgraph.input_composer import InputComposer
from sat_rs_vlm.taskgraph.operators import (
    CountExecutor,
    GeometryExecutor,
    OperatorContext,
    SelectExecutor,
    SemanticExecutor,
)
from sat_rs_vlm.taskgraph.providers import (
    DetectionRequest,
    DetectionSet,
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


def test_region_candidates_reject_non_finite_scores_and_latency(tmp_path: Path) -> None:
    image = _image(tmp_path / "candidate.png", (32, 32))
    region = Region(image, (0, 0, 16, 16))

    with pytest.raises(ValueError, match="relevance_score"):
        RegionCandidate(region, float("nan"))
    with pytest.raises(ValueError, match="latency_ms"):
        RegionCandidates((), "fixture", float("inf"))
    with pytest.raises(ValueError, match="provider"):
        RegionCandidates((), " ", 0.0)


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
    assert len(result.candidates) == 4
    assert result.candidates[0].provenance["tile"] == {
        "level": 1,
        "index": 3,
        "row": 1,
        "column": 1,
        "grid_size": 2,
    }
    assert result.candidates[0].provenance["bbox_xyxy_global"] == [50.0, 40.0, 80.0, 70.0]
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

        def score_regions(self, image_path: Path, query: str, regions_xyxy: list) -> object:
            self.boxes = list(regions_xyxy)
            return SimpleNamespace(
                scores=[float(index) for index in range(len(regions_xyxy))],
                model_id="fake",
                latency_ms=0.0,
                metadata={},
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
    assert result.candidates[0].provenance["candidate_geometry"] == {
        "layout": "uniform_sliding_grid",
        "window_ratio": 0.5,
        "overlapping": True,
    }


def test_count_retriever_gate_filters_regions_and_deduplicates(tmp_path: Path) -> None:
    image = _image(tmp_path / "count_gate.png", (90, 90))
    retriever = FakeRegionRetriever(
        [
            ((0, 0, 30, 30), 0.25),
            ((30, 0, 60, 30), 0.10),
            ((60, 0, 90, 30), 0.20),
        ]
    )

    class Detection:
        provider_name = "fake_detector"

        def __init__(self) -> None:
            self.calls: list[DetectionRequest] = []

        def detect(self, request: DetectionRequest) -> DetectionSet:
            self.calls.append(request)
            assert isinstance(request.scope, Region)
            box = request.scope.bbox_xyxy_global
            entity = Entity(Region(image, box), "aircraft", 0.9)
            return DetectionSet(EntitySet((entity, entity)), 0.0, self.provider_name)

        def close(self) -> None:
            return None

    detection = Detection()
    executor = CountExecutor(
        detection,
        retriever,
        gate_enabled=True,
        gate_threshold=0.17203009128570557,
        gate_max_regions=3,
    )
    node = _node(
        "COUNT",
        {"image": "$image0"},
        {"target": {"category": "aircraft", "attributes": {}}, "entire": True},
    )
    composer = InputComposer(tmp_path / "count_inputs")
    try:
        outcome = executor.execute(
            node,
            {"image": image},
            OperatorContext("count", (), composer),
        )
    finally:
        composer.close()
    assert outcome.value.value == 2
    assert len(detection.calls) == 2
    assert outcome.value.provenance["gate"]["rejected_regions"] == 1


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
        assert isinstance(result.value, EntitySet)
        assert result.value.entities == (candidates.entities[0], candidates.entities[2])
        assert provider.calls[0].model_input.metadata["canvas_kind"] == "CANDIDATE_CANVAS"
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
        model_input = provider.calls[0].model_input
        assert result.value == Label("LEFT_OF", {"provider": "fake_vlm"})
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
