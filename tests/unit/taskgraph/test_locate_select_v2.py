"""Contract tests, explicitly mocked semantic predictions (not model accuracy)."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from PIL import Image

from sat_rs_vlm.taskgraph.execution_plan import FinalChoiceFusionConfig
from sat_rs_vlm.taskgraph.executor import CapabilityRouter, ExecutorBinding, GraphExecutor
from sat_rs_vlm.taskgraph.input_composer import InputComposer
from sat_rs_vlm.taskgraph.operators import (
    CountExecutor,
    LocateExecutor,
    OperatorContext,
    SelectExecutor,
)
from sat_rs_vlm.taskgraph.providers import (
    FakeCountingProvider,
    FakeDetectionProvider,
    FakeSemanticVLMProvider,
    FiniteDecisionResult,
    VLMResult,
)
from sat_rs_vlm.taskgraph.referent_refinement import ReferentRefinementConfig, ReferentRefiner
from sat_rs_vlm.taskgraph.runtime_types import (
    Entity,
    EntitySet,
    ImageRef,
    Region,
    RegionSet,
    SelectStatus,
)
from sat_rs_vlm.taskgraph.schema import GraphNode, OperatorName, SelectParams, TargetSpec, TaskGraph
from sat_rs_vlm.taskgraph.semantic_selection import SemanticSelectionConfig
from sat_rs_vlm.taskgraph.store import RuntimeStore


class SemanticFixture(FakeSemanticVLMProvider):
    def __init__(self, decisions=("A",), pointer='{"status":"OK","bbox":[32,32,128,128]}'):
        super().__init__()
        self.decisions = iter(decisions)
        self.pointer = pointer

    def infer(self, request):
        self.calls.append(request)
        return VLMResult(self.pointer, "fixture_vlm")

    def reason_and_decide(self, request):
        self.semantic_calls.append(request)
        verdict = next(self.decisions, "NONE" if "NONE" in request.candidate_ids else "NO_MATCH")
        if verdict == "INVALID":
            raise ValueError("invalid model result")
        return FiniteDecisionResult(
            (verdict,),
            {id: 1.0 if id == verdict else -1.0 for id in request.candidate_ids},
            "SINGLE",
            "mock reasoning",
            self.provider_name,
            "fixture",
            "finite_mock",
            True,
        )


@pytest.fixture
def scene(tmp_path):
    path = tmp_path / "scene.png"
    Image.new("RGB", (640, 480), "green").save(path)
    image = ImageRef(str(path), 640, 480)
    composer = InputComposer(tmp_path / "visuals", entity_set_max_side=640)
    yield image, composer
    composer.close()


def candidates(image, count=3):
    return EntitySet(
        tuple(
            Entity(
                Region(image, (10.0 + i * 15.0, 50.0, 20.0 + i * 15.0, 60.0)),
                "building",
                0.9,
                {"candidate_id": f"id-{i}"},
            )
            for i in range(count)
        )
    )


def select_node(cardinality="SINGLE", query="the building with trees to its right and grass below"):
    return GraphNode(
        id="n2",
        op="SELECT",
        inputs={"candidates": "$n1"},
        params={"mode": "SEMANTIC", "query": query, "selection_type": cardinality},
    )


@pytest.mark.parametrize(
    "verdict,status,count",
    [
        ("A", "OK", 1),
        ("NONE", "EMPTY", 0),
        ("AMBIGUOUS", "AMBIGUOUS", 0),
        ("UNRESOLVED", "UNRESOLVED", 0),
        ("INVALID", "UNRESOLVED", 0),
    ],
)
def test_single_finite_states(scene, verdict, status, count):
    image, composer = scene
    provider = SemanticFixture((verdict,))
    result = (
        SelectExecutor(provider)
        .execute(
            select_node(),
            {"candidates": candidates(image)},
            OperatorContext("secret downstream final", (), composer),
        )
        .value
    )
    assert result.status.value == status
    assert len(result.selected.entities) == count
    if count:
        assert result.selected.entities[0].provenance["candidate_id"] == "id-0"
    request = provider.semantic_calls[0]
    assert request.purpose == "select_semantic"
    assert select_node().params["query"] in request.model_input.question
    assert "secret downstream final" not in request.model_input.question
    assert request.candidate_ids == ("A", "B", "C", "NONE", "AMBIGUOUS", "UNRESOLVED")
    assert provider.calls == []  # no free generation / full-image detection


@pytest.mark.parametrize(
    "decisions,status,ids",
    [
        (("MATCH", "NO_MATCH", "MATCH"), "OK", ["id-0", "id-2"]),
        (("NO_MATCH",) * 3, "EMPTY", []),
        (("MATCH", "UNCERTAIN", "NO_MATCH"), "UNRESOLVED", ["id-0"]),
    ],
)
def test_multi_true_three_state_verification(scene, decisions, status, ids):
    image, composer = scene
    provider = SemanticFixture(decisions)
    result = (
        SelectExecutor(provider)
        .execute(
            select_node("MULTI"),
            {"candidates": candidates(image)},
            OperatorContext("", (), composer),
        )
        .value
    )
    assert result.status.value == status
    assert [e.provenance["candidate_id"] for e in result.selected.entities] == ids
    assert all(
        r.candidate_ids == ("MATCH", "NO_MATCH", "UNCERTAIN") and r.decision_mode == "SINGLE"
        for r in provider.semantic_calls
    )


def test_all_candidates_batched_with_exact_ids_and_pixel_budget(scene):
    image, composer = scene
    provider = SemanticFixture(("NONE", "NONE", "B"))
    result = (
        SelectExecutor(provider, selection_config=SemanticSelectionConfig(batch_size=8))
        .execute(
            select_node(), {"candidates": candidates(image, 19)}, OperatorContext("", (), composer)
        )
        .value
    )
    assert result.status is SelectStatus.OK
    assert result.selected.entities[0].provenance["candidate_id"] == "id-17"
    assert len(provider.semantic_calls) == 3
    for request in provider.semantic_calls:
        metadata = request.model_input.metadata
        assert len(metadata["candidate_mapping"]) <= 8
        assert metadata["total_render_pixels"] <= 409600
        assert len(request.model_input.visual_inputs) == 2
        assert metadata["context_overview"]
        assert all(
            w % 32 == h % 32 == 0 and w * h >= 65536 for w, h in metadata["selection_render_sizes"]
        )


def test_two_single_matches_across_batches_are_ambiguous(scene):
    image, composer = scene
    result = (
        SelectExecutor(
            SemanticFixture(("A", "A")), selection_config=SemanticSelectionConfig(batch_size=2)
        )
        .execute(
            select_node(), {"candidates": candidates(image, 3)}, OperatorContext("", (), composer)
        )
        .value
    )
    assert result.status is SelectStatus.AMBIGUOUS


def test_budget_exhaustion_never_counts_partial(scene):
    image, composer = scene
    result = (
        SelectExecutor(
            SemanticFixture(("MATCH",) * 4),
            selection_config=SemanticSelectionConfig(batch_size=2, max_candidates=4),
        )
        .execute(
            select_node("MULTI"),
            {"candidates": candidates(image, 5)},
            OperatorContext("", (), composer),
        )
        .value
    )
    assert result.status is SelectStatus.UNRESOLVED
    assert result.provenance["unseen_candidate_ids"] == ["id-4"]
    counter = CountExecutor(FakeCountingProvider())
    count_node = GraphNode(
        id="n3",
        op="COUNT",
        inputs={"entities": "$n2"},
        params={"target": {"category": "building"}, "entire": True},
    )
    with pytest.raises(ValueError):
        counter.execute(count_node, {"entities": result}, OperatorContext("", (), composer))


def test_zero_candidates_and_regions_preserve_type(scene):
    image, composer = scene
    provider = SemanticFixture(("B",))
    executor = SelectExecutor(provider)
    empty = executor.execute(
        select_node(), {"candidates": EntitySet(())}, OperatorContext("", (), composer)
    ).value
    assert empty.status is SelectStatus.EMPTY and not provider.semantic_calls
    source = candidates(image)
    regions = RegionSet(tuple(e.region for e in source.entities))
    result = executor.execute(
        select_node(), {"candidates": regions}, OperatorContext("", (), composer)
    ).value
    assert result.selected.regions[0].bbox_xyxy_global == regions.regions[1].bbox_xyxy_global


@pytest.mark.parametrize(
    "raw",
    [
        "numbers 1 2 3 4",
        '{"status":"OK","bbox":[100,20,10,80]}',
        '{"status":"OK","bbox":[-1,2,20,40]}',
        '{"status":"OK","bbox":[1,2,10000,40]}',
        '{"status":"OK","bbox":[1,2,NaN,40]}',
        '{"status":"OK","bbox":[1,2,1.1,2.1]}',
        '{"status":"NONE","bbox":[1,2,3,4]}',
    ],
)
def test_strict_locate_invalid_outputs_are_unresolved(scene, raw):
    image, composer = scene
    result = ReferentRefiner(SemanticFixture(pointer=raw), composer).vlm_referent(
        image, question="must not leak", target=TargetSpec(category="building")
    )
    assert result.entities.entities == ()
    assert result.metadata["resolution_status"] == "UNRESOLVED"
    assert not result.metadata["candidate_pool_complete"]


@pytest.mark.parametrize("status", ["NONE", "AMBIGUOUS", "UNRESOLVED"])
def test_locate_failure_status_does_not_create_scope_entity(scene, status):
    image, composer = scene
    result = ReferentRefiner(
        SemanticFixture(pointer=json.dumps({"status": status})), composer
    ).vlm_referent(image, target=TargetSpec(category="building"))
    assert not result.entities.entities
    assert result.metadata["resolution_status"] == ("EMPTY" if status == "NONE" else status)


def test_local_scope_pixel_inverse_mapping_and_no_query_leak(scene):
    image, composer = scene
    provider = SemanticFixture(pointer='{"status":"OK","bbox":[32,32,128,128]}')
    scope = Region(image, (320.0, 240.0, 640.0, 480.0))
    result = ReferentRefiner(provider, composer).vlm_referent(
        scope,
        question="secret downstream SELECT query 123",
        target=TargetSpec(category="building", attributes={"color": "white"}),
    )
    transform = result.metadata["visual_transform"]
    sx, sy = transform["scale_xy"]
    assert result.entities.entities[0].region.bbox_xyxy_global == (
        320 + 32 / sx,
        240 + 32 / sy,
        320 + 128 / sx,
        240 + 128 / sy,
    )
    prompt = provider.calls[0].model_input.question
    assert "secret downstream" not in prompt
    assert '"color": "white"' in prompt


def test_topmost_low_score_beyond_sixteen_survives_and_graph_is_unchanged(scene):
    image, composer = scene
    boxes = [(10 + i * 15, 300, 20 + i * 15, 310) for i in range(19)] + [(100, 5, 120, 25)]
    detector = FakeDetectionProvider(boxes)
    provider = SemanticFixture()
    locate = LocateExecutor(detector, None, refiner=ReferentRefiner(provider, composer))
    selector = SelectExecutor(provider)
    nodes = [
        GraphNode(
            id="n1",
            op="LOCATE",
            inputs={"image": "$image0"},
            params={"target": {"category": "building"}},
        ),
        GraphNode(
            id="n2",
            op="SELECT",
            inputs={"candidates": "$n1"},
            params={"mode": "EXTREME", "direction": "TOPMOST"},
        ),
    ]
    graph = TaskGraph(
        question="how many is NOT a routing signal",
        question_type="FREE_FORM",
        inputs={"image0": {"uri_or_key": image.uri_or_key}},
        nodes=nodes,
        final={"sources": ["$n2"], "question": "", "answer_type": "TEXT"},
    )
    original = graph.model_dump(mode="json")
    store = RuntimeStore({"image0": image})
    trace = GraphExecutor(
        CapabilityRouter(
            {
                OperatorName.LOCATE: ExecutorBinding(locate),
                OperatorName.SELECT: ExecutorBinding(selector),
            }
        ),
        FinalChoiceFusionConfig(mode="off"),
    ).execute(
        graph,
        store,
        sample_id="mock",
        execution_mode="TASKGRAPH",
        context=OperatorContext("", (), composer),
    )
    assert graph.model_dump(mode="json") == original
    assert len(store.get("n1").entities) == 20
    assert store.get("n2").selected.entities[0].region.bbox_xyxy_global[1] == 5
    assert not provider.calls and not provider.semantic_calls
    assert not detector.calls[0].apply_locate_policy and not detector.calls[0].use_clip_rerank
    assert trace.nodes[0].trace_metadata["planned_backend"] == "LAE_DINO"


def test_intrinsic_attributes_are_not_clip_topk_truth(scene):
    image, composer = scene
    detector = FakeDetectionProvider([(10, 10, 30, 30), (60, 60, 80, 80)])
    node = GraphNode(
        id="n1",
        op="LOCATE",
        inputs={"image": "$image0"},
        params={"target": {"category": "car", "attributes": {"color": "red"}}},
    )
    count = GraphNode(
        id="n2",
        op="COUNT",
        inputs={"entities": "$n1"},
        params={"target": {"category": "car"}, "entire": True},
    )
    context = OperatorContext("unrelated full question", (), composer, graph_nodes=(node, count))
    unverified = (
        LocateExecutor(detector, None, refiner=ReferentRefiner(SemanticFixture(), composer))
        .execute(node, {"image": image}, context)
        .value
    )
    assert unverified.provenance["candidate_pool_complete"] is False
    with pytest.raises(ValueError):
        CountExecutor(FakeCountingProvider()).execute(count, {"entities": unverified}, context)
    provider = SemanticFixture(("MATCH", "NO_MATCH"))
    verified = (
        LocateExecutor(
            detector,
            None,
            refiner=ReferentRefiner(
                provider, composer, ReferentRefinementConfig(attribute_policy="vlm")
            ),
        )
        .execute(node, {"image": image}, context)
        .value
    )
    assert len(verified.entities) == 1
    assert verified.provenance["candidate_pool_complete"]
    assert verified.entities[0].provenance["attribute_verdict"] == "MATCH"
    assert all(req.purpose == "locate_attribute_verification" for req in provider.semantic_calls)
    assert (
        CountExecutor(FakeCountingProvider())
        .execute(count, {"entities": verified}, context)
        .value.value
        == 1
    )


@pytest.mark.parametrize(
    "params",
    [
        {"mode": "SEMANTIC", "query": " ", "selection_type": "SINGLE"},
        {"mode": "SEMANTIC", "query": "x"},
        {"mode": "SEMANTIC", "query": "x", "selection_type": "SINGLE", "relation": "NEAR"},
        {"mode": "RELATION", "relation": "NEAR", "query": "x"},
        {"mode": "SEMANTIC", "query": "x" * 2049, "selection_type": "SINGLE"},
    ],
)
def test_semantic_schema_rejects_invalid_contract(params):
    with pytest.raises(ValueError):
        SelectParams.model_validate(params)


def test_twenty_candidates_relation_count_does_not_redetect_or_truncate(scene):
    image, composer = scene
    source = candidates(image, 20)
    provider = SemanticFixture()
    selector = SelectExecutor(provider)
    node = GraphNode(
        id="n2",
        op="SELECT",
        inputs={"candidates": "$n1", "reference": "$n9"},
        params={"mode": "RELATION", "relation": "LEFT_OF", "margin": 0},
    )
    result = selector.execute(
        node,
        {"candidates": source, "reference": Region(image, (500.0, 20.0, 540.0, 60.0))},
        OperatorContext("", (), composer),
    ).value
    assert result.status is SelectStatus.OK
    assert len(result.selected.entities) == 20
    counter_provider = FakeCountingProvider()
    count = GraphNode(
        id="n3",
        op="COUNT",
        inputs={"entities": "$n2"},
        params={"target": {"category": "building"}, "entire": False},
    )
    assert (
        CountExecutor(counter_provider)
        .execute(count, {"entities": result}, OperatorContext("", (), composer))
        .value.value
        == 20
    )
    assert not provider.semantic_calls and not provider.calls and not counter_provider.calls


def test_semantic_select_uses_base_candidates_despite_unresolved_instance_regions(scene):
    image, composer = scene
    source = candidates(image, 2)
    source.provenance.update(
        {
            "candidate_pool_complete": True,
            "instance_resolution_complete": False,
            "unresolved_instance_regions": [{"status": "UNRESOLVED"}],
        }
    )
    provider = SemanticFixture(("A",))
    selected = SelectExecutor(provider).execute(
        select_node(query="the first of the retained candidates"),
        {"candidates": source},
        OperatorContext("", (), composer),
    ).value

    assert selected.status is SelectStatus.OK
    assert selected.selected.entities[0].provenance["candidate_id"] == "id-0"


def test_ordinal_area_rank_and_group_retain_the_full_pool(scene):
    from sat_rs_vlm.taskgraph.operators import GeometryExecutor

    image, composer = scene
    pool = candidates(image, 19)
    pool = EntitySet(
        (
            *pool.entities,
            Entity(
                Region(image, (400.0, 400.0, 500.0, 470.0)),
                "building",
                0.001,
                {"candidate_id": "last"},
            ),
        )
    )
    context = OperatorContext("", (), composer)
    selector = SelectExecutor(SemanticFixture())
    rank = GraphNode(
        id="n2",
        op="SELECT",
        inputs={"candidates": "$n1"},
        params={"mode": "RANK", "criterion": "bbox_area", "rank": 1, "order": "DESCENDING"},
    )
    assert (
        selector.execute(rank, {"candidates": pool}, context)
        .value.selected.entities[0]
        .provenance["candidate_id"]
        == "last"
    )
    # Unique coordinates make the fourth item a deterministic geometric choice.
    ordinal_pool = EntitySet(
        tuple(
            Entity(
                Region(image, (20.0, i * 20.0, 30.0, i * 20.0 + 10)),
                "building",
                0.5,
                {"candidate_id": str(i)},
            )
            for i in range(20)
        )
    )
    ordinal = GraphNode(
        id="n2",
        op="SELECT",
        inputs={"candidates": "$n1"},
        params={"mode": "ORDINAL", "index": 4, "order": "BOTTOM_TO_TOP"},
    )
    assert (
        selector.execute(ordinal, {"candidates": ordinal_pool}, context)
        .value.selected.entities[0]
        .provenance["candidate_id"]
        == "16"
    )
    group = GraphNode(id="n2", op="GROUP", inputs={"entities": "$n1"}, params={"mode": "CLUSTER"})
    grouped = GeometryExecutor().execute(group, {"entities": pool}, context).value
    assert len(grouped.entities) == 20
    assert sum(len(group["source_indices"]) for group in grouped.provenance["groups"]) == 20


def test_detector_invalid_coordinates_mark_pool_partial(scene):
    from sat_rs_vlm.integrations.detectors.protocol import ProposalResult
    from sat_rs_vlm.taskgraph.providers import DetectionRequest, ProposalDetectionAdapter

    image, _ = scene

    class Detector:
        provider_name = "fixture"

        def predict(self, path, query):
            return ProposalResult(
                [[1, 1, 20, 20], [-1, 1, 20, 20]], [0.9, 0.8], 0.0, "fixture", "fixture"
            )

    result = ProposalDetectionAdapter(Detector()).detect(
        DetectionRequest(
            image, TargetSpec(category="building"), apply_locate_policy=False, use_clip_rerank=False
        )
    )
    assert len(result.detections.entities) == 1
    assert result.detections.provenance["candidate_pool_complete"] is False
    assert result.metadata["invalid_candidate_indices"] == [1]


def test_locate_single_fallback_can_be_disabled_and_never_uses_downstream_query(scene):
    image, composer = scene
    provider = SemanticFixture(pointer="bad output")
    detector = FakeDetectionProvider([(20, 20, 30, 30)])
    refiner = ReferentRefiner(
        provider, composer, ReferentRefinementConfig(single_detector_fallback=False)
    )
    node = GraphNode(
        id="n1",
        op="LOCATE",
        inputs={"image": "$image0"},
        params={"target": {"category": "building"}},
    )
    result = LocateExecutor(detector, None, refiner=refiner).execute(
        node, {"image": image}, OperatorContext("leaked query", (), composer)
    )
    assert not detector.calls
    assert not result.value.entities
    assert result.trace_metadata["actual_backend"] == "VLM_SINGLE"


def test_unresolved_semantic_final_does_not_trigger_whole_image_answer(scene):
    from sat_rs_vlm.taskgraph.runtime import RuntimeRequest, fake_runtime
    from sat_rs_vlm.taskgraph.runtime_types import CandidateIntegrityError

    image, _ = scene
    nodes = [
        {
            "id": "n1",
            "op": "LOCATE",
            "inputs": {"image": "$image0"},
            "params": {"target": {"category": "building"}},
        },
        select_node().model_dump(mode="json"),
    ]
    graph = {
        "question": "Where is the selected building?",
        "question_type": "MULTIPLE_CHOICE_SINGLE",
        "choices": ["A left", "B right"],
        "inputs": {"image0": {"uri_or_key": image.uri_or_key}},
        "nodes": nodes,
        "final": {
            "sources": ["$n2"],
            "question": "Where is the selected building?",
            "answer_type": "CHOICE_SINGLE",
        },
    }
    runtime = fake_runtime(detection_boxes=[(10, 10, 20, 20)])
    runtime.graph_executor.router.bindings[OperatorName.SELECT] = ExecutorBinding(
        SelectExecutor(SemanticFixture(("UNRESOLVED",)))
    )
    try:
        with pytest.raises(CandidateIntegrityError):
            runtime.run(
                RuntimeRequest(
                    "mock",
                    "MME_RealWorld_RS",
                    "vqa",
                    graph["question"],
                    (image.uri_or_key,),
                    tuple(graph["choices"]),
                    graph=graph,
                )
            )
        assert not runtime.providers.choice.choice_calls
    finally:
        runtime.close()


def test_five_legal_end_to_end_graphs_and_full_qwen_wrappers(scene, tmp_path):
    from scripts.taskgraph.audit_locate_select_prompts import CaptureEngine, capture_end_to_end

    from sat_rs_vlm.infrastructure.config import ModelConfig
    from sat_rs_vlm.taskgraph.providers import LazyQwenSemanticProvider

    image, composer = scene
    provider = LazyQwenSemanticProvider(ModelConfig(model_id="MOCK_NO_WEIGHTS"))
    engine = CaptureEngine()
    provider._engine = engine
    records = capture_end_to_end(tmp_path, image, composer, provider, engine)
    assert len(records) == 5
    assert records[0]["result"]["value"] == "white"
    assert records[1]["trace"]["nodes"][0]["output_summary"]["count"] == 20
    assert records[2]["result"]["provenance"]["selected_candidate_ids"] == ["candidate_0002"]
    assert records[3]["result"]["value"] == 1
    assert records[4]["result"]["value"] == 10
    assert [
        node["trace_metadata"]["planned_backend"]
        for node in records[4]["trace"]["nodes"]
        if node["operator"] == "LOCATE"
    ] == ["VLM_SINGLE", "LAE_DINO"]
    for record in records:
        for call in record["calls"]:
            assert "Visual inputs:" in call["prompt"]
            assert "Question:" in call["prompt"]
            assert call["messages"][0]["role"] == "user"
            if "Task: LOCATE_SINGLE" in call["prompt"]:
                assert "Full final question" not in call["prompt"]
                assert "trees to its right" not in call["prompt"]


@pytest.mark.parametrize("empty_input", [False, True])
@pytest.mark.parametrize("region_set", [False, True])
def test_empty_semantic_selection_cannot_trigger_attribute_guess(scene, empty_input, region_set):
    from sat_rs_vlm.taskgraph.executor import TaskGraphExecutionError
    from sat_rs_vlm.taskgraph.operators import SemanticExecutor

    image, composer = scene
    provider = SemanticFixture(("NONE",))
    pool = candidates(image, 0 if empty_input else 3)
    if region_set:
        pool = RegionSet(tuple(entity.region for entity in pool.entities))
    selected = (
        SelectExecutor(provider)
        .execute(select_node(), {"candidates": pool}, OperatorContext("", (), composer))
        .value
    )
    assert selected.status is SelectStatus.EMPTY
    assert selected.selected.provenance["semantic_selection_ancestor"]
    if empty_input:
        assert not provider.semantic_calls
    attribute = GraphNode(
        id="n3", op="ATTRIBUTE", inputs={"entity": "$n2"}, params={"attribute": "color"}
    )
    router = CapabilityRouter({OperatorName.ATTRIBUTE: ExecutorBinding(SemanticExecutor(provider))})
    with pytest.raises(TaskGraphExecutionError, match="no located/selected target"):
        router.execute(attribute, {"entity": selected}, OperatorContext("", (), composer))
    assert not provider.calls


def test_empty_semantic_set_counts_zero_without_redetection(scene):
    image, composer = scene
    provider = SemanticFixture(("NO_MATCH",) * 3)
    selected = (
        SelectExecutor(provider)
        .execute(
            select_node("MULTI"),
            {"candidates": candidates(image)},
            OperatorContext("", (), composer),
        )
        .value
    )
    counting = FakeCountingProvider()
    router = CapabilityRouter({OperatorName.COUNT: ExecutorBinding(CountExecutor(counting))})
    node = GraphNode(
        id="n3",
        op="COUNT",
        inputs={"entities": "$n2"},
        params={"entire": False, "target": {"category": "building"}},
    )
    result, _ = router.execute(node, {"entities": selected}, OperatorContext("", (), composer))
    assert result.value.value == 0
    assert not counting.calls


def test_nested_empty_semantic_final_does_not_guess_from_whole_image(scene):
    from sat_rs_vlm.taskgraph.runtime import RuntimeRequest, fake_runtime
    from sat_rs_vlm.taskgraph.runtime_types import CandidateIntegrityError

    image, _ = scene
    graph = {
        "question": "Where is the selected building?",
        "question_type": "MULTIPLE_CHOICE_SINGLE",
        "choices": ["A left", "B right"],
        "inputs": {"image0": {"uri_or_key": image.uri_or_key}},
        "nodes": [
            {
                "id": "n1",
                "op": "LOCATE",
                "inputs": {"image": "$image0"},
                "params": {"target": {"category": "building"}},
            },
            select_node().model_dump(mode="json"),
            {
                "id": "n3",
                "op": "SELECT",
                "inputs": {"candidates": "$n2"},
                "params": {"mode": "EXTREME", "direction": "TOPMOST"},
            },
        ],
        "final": {
            "sources": ["$n3"],
            "question": "Where is the selected building?",
            "answer_type": "CHOICE_SINGLE",
        },
    }
    runtime = fake_runtime(detection_boxes=[(10, 10, 20, 20)])
    runtime.graph_executor.router.bindings[OperatorName.SELECT] = ExecutorBinding(
        SelectExecutor(SemanticFixture(("NONE",)))
    )
    request = RuntimeRequest(
        "mock",
        "MME_RealWorld_RS",
        "vqa",
        graph["question"],
        (image.uri_or_key,),
        tuple(graph["choices"]),
        graph=graph,
    )
    try:
        with pytest.raises(CandidateIntegrityError):
            runtime.run(request)
        assert not runtime.providers.choice.choice_calls
        deferred = runtime.run(replace(request, defer_final_answer=True))
        assert not deferred.output.entities
    finally:
        runtime.close()


def test_processor_pixel_cap_is_respected_for_single_and_semantic_selection(scene):
    from types import SimpleNamespace

    image, composer = scene
    provider = SemanticFixture()
    provider.model_config = SimpleNamespace(max_pixels=100000)
    located = ReferentRefiner(provider, composer).vlm_referent(
        image, target=TargetSpec(category="building")
    )
    width, height = located.metadata["visual_transform"]["render_size"]
    assert width * height <= 100000
    selected = (
        SelectExecutor(provider)
        .execute(
            select_node(), {"candidates": candidates(image)}, OperatorContext("", (), composer)
        )
        .value
    )
    assert selected.status is SelectStatus.OK
    assert selected.provenance["budget"]["effective_max_pixels"] == 200000
    visuals = selected.provenance["batches"][0]["visual_metadata"]
    assert visuals["total_render_pixels"] <= 200000
    assert all(w * h <= 100000 for w, h in visuals["selection_render_sizes"])


def test_locate_none_cannot_trigger_attribute_guess(scene):
    from sat_rs_vlm.taskgraph.executor import TaskGraphExecutionError
    from sat_rs_vlm.taskgraph.operators import SemanticExecutor

    image, composer = scene
    provider = SemanticFixture(pointer='{"status":"NONE"}')
    located = (
        LocateExecutor(FakeDetectionProvider(), None, refiner=ReferentRefiner(provider, composer))
        .execute(
            GraphNode(
                id="n1",
                op="LOCATE",
                inputs={"image": "$image0"},
                params={"target": {"category": "building"}},
            ),
            {"image": image},
            OperatorContext("", (), composer),
        )
        .value
    )
    assert not located.entities
    router = CapabilityRouter({OperatorName.ATTRIBUTE: ExecutorBinding(SemanticExecutor(provider))})
    node = GraphNode(
        id="n2", op="ATTRIBUTE", inputs={"entity": "$n1"}, params={"attribute": "color"}
    )
    with pytest.raises(TaskGraphExecutionError, match="no located/selected target"):
        router.execute(node, {"entity": located}, OperatorContext("", (), composer))
    assert len(provider.calls) == 1  # Only LOCATE; ATTRIBUTE never runs.


@pytest.mark.parametrize(
    "metadata,global_limit",
    [({"candidate_pool_complete": False}, None), ({"candidate_truncated": True}, None), ({}, 1)],
)
def test_detector_limits_propagate_through_tiles_to_count(scene, metadata, global_limit):
    from sat_rs_vlm.integrations.detectors.protocol import ProposalResult
    from sat_rs_vlm.integrations.detectors.tiled import TiledProposalProvider
    from sat_rs_vlm.taskgraph.providers import DetectionRequest, ProposalDetectionAdapter

    image, composer = scene

    class Detector:
        provider_name = "fixture"
        model_id = "fixture"

        def predict(self, path, query):
            return ProposalResult(
                [[1, 1, 20, 20], [100, 100, 120, 120]],
                [0.9, 0.8],
                0.0,
                "fixture",
                "fixture",
                metadata=metadata,
            )

        def close(self):
            pass

    tiled = TiledProposalProvider(
        Detector(), {"tile_size": 640, "global_top_k": global_limit}, base_provider_name="fixture"
    )
    try:
        result = ProposalDetectionAdapter(tiled).detect(
            DetectionRequest(
                image,
                TargetSpec(category="building"),
                apply_locate_policy=False,
                use_clip_rerank=False,
            )
        )
        assert result.detections.provenance["candidate_pool_complete"] is False
        node = GraphNode(
            id="n2",
            op="COUNT",
            inputs={"entities": "$n1"},
            params={"target": {"category": "building"}, "entire": False},
        )
        with pytest.raises(ValueError, match="incomplete"):
            CountExecutor(FakeCountingProvider()).execute(
                node, {"entities": result.detections}, OperatorContext("", (), composer)
            )
    finally:
        tiled.close()


def test_entity_region_ids_stay_aligned_after_detector_sort_and_invalid_filter(scene):
    from sat_rs_vlm.integrations.detectors.protocol import ProposalResult
    from sat_rs_vlm.taskgraph.providers import DetectionRequest, ProposalDetectionAdapter

    image, composer = scene

    class Detector:
        provider_name = "fixture"

        def predict(self, path, query):
            return ProposalResult(
                [[1, 1, 20, 20], [-1, 1, 20, 20], [100, 100, 120, 120]],
                [0.1, 0.9, 0.8],
                0.0,
                "fixture",
                "fixture",
            )

    detected = (
        ProposalDetectionAdapter(Detector())
        .detect(
            DetectionRequest(
                image,
                TargetSpec(category="building"),
                apply_locate_policy=False,
                use_clip_rerank=False,
            )
        )
        .detections
    )
    region_pool = RegionSet(tuple(entity.region for entity in detected.entities))
    assert [region.provenance["candidate_id"] for region in region_pool.regions] == [
        entity.provenance["candidate_id"] for entity in detected.entities
    ]
    selected = (
        SelectExecutor(SemanticFixture(("A",)))
        .execute(select_node(), {"candidates": region_pool}, OperatorContext("", (), composer))
        .value
    )
    assert selected.selected.regions[0].bbox_xyxy_global == (100, 100, 120, 120)
    assert selected.provenance["selected_candidate_ids"] == [
        detected.entities[0].provenance["candidate_id"]
    ]
