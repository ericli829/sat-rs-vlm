"""Tests for detector-recall fallback to the semantic VLM (empty-detection tolerance)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from sat_rs_vlm.taskgraph.input_composer import InputComposer
from sat_rs_vlm.taskgraph.operators import (
    LocateExecutor,
    OperatorContext,
    SemanticExecutor,
)
from sat_rs_vlm.taskgraph.providers import (
    FakeDetectionProvider,
    FakeRegionRetriever,
    FakeSemanticVLMProvider,
)
from sat_rs_vlm.taskgraph.referent_refinement import (
    ReferentRefinementConfig,
    ReferentRefiner,
)
from sat_rs_vlm.taskgraph.runtime_types import (
    ChoiceResult,
    EntitySet,
    ImageRef,
    Label,
    Region,
)
from sat_rs_vlm.taskgraph.schema import GraphNode


def _image(tmp_path: Path) -> ImageRef:
    path = tmp_path / "source.png"
    Image.new("RGB", (160, 120), "white").save(path)
    return ImageRef(str(path), width=160, height=120)


def _attach_node() -> GraphNode:
    return GraphNode.model_validate(
        {
            "id": "n3",
            "op": "ATTRIBUTE",
            "inputs": {"entity": "$n2"},
            "params": {"attribute": "color"},
        }
    )


def _context(
    tmp_path: Path, question: str = "What color is the minibus?"
) -> tuple[InputComposer, OperatorContext]:
    composer = InputComposer(tmp_path / "inputs")
    context = OperatorContext(
        question,
        (),
        composer,
        final_question=question,
        final_sources=("$n3",),
        graph_nodes=(_attach_node(),),
    )
    return composer, context


def _detector_locator(
    composer: InputComposer,
    refiner_enabled: bool = True,
) -> tuple[LocateExecutor, FakeSemanticVLMProvider]:
    semantic = FakeSemanticVLMProvider({})
    refiner = ReferentRefiner(
        semantic,
        composer,
        ReferentRefinementConfig(enabled=refiner_enabled),
    )
    locate = LocateExecutor(
        FakeDetectionProvider(boxes=[]),
        FakeRegionRetriever(()),
        max_candidates=8,
        refiner=refiner,
    )
    return locate, semantic


def test_empty_detection_falls_back_to_semantic_visual_entity(tmp_path: Path) -> None:
    image = _image(tmp_path)
    scope = Region(image, (5, 5, 155, 115), {"operator": "REGION"})
    composer, context = _context(tmp_path)
    locate, semantic = _detector_locator(composer)
    try:
        node = GraphNode.model_validate(
            {
                "id": "n2",
                "op": "LOCATE",
                "inputs": {"image": "$n1"},
                "params": {"target": {"category": "minibus", "attributes": {}}},
            }
        )
        outcome = locate.execute(node, {"image": scope}, context)
        assert isinstance(outcome.value, EntitySet)
        # The fallback produced one visual candidate covering the scope.
        assert len(outcome.value.entities) == 1
        entity = outcome.value.entities[0]
        assert entity.provenance.get("fallback_kind") == "semantic_visual"
        assert entity.provenance.get("fallback_required") is True
        assert entity.region.bbox_xyxy_global == (5.0, 5.0, 155.0, 115.0)
        # The fallback is visible on the EntitySet provenance for tracing.
        assert outcome.value.provenance.get("fallback_triggered") is True
        assert outcome.value.provenance.get("fallback_reason") == (
            "EMPTY_PROPOSALS_AND_REGIONAL_FALLBACK"
        )
        # The semantic VLM can compose a visual from the fallback entity.
        composed = composer.compose_named(
            {"entity": outcome.value},
            question="What color is the minibus?",
        )
        assert len(composed.visual_inputs) >= 1
    finally:
        composer.close()


def test_empty_detection_no_visual_fallback_when_refiner_disabled(tmp_path: Path) -> None:
    image = _image(tmp_path)
    scope = Region(image, (0, 0, 160, 120), {})
    composer, context = _context(tmp_path)
    locate, _ = _detector_locator(composer, refiner_enabled=False)
    try:
        node = GraphNode.model_validate(
            {
                "id": "n2",
                "op": "LOCATE",
                "inputs": {"image": "$n1"},
                "params": {"target": {"category": "minibus", "attributes": {}}},
            }
        )
        outcome = locate.execute(node, {"image": scope}, context)
        assert isinstance(outcome.value, EntitySet)
        assert len(outcome.value.entities) == 0
    finally:
        composer.close()


def test_empty_detection_semantic_executor_answers_from_fallback_scope(
    tmp_path: Path,
) -> None:
    image = _image(tmp_path)
    scope = Region(image, (5, 5, 155, 115), {"operator": "REGION"})
    composer, context = _context(tmp_path)
    locate, semantic = _detector_locator(composer)
    try:
        node = GraphNode.model_validate(
            {
                "id": "n2",
                "op": "LOCATE",
                "inputs": {"image": "$n1"},
                "params": {"target": {"category": "minibus", "attributes": {}}},
            }
        )
        outcome = locate.execute(node, {"image": scope}, context)
        # ATTRIBUTE consumes the fallback entity set and must not raise on
        # empty materialization.
        attribute = SemanticExecutor(semantic, model_role="semantic_2b")
        result = attribute.execute(
            _attach_node(),
            {"entity": outcome.value},
            context,
        )
        assert isinstance(result.value, Label)
    finally:
        composer.close()


def test_final_choice_falls_back_to_image_vlm_on_empty_evidence(
    tmp_path: Path,
) -> None:
    """A choice question whose final source is an EMPTY SelectResult must still
    produce an answer via the input images instead of failing the sample."""
    from sat_rs_vlm.taskgraph.runtime import (
        RuntimeRequest,
        fake_runtime,
    )

    image_path = tmp_path / "source.png"
    Image.new("RGB", (160, 120), "white").save(image_path)
    question = "What color is the flag nearest the bridge?"
    options = ("(A) Red", "(B) Blue", "(C) White")
    graph = {
        "version": "taskgraph-v1.1",
        "question": question,
        "question_type": "MULTIPLE_CHOICE_SINGLE",
        "choices": list(options),
        "inputs": {"image0": {"type": "image", "uri_or_key": "fixture"}},
        "intent": "ATTRIBUTE_QUERY",
        "nodes": [
            {
                "id": "n1",
                "op": "REGION",
                "inputs": {"image": "$image0"},
                "params": {"position": "TOP_RIGHT"},
            },
            {
                "id": "n2",
                "op": "LOCATE",
                "inputs": {"image": "$n1"},
                "params": {"target": {"category": "bridge", "attributes": {}}},
            },
            {
                "id": "n3",
                "op": "LOCATE",
                "inputs": {"image": "$n1"},
                "params": {"target": {"category": "flag", "attributes": {}}},
            },
            {
                "id": "n4",
                "op": "SELECT",
                "inputs": {"candidates": "$n3", "reference": "$n2"},
                "params": {"mode": "RELATION", "relation": "NEAR"},
            },
        ],
        "final": {
            "sources": ["$n4"],
            "question": "What colors are visible on the selected flag?",
            "answer_type": "CHOICE_SINGLE",
        },
    }
    runtime = fake_runtime(detection_boxes=[], choice_responses={"choice": "C"})
    try:
        result = runtime.run(
            RuntimeRequest(
                "recall-fallback",
                "MME_RealWorld_RS",
                "color",
                question,
                (str(image_path),),
                tuple(options),
                graph=graph,
            )
        )
        assert isinstance(result.output, ChoiceResult)
        assert result.output.choice_id == "C"
        assert bool(result.trace.telemetry.get("final_evidence_fallback"))
    finally:
        runtime.close()
