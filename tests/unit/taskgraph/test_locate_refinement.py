from __future__ import annotations

from pathlib import Path

from PIL import Image

from sat_rs_vlm.taskgraph.choice import ChoiceRequest, ChoiceResolver
from sat_rs_vlm.taskgraph.choice_config import ChoiceSystemConfig
from sat_rs_vlm.taskgraph.input_composer import InputComposer
from sat_rs_vlm.taskgraph.operators import CountExecutor, LocateExecutor, OperatorContext
from sat_rs_vlm.taskgraph.providers import (
    FakeDetectionProvider,
    FakeRegionRetriever,
    FakeSemanticVLMProvider,
)
from sat_rs_vlm.taskgraph.referent_refinement import (
    ReferentRefinementConfig,
    ReferentRefiner,
)
from sat_rs_vlm.taskgraph.runtime_types import Entity, EntitySet, ImageRef, Region
from sat_rs_vlm.taskgraph.schema import AnswerType, GraphNode, TargetSpec
from sat_rs_vlm.taskgraph.spatial_choice import SpatialPositionChoiceResolver


def _entities(tmp_path: Path, count: int = 5) -> EntitySet:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (100, 100), "white").save(image_path)
    image = ImageRef(str(image_path), width=100, height=100)
    values = tuple(
        Entity(
            Region(image, (index * 15.0, 10.0, index * 15.0 + 10.0, 20.0)),
            "building",
            0.5,
            {"candidate_id": f"candidate_{index + 1:04d}"},
        )
        for index in range(count)
    )
    return EntitySet(values, {"resolution_status": "MULTIPLE_VALID"})


def test_attribute_candidates_are_semantically_refined_to_singleton(tmp_path: Path) -> None:
    semantic = FakeSemanticVLMProvider(
        choice_scores={"referent_refinement": {"A": 0.1, "B": 0.2, "C": 0.9, "D": 0.1, "E": 0.1}}
    )
    composer = InputComposer(tmp_path / "artifacts")
    refiner = ReferentRefiner(
        semantic,
        composer,
        ReferentRefinementConfig(enabled=True, geometry_weight=0.0),
    )

    result = refiner.refine(
        _entities(tmp_path),
        question="What color is the white cylindrical building?",
        target=TargetSpec(
            category="building", attributes={"color": "white", "shape": "cylindrical"}
        ),
        trigger_reason="target_attributes",
    )

    assert len(result.entities.entities) == 1
    assert result.metadata["resolution_status"] == "REFINED_RESOLVED"
    assert result.metadata["selected_candidate_ids"] == ["candidate_0003"]
    assert result.metadata["input_candidate_count"] == 5
    assert semantic.choice_calls[0].purpose == "referent_refinement"
    composer.close()


def test_unresolved_refinement_keeps_bounded_candidates_for_fallback(tmp_path: Path) -> None:
    semantic = FakeSemanticVLMProvider(
        choice_scores={"referent_refinement": {"A": 1.0, "B": 1.0, "C": 0.1, "D": 0.1, "E": 0.1}}
    )
    composer = InputComposer(tmp_path / "artifacts")
    refiner = ReferentRefiner(
        semantic,
        composer,
        ReferentRefinementConfig(enabled=True, geometry_weight=0.0, minimum_margin=0.2),
    )

    result = refiner.refine(
        _entities(tmp_path),
        question="What color is the selected boat?",
        target=TargetSpec(category="boat"),
        trigger_reason="direct_final_visual_source",
    )

    assert len(result.entities.entities) == 5
    assert result.metadata["resolution_status"] == "UNRESOLVED"
    assert result.entities.provenance["fallback_required"] is True
    composer.close()


def test_empty_proposals_get_one_scoped_visual_fallback_entity(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (100, 80), "white").save(image_path)
    image = ImageRef(str(image_path), width=100, height=80)
    semantic = FakeSemanticVLMProvider()
    composer = InputComposer(tmp_path / "artifacts")
    refiner = ReferentRefiner(
        semantic,
        composer,
        ReferentRefinementConfig(enabled=True),
    )

    result = refiner.visual_fallback(
        image,
        question="What color is the plane?",
        target=TargetSpec(category="plane"),
        reason="EMPTY_PROPOSALS",
    )

    assert len(result.entities.entities) == 1
    assert result.entities.provenance["fallback_kind"] == "semantic_visual"
    assert result.metadata["resolution_status"] == "SEMANTIC_FALLBACK_RESOLVED"
    composer.close()


def test_position_geometry_resolver_uses_global_bbox_without_model(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (100, 100), "white").save(image_path)
    image = ImageRef(str(image_path), width=100, height=100)
    source = EntitySet(
        (Entity(Region(image, (75.0, 40.0, 95.0, 60.0)), "boat", 0.9),),
        {"resolution_status": "REFINED_RESOLVED"},
    )
    semantic = FakeSemanticVLMProvider(default="A")
    resolver = ChoiceResolver(semantic, InputComposer(tmp_path / "artifacts"), ChoiceSystemConfig())

    result = resolver.resolve(
        ChoiceRequest(
            (source,),
            "Where is the boat?",
            ("(A) In the left area of the picture", "(B) In the right area of the picture"),
            AnswerType.CHOICE_SINGLE,
        )
    )

    assert result.choice_id == "B"
    assert result.provenance["provider"] == "spatial_position_geometry"
    assert semantic.choice_calls == []


def test_position_geometry_handles_middle_right_edge_and_absence(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (100, 100), "white").save(image_path)
    image = ImageRef(str(image_path), width=100, height=100)
    resolver = SpatialPositionChoiceResolver()
    source = EntitySet(
        (Entity(Region(image, (92.0, 45.0, 98.0, 55.0)), "boat", 0.9),),
        {"resolution_status": "PRIMARY_RESOLVED"},
    )

    edge = resolver.resolve(
        (source,),
        (
            "(A) In the left area of the picture",
            "(B) In the middle of the right edge of the picture.",
        ),
        question="Where is the boat?",
    )
    empty = resolver.resolve(
        (EntitySet((), {"resolution_status": "EMPTY"}),),
        ("(A) In the right area of the picture", "(B) This image doesn't feature the position."),
        question="Where is the boat?",
    )

    assert edge is not None and edge.selected_ids == ("B",)
    assert empty is not None and empty.selected_ids == ("B",)


def _locate_node() -> GraphNode:
    return GraphNode.model_validate(
        {
            "id": "n1",
            "op": "LOCATE",
            "inputs": {"image": "$image"},
            "params": {"target": {"category": "building"}},
        }
    )


def test_empty_detector_enters_traced_visual_fallback(tmp_path: Path) -> None:
    image_path = tmp_path / "empty.png"
    Image.new("RGB", (100, 80), "white").save(image_path)
    image = ImageRef(str(image_path), width=100, height=80)
    semantic = FakeSemanticVLMProvider()
    composer = InputComposer(tmp_path / "artifacts")
    refiner = ReferentRefiner(
        semantic, composer, ReferentRefinementConfig(enabled=True)
    )
    node = _locate_node()
    context = OperatorContext(
        "What color is the building?",
        ("(A) red", "(B) blue"),
        composer,
        final_sources=("$n1",),
        final_question="What color is the building?",
        graph_nodes=(node,),
    )

    outcome = LocateExecutor(
        FakeDetectionProvider(), FakeRegionRetriever(), refiner=refiner
    ).execute(node, {"image": image}, context)

    assert len(outcome.value.entities) == 1
    assert outcome.trace_metadata["primary_candidate_count"] == 0
    assert outcome.trace_metadata["fallback_reason"] == "EMPTY_PROPOSALS"
    assert outcome.trace_metadata["final_resolution_status"] == "SEMANTIC_FALLBACK_RESOLVED"
    assert outcome.value.provenance["fallback_required"] is True
    composer.close()


def test_unresolved_candidates_use_one_bounded_semantic_choice_fallback(tmp_path: Path) -> None:
    semantic = FakeSemanticVLMProvider(
        choice_scores={
            "referent_refinement": {"A": 1.0, "B": 1.0, "C": 0.1, "D": 0.1, "E": 0.1},
            "semantic_candidate_fallback": {"A": 0.1, "B": 0.9, "C": 0.1, "D": 0.1, "E": 0.1},
        }
    )
    composer = InputComposer(tmp_path / "artifacts")
    refiner = ReferentRefiner(
        semantic,
        composer,
        ReferentRefinementConfig(enabled=True, geometry_weight=0.0, minimum_margin=0.2),
    )
    unresolved = refiner.refine(
        _entities(tmp_path),
        question="What color is the selected boat?",
        target=TargetSpec(category="boat"),
        trigger_reason="direct_final_visual_source",
    )

    resolver = ChoiceResolver(semantic, composer, ChoiceSystemConfig())
    result = resolver.resolve(
        ChoiceRequest(
            (unresolved.entities,),
            "What color is the selected boat?",
            ("(A) red", "(B) blue", "(C) green", "(D) white", "(E) black"),
            AnswerType.CHOICE_SINGLE,
        )
    )

    assert result.choice_id == "B"
    assert [call.purpose for call in semantic.choice_calls] == [
        "referent_refinement",
        "semantic_candidate_fallback",
    ]
    assert result.provenance["fallback_scope"] == "bounded_candidate_set"
    assert result.provenance["final_resolution_status"] == "SEMANTIC_FALLBACK_RESOLVED"
    composer.close()


def test_locate_count_preserves_all_candidates_and_skips_semantic_refinement(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "count.png"
    Image.new("RGB", (100, 100), "white").save(image_path)
    image = ImageRef(str(image_path), width=100, height=100)
    semantic = FakeSemanticVLMProvider()
    composer = InputComposer(tmp_path / "artifacts")
    refiner = ReferentRefiner(
        semantic, composer, ReferentRefinementConfig(enabled=True)
    )
    locate_node = _locate_node()
    count_node = GraphNode.model_validate(
        {
            "id": "n2",
            "op": "COUNT",
            "inputs": {"entities": "$n1"},
            "params": {"target": {"category": "building"}, "entire": False},
        }
    )
    context = OperatorContext(
        "How many buildings are there?",
        ("(A) 5", "(B) 2"),
        composer,
        final_sources=("$n2",),
        final_question="How many buildings are there?",
        graph_nodes=(locate_node, count_node),
    )
    boxes = [(index * 15.0, 10.0, index * 15.0 + 10.0, 20.0) for index in range(5)]
    outcome = LocateExecutor(
        FakeDetectionProvider(boxes), FakeRegionRetriever(), refiner=refiner
    ).execute(locate_node, {"image": image}, context)
    counted = CountExecutor(FakeDetectionProvider()).execute(
        count_node,
        {"entities": outcome.value},
        context,
    )

    assert len(outcome.value.entities) == 5
    assert outcome.trace_metadata["refinement_status"] == "MULTIPLE_VALID"
    assert semantic.choice_calls == []
    assert counted.value.value == 5
    composer.close()


def test_refinement_provider_error_is_bounded_as_unresolved(tmp_path: Path) -> None:
    class RaisingSemantic(FakeSemanticVLMProvider):
        def reason_and_choose(self, request: object) -> object:
            raise RuntimeError("semantic unavailable")

    semantic = RaisingSemantic()
    composer = InputComposer(tmp_path / "artifacts")
    refiner = ReferentRefiner(
        semantic, composer, ReferentRefinementConfig(enabled=True)
    )

    result = refiner.refine(
        _entities(tmp_path),
        question="What color is the selected boat?",
        target=TargetSpec(category="boat"),
        trigger_reason="target_attributes",
    )

    assert result.metadata["resolution_status"] == "UNRESOLVED"
    assert result.entities.provenance["fallback_required"] is True
    assert "semantic unavailable" in result.metadata["failure_reason"]
    composer.close()
