"""SELECT v1: deterministic spatial selection with auditable semantic fallback."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from sat_rs_vlm.infrastructure.config import ModelConfig
from sat_rs_vlm.taskgraph.input_composer import InputComposer
from sat_rs_vlm.taskgraph.operators import CountExecutor, OperatorContext, SelectExecutor
from sat_rs_vlm.taskgraph.providers import (
    FakeDetectionProvider,
    FakeSemanticVLMProvider,
    LazyQwenSemanticProvider,
    ModelInput,
    VLMRequest,
    parse_selection_indices,
)
from sat_rs_vlm.taskgraph.runtime_types import (
    Entity,
    EntitySet,
    ImageRef,
    Region,
    ScalarInt,
    SelectResult,
    SelectResultConsumptionError,
    SelectStatus,
    unwrap_select_result,
)
from sat_rs_vlm.taskgraph.schema import GraphNode


def _node(params: dict[str, object]) -> GraphNode:
    return GraphNode.model_validate(
        {
            "id": "n1",
            "op": "SELECT",
            "inputs": {"candidates": "$c", "reference": "$r"},
            "params": params,
        }
    )


def _image(tmp_path: Path) -> ImageRef:
    path = tmp_path / "source.png"
    Image.new("RGB", (200, 120), "white").save(path)
    return ImageRef(str(path), width=200, height=120)


def _entity(image: ImageRef, box: tuple[float, float, float, float], candidate_id: str) -> Entity:
    return Entity(Region(image, box), "target", 0.9, {"candidate_id": candidate_id})


def _context(tmp_path: Path) -> tuple[InputComposer, OperatorContext]:
    composer = InputComposer(tmp_path / "model_inputs")
    return composer, OperatorContext("select", (), composer)


def test_clear_left_relation_uses_geometry_and_preserves_candidate_ids(tmp_path: Path) -> None:
    image = _image(tmp_path)
    candidates = EntitySet(
        (_entity(image, (10, 40, 30, 60), "det-7"), _entity(image, (150, 40, 170, 60), "det-9"))
    )
    reference = _entity(image, (90, 40, 110, 60), "ref-1")
    provider = FakeSemanticVLMProvider({"selection": "B"})
    composer, context = _context(tmp_path)
    try:
        output = SelectExecutor(provider).execute(
            _node({"mode": "RELATION", "relation": "LEFT_OF"}),
            {"candidates": candidates, "reference": reference},
            context,
        )
        assert isinstance(output.value, SelectResult)
        assert output.value.status is SelectStatus.OK
        assert output.value.method == "geometry"
        assert isinstance(output.value.selected, EntitySet)
        assert output.value.selected.entities[0].provenance["candidate_id"] == "det-7"
        assert provider.calls == []
    finally:
        composer.close()


def test_boundary_relation_and_near_use_kv_cached_choice(tmp_path: Path) -> None:
    image = _image(tmp_path)
    candidates = EntitySet((_entity(image, (98, 40, 118, 60), "det-1"),))
    reference = _entity(image, (100, 40, 120, 60), "ref-1")
    provider = FakeSemanticVLMProvider({"selection": "A"})
    composer, context = _context(tmp_path)
    try:
        boundary = SelectExecutor(provider).execute(
            _node({"mode": "RELATION", "relation": "LEFT_OF"}),
            {"candidates": candidates, "reference": reference},
            context,
        )
        assert isinstance(boundary.value, SelectResult)
        assert boundary.value.method == "qwen3_vl_kv_cached_choice"
        assert (
            provider.choice_calls[0].model_input.metadata["candidate_mapping"]["A"]["candidate_id"]
            == "det-1"
        )

        near = SelectExecutor(provider).execute(
            _node({"mode": "RELATION", "relation": "NEAR"}),
            {"candidates": candidates, "reference": reference},
            context,
        )
        assert isinstance(near.value, SelectResult)
        assert near.value.method == "qwen3_vl_kv_cached_choice"
        assert len(provider.choice_calls) == 2
        assert provider.calls == []
    finally:
        composer.close()


def test_deterministic_relation_with_plural_reference_falls_back_to_semantic(
    tmp_path: Path,
) -> None:
    image = _image(tmp_path)
    candidates = EntitySet(
        (
            _entity(image, (10, 40, 30, 60), "det-1"),
            _entity(image, (150, 40, 170, 60), "det-2"),
        )
    )
    # Plural reference: LOCATE("river") returned multiple regions.
    plural_reference = EntitySet(
        (
            _entity(image, (40, 10, 60, 30), "ref-1"),
            _entity(image, (80, 10, 100, 30), "ref-2"),
        )
    )
    provider = FakeSemanticVLMProvider({"selection": "A"})
    composer, context = _context(tmp_path)
    try:
        output = SelectExecutor(provider).execute(
            _node({"mode": "RELATION", "relation": "INSIDE"}),
            {"candidates": candidates, "reference": plural_reference},
            context,
        )
        assert isinstance(output.value, SelectResult)
        # No single reference: the semantic VLM chooses from the grey subset.
        assert output.value.method == "qwen3_vl_kv_cached_choice"
        assert output.value.status is SelectStatus.OK
    finally:
        composer.close()


def test_relation_with_no_geometric_match_falls_back_to_semantic(tmp_path: Path) -> None:
    image = _image(tmp_path)
    # Candidates far from the reference: geometry finds no positive match.
    candidates = EntitySet((_entity(image, (150, 40, 170, 60), "det-1"),))
    reference = _entity(image, (10, 10, 30, 30), "ref-1")
    provider = FakeSemanticVLMProvider({"selection": "A"})
    composer, context = _context(tmp_path)
    try:
        output = SelectExecutor(provider).execute(
            _node({"mode": "RELATION", "relation": "INSIDE"}),
            {"candidates": candidates, "reference": reference},
            context,
        )
        assert isinstance(output.value, SelectResult)
        assert output.value.method == "qwen3_vl_kv_cached_choice"
        assert output.value.status is SelectStatus.OK
    finally:
        composer.close()


def test_subregion_is_computed_from_scope_and_reference_not_reference_inner_half(
    tmp_path: Path,
) -> None:
    image = _image(tmp_path)
    scope = Region(image, (20, 10, 180, 110))
    reference = Region(image, (90, 40, 120, 70))
    candidate_scope = EntitySet((_entity(image, (30, 20, 40, 30), "det-1"),))
    node = _node({"mode": "SUBREGION", "subregion": "LEFT_SIDE", "margin": 5.0})
    provider = FakeSemanticVLMProvider({})
    composer, context = _context(tmp_path)
    try:
        output = SelectExecutor(provider).execute(
            node,
            {"candidates": candidate_scope, "reference": reference, "scope": scope},
            context,
        )
        assert isinstance(output.value, SelectResult)
        assert output.value.status is SelectStatus.OK
        assert isinstance(output.value.selected, Region)
        assert output.value.selected.bbox_xyxy_global == (20.0, 10.0, 95.0, 110.0)
        assert output.value.selected.provenance["coordinate_system"] == "bbox_xyxy_global"
        assert provider.calls == []
    finally:
        composer.close()


def test_unresolved_selection_is_not_silently_counted(tmp_path: Path) -> None:
    image = _image(tmp_path)
    candidates = EntitySet((_entity(image, (10, 10, 20, 20), "det-1"),))
    unresolved = SelectResult(candidates, SelectStatus.UNRESOLVED, "qwen3_vl", "invalid response")
    count_node = GraphNode.model_validate(
        {
            "id": "n2",
            "op": "COUNT",
            "inputs": {"entities": "$select"},
            "params": {"target": {"category": "car"}, "entire": False},
        }
    )
    composer, context = _context(tmp_path)
    try:
        try:
            CountExecutor(FakeDetectionProvider()).execute(
                count_node, {"entities": unresolved}, context
            )
        except ValueError as exc:
            assert "UNRESOLVED" in str(exc)
        else:
            raise AssertionError("COUNT accepted an unresolved SELECT result")

        selected = SelectResult(candidates, SelectStatus.OK, "geometry")
        output = CountExecutor(FakeDetectionProvider()).execute(
            count_node, {"entities": selected}, context
        )
        assert output.value == ScalarInt(1, {"provider": "cardinality", "source": "EntitySet"})
    finally:
        composer.close()


def test_equal_rank_is_reported_as_ambiguous(tmp_path: Path) -> None:
    image = _image(tmp_path)
    candidates = EntitySet(
        (
            _entity(image, (10, 10, 30, 30), "det-1"),
            _entity(image, (40, 10, 60, 30), "det-2"),
        )
    )
    node = _node({"mode": "RANK", "criterion": "bbox_area", "rank": 1, "order": "DESCENDING"})
    provider = FakeSemanticVLMProvider({})
    composer, context = _context(tmp_path)
    try:
        output = SelectExecutor(provider).execute(node, {"candidates": candidates}, context)
        assert isinstance(output.value, SelectResult)
        assert output.value.status is SelectStatus.AMBIGUOUS
        assert output.value.reason == "rank_tie"
        assert isinstance(output.value.selected, EntitySet)
        assert len(output.value.selected.entities) == 2
    finally:
        composer.close()


def test_subregion_uses_single_selected_candidate_when_reference_is_absent(
    tmp_path: Path,
) -> None:
    image = _image(tmp_path)
    candidate_scope = EntitySet((_entity(image, (90, 40, 120, 70), "det-1"),))
    node = _node({"mode": "SUBREGION", "subregion": "ABOVE", "margin": 5.0})
    provider = FakeSemanticVLMProvider({})
    composer, context = _context(tmp_path)
    try:
        output = SelectExecutor(provider).execute(
            node,
            {"candidates": candidate_scope},
            context,
        )
        assert isinstance(output.value, SelectResult)
        assert output.value.status is SelectStatus.OK
        assert isinstance(output.value.selected, Region)
        # ABOVE the single selected candidate, clipped to the image scope.
        assert output.value.selected.bbox_xyxy_global == (0.0, 0.0, 200.0, 45.0)
    finally:
        composer.close()


def test_subregion_requires_reference_when_candidates_are_multi(tmp_path: Path) -> None:
    image = _image(tmp_path)
    candidate_scope = EntitySet(
        (
            _entity(image, (10, 10, 20, 20), "det-1"),
            _entity(image, (40, 10, 50, 20), "det-2"),
        )
    )
    node = _node({"mode": "SUBREGION", "subregion": "ABOVE", "margin": 5.0})
    provider = FakeSemanticVLMProvider({})
    composer, context = _context(tmp_path)
    try:
        output = SelectExecutor(provider).execute(
            node,
            {"candidates": candidate_scope},
            context,
        )
        assert isinstance(output.value, SelectResult)
        # Multi-candidate group falls back to the union extent as reference.
        assert output.value.status is SelectStatus.OK
        assert isinstance(output.value.selected, Region)
        assert output.value.selected.bbox_xyxy_global == (0.0, 0.0, 200.0, 15.0)
    finally:
        composer.close()


def test_select_consumption_error_includes_upstream_reason(tmp_path: Path) -> None:
    image = _image(tmp_path)
    candidates = EntitySet((_entity(image, (10, 10, 20, 20), "det-1"),))
    unresolved = SelectResult(
        candidates,
        SelectStatus.UNRESOLVED,
        "geometry",
        "RELATION requires exactly one reference",
    )
    with pytest.raises(SelectResultConsumptionError) as error:
        unwrap_select_result(
            unresolved,
            allow_empty=False,
            consumer="SELECT.candidates",
        )
    context = str(error.value)
    assert "UNRESOLVED" in context
    assert "RELATION requires exactly one reference" in context


def test_selection_parser_accepts_explicit_ids_but_rejects_prose_counts() -> None:
    assert parse_selection_indices("候选A和C", 3) == (0, 2)
    assert parse_selection_indices("candidate 1, candidate 3", 3) == (0, 2)
    assert parse_selection_indices("NONE", 3) == ()
    try:
        parse_selection_indices("There are 2 objects near the reference.", 3)
    except ValueError as exc:
        assert "no candidate ids" in str(exc)
    else:
        raise AssertionError("a prose count must not be decoded as candidate B")


def test_real_selection_provider_enables_finite_candidate_constraint() -> None:
    class RecordingEngine:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def generate_text(self, **kwargs: object) -> str:
            self.calls.append(kwargs)
            return "A,C"

    provider = LazyQwenSemanticProvider(ModelConfig(model_id="local/qwen3-vl"))
    engine = RecordingEngine()
    provider._engine = engine  # type: ignore[assignment]  # No model is loaded in this unit test.
    result = provider.infer(
        VLMRequest(
            ModelInput(
                visual_inputs=(),
                structured_context="",
                question="select",
                metadata={"candidate_mapping": {"A": {}, "B": {}, "C": {}}},
            ),
            "selection",
        )
    )
    assert result.text == "A,C"
    assert result.metadata["constrained_decoding"] is True
    assert result.metadata["allowed_output_count"] == 8
    assert engine.calls[0]["allowed_outputs"] == (
        "NONE",
        "A",
        "B",
        "C",
        "A,B",
        "A,C",
        "B,C",
        "A,B,C",
    )
