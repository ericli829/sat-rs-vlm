"""SELECT hardening contracts across geometry, semantics, and downstream use."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from sat_rs_vlm.taskgraph import RuntimeRequest, fake_runtime
from sat_rs_vlm.taskgraph.executor import TaskGraphExecutionError
from sat_rs_vlm.taskgraph.input_composer import InputComposer
from sat_rs_vlm.taskgraph.operators import OperatorContext, SelectExecutor
from sat_rs_vlm.taskgraph.providers import (
    CachedChoiceUnavailableError,
    ChoiceScoringRequest,
    FakeSemanticVLMProvider,
    ModelInput,
    VLMRequest,
    VLMResult,
)
from sat_rs_vlm.taskgraph.runtime_types import (
    Entity,
    EntitySet,
    ImageRef,
    Region,
    RegionSet,
    SelectResult,
    SelectStatus,
)
from sat_rs_vlm.taskgraph.schema import GraphNode


def _image(path: Path, size: tuple[int, int] = (240, 160)) -> ImageRef:
    Image.new("RGB", size, "white").save(path)
    return ImageRef(str(path), width=size[0], height=size[1])


def _entity(
    image: ImageRef,
    box: tuple[float, float, float, float],
    candidate_id: str,
    *,
    score: float | None = 0.9,
) -> Entity:
    return Entity(Region(image, box), "target", score, {"candidate_id": candidate_id})


def _select_node(params: dict[str, object]) -> GraphNode:
    return GraphNode.model_validate(
        {
            "id": "n1",
            "op": "SELECT",
            "inputs": {"candidates": "$c", "reference": "$r"},
            "params": params,
        }
    )


def _node(
    op: str,
    inputs: dict[str, str | list[str]],
    params: dict[str, object],
) -> GraphNode:
    return GraphNode.model_validate(
        {"id": "n1", "op": op, "inputs": inputs, "params": params}
    )


def _execute_select(
    tmp_path: Path,
    provider: object,
    node: GraphNode,
    inputs: dict[str, object],
) -> SelectResult:
    composer = InputComposer(tmp_path / "inputs")
    try:
        outcome = SelectExecutor(provider).execute(  # type: ignore[arg-type]
            node,
            inputs,  # type: ignore[arg-type]
            OperatorContext("select", (), composer),
        )
        assert isinstance(outcome.value, SelectResult)
        return outcome.value
    finally:
        composer.close()


@pytest.mark.parametrize(
    ("candidate_count", "expected"),
    [
        (0, SelectStatus.EMPTY),
        (1, SelectStatus.OK),
        (2, SelectStatus.AMBIGUOUS),
    ],
)
def test_geometry_single_applies_zero_one_many_cardinality(
    tmp_path: Path,
    candidate_count: int,
    expected: SelectStatus,
) -> None:
    image = _image(tmp_path / "single.png")
    candidates = EntitySet(
        tuple(
            _entity(image, (10 + index * 25, 60, 25 + index * 25, 80), f"det-{index}")
            for index in range(candidate_count)
        )
    )
    reference = _entity(image, (120, 60, 140, 80), "ref")
    result = _execute_select(
        tmp_path,
        FakeSemanticVLMProvider({}),
        _select_node(
            {
                "mode": "RELATION",
                "relation": "LEFT_OF",
                "selection_type": "SINGLE",
            }
        ),
        {"candidates": candidates, "reference": reference},
    )
    assert result.status is expected


@pytest.mark.parametrize(
    ("selection_type", "scores", "expected"),
    [
        ("SINGLE", {"A": -1.0, "B": -0.5}, SelectStatus.EMPTY),
        ("SINGLE", {"A": 1.0, "B": -0.5}, SelectStatus.OK),
        ("SINGLE", {"A": 1.0, "B": 0.5}, SelectStatus.AMBIGUOUS),
        ("MULTI", {"A": -1.0, "B": -0.5}, SelectStatus.EMPTY),
        ("MULTI", {"A": 1.0, "B": -0.5}, SelectStatus.OK),
        ("MULTI", {"A": 1.0, "B": 0.5}, SelectStatus.OK),
    ],
)
def test_semantic_select_always_verifies_independently_then_applies_cardinality(
    tmp_path: Path,
    selection_type: str,
    scores: dict[str, float],
    expected: SelectStatus,
) -> None:
    image = _image(tmp_path / f"semantic-{selection_type}.png")
    candidates = EntitySet(
        (
            _entity(image, (20, 40, 40, 60), "det-a"),
            _entity(image, (80, 40, 100, 60), "det-b"),
        )
    )
    reference = _entity(image, (130, 40, 150, 60), "ref")
    provider = FakeSemanticVLMProvider(
        {}, choice_scores={"select_relation": scores}
    )
    result = _execute_select(
        tmp_path,
        provider,
        _select_node(
            {
                "mode": "RELATION",
                "relation": "NEAR",
                "selection_type": selection_type,
            }
        ),
        {"candidates": candidates, "reference": reference},
    )
    assert result.status is expected
    assert provider.choice_calls[0].answer_type == "CHOICE_MULTI"
    assert provider.choice_calls[0].multi_verify_template


@pytest.mark.parametrize(
    "case",
    ["candidate_set", "candidate_reference", "candidate_scope", "region_set"],
)
def test_select_rejects_every_cross_image_combination(tmp_path: Path, case: str) -> None:
    image_a = _image(tmp_path / "a.png")
    image_b = _image(tmp_path / "b.png")
    reference = _entity(image_a, (120, 40, 140, 60), "ref")
    scope: ImageRef | Region = image_a
    if case == "candidate_set":
        candidates: EntitySet | RegionSet = EntitySet(
            (
                _entity(image_a, (10, 40, 30, 60), "a"),
                _entity(image_b, (40, 40, 60, 60), "b"),
            )
        )
    elif case == "region_set":
        candidates = RegionSet(
            (
                Region(image_a, (10, 40, 30, 60)),
                Region(image_b, (40, 40, 60, 60)),
            )
        )
    else:
        candidates = EntitySet((_entity(image_a, (10, 40, 30, 60), "a"),))
        if case == "candidate_reference":
            reference = _entity(image_b, (120, 40, 140, 60), "ref")
        else:
            scope = Region(image_b, (0, 0, 200, 120))
    result = _execute_select(
        tmp_path,
        FakeSemanticVLMProvider({}),
        _select_node({"mode": "RELATION", "relation": "LEFT_OF"}),
        {"candidates": candidates, "reference": reference, "scope": scope},
    )
    assert result.status is SelectStatus.UNRESOLVED
    assert result.reason == "cross_image_select_inputs"


def test_relation_fallback_sends_only_grey_and_preserves_clear_geometry(
    tmp_path: Path,
) -> None:
    image = _image(tmp_path / "grey.png")
    candidates = EntitySet(
        (
            _entity(image, (10, 50, 30, 70), "det-a"),
            _entity(image, (98, 50, 118, 70), "det-b"),
            _entity(image, (170, 50, 190, 70), "det-c"),
        )
    )
    reference = _entity(image, (100, 50, 120, 70), "ref")
    provider = FakeSemanticVLMProvider({"selection": "A"})
    result = _execute_select(
        tmp_path,
        provider,
        _select_node({"mode": "RELATION", "relation": "LEFT_OF"}),
        {"candidates": candidates, "reference": reference},
    )
    mapping = provider.choice_calls[0].model_input.metadata["candidate_mapping"]
    assert list(mapping) == ["A"]
    assert mapping["A"]["candidate_id"] == "det-b"
    assert result.provenance["clear_positive_candidate_ids"] == ["det-a"]
    assert result.provenance["grey_candidate_ids"] == ["det-b"]
    assert result.provenance["semantic_positive_candidate_ids"] == ["det-b"]
    assert result.provenance["final_candidate_ids"] == ["det-a", "det-b"]
    assert "det-c" not in result.provenance["final_candidate_ids"]


def test_single_with_two_clear_positives_skips_grey_verification(tmp_path: Path) -> None:
    image = _image(tmp_path / "single-clear.png")
    candidates = EntitySet(
        (
            _entity(image, (10, 50, 30, 70), "det-a"),
            _entity(image, (45, 50, 65, 70), "det-b"),
            _entity(image, (98, 50, 118, 70), "det-grey"),
        )
    )
    reference = _entity(image, (100, 50, 120, 70), "ref")
    provider = FakeSemanticVLMProvider({"selection": "A"})
    result = _execute_select(
        tmp_path,
        provider,
        _select_node(
            {
                "mode": "RELATION",
                "relation": "LEFT_OF",
                "selection_type": "SINGLE",
            }
        ),
        {"candidates": candidates, "reference": reference},
    )
    assert result.status is SelectStatus.AMBIGUOUS
    assert result.provenance["clear_positive_candidate_ids"] == ["det-a", "det-b"]
    assert provider.choice_calls == []


def test_region_scope_default_margin_uses_scope_dimensions(tmp_path: Path) -> None:
    image = _image(tmp_path / "uhr.png", (10000, 10000))
    scope = Region(image, (1000, 1000, 1800, 1800))
    candidates = EntitySet((_entity(image, (1100, 1200, 1120, 1220), "det"),))
    reference = _entity(image, (1500, 1200, 1520, 1220), "ref")
    result = _execute_select(
        tmp_path,
        FakeSemanticVLMProvider({}),
        _select_node({"mode": "RELATION", "relation": "LEFT_OF"}),
        {"candidates": candidates, "reference": reference, "scope": scope},
    )
    assert result.provenance["margin_px"] == 16.0


class _UnavailableProvider:
    provider_name = "unavailable"

    def __init__(self, *, constrained: bool = True, text: str = "A") -> None:
        self.constrained = constrained
        self.text = text
        self.reason_calls = 0
        self.infer_calls = 0

    def reason_and_choose(self, request: ChoiceScoringRequest) -> object:
        self.reason_calls += 1
        raise CachedChoiceUnavailableError("cached API unavailable")

    def infer(self, request: VLMRequest) -> VLMResult:
        self.infer_calls += 1
        return VLMResult(
            self.text,
            self.provider_name,
            metadata={"constrained_decoding": self.constrained},
        )

    def close(self) -> None:
        return None


class _BrokenProvider(_UnavailableProvider):
    def reason_and_choose(self, request: ChoiceScoringRequest) -> object:
        self.reason_calls += 1
        raise RuntimeError("CUDA OOM sentinel")


def test_explicit_cache_unavailable_uses_finite_constrained_fallback(tmp_path: Path) -> None:
    image = _image(tmp_path / "fallback.png")
    candidates = EntitySet((_entity(image, (20, 40, 40, 60), "det-a"),))
    reference = _entity(image, (120, 40, 140, 60), "ref")
    provider = _UnavailableProvider()
    result = _execute_select(
        tmp_path,
        provider,
        _select_node({"mode": "RELATION", "relation": "NEAR"}),
        {"candidates": candidates, "reference": reference},
    )
    assert result.status is SelectStatus.OK
    assert result.method == "qwen3_vl_token_mask_fallback"
    assert result.provenance["fallback_used"] is True
    assert result.provenance["fallback_type"] == "CachedChoiceUnavailableError"


def test_unexpected_cached_choice_runtime_error_is_not_swallowed(tmp_path: Path) -> None:
    image = _image(tmp_path / "broken.png")
    candidates = EntitySet((_entity(image, (20, 40, 40, 60), "det-a"),))
    reference = _entity(image, (120, 40, 140, 60), "ref")
    provider = _BrokenProvider()
    with pytest.raises(RuntimeError, match="CUDA OOM sentinel"):
        _execute_select(
            tmp_path,
            provider,
            _select_node({"mode": "RELATION", "relation": "NEAR"}),
            {"candidates": candidates, "reference": reference},
        )
    assert provider.infer_calls == 0


def test_more_than_eight_candidates_never_use_unrestricted_fallback(tmp_path: Path) -> None:
    image = _image(tmp_path / "many.png", (500, 200))
    candidates = EntitySet(
        tuple(
            _entity(image, (10 + index * 40, 40, 30 + index * 40, 60), f"det-{index}")
            for index in range(9)
        )
    )
    reference = _entity(image, (430, 40, 450, 60), "ref")
    provider = _UnavailableProvider()
    result = _execute_select(
        tmp_path,
        provider,
        _select_node({"mode": "RELATION", "relation": "NEAR"}),
        {"candidates": candidates, "reference": reference},
    )
    assert result.status is SelectStatus.UNRESOLVED
    assert result.reason == "safe_fallback_unavailable"
    assert provider.infer_calls == 0


def test_rank_criterion_is_finite_and_missing_score_is_unresolved(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        _select_node(
            {
                "mode": "RANK",
                "criterion": "detector_confidence_typo",
                "rank": 1,
                "order": "DESCENDING",
            }
        )
    image = _image(tmp_path / "rank.png")
    candidates = EntitySet(
        (
            _entity(image, (10, 10, 30, 30), "a", score=0.9),
            _entity(image, (40, 10, 60, 30), "b", score=None),
        )
    )
    result = _execute_select(
        tmp_path,
        FakeSemanticVLMProvider({}),
        _select_node(
            {
                "mode": "RANK",
                "criterion": "score",
                "rank": 1,
                "order": "DESCENDING",
            }
        ),
        {"candidates": candidates},
    )
    assert result.status is SelectStatus.UNRESOLVED
    assert result.reason == "rank_score_missing"


def test_ordinal_tie_uses_only_primary_axis(tmp_path: Path) -> None:
    image = _image(tmp_path / "ordinal.png")
    candidates = EntitySet(
        (
            _entity(image, (10, 20, 30, 40), "a"),
            _entity(image, (80, 20, 100, 40), "b"),
        )
    )
    result = _execute_select(
        tmp_path,
        FakeSemanticVLMProvider({}),
        _select_node({"mode": "ORDINAL", "index": 1, "order": "TOP_TO_BOTTOM"}),
        {"candidates": candidates},
    )
    assert result.status is SelectStatus.AMBIGUOUS
    assert isinstance(result.selected, EntitySet)
    assert len(result.selected.entities) == 2


def test_downstream_router_unwraps_single_and_set_consumers(tmp_path: Path) -> None:
    image = _image(tmp_path / "downstream.png")
    entities = EntitySet(
        (
            _entity(image, (10, 10, 30, 30), "a"),
            _entity(image, (50, 10, 90, 50), "b"),
        )
    )
    singleton = SelectResult(EntitySet((entities.entities[0],)), SelectStatus.OK, "geometry")
    selected_set = SelectResult(entities, SelectStatus.OK, "geometry")
    runtime = fake_runtime(
        semantic_responses={
            "attribute": "red",
            "classify": "vehicle",
            "multilabel_classify": "red,vehicle",
            "motion": "yes",
            "relation": "LEFT_OF",
            "vlm_reason": "supported",
        }
    )
    context = OperatorContext("question", (), runtime.composer)
    router = runtime.graph_executor.router
    try:
        semantic_cases = (
            ("ATTRIBUTE", {"entity": singleton}, {"attribute": "color"}),
            ("CLASSIFY", {"source": singleton}, {}),
            (
                "MULTILABEL_CLASSIFY",
                {"source": singleton},
                {"label_space": ["red", "vehicle"]},
            ),
            ("MOTION", {"source": singleton}, {}),
            (
                "RELATION",
                {"subject": singleton, "reference": entities.entities[1]},
                {},
            ),
            (
                "VLM_REASON",
                {"evidence": singleton},
                {"question": "$question"},
            ),
        )
        for op, values, params in semantic_cases:
            outcome, _ = router.execute(
                _node(op, {key: f"${key}" for key in values}, params),
                values,
                context,
            )
            assert outcome.value is not None

        count, _ = router.execute(
            _node(
                "COUNT",
                {"entities": "$selected"},
                {
                    "target": {"category": "target", "attributes": {}},
                    "entire": False,
                },
            ),
            {"entities": selected_set},
            context,
        )
        assert count.value.value == 2

        grouped, _ = router.execute(
            _node("GROUP", {"entities": "$selected"}, {"mode": "ROW"}),
            {"entities": selected_set},
            context,
        )
        assert isinstance(grouped.value, EntitySet)

        second_select, _ = router.execute(
            _node(
                "SELECT",
                {"candidates": "$selected"},
                {
                    "mode": "RANK",
                    "criterion": "bbox_area",
                    "rank": 1,
                    "order": "DESCENDING",
                },
            ),
            {"candidates": selected_set},
            context,
        )
        assert isinstance(second_select.value, SelectResult)
        assert second_select.value.status is SelectStatus.OK
    finally:
        runtime.close()


def test_ambiguous_select_is_rejected_by_single_consumer_and_composer(tmp_path: Path) -> None:
    image = _image(tmp_path / "ambiguous.png")
    entities = EntitySet(
        (
            _entity(image, (10, 10, 30, 30), "a"),
            _entity(image, (50, 10, 70, 30), "b"),
        )
    )
    ambiguous = SelectResult(entities, SelectStatus.AMBIGUOUS, "geometry", "tie")
    runtime = fake_runtime()
    context = OperatorContext("question", (), runtime.composer)
    try:
        with pytest.raises(TaskGraphExecutionError, match="AMBIGUOUS"):
            runtime.graph_executor.router.execute(
                _node(
                    "ATTRIBUTE",
                    {"entity": "$selected"},
                    {"attribute": "color"},
                ),
                {"entity": ambiguous},
                context,
            )
        with pytest.raises(ValueError, match="AMBIGUOUS"):
            runtime.composer.compose([ambiguous], question="final")
    finally:
        runtime.close()


def test_unresolved_select_final_source_is_rejected(tmp_path: Path) -> None:
    image_path = tmp_path / "final.png"
    _image(image_path)
    graph = {
        "version": "taskgraph-v1.1",
        "question": "Which target is largest?",
        "question_type": "FREE_FORM",
        "choices": None,
        "inputs": {"image0": {"type": "image", "uri_or_key": "fixture"}},
        "nodes": [
            {
                "id": "n1",
                "op": "REGION",
                "inputs": {"image": "$image0"},
                "params": {"position": "TOP"},
            },
            {
                "id": "n2",
                "op": "REGION",
                "inputs": {"image": "$image0"},
                "params": {"position": "CENTER"},
            },
            {
                "id": "n3",
                "op": "SELECT",
                "inputs": {
                    "candidates": "$n1",
                    "reference": "$n2",
                    "scope": "$image0",
                },
                "params": {
                    "mode": "SUBREGION",
                    "subregion": "OUTSIDE",
                },
            },
        ],
        "final": {
            "sources": ["$n3"],
            "question": "Return the selected target.",
            "answer_type": "TEXT",
        },
    }
    runtime = fake_runtime()
    try:
        with pytest.raises(ValueError, match="UNRESOLVED"):
            runtime.run(
                RuntimeRequest(
                    "select-final",
                    "XLRS_Bench",
                    "attribute",
                    "Which target is largest?",
                    (str(image_path),),
                    graph=graph,
                )
            )
    finally:
        runtime.close()


def test_choice_request_shape_remains_cache_compatible() -> None:
    request = ChoiceScoringRequest(
        model_input=ModelInput((), "", "select"),
        answer_type="CHOICE_MULTI",
        choice_ids=("A",),
        option_texts=("Candidate A",),
        single_choice_suffix="Final:",
        multi_verify_template="{choice_id}: {option_text}",
        purpose="select_relation",
    )
    assert request.answer_type == "CHOICE_MULTI"
