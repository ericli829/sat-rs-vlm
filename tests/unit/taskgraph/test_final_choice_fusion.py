from __future__ import annotations

from pathlib import Path

import pytest

from sat_rs_vlm.taskgraph import RuntimeRequest, fake_runtime
from sat_rs_vlm.taskgraph.choice_config import ChoiceSystemConfig
from sat_rs_vlm.taskgraph.runtime_types import Answer, ChoiceScoreResult, Label

IMAGE = str(Path("tests/fixtures/miniature_dataset/images/vqa.ppm").resolve())


def _graph(
    nodes: list[dict[str, object]],
    *,
    sources: list[str],
    options: tuple[str, ...],
    answer_type: str = "CHOICE_SINGLE",
) -> dict[str, object]:
    question_type = {
        "CHOICE_SINGLE": "MULTIPLE_CHOICE_SINGLE",
        "CHOICE_MULTI": "MULTIPLE_CHOICE_MULTI",
    }.get(answer_type, "FREE_FORM")
    return {
        "version": "taskgraph-v1.1",
        "question": "What does the selected visual evidence show?",
        "question_type": question_type,
        "choices": list(options) if options else None,
        "inputs": {"image0": {"type": "image", "uri_or_key": "fixture"}},
        "intent": "COMPLEX_REASONING",
        "nodes": nodes,
        "final": {
            "sources": sources,
            "question": "Choose the option supported by the resolved evidence.",
            "answer_type": answer_type,
        },
    }


def _request(graph: dict[str, object], options: tuple[str, ...], sample_id: str) -> RuntimeRequest:
    return RuntimeRequest(
        sample_id,
        "XLRS_Bench",
        "complex_reasoning",
        str(graph["question"]),
        (IMAGE,),
        options,
        graph=graph,
    )


def _vlm(node_id: str, *, evidence: str | None = None) -> dict[str, object]:
    return {
        "id": node_id,
        "op": "VLM_REASON",
        "inputs": {"evidence": evidence} if evidence else {"image": "$image0"},
        "params": {"question": "$question", "choices": None},
    }


def _locate_select() -> list[dict[str, object]]:
    return [
        {
            "id": "n1",
            "op": "LOCATE",
            "inputs": {"image": "$image0"},
            "params": {"target": {"category": "building", "attributes": {}}},
        },
        {
            "id": "n2",
            "op": "SELECT",
            "inputs": {"candidates": "$n1"},
            "params": {"mode": "EXTREME", "direction": "LEFTMOST"},
        },
    ]


def test_final_vlm_reason_fuses_and_choice_resolver_makes_no_second_call() -> None:
    options = ("A airport", "B harbor", "C residential")
    graph = _graph([_vlm("n1")], sources=["$n1"], options=options)
    runtime = fake_runtime(
        semantic_responses={
            "final_vlm_reason_choice_fusion_reasoning": "A prose mention is irrelevant."
        },
        semantic_choice_scores={"final_vlm_reason_choice_fusion": {"A": -2.0, "B": 6.0, "C": 0.0}},
    )
    try:
        result = runtime.run(_request(graph, options, "final-vlm"))
        fused = result.store.get("$n1")
        assert isinstance(fused, ChoiceScoreResult)
        assert fused.selected_ids == ("B",)
        assert fused.cache_reused is True
        assert fused.metadata["visual_prefill_count"] == 1
        assert fused.metadata["reasoning_pass_count"] == 1
        assert result.output.choice_id == "B"
        assert len(runtime.providers.semantic_2b.choice_calls) == 1
        assert runtime.providers.semantic_2b.calls == []
        trace = result.trace.nodes[0]
        assert trace.execution_mode == "final_choice_fused"
        assert trace.semantic_method == "kv_cached_final_choice"
        assert trace.final_choice_fusion is True
        assert trace.fusion_reason == "eligible"
    finally:
        runtime.close()


def test_final_attribute_fusion_sees_original_options_and_transports_score() -> None:
    options = ("A light gray", "B white", "C silver", "D beige")
    nodes = _locate_select() + [
        {
            "id": "n3",
            "op": "ATTRIBUTE",
            "inputs": {"entity": "$n2"},
            "params": {"attribute": "color", "part": None},
        }
    ]
    graph = _graph(nodes, sources=["$n3"], options=options)
    runtime = fake_runtime(
        detection_boxes=[[1, 1, 8, 8]],
        semantic_choice_scores={
            "final_attribute_choice_fusion": {"A": 0.0, "B": 4.0, "C": 1.0, "D": -1.0}
        },
    )
    try:
        result = runtime.run(_request(graph, options, "final-attribute"))
        fused = result.store.get("$n3")
        assert isinstance(fused, ChoiceScoreResult)
        assert result.output.choice_id == "B"
        assert len(runtime.providers.semantic_2b.choice_calls) == 1
        request = runtime.providers.semantic_2b.choice_calls[0]
        assert request.purpose == "final_attribute_choice_fusion"
        assert request.model_input.options == options
        assert "Residual final question" in request.model_input.question
        assert (
            "Choose the option supported by the resolved evidence." in request.model_input.question
        )
        assert str(graph["question"]) not in request.model_input.question
        assert "already selected target" in request.model_input.question
        assert request.model_input.metadata["source_types"] == ["Entity"]
    finally:
        runtime.close()


def test_intermediate_attribute_stays_label_before_final_vlm_fusion() -> None:
    options = ("A white", "B black")
    nodes = _locate_select() + [
        {
            "id": "n3",
            "op": "ATTRIBUTE",
            "inputs": {"entity": "$n2"},
            "params": {"attribute": "color", "part": None},
        },
        _vlm("n4", evidence="$n3"),
    ]
    graph = _graph(nodes, sources=["$n4"], options=options)
    runtime = fake_runtime(
        detection_boxes=[[1, 1, 8, 8]],
        semantic_responses={"attribute": "white"},
        semantic_choice_scores={"final_vlm_reason_choice_fusion": {"A": 3.0, "B": -1.0}},
    )
    try:
        result = runtime.run(_request(graph, options, "intermediate-attribute"))
        attribute = result.store.get("$n3")
        assert isinstance(attribute, Label)
        assert attribute.value == "white"
        assert result.trace.nodes[2].final_choice_fusion is False
        assert result.trace.nodes[2].fusion_reason == "not_final_source"
        assert isinstance(result.store.get("$n4"), ChoiceScoreResult)
    finally:
        runtime.close()


def test_fanout_disables_final_source_fusion() -> None:
    options = ("A white", "B black")
    nodes = _locate_select() + [
        {
            "id": "n3",
            "op": "ATTRIBUTE",
            "inputs": {"entity": "$n2"},
            "params": {"attribute": "color", "part": None},
        },
        _vlm("n4", evidence="$n3"),
    ]
    graph = _graph(nodes, sources=["$n3"], options=options)
    runtime = fake_runtime(
        detection_boxes=[[1, 1, 8, 8]],
        semantic_responses={"attribute": "white", "vlm_reason": "downstream evidence"},
    )
    try:
        result = runtime.run(_request(graph, options, "fanout"))
        assert isinstance(result.store.get("$n3"), Label)
        assert result.output.choice_id == "A"
        trace = result.trace.nodes[2]
        assert trace.final_choice_fusion is False
        assert trace.fusion_reason == "has_downstream_consumer"
        assert runtime.providers.semantic_2b.choice_calls == []
    finally:
        runtime.close()


def test_multiple_final_sources_disable_fusion_and_use_one_final_choice_call() -> None:
    options = ("A first", "B second")
    graph = _graph(
        [_vlm("n1"), _vlm("n2")],
        sources=["$n1", "$n2"],
        options=options,
    )
    runtime = fake_runtime(
        semantic_responses={"vlm_reason": "typed open evidence"},
        semantic_choice_scores={"final_choice": {"A": -1.0, "B": 2.0}},
    )
    try:
        result = runtime.run(_request(graph, options, "multiple-sources"))
        assert isinstance(result.store.get("$n1"), Answer)
        assert isinstance(result.store.get("$n2"), Answer)
        assert result.output.choice_id == "B"
        assert [item.fusion_reason for item in result.trace.nodes] == [
            "multiple_final_sources",
            "multiple_final_sources",
        ]
        assert len(runtime.providers.semantic_2b.choice_calls) == 1
        assert runtime.providers.semantic_2b.choice_calls[0].purpose == "final_choice"
    finally:
        runtime.close()


def test_free_form_final_disables_fusion_and_keeps_answer() -> None:
    graph = _graph([_vlm("n1")], sources=["$n1"], options=(), answer_type="TEXT")
    runtime = fake_runtime(semantic_responses={"vlm_reason": "free answer"})
    try:
        result = runtime.run(_request(graph, (), "free-form"))
        assert isinstance(result.output, Answer)
        assert result.output.text == "free answer"
        assert result.trace.nodes[0].fusion_reason == "free_form_final"
        assert result.trace.nodes[0].execution_mode == "free_text"
        assert runtime.providers.semantic_2b.choice_calls == []
    finally:
        runtime.close()


@pytest.mark.parametrize(
    ("scores", "selected"),
    [
        ({"A": 2.0, "B": -1.0, "C": -2.0}, ("A",)),
        ({"A": 2.0, "B": -1.0, "C": 3.0}, ("A", "C")),
        ({"A": -2.0, "B": -1.0, "C": -3.0}, ()),
    ],
)
def test_choice_multi_final_fusion_preserves_zero_one_many(
    scores: dict[str, float],
    selected: tuple[str, ...],
) -> None:
    options = ("A airport", "B harbor", "C residential")
    graph = _graph(
        [_vlm("n1")],
        sources=["$n1"],
        options=options,
        answer_type="CHOICE_MULTI",
    )
    runtime = fake_runtime(
        semantic_choice_scores={"final_vlm_reason_choice_fusion": scores},
        choice_config=ChoiceSystemConfig(
            multi_select_threshold=0.0,
            multi_empty_policy="UNRESOLVED",
        ),
    )
    try:
        result = runtime.run(_request(graph, options, f"multi-{len(selected)}"))
        assert result.output.selected_ids == selected
        assert result.output.choice_id is None
        assert result.output.answer_type == "CHOICE_MULTI"
        assert len(runtime.providers.semantic_2b.choice_calls) == 1
        if not selected:
            assert result.output.provenance["empty_multi_status"] == "UNRESOLVED"
    finally:
        runtime.close()


@pytest.mark.parametrize(
    ("operator", "prefix_nodes", "inputs", "params", "purpose"),
    [
        (
            "CLASSIFY",
            [],
            {"source": "$image0"},
            {"label_space": ["airport", "harbor"]},
            "final_classify_choice_fusion",
        ),
        (
            "MULTILABEL_CLASSIFY",
            [],
            {"source": "$image0"},
            {"label_space": ["airport", "harbor"]},
            "final_multilabel_classify_choice_fusion",
        ),
        (
            "MOTION",
            [
                {
                    "id": "n1",
                    "op": "REGION",
                    "inputs": {"image": "$image0"},
                    "params": {"position": "TOP"},
                }
            ],
            {"source": "$n1"},
            {},
            "final_motion_choice_fusion",
        ),
        (
            "RELATION",
            [
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
                    "params": {"position": "BOTTOM"},
                },
            ],
            {"subject": "$n1", "reference": "$n2"},
            {},
            "final_relation_choice_fusion",
        ),
    ],
)
def test_other_final_semantic_operators_fuse_directly_to_original_options(
    operator: str,
    prefix_nodes: list[dict[str, object]],
    inputs: dict[str, str],
    params: dict[str, object],
    purpose: str,
) -> None:
    final_id = f"n{len(prefix_nodes) + 1}"
    nodes = prefix_nodes + [{"id": final_id, "op": operator, "inputs": inputs, "params": params}]
    options = ("A first semantic option", "B second semantic option")
    graph = _graph(nodes, sources=[f"${final_id}"], options=options)
    runtime = fake_runtime(semantic_choice_scores={purpose: {"A": -1.0, "B": 2.0}})
    try:
        result = runtime.run(_request(graph, options, f"final-{operator.casefold()}"))
        fused = result.store.get(f"${final_id}")
        assert isinstance(fused, ChoiceScoreResult)
        assert fused.selected_ids == ("B",)
        assert runtime.providers.semantic_2b.choice_calls[-1].purpose == purpose
        assert runtime.providers.semantic_2b.choice_calls[-1].model_input.options == options
    finally:
        runtime.close()
