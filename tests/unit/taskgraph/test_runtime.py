from __future__ import annotations

import json
from pathlib import Path

import pytest

from sat_rs_vlm.taskgraph import RuntimeRequest, fake_runtime, parse_taskgraph
from sat_rs_vlm.taskgraph.routing import ExecutionMode
from sat_rs_vlm.taskgraph.runtime_types import (
    Answer,
    ChoiceResult,
    ChoiceScoreResult,
    ScalarInt,
)

IMAGE = str(Path("tests/fixtures/miniature_dataset/images/counting.ppm").resolve())
SECOND_IMAGE = str(Path("tests/fixtures/miniature_dataset/images/vqa.ppm").resolve())


def _graph(
    *,
    question: str,
    options: list[str],
    nodes: list[dict],
    sources: list[str],
    final_question: str,
    intent: str = "OTHER",
) -> dict:
    return {
        "version": "taskgraph-v1.1",
        "question": question,
        "question_type": "MULTIPLE_CHOICE_SINGLE",
        "choices": options,
        "inputs": {"image0": {"type": "image", "uri_or_key": "fixture"}},
        "intent": intent,
        "nodes": nodes,
        "final": {
            "sources": sources,
            "question": final_question,
            "answer_type": "CHOICE_SINGLE",
        },
    }


def _locate(node_id: str, image_ref: str, category: str) -> dict:
    return {
        "id": node_id,
        "op": "LOCATE",
        "inputs": {"image": image_ref},
        "params": {"target": {"category": category, "attributes": {}}},
    }


def test_case_a_high_res_count_is_structured_text_only_choice() -> None:
    question = "How many ships are there?"
    options = ["A 5", "B 6", "C 7", "D 8"]
    graph = _graph(
        question=question,
        options=options,
        nodes=[
            {
                "id": "n1",
                "op": "COUNT",
                "inputs": {"image": "$image0"},
                "params": {
                    "target": {"category": "ship", "attributes": {}},
                    "entire": True,
                },
            }
        ],
        sources=["$n1"],
        final_question="Which option matches this count?",
        intent="SIMPLE_COUNT",
    )
    boxes = [[index, 0, index + 0.5, 0.5] for index in range(7)]
    runtime = fake_runtime(detection_boxes=boxes, choice_responses={"choice": "C"})
    try:
        result = runtime.run(
            RuntimeRequest(
                "case-a",
                "MME_RealWorld_RS",
                "count",
                question,
                (IMAGE,),
                tuple(options),
                graph=graph,
            )
        )
        assert isinstance(result.output, ChoiceResult)
        assert result.output.choice_id == "C"
        assert isinstance(result.store.get("$n1"), ScalarInt)
        assert result.store.get("$n1").value == 7
        assert runtime.choice_resolver.last_model_input.visual_inputs == ()
        assert "value: 7" in runtime.choice_resolver.last_model_input.structured_context
        assert runtime.providers.choice.choice_calls == []
    finally:
        runtime.close()


def test_taskgraph_choice_multi_is_indeterminate_and_not_blocked_by_legacy_label() -> None:
    question = "Select all statements that apply to the number of ships."
    options = ["A At least 1", "B At least 2", "C At least 4", "D More than 10"]
    graph = _graph(
        question=question,
        options=options,
        nodes=[
            {
                "id": "n1",
                "op": "COUNT",
                "inputs": {"image": "$image0"},
                "params": {
                    "target": {"category": "ship", "attributes": {}},
                    "entire": True,
                },
            }
        ],
        sources=["$n1"],
        final_question="Select every statement supported by the resolved count.",
        intent="SIMPLE_COUNT",
    )
    graph["final"]["answer_type"] = "CHOICE_MULTI"
    runtime = fake_runtime(
        detection_boxes=[[0, 0, 1, 1]] * 4,
        choice_responses={"choice_multi": json.dumps({"choice_ids": ["A", "B", "C"]})},
    )
    try:
        result = runtime.run(
            RuntimeRequest(
                "case-a-multi",
                "MME_RealWorld_RS",
                "count",
                question,
                (IMAGE,),
                tuple(options),
                graph=graph,
            )
        )
        assert result.output.choice_ids == ("A", "B", "C")
        assert result.output.choice_id is None
        assert result.output.answer_type == "CHOICE_MULTI"
        assert result.trace.choice_result["answer_type"] == "CHOICE_MULTI"
        assert result.trace.choice_result["choice_id"] is None
        assert result.trace.choice_result["selected_ids"] == ["A", "B", "C"]
        assert runtime.choice_resolver.last_score_result.cache_reused is True
    finally:
        runtime.close()


def test_case_b_compound_color_keeps_original_options_and_selected_visual() -> None:
    question = "What color is the house in the middle area at the top with roads in front and back?"
    options = [
        "A White and purple",
        "B Green and blue",
        "C Black and white",
        "D Gold and green",
        "E This image doesn't feature the color.",
    ]
    graph = _graph(
        question=question,
        options=options,
        nodes=[
            {
                "id": "n1",
                "op": "REGION",
                "inputs": {"image": "$image0"},
                "params": {"position": "TOP_CENTER"},
            },
            _locate("n2", "$n1", "house"),
            {
                "id": "n3",
                "op": "SELECT",
                "inputs": {"candidates": "$n2"},
                "params": {
                    "mode": "EXTREME",
                    "direction": "TOPMOST",
                },
            },
        ],
        sources=["$n3"],
        final_question="What colors are visible on the selected house?",
        intent="ATTRIBUTE_QUERY",
    )
    runtime = fake_runtime(
        detection_boxes=[[1, 1, 6, 6], [8, 5, 12, 10]],
        choice_responses={"choice": "A"},
    )
    try:
        result = runtime.run(
            RuntimeRequest(
                "case-b",
                "MME_RealWorld_RS",
                "attribute",
                question,
                (IMAGE,),
                tuple(options),
                graph=graph,
            )
        )
        assert result.output.choice_id == "A"
        model_input = runtime.choice_resolver.last_model_input
        assert model_input.options == tuple(options)
        assert model_input.question == "What colors are visible on the selected house?"
        assert len(model_input.visual_inputs) == 1
        assert len(runtime.providers.semantic_2b.choice_calls) == 1
    finally:
        runtime.close()


def test_case_c_relational_select_flows_into_entityset_count() -> None:
    question = "How many umbrellas are next to the building?"
    options = ["A 1", "B 2", "C 3"]
    graph = _graph(
        question=question,
        options=options,
        nodes=[
            {
                "id": "n1",
                "op": "REGION",
                "inputs": {"image": "$image0"},
                "params": {"position": "TOP"},
            },
            _locate("n2", "$n1", "building"),
            _locate("n3", "$image0", "umbrella"),
            {
                "id": "n4",
                "op": "SELECT",
                "inputs": {"candidates": "$n3", "reference": "$n2"},
                "params": {
                    "mode": "RELATION",
                    "relation": "NEXT_TO",
                },
            },
            {
                "id": "n5",
                "op": "COUNT",
                "inputs": {"entities": "$n4"},
                "params": {
                    "target": {"category": "umbrella", "attributes": {}},
                    "entire": False,
                },
            },
        ],
        sources=["$n5"],
        final_question="Which option matches this count?",
        intent="RELATIONAL_COUNT",
    )
    runtime = fake_runtime(
        detection_boxes=[[1, 1, 3, 3], [4, 1, 6, 3], [8, 1, 10, 3]],
        semantic_responses={"selection": "0, 2"},
        choice_responses={"choice": "B"},
    )
    try:
        result = runtime.run(
            RuntimeRequest(
                "case-c",
                "XLRS_Bench",
                "relational_count",
                question,
                (IMAGE,),
                tuple(options),
                graph=graph,
            )
        )
        assert result.output.choice_id == "B"
        assert result.store.get("$n5").value == 2
        count_trace = next(item for item in result.trace.nodes if item.node_id == "n5")
        assert count_trace.input_refs == {"entities": "$n4"}
        assert count_trace.provider == "cardinality"
    finally:
        runtime.close()


def test_case_d_object_relation_uses_semantic_relation_provider() -> None:
    question = "Where is A relative to B?"
    options = ["A left", "B right"]
    graph = _graph(
        question=question,
        options=options,
        nodes=[
            _locate("n1", "$image0", "A"),
            _locate("n2", "$image0", "B"),
            {
                "id": "n3",
                "op": "RELATION",
                "inputs": {"subject": "$n1", "reference": "$n2"},
                "params": {},
            },
        ],
        sources=["$n3"],
        final_question="Which option matches the determined spatial relation?",
        intent="OBJECT_RELATION",
    )
    runtime = fake_runtime(
        detection_boxes=[[1, 1, 3, 3]],
        semantic_responses={"relation": "RIGHT_OF"},
        choice_responses={"choice": "B"},
    )
    try:
        result = runtime.run(
            RuntimeRequest(
                "case-d", "XLRS_Bench", "relation", question, (IMAGE,), tuple(options), graph=graph
            )
        )
        assert result.output.choice_id == "B"
        fused = result.store.get("$n3")
        assert isinstance(fused, ChoiceScoreResult)
        assert fused.selected_ids == ("B",)
        assert fused.cache_reused is True
        assert len(runtime.providers.semantic_2b.choice_calls) == 1
    finally:
        runtime.close()


def test_case_e_route_accepts_entityset_start_and_goal() -> None:
    question = "Which route reaches the goal?"
    options = ["A north then east", "B south then west"]
    graph = _graph(
        question=question,
        options=options,
        nodes=[
            {
                "id": "n1",
                "op": "REGION",
                "inputs": {"image": "$image0"},
                "params": {"position": "TOP_LEFT"},
            },
            _locate("n2", "$n1", "start building"),
            {
                "id": "n3",
                "op": "REGION",
                "inputs": {"image": "$image0"},
                "params": {"position": "BOTTOM_RIGHT"},
            },
            _locate("n4", "$n3", "goal building"),
            {
                "id": "n5",
                "op": "BUILD_ROUTE_CONTEXT",
                "inputs": {"image": "$image0", "start": "$n2", "goal": "$n4"},
                "params": {},
            },
            {
                "id": "n6",
                "op": "ROUTE_REASON",
                "inputs": {"context": "$n5"},
                "params": {"question": "$question", "choices": "$choices"},
            },
        ],
        sources=["$n6"],
        final_question="Which option describes the best route?",
        intent="ROUTE_PLANNING",
    )
    runtime = fake_runtime(
        detection_boxes=[[1, 1, 3, 3], [4, 4, 6, 6]],
        route_responses={"route_reason": "route evidence"},
        choice_responses={"choice": "A"},
    )
    try:
        result = runtime.run(
            RuntimeRequest(
                "case-e", "XLRS_Bench", "route", question, (IMAGE,), tuple(options), graph=graph
            )
        )
        assert result.output.choice_id == "A"
        assert result.trace.nodes[-1].provider == "fake_vlm"
        assert len(runtime.providers.route_4b.choice_calls) == 1
        assert runtime.providers.route_4b.choice_calls[0].purpose == "route_choice"
        assert runtime.providers.route_4b.choice_calls[0].model_input.options == tuple(options)
        assert runtime.providers.semantic_2b.choice_calls == []
        route_output = result.store.get("$n6")
        assert route_output.cache_reused is True
    finally:
        runtime.close()


def test_case_f_vrs_vqa_bypasses_planner_and_dag() -> None:
    runtime = fake_runtime(semantic_responses={"direct_vlm": "direct answer"})
    try:
        result = runtime.run(
            RuntimeRequest("case-f", "VRSBench", "vqa", "What is visible?", (IMAGE,))
        )
        assert result.execution_mode is ExecutionMode.DIRECT_VLM
        assert isinstance(result.output, Answer)
        assert result.output.text == "direct answer"
        assert result.trace.taskgraph is None
    finally:
        runtime.close()


def test_case_g_vrs_count_bypasses_taskgraph_and_keeps_choice_text_only() -> None:
    runtime = fake_runtime(
        detection_boxes=[[1, 1, 2, 2], [3, 1, 4, 2]],
        choice_responses={"choice": "B"},
    )
    try:
        result = runtime.run(
            RuntimeRequest(
                "case-g",
                "VRSBench",
                "count",
                "How many cars?",
                (IMAGE,),
                ("A 1", "B 2"),
                target_category="car",
            )
        )
        assert result.execution_mode is ExecutionMode.DIRECT_DETECTION
        assert result.output.choice_id == "B"
        assert runtime.choice_resolver.last_model_input.visual_inputs == ()
    finally:
        runtime.close()


def test_case_h_levir_two_image_direct_path_bypasses_taskgraph() -> None:
    runtime = fake_runtime(semantic_responses={"direct_vlm": "change description"})
    try:
        result = runtime.run(
            RuntimeRequest(
                "case-h",
                "LEVIR_CC",
                "change_caption",
                "Describe the change.",
                (IMAGE, SECOND_IMAGE),
            )
        )
        assert result.execution_mode is ExecutionMode.DIRECT_VLM
        assert result.output.text == "change description"
        assert result.trace.taskgraph is None
    finally:
        runtime.close()


def test_production_schema_parses_lab_v1_1_canonical_fixture() -> None:
    fixture = Path("tests/fixtures/taskgraph/planner_fixtures.json")
    graph = next(iter(__import__("json").loads(fixture.read_text(encoding="utf-8")).values()))
    lab_schema = pytest.importorskip("taskgraph_lab.taskgraph.schema")
    lab_schema.TaskGraph.model_validate(graph)
    parsed = parse_taskgraph(graph)
    assert parsed.version == "taskgraph-v1.1"
    assert parsed.final.sources == ["$n1"]
