from __future__ import annotations

import json
from pathlib import Path

from taskgraph_lab.taskgraph.canonicalize import canonicalize_target
from taskgraph_lab.taskgraph.validator import validate_candidate


def _target(category: str, **attributes: str) -> dict:
    return {"category": category, "attributes": attributes}


def _count_graph(final: dict | None = None) -> dict:
    return {
        "intent": "SIMPLE_COUNT",
        "nodes": [
            {
                "id": "n1",
                "op": "COUNT",
                "inputs": {"image": "$image0"},
                "params": {"target": _target("ship", size="large"), "entire": True},
            }
        ],
        "final": final
        or {
            "sources": ["$n1"],
            "answer_type": "CHOICE_SINGLE",
        },
    }


def test_final_sources_single_source() -> None:
    graph = _count_graph()
    target, report = validate_candidate(
        graph,
        inputs={"image0": {}},
        question_type="MULTIPLE_CHOICE_SINGLE",
    )
    assert report.valid, report.model_dump()
    assert target is not None and target.final.sources == ["$n1"]
    assert target.final.question is None


def test_legacy_choice_label_does_not_override_teacher_cardinality() -> None:
    graph = {
        "intent": "MULTILABEL_CLASSIFICATION",
        "nodes": [
            {
                "id": "n1",
                "op": "MULTILABEL_CLASSIFY",
                "inputs": {"input": "$image0"},
                "params": {"label_space": ["farmland", "residential"]},
            }
        ],
        "final": {"sources": ["$n1"], "answer_type": "CHOICE_MULTI"},
    }
    for question_type in ("MULTIPLE_CHOICE", "MULTIPLE_CHOICE_SINGLE"):
        _, report = validate_candidate(
            graph,
            inputs={"image0": {}},
            question="Select all land use types shown.",
            question_type=question_type,
        )
        assert report.valid, report.model_dump()


def test_multiple_choice_still_requires_a_choice_family_answer_type() -> None:
    graph = _count_graph()
    graph["final"]["answer_type"] = "INTEGER"
    _, report = validate_candidate(
        graph,
        inputs={"image0": {}},
        question_type="MULTIPLE_CHOICE",
    )
    assert "choice_answer_type_mismatch" in {issue.code for issue in report.errors}


def test_structured_final_without_question_stays_absent_when_canonicalized() -> None:
    canonical = canonicalize_target(_count_graph())
    assert "question" not in canonical["final"]


def test_generic_structured_final_question_is_non_minimal() -> None:
    graph = _count_graph(
        {
            "sources": ["$n1"],
            "question": "Which option matches this count?",
            "answer_type": "CHOICE_SINGLE",
        }
    )
    _, report = validate_candidate(
        graph,
        inputs={"image0": {}},
        question_type="MULTIPLE_CHOICE_SINGLE",
    )
    assert report.valid, report.model_dump()
    assert any(issue.code == "non_minimal_structured_final_question" for issue in report.warnings)


def test_final_sources_multiple_sources_are_all_graph_sinks() -> None:
    graph = {
        "intent": "COMPLEX_REASONING",
        "nodes": [
            {
                "id": "n1",
                "op": "LOCATE",
                "inputs": {"image": "$image0"},
                "params": {"target": _target("building")},
            },
            {
                "id": "n2",
                "op": "LOCATE",
                "inputs": {"image": "$image0"},
                "params": {"target": _target("car")},
            },
        ],
        "final": {
            "sources": ["$n1", "$n2"],
            "question": "Which option best describes the selected objects?",
            "answer_type": "CHOICE_SINGLE",
        },
    }
    _, report = validate_candidate(
        graph,
        inputs={"image0": {}},
        question_type="MULTIPLE_CHOICE_SINGLE",
    )
    assert report.valid, report.model_dump()
    assert not any(issue.code == "dead_node" for issue in report.errors)


def test_visual_final_requires_residual_question() -> None:
    graph = {
        "intent": "ATTRIBUTE_QUERY",
        "nodes": [
            {
                "id": "n1",
                "op": "LOCATE",
                "inputs": {"image": "$image0"},
                "params": {"target": _target("house")},
            }
        ],
        "final": {"sources": ["$n1"], "answer_type": "CHOICE_SINGLE"},
    }
    _, report = validate_candidate(
        graph,
        inputs={"image0": {}},
        question_type="MULTIPLE_CHOICE_SINGLE",
    )
    assert any(issue.code == "missing_residual_final_question" for issue in report.errors)


def test_terminal_semantic_reasoning_uses_final_question_without_vlm_node() -> None:
    graph = {
        "intent": "COMPLEX_REASONING",
        "nodes": [
            {
                "id": "n1",
                "op": "LOCATE",
                "inputs": {"image": "$image0"},
                "params": {"target": _target("pond")},
            },
            {
                "id": "n2",
                "op": "LOCATE",
                "inputs": {"image": "$image0"},
                "params": {"target": _target("farmland")},
            },
        ],
        "final": {
            "sources": ["$n1", "$n2"],
            "question": (
                "What is the most likely purpose of these ponds in this agricultural setting?"
            ),
            "answer_type": "CHOICE_SINGLE",
        },
    }
    _, report = validate_candidate(
        graph,
        inputs={"image0": {}},
        question_type="MULTIPLE_CHOICE_SINGLE",
    )
    assert report.valid, report.model_dump()
    assert all(node["op"] != "VLM_REASON" for node in graph["nodes"])


def test_invalid_final_ref_is_rejected() -> None:
    graph = _count_graph(
        {
            "sources": ["$n9"],
            "question": "Which option matches this count?",
            "answer_type": "CHOICE_SINGLE",
        }
    )
    _, report = validate_candidate(
        graph,
        inputs={"image0": {}},
        question_type="MULTIPLE_CHOICE_SINGLE",
    )
    assert any(issue.code == "missing_final_ref" for issue in report.errors)


def test_final_question_must_not_be_empty() -> None:
    graph = _count_graph({"sources": ["$n1"], "question": "  ", "answer_type": "CHOICE_SINGLE"})
    _, report = validate_candidate(graph, inputs={"image0": {}})
    assert not report.schema_valid
    assert "final.question" in report.errors[0].message


def test_final_sources_must_not_be_empty() -> None:
    graph = _count_graph(
        {
            "sources": [],
            "question": "Which option matches this count?",
            "answer_type": "CHOICE_SINGLE",
        }
    )
    _, report = validate_candidate(graph, inputs={"image0": {}})
    assert not report.schema_valid
    assert "final.sources" in report.errors[0].message


def test_runtime_type_answer_is_not_a_valid_answer_type() -> None:
    graph = _count_graph()
    graph["final"]["answer_type"] = "Answer"
    _, report = validate_candidate(graph, inputs={"image0": {}})
    assert not report.schema_valid
    assert "final.answer_type" in report.errors[0].message


def test_count_final_question_cannot_contain_runtime_number() -> None:
    graph = _count_graph(
        {
            "sources": ["$n1"],
            "question": "The detected count is 7. Which option matches?",
            "answer_type": "CHOICE_SINGLE",
        }
    )
    _, report = validate_candidate(
        graph,
        inputs={"image0": {}},
        question_type="MULTIPLE_CHOICE_SINGLE",
    )
    assert any(issue.code == "runtime_prediction_in_final_question" for issue in report.errors)


def test_original_choices_cannot_enter_teacher_target() -> None:
    graph = _count_graph()
    graph["final"]["choices"] = ["(A) 7", "(B) 8"]
    _, report = validate_candidate(graph, inputs={"image0": {}})
    assert not report.schema_valid
    assert any("final.choices" in issue.message for issue in report.errors)

    graph = {
        "intent": "COMPLEX_REASONING",
        "nodes": [
            {
                "id": "n1",
                "op": "VLM_REASON",
                "inputs": {"image": "$image0"},
                "params": {
                    "question": "$question",
                    "choices": ["(A) copied", "(B) options"],
                },
            }
        ],
        "final": {
            "sources": ["$n1"],
            "question": "Which option best matches the resolved evidence?",
            "answer_type": "CHOICE_SINGLE",
        },
    }
    _, report = validate_candidate(
        graph,
        inputs={"image0": {}},
        question_type="MULTIPLE_CHOICE_SINGLE",
    )
    assert any(issue.code == "original_choices_copied_into_target" for issue in report.errors)


def test_vlm_reason_evidence_list_preserves_each_ref_type() -> None:
    graph = {
        "intent": "COMPLEX_REASONING",
        "nodes": [
            {
                "id": "n1",
                "op": "REGION",
                "inputs": {"image": "$image0"},
                "params": {"position": "TOP"},
            },
            {
                "id": "n2",
                "op": "LOCATE",
                "inputs": {"image": "$n1"},
                "params": {"target": _target("building")},
            },
            {
                "id": "n3",
                "op": "VLM_REASON",
                "inputs": {"image": "$image0", "evidence": ["$n1", "$n2"]},
                "params": {"question": "$question"},
            },
        ],
        "final": {
            "sources": ["$n3"],
            "question": "What conclusion follows from the selected evidence?",
            "answer_type": "TEXT",
        },
    }
    _, report = validate_candidate(graph, inputs={"image0": {}})
    assert report.valid, report.model_dump()
    evidence = report.resolved_input_types["n3"]["evidence"]
    assert evidence == [
        {"ref": "$n1", "types": ["Region"]},
        {"ref": "$n2", "types": ["EntitySet"]},
    ]


def test_relation_requires_named_multi_inputs() -> None:
    graph = {
        "intent": "OBJECT_RELATION",
        "nodes": [
            {
                "id": "n1",
                "op": "LOCATE",
                "inputs": {"image": "$image0"},
                "params": {"target": _target("building")},
            },
            {
                "id": "n2",
                "op": "LOCATE",
                "inputs": {"image": "$image0"},
                "params": {"target": _target("car")},
            },
            {
                "id": "n3",
                "op": "RELATION",
                "inputs": {"subject": "$n1", "reference": "$n2"},
                "params": {},
            },
        ],
        "final": {
            "sources": ["$n3"],
            "answer_type": "CHOICE_SINGLE",
        },
    }
    _, report = validate_candidate(
        graph,
        inputs={"image0": {}},
        question_type="MULTIPLE_CHOICE_SINGLE",
    )
    assert report.valid, report.model_dump()

    graph["nodes"][2]["inputs"] = {"subject": ["$n1", "$n2"]}
    _, invalid = validate_candidate(graph, inputs={"image0": {}})
    assert not invalid.schema_valid


def test_build_route_context_accepts_three_named_inputs() -> None:
    graph = {
        "intent": "ROUTE_PLANNING",
        "nodes": [
            {
                "id": "n1",
                "op": "LOCATE",
                "inputs": {"image": "$image0"},
                "params": {"target": _target("roundabout")},
            },
            {
                "id": "n2",
                "op": "LOCATE",
                "inputs": {"image": "$image0"},
                "params": {"target": _target("pond")},
            },
            {
                "id": "n3",
                "op": "BUILD_ROUTE_CONTEXT",
                "inputs": {"image": "$image0", "start": "$n1", "goal": "$n2"},
                "params": {},
            },
        ],
        "final": {
            "sources": ["$n3"],
            "question": (
                "Which option describes the best route between the selected start and goal?"
            ),
            "answer_type": "CHOICE_SINGLE",
        },
    }
    _, report = validate_candidate(
        graph,
        inputs={"image0": {}},
        question_type="MULTIPLE_CHOICE_SINGLE",
    )
    assert report.valid, report.model_dump()
    assert set(report.resolved_input_types["n3"]) == {"image", "start", "goal"}

    graph["final"] = {"sources": ["$n3"], "answer_type": "CHOICE_SINGLE"}
    _, missing_question = validate_candidate(
        graph,
        inputs={"image0": {}},
        question_type="MULTIPLE_CHOICE_SINGLE",
    )
    assert any(issue.code == "missing_residual_final_question" for issue in missing_question.errors)


def test_color_compound_options_can_use_selected_visual_source() -> None:
    graph = {
        "intent": "ATTRIBUTE_QUERY",
        "nodes": [
            {
                "id": "n1",
                "op": "REGION",
                "inputs": {"image": "$image0"},
                "params": {"position": "TOP_CENTER"},
            },
            {
                "id": "n2",
                "op": "LOCATE",
                "inputs": {"image": "$n1"},
                "params": {"target": _target("house")},
            },
        ],
        "final": {
            "sources": ["$n2"],
            "question": "What colors are visible on the selected house?",
            "answer_type": "CHOICE_SINGLE",
        },
    }
    canonical = canonicalize_target(graph)
    _, report = validate_candidate(
        canonical,
        inputs={"image0": {}},
        question_type="MULTIPLE_CHOICE_SINGLE",
    )
    assert report.valid, report.model_dump()
    assert "choices" not in canonical["final"]
    assert all("choices" not in node["params"] for node in canonical["nodes"])


def test_authoritative_count_does_not_reintroduce_visual_source() -> None:
    graph = _count_graph()
    graph["nodes"].append(
        {
            "id": "n2",
            "op": "REGION",
            "inputs": {"image": "$image0"},
            "params": {"position": "CENTER"},
        }
    )
    graph["final"]["sources"] = ["$n1", "$n2"]
    _, report = validate_candidate(
        graph,
        inputs={"image0": {}},
        question_type="MULTIPLE_CHOICE_SINGLE",
    )
    assert any(issue.code == "authoritative_count_visual_reintroduction" for issue in report.errors)


def test_direct_image_final_source_is_rejected_by_schema() -> None:
    graph = _count_graph()
    graph["final"]["sources"] = ["$image0"]
    _, report = validate_candidate(graph, inputs={"image0": {}})
    assert not report.schema_valid
    assert "only $nX references" in report.errors[0].message


def test_final_choice_few_shots_are_valid_and_do_not_copy_choices() -> None:
    path = Path(__file__).parents[1] / "prompts" / "few_shot_final_choice.txt"
    targets = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith('{"intent"')
    ]
    assert len(targets) == 4
    for target in targets:
        _, report = validate_candidate(
            target,
            inputs={"image0": {}},
            question_type="MULTIPLE_CHOICE_SINGLE",
        )
        assert report.valid, report.model_dump()
        assert "source" not in target["final"]
        assert '"choices"' not in json.dumps(target)
