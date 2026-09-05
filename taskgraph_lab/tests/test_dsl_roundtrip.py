from __future__ import annotations

import json
from pathlib import Path

import pytest

from taskgraph_lab.taskgraph.canonicalize import canonicalize_target
from taskgraph_lab.taskgraph.dsl import compile_taskgraph_to_dsl, parse_taskgraph_dsl
from taskgraph_lab.tests.dsl_fixtures import representative_graphs

GRAPHS = representative_graphs()


@pytest.mark.parametrize("name", sorted(GRAPHS))
def test_json_dsl_json_round_trip(name: str) -> None:
    original = canonicalize_target(GRAPHS[name])
    dsl = compile_taskgraph_to_dsl(original)
    restored = canonicalize_target(parse_taskgraph_dsl(dsl))
    assert restored == original


@pytest.mark.parametrize("name", sorted(GRAPHS))
def test_canonical_dsl_is_byte_stable(name: str) -> None:
    dsl = compile_taskgraph_to_dsl(GRAPHS[name])
    assert compile_taskgraph_to_dsl(parse_taskgraph_dsl(dsl)) == dsl


def test_whitespace_flexible_input_normalizes_to_canonical_dsl() -> None:
    loose = """
        INTENT( SIMPLE_COUNT )
        n1 = COUNT( $image0 , T( "ship", size = "large" ) , true )
        FINAL( $n1 , CHOICE_SINGLE )
    """
    canonical = (
        "INTENT(SIMPLE_COUNT)\n"
        'n1=COUNT($image0,T("ship",size="large"),true)\n'
        "FINAL($n1,CHOICE_SINGLE)"
    )
    assert compile_taskgraph_to_dsl(parse_taskgraph_dsl(loose)) == canonical


def test_ambiguous_count_role_uses_explicit_reversible_surface() -> None:
    dsl = compile_taskgraph_to_dsl(GRAPHS["relational_count"])
    assert 'n5=COUNT_ENTITIES($n4,T("sun umbrella",color="red"),false)' in dsl


def test_teacher_few_shot_targets_round_trip_without_rewriting_fixtures() -> None:
    path = Path(__file__).parents[1] / "prompts" / "few_shot_final_choice.txt"
    targets = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith('{"intent"')
    ]
    assert len(targets) == 4
    for target in targets:
        restored = canonicalize_target(
            parse_taskgraph_dsl(compile_taskgraph_to_dsl(target))
        )
        assert restored == canonicalize_target(target)


def test_compiler_is_deterministic_and_target_attributes_are_sorted() -> None:
    graph = GRAPHS["escaped_and_all_attributes"]
    first = compile_taskgraph_to_dsl(graph)
    second = compile_taskgraph_to_dsl(graph)
    assert first == second
    target_text = first.split("T(", 1)[1].split(")", 1)[0]
    keys = [item.split("=", 1)[0] for item in target_text.split(",")[1:]]
    assert keys == sorted(keys)


def test_structured_and_question_finals_have_distinct_surface_forms() -> None:
    assert compile_taskgraph_to_dsl(GRAPHS["whole_image_count"]).endswith(
        "FINAL($n1,CHOICE_SINGLE)"
    )
    route_dsl = compile_taskgraph_to_dsl(GRAPHS["route_context"])
    assert "FINAL_QUESTION($n3,CHOICE_SINGLE," in route_dsl


def test_optional_intent_absence_round_trips() -> None:
    graph = dict(GRAPHS["whole_image_count"])
    graph.pop("intent")
    dsl = compile_taskgraph_to_dsl(graph)
    assert not dsl.startswith("INTENT(")
    assert "intent" not in canonicalize_target(parse_taskgraph_dsl(dsl))
