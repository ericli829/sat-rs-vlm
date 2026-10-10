from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from sat_rs_vlm.taskgraph.schema import GraphNode as ProductionNode
from taskgraph_lab.taskgraph.canonicalize import canonicalize_target
from taskgraph_lab.taskgraph.dsl import (
    CanonicalDSLPrefixGrammar,
    DSLCompileError,
    DSLParseError,
    compile_taskgraph_to_dsl,
    parse_taskgraph_dsl,
    parse_taskgraph_dsl_payload,
)
from taskgraph_lab.taskgraph.schema import PlannerTarget
from taskgraph_lab.taskgraph.type_checker import check_types
from taskgraph_lab.taskgraph.validator import validate_candidate
from taskgraph_lab.tools.export_sft import main as export_sft_main

QUERIES = (
    "the building near the road",
    '右边有树、下面有草地的建筑，屋顶写着 "A"。',
    'roof "A"; path C:\\tiles\\north; newline\nnext\tcolumn\rreturn\bback\fform / slash',
    'literal $n99, null, SINGLE and "),MULTI,$image1) are part of the query',
)


def _graph(*, scope=None, reference=None, cardinality="SINGLE", query=QUERIES[0]):
    inputs = {"candidates": "$n2"}
    if reference is not None:
        inputs["reference"] = reference
    if scope is not None:
        inputs["scope"] = scope
    return {
        "intent": "OTHER",
        "nodes": [
            {
                "id": "n1",
                "op": "REGION",
                "inputs": {"image": "$image0"},
                "params": {"position": "CENTER"},
            },
            {
                "id": "n2",
                "op": "LOCATE",
                "inputs": {"image": "$n1"},
                "params": {"target": {"category": "building", "attributes": {}}},
            },
            {
                "id": "n3",
                "op": "SELECT",
                "inputs": inputs,
                "params": {
                    "mode": "SEMANTIC",
                    "query": query,
                    "selection_type": cardinality,
                },
            },
        ],
        "final": {
            "sources": ["$n3"],
            "question": "Which candidates match the description?",
            "answer_type": "CHOICE_SINGLE",
        },
    }


def _program(expression):
    return (
        "n1=REGION($image0,CENTER)\n"
        'n2=LOCATE($n1,T("building"))\n'
        f"n3={expression}\n"
        'FINAL_QUESTION($n3,CHOICE_SINGLE,"Which candidates match?")'
    )


@pytest.mark.parametrize("scope", [None, "$image0", "$n1"])
@pytest.mark.parametrize("reference", [None, "$n1", "$n2"])
@pytest.mark.parametrize("cardinality", ["SINGLE", "MULTI"])
@pytest.mark.parametrize("query", QUERIES)
def test_json_dsl_json_preserves_every_semantic_field(scope, reference, cardinality, query):
    original = _graph(scope=scope, reference=reference, cardinality=cardinality, query=query)
    canonical = canonicalize_target(original)
    dsl = compile_taskgraph_to_dsl(original)
    restored = parse_taskgraph_dsl(dsl)
    assert canonicalize_target(restored) == canonical
    assert compile_taskgraph_to_dsl(restored) == dsl
    select = restored.nodes[-1]
    assert select.inputs == original["nodes"][-1]["inputs"]
    assert select.params["query"] == query
    assert "query" not in select.inputs and "scope" not in select.params
    suffix = f",{scope})" if scope is not None else ")"
    expected = (
        f"n3=SELECT_SEM($n2,{reference or 'null'},"
        f"{json.dumps(query, ensure_ascii=False)},{cardinality}{suffix}"
    )
    assert expected in dsl.splitlines()
    production = ProductionNode.model_validate(select.model_dump(mode="json"))
    assert production.inputs == select.inputs
    assert production.params["query"] == query


@pytest.mark.parametrize("scope", [None, "$image0", "$n1"])
@pytest.mark.parametrize("cardinality", ["SINGLE", "MULTI"])
@pytest.mark.parametrize("ensure_ascii", [False, True])
def test_escaped_queries_and_optional_scope_have_valid_decoding_prefixes(
    scope, cardinality, ensure_ascii
):
    query = QUERIES[1] + "\n" + QUERIES[2]
    scope_arg = f",{scope}" if scope is not None else ""
    dsl = _program(
        f"SELECT_SEM($n2,$n1,{json.dumps(query, ensure_ascii=ensure_ascii)},"
        f"{cardinality}{scope_arg})"
    )
    assert parse_taskgraph_dsl(dsl).nodes[-1].params["query"] == query
    grammar = CanonicalDSLPrefixGrammar(("image0",))
    assert grammar.accepts(dsl)
    for offset in range(len(dsl) + 1):
        assert grammar.analyze(dsl[:offset]).valid_prefix, (offset, dsl[:offset])
    assert not grammar.analyze(dsl[:-1]).complete


@pytest.mark.parametrize(
    "scope_node",
    [
        "n1=REGION($image0,CENTER)",
        "n1=REGION_BBOX($image0,[10,20,100,200])",
        'n1=FIND_MARKER($image0,"circle","red")',
    ],
)
def test_existing_region_producers_can_supply_scope(scope_node):
    dsl = _program('SELECT_SEM($n2,$n1,"near the road",SINGLE,$n1)')
    dsl = dsl.replace("n1=REGION($image0,CENTER)", scope_node)
    assert parse_taskgraph_dsl(dsl).nodes[-1].inputs["scope"] == "$n1"
    assert CanonicalDSLPrefixGrammar(("image0",)).accepts(dsl)


def test_subregion_scope_and_selected_reference_keep_separate_roles():
    dsl = (
        'n1=LOCATE($image0,T("building"))\n'
        "n2=SELECT_EXTREME($n1,null,TOPMOST)\n"
        "n3=SELECT_SUBREGION($n2,null,BELOW)\n"
        'n4=LOCATE($n3,T("vehicle"))\n'
        'n5=SELECT_SEM($n4,$n2,"near the building",MULTI,$n3)\n'
        'n6=COUNT_ENTITIES($n5,T("vehicle"),false)\n'
        "FINAL($n6,INTEGER)"
    )
    target = parse_taskgraph_dsl(dsl)
    assert target.nodes[4].inputs == {"candidates": "$n4", "reference": "$n2", "scope": "$n3"}
    assert compile_taskgraph_to_dsl(target) == dsl
    assert CanonicalDSLPrefixGrammar(("image0",)).accepts(dsl)


@pytest.mark.parametrize("producer", ["REGION", "FIND_MARKER"])
def test_region_and_region_set_candidates_also_roundtrip(producer):
    node = (
        "n1=REGION($image0,CENTER)"
        if producer == "REGION"
        else 'n1=FIND_MARKER($image0,"circle","red")'
    )
    dsl = (
        f"{node}\n"
        'n2=SELECT_SEM($n1,null,"the region near trees",SINGLE,$image0)\n'
        'FINAL_QUESTION($n2,CHOICE_SINGLE,"Which region?")'
    )
    assert compile_taskgraph_to_dsl(parse_taskgraph_dsl(dsl)) == dsl
    assert CanonicalDSLPrefixGrammar(("image0",)).accepts(dsl)


@pytest.mark.parametrize(
    "expression",
    [
        'SELECT_SEM($n2,null,"near road")',
        'SELECT_SEM($n2,null,"near road",SINGLE,$image0,$n1)',
        'SELECT_SEM($n2,null,"near road",SINGLE,null)',
        'SELECT_SEM($n2,null,"near road",SINGLE,"$image0")',
        'SELECT_SEM($n2,null,"near road",SINGLE,[$image0])',
        'SELECT_SEM($n2,null,"near road",SINGLE,scope=$image0)',
        'SELECT_SEM($n2,null,"near road",SINGLE,$image0,)',
        'SELECT_SEM($n2,null,"near road",$image0,SINGLE)',
        'SELECT_SEM($n2,null,"near road","SINGLE",$image0)',
        'SELECT_SEM($n2,null,"near road",INVALID,$image0)',
        'SELECT_SEM(null,null,"near road",SINGLE,$image0)',
        'SELECT_SEM([$n2],null,"near road",SINGLE,$image0)',
        'SELECT_SEM($n2,"$n1","near road",SINGLE,$image0)',
        "SELECT_SEM($n2,null,near_road,SINGLE,$image0)",
        "SELECT_SEM($n2,null,$n1,SINGLE,$image0)",
        "SELECT_SEM($n2,null,null,SINGLE,$image0)",
        "SELECT_SEM($n2,null,42,SINGLE,$image0)",
        'SELECT_SEM($n2,null,"bad\\q",SINGLE,$image0)',
        'SELECT_SEM($n2,null,"bad\\u12zz",SINGLE,$image0)',
        'SELECT_SEM($n2,null,"literal\nnewline",SINGLE,$image0)',
        'SELECT_SEM($n2,null,"unterminated,SINGLE,$image0)',
    ],
)
def test_invalid_scoped_surface_is_rejected_by_parser_and_decoding_grammar(expression):
    dsl = _program(expression)
    with pytest.raises(DSLParseError):
        parse_taskgraph_dsl(dsl)
    assert not CanonicalDSLPrefixGrammar(("image0",)).accepts(dsl)


@pytest.mark.parametrize("query", ["", " \t\n", "x" * 2049])
def test_scoped_query_schema_limits_are_unchanged(query):
    dsl = _program(f"SELECT_SEM($n2,null,{json.dumps(query)},SINGLE,$image0)")
    with pytest.raises(DSLParseError):
        parse_taskgraph_dsl(dsl)
    with pytest.raises(DSLCompileError):
        compile_taskgraph_to_dsl(_graph(scope="$image0", query=query))


def test_maximum_length_query_roundtrips_with_scope():
    graph = _graph(scope="$n1", query="建" * 2048)
    restored = parse_taskgraph_dsl(compile_taskgraph_to_dsl(graph))
    assert restored.nodes[-1].params["query"] == "建" * 2048


@pytest.mark.parametrize(
    "literal",
    [
        r'"near \/ road; \"roof\"; \\ path"',
        r'"\u5efa\u7b51 near trees \ud83d\ude80"',
        r'"line\nnext\ttab\rreturn\bback\fform"',
    ],
)
def test_valid_json_escape_spellings_decode_and_compile_without_query_loss(literal):
    dsl = _program(f"SELECT_SEM($n2,null,{literal},SINGLE,$n1)")
    original = parse_taskgraph_dsl(dsl)
    canonical = compile_taskgraph_to_dsl(original)
    restored = parse_taskgraph_dsl(canonical)
    assert restored.nodes[-1].params["query"] == json.loads(literal)
    grammar = CanonicalDSLPrefixGrammar(("image0",))
    assert grammar.accepts(dsl) and grammar.accepts(canonical)


@pytest.mark.parametrize(
    ("expression", "role"),
    [
        ('SELECT_SEM($image0,null,"near road",SINGLE,$n1)', "candidates"),
        ('SELECT_SEM($n2,$image0,"near road",SINGLE,$n1)', "reference"),
        ('SELECT_SEM($n2,null,"near road",SINGLE,$n2)', "scope"),
    ],
)
def test_input_roles_are_type_checked_independently(expression, role):
    dsl = _program(expression)
    payload = parse_taskgraph_dsl_payload(dsl)
    result = check_types(PlannerTarget.model_validate(payload), {"image0"})
    assert any(
        issue.code == "input_type_mismatch" and role in issue.message for issue in result.errors
    )
    with pytest.raises(DSLParseError, match="input_type_mismatch"):
        parse_taskgraph_dsl(dsl)
    with pytest.raises(DSLCompileError, match="input_type_mismatch"):
        compile_taskgraph_to_dsl(payload)
    assert not CanonicalDSLPrefixGrammar(("image0",)).analyze(dsl).valid_prefix


@pytest.mark.parametrize(
    "producer",
    [
        'n3=COUNT_ENTITIES($n2,T("building"),false)',
        "n3=CLASSIFY($n1)",
        "n3=MOTION($n1)",
    ],
)
def test_scalar_label_boolean_cannot_be_scope(producer):
    dsl = (
        "n1=REGION($image0,CENTER)\n"
        'n2=LOCATE($n1,T("building"))\n'
        f"{producer}\n"
        'n4=SELECT_SEM($n2,null,"near road",SINGLE,$n3)\n'
        'FINAL_QUESTION($n4,CHOICE_SINGLE,"Which candidates?")'
    )
    payload = parse_taskgraph_dsl_payload(dsl)
    _, report = validate_candidate(payload, inputs={"image0": {}})
    assert not report.type_valid
    assert any(
        issue.code == "input_type_mismatch" and "scope" in issue.message for issue in report.errors
    )
    with pytest.raises(DSLParseError, match="input_type_mismatch"):
        parse_taskgraph_dsl(dsl)
    assert not CanonicalDSLPrefixGrammar(("image0",)).accepts(dsl)


@pytest.mark.parametrize("scope", ["$n3", "$n4", "$n99"])
def test_self_future_and_missing_scope_refs_are_rejected(scope):
    dsl = _program(f'SELECT_SEM($n2,null,"near road",SINGLE,{scope})')
    if scope == "$n4":
        dsl = dsl.replace("FINAL_QUESTION(", "n4=REGION($image0,BOTTOM)\nFINAL_QUESTION(")
    with pytest.raises(DSLParseError, match="forward_reference|missing_node_ref|dag_cycle"):
        parse_taskgraph_dsl(dsl)
    assert not CanonicalDSLPrefixGrammar(("image0",)).analyze(dsl).valid_prefix


def test_scope_on_second_image_requires_caller_input_but_roundtrips_when_present():
    graph = _graph(scope="$image1")
    dsl = compile_taskgraph_to_dsl(graph)
    assert parse_taskgraph_dsl(dsl).nodes[-1].inputs["scope"] == "$image1"
    _, missing = validate_candidate(graph, inputs={"image0": {}})
    assert "missing_input_ref" in {issue.code for issue in missing.errors}
    _, present = validate_candidate(graph, inputs={"image0": {}, "image1": {}})
    assert present.valid
    assert not CanonicalDSLPrefixGrammar(("image0",)).accepts(dsl)
    assert CanonicalDSLPrefixGrammar(("image0", "image1")).accepts(dsl)


def test_scope_is_remapped_with_nodes_but_query_is_never_rewritten():
    graph = _graph(scope="$n1", reference="$n1", query="literal $n1 and $n99")
    graph["nodes"][0]["id"] = "n7"
    graph["nodes"][1]["inputs"]["image"] = "$n7"
    graph["nodes"][2]["inputs"].update(scope="$n7", reference="$n7")
    restored = parse_taskgraph_dsl(compile_taskgraph_to_dsl(graph))
    assert restored.nodes[-1].inputs["scope"] == "$n1"
    assert restored.nodes[-1].inputs["reference"] == "$n1"
    assert restored.nodes[-1].params["query"] == "literal $n1 and $n99"


def test_scope_survives_canonical_json_and_sft_export(tmp_path: Path, monkeypatch):
    graph = canonicalize_target(_graph(scope="$n1", reference="$n2", cardinality="MULTI"))
    record = {
        "sample_id": "semantic-scope-fixture",
        "input": {"question": "Which buildings match?"},
        "target": graph,
        "planner_dsl": compile_taskgraph_to_dsl(graph),
        "metadata": {},
    }
    source, output = tmp_path / "accepted.jsonl", tmp_path / "exported.jsonl"
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv", ["export_sft", "--input", str(source), "--output", str(output)]
    )
    assert export_sft_main() == 0
    exported = json.loads(output.read_text(encoding="utf-8"))
    assert exported["target"] == graph
    assert canonicalize_target(parse_taskgraph_dsl(exported["planner_dsl"])) == graph


@pytest.mark.parametrize(
    "expression",
    [
        "SELECT_REL($n2,$n1,NEAR,$image0)",
        'SELECT_RANK($n2,null,"bbox_area",1,DESCENDING,$image0)',
        "SELECT_ORD($n2,null,1,LEFT_TO_RIGHT,$image0)",
        "SELECT_EXTREME($n2,null,TOPMOST,$image0)",
        "SELECT_SUBREGION($n2,null,ABOVE,$image0)",
    ],
)
def test_new_optional_argument_does_not_change_other_select_surfaces(expression):
    dsl = _program(expression)
    with pytest.raises(DSLParseError, match="positional arguments"):
        parse_taskgraph_dsl(dsl)
    assert not CanonicalDSLPrefixGrammar(("image0",)).accepts(dsl)
