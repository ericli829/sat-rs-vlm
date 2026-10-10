import json

import pytest

from sat_rs_vlm.taskgraph.schema import GraphNode
from taskgraph_lab.taskgraph.dsl import (
    CanonicalDSLPrefixGrammar,
    compile_taskgraph_to_dsl,
    parse_taskgraph_dsl,
)
from taskgraph_lab.taskgraph.schema import PlannerTarget
from taskgraph_lab.taskgraph.validator import validate_candidate

QUERY = '建筑右边有树，下面有草地；屋顶写着 "A"。\n保留完整条件。'


def program(cardinality="SINGLE", query=QUERY, reference="null"):
    return (
        'n1=LOCATE($image0,T("building"))\n'
        f"n2=SELECT_SEM($n1,{reference},{json.dumps(query, ensure_ascii=False)},{cardinality})\n"
        'FINAL_QUESTION($n2,CHOICE_SINGLE,"What color is the selected building?")'
    )


@pytest.mark.parametrize("cardinality", ["SINGLE", "MULTI"])
def test_escaped_unicode_roundtrip_and_every_prefix(cardinality):
    dsl = program(cardinality)
    target = parse_taskgraph_dsl(dsl)
    select = target.nodes[1]
    assert select.params["query"] == QUERY
    assert "query" not in select.inputs
    assert select.params["selection_type"] == cardinality
    compiled = compile_taskgraph_to_dsl(target)
    assert parse_taskgraph_dsl(compiled).model_dump() == target.model_dump()
    grammar = CanonicalDSLPrefixGrammar(("image0",))
    assert grammar.accepts(compiled)
    assert all(
        grammar.analyze(compiled[:offset]).valid_prefix for offset in range(len(compiled) + 1)
    )
    production = GraphNode.model_validate(select.model_dump(mode="json"))
    assert production.params["query"] == QUERY


@pytest.mark.parametrize(
    "dsl",
    [
        program(query=" "),
        program(query="x" * 2049),
        program().replace(",SINGLE)", ",INVALID)"),
        program().replace(json.dumps(QUERY, ensure_ascii=False), "$n1"),
    ],
)
def test_parser_rejects_invalid_semantic_strings(dsl):
    with pytest.raises(ValueError):
        parse_taskgraph_dsl(dsl)


def test_attribute_allows_semantic_single_but_rejects_multi():
    for cardinality, expected in [("SINGLE", True), ("MULTI", False)]:
        target = parse_taskgraph_dsl(program(cardinality))
        raw = target.model_dump(mode="json", exclude_none=True)
        raw["nodes"].append(
            {
                "id": "n3",
                "op": "ATTRIBUTE",
                "inputs": {"entity": "$n2"},
                "params": {"attribute": "color"},
            }
        )
        raw["final"]["sources"] = ["$n3"]
        _, result = validate_candidate(raw, inputs={"image0": {}})
        assert result.valid == expected


def test_reference_and_scope_roundtrip_without_being_silently_dropped():
    target = parse_taskgraph_dsl(program(reference="$n1"))
    compiled = compile_taskgraph_to_dsl(target)
    assert parse_taskgraph_dsl(compiled).nodes[1].inputs["reference"] == "$n1"
    raw = target.model_dump(mode="json", exclude_none=True)
    raw["nodes"][1]["inputs"]["scope"] = "$image0"
    parsed = PlannerTarget.model_validate(raw)
    assert validate_candidate(raw, inputs={"image0": {}})[1].valid
    restored = parse_taskgraph_dsl(compile_taskgraph_to_dsl(parsed))
    assert restored.nodes[1].inputs == {
        "candidates": "$n1", "reference": "$n1", "scope": "$image0"
    }


def test_prefix_does_not_accept_unquoted_query_or_unknown_cardinality():
    grammar = CanonicalDSLPrefixGrammar(("image0",))
    assert not grammar.accepts(program().replace(json.dumps(QUERY, ensure_ascii=False), "$n1"))
    assert not grammar.accepts(program().replace(",SINGLE)", ",OTHER)"))
