from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from taskgraph_lab.taskgraph.canonicalize import canonicalize_target
from taskgraph_lab.taskgraph.dsl import (
    CanonicalDSLPrefixGrammar,
    DSLCompileError,
    DSLParseError,
    compile_taskgraph_to_dsl,
    parse_taskgraph_dsl,
)
from taskgraph_lab.taskgraph.enums import AnswerType, RuntimeType
from taskgraph_lab.taskgraph.schema import FinalSpec, PlannerTarget, TaskGraph
from taskgraph_lab.taskgraph.type_checker import OUTPUT_TYPES, check_types
from taskgraph_lab.taskgraph.validator import validate_candidate


def _locate() -> dict:
    return {
        "id": "n1",
        "op": "LOCATE",
        "inputs": {"image": "$image0"},
        "params": {"target": {"category": "building", "attributes": {}}},
    }


def _graph(nodes: list[dict] | None = None) -> dict:
    nodes = nodes or [_locate()]
    return {
        "intent": "OTHER",
        "nodes": nodes,
        "final": {"sources": [f"${nodes[-1]['id']}"], "answer_type": "BBOX"},
    }


def _select(mode: str, **params: object) -> dict:
    return {
        "id": "n2",
        "op": "SELECT",
        "inputs": {"candidates": "$n1"},
        "params": {"mode": mode, **params},
    }


SINGLETON_SELECTS = [
    _select("EXTREME", direction="TOPMOST"),
    _select("ORDINAL", index=2, order="LEFT_TO_RIGHT"),
    _select("RANK", criterion="bbox_area", rank=1, order="DESCENDING"),
    _select("SEMANTIC", query="the building beside the road", selection_type="SINGLE"),
]


@pytest.mark.parametrize("selection", SINGLETON_SELECTS)
def test_bbox_singleton_select_roundtrip_and_all_prefixes(selection: dict) -> None:
    graph = _graph([_locate(), selection])
    dsl = compile_taskgraph_to_dsl(graph)
    assert dsl.endswith("FINAL($n2,BBOX)")
    parsed = parse_taskgraph_dsl(dsl)
    assert canonicalize_target(parsed) == canonicalize_target(graph)
    grammar = CanonicalDSLPrefixGrammar(["image0"])
    assert grammar.accepts(dsl)
    for offset in range(len(dsl) + 1):
        assert grammar.analyze(dsl[:offset]).valid_prefix, (offset, dsl)


def test_bbox_direct_locate_is_short_and_preserves_locate_set_type() -> None:
    graph = _graph()
    target, report = validate_candidate(
        graph, inputs={"image0": {}}, question="Return the bounding box of the building."
    )
    assert target is not None and report.valid, report.model_dump()
    assert target.final.question is None
    assert not report.warnings
    assert report.inferred_types["$n1"] == [RuntimeType.ENTITY_SET.value]
    assert OUTPUT_TYPES[target.nodes[0].op] == {RuntimeType.ENTITY_SET}
    dsl = compile_taskgraph_to_dsl(target)
    assert dsl == 'INTENT(OTHER)\nn1=LOCATE($image0,T("building"))\nFINAL($n1,BBOX)'
    assert canonicalize_target(parse_taskgraph_dsl(dsl)) == canonicalize_target(graph)


@pytest.mark.parametrize(
    "node",
    [
        {
            "id": "n1",
            "op": "REGION",
            "inputs": {"image": "$image0"},
            "params": {"position": "TOP_RIGHT"},
        },
        {
            "id": "n1",
            "op": "REGION_FROM_BBOX",
            "inputs": {"image": "$image0"},
            "params": {"bbox": [10, 20, 30, 40]},
        },
    ],
)
def test_bbox_region_producers_are_valid_without_residual_question(node: dict) -> None:
    graph = _graph([node])
    _, report = validate_candidate(graph, inputs={"image0": {}})
    assert report.valid, report.model_dump()
    assert canonicalize_target(parse_taskgraph_dsl(compile_taskgraph_to_dsl(graph))) == (
        canonicalize_target(graph)
    )


def test_bbox_region_then_locate_roundtrip() -> None:
    region = {
        "id": "n1",
        "op": "REGION",
        "inputs": {"image": "$image0"},
        "params": {"position": "BOTTOM_RIGHT"},
    }
    locate = _locate()
    locate["id"] = "n2"
    locate["inputs"]["image"] = "$n1"
    graph = _graph([region, locate])
    assert canonicalize_target(parse_taskgraph_dsl(compile_taskgraph_to_dsl(graph))) == (
        canonicalize_target(graph)
    )


def test_bbox_select_subregion_is_a_single_region() -> None:
    region = {
        "id": "n1",
        "op": "REGION",
        "inputs": {"image": "$image0"},
        "params": {"position": "CENTER"},
    }
    graph = _graph([region, _select("SUBREGION", subregion="ABOVE")])
    _, report = validate_candidate(graph, inputs={"image0": {}})
    assert report.valid, report.model_dump()


@pytest.mark.parametrize(
    "producer",
    [
        _select("SEMANTIC", query="all buildings beside roads", selection_type="MULTI"),
        {
            "id": "n2",
            "op": "SELECT",
            "inputs": {"candidates": "$n1", "reference": "$n1"},
            "params": {"mode": "RELATION", "relation": "LEFT_OF"},
        },
        {
            "id": "n2",
            "op": "GROUP",
            "inputs": {"entities": "$n1"},
            "params": {"mode": "CLUSTER"},
        },
        {
            "id": "n2",
            "op": "COUNT",
            "inputs": {"entities": "$n1"},
            "params": {
                "target": {"category": "building", "attributes": {}},
                "entire": False,
            },
        },
    ],
)
def test_bbox_rejects_set_and_nonvisual_producers(producer: dict) -> None:
    graph = _graph([_locate(), producer])
    target, report = validate_candidate(graph, inputs={"image0": {}})
    assert target is not None
    assert not report.valid
    assert "bbox_requires_single_visual_source" in {issue.code for issue in report.errors}
    with pytest.raises(DSLCompileError, match="bbox_requires_single_visual_source"):
        compile_taskgraph_to_dsl(graph)


@pytest.mark.parametrize("sources", [[], ["$n1", "$n2"], ["$image0"]])
def test_bbox_final_schema_requires_one_node_source(sources: list[str]) -> None:
    with pytest.raises(ValidationError):
        FinalSpec(sources=sources, answer_type=AnswerType.BBOX)


@pytest.mark.parametrize(
    "dsl",
    [
        'n1=LOCATE($image0,T("ship"))\nFINAL($n2,BBOX)',
        'n1=LOCATE($image0,T("ship"))\nFINAL($image0,BBOX)',
        'n1=LOCATE($image0,T("ship"))\nFINAL($n1,BOX)',
        'n1=LOCATE($image0,T("ship"))\nFINAL($n1,BBOX,"extra")',
        'n1=LOCATE($image0,T("ship"))\n'
        'n2=SELECT_SEM($n1,null,"ships beside roads",MULTI)\nFINAL($n2,BBOX)',
    ],
)
def test_bbox_parser_rejects_invalid_final_contracts(dsl: str) -> None:
    with pytest.raises(DSLParseError):
        parse_taskgraph_dsl(dsl)


def test_bbox_preserves_free_form_input_contract() -> None:
    graph = _graph()
    full = TaskGraph.model_validate(
        {
            **graph,
            "question": "Locate the building.",
            "question_type": "FREE_FORM",
            "choices": None,
            "inputs": {"image0": {"type": "image", "uri_or_key": "synthetic.png"}},
        }
    )
    assert full.final.answer_type is AnswerType.BBOX
    _, report = validate_candidate(graph, inputs=full.inputs, question_type="MULTIPLE_CHOICE")
    assert "choice_answer_type_mismatch" in {issue.code for issue in report.errors}


def test_bbox_does_not_relax_other_visual_final_contracts() -> None:
    graph = _graph()
    graph["final"]["answer_type"] = "CHOICE_SINGLE"
    _, report = validate_candidate(graph, inputs={"image0": {}})
    assert "missing_residual_final_question" in {issue.code for issue in report.errors}
    graph["final"]["question"] = "What color is this building?"
    _, report = validate_candidate(graph, inputs={"image0": {}})
    assert report.valid, report.model_dump()


def test_bbox_type_checker_rejects_multiple_sources_independently_of_schema() -> None:
    target = PlannerTarget.model_validate(_graph())
    final = target.final.model_copy(update={"sources": ["$n1", "$n1"]})
    bad = target.model_copy(update={"final": final})
    report = check_types(bad, {"image0"})
    assert "bbox_requires_single_visual_source" in {issue.code for issue in report.errors}


def test_bbox_invalid_source_is_not_hidden_by_a_valid_union_member() -> None:
    graph = _graph([_locate(), copy.deepcopy(SINGLETON_SELECTS[-1])])
    graph["nodes"][-1]["params"]["selection_type"] = "MULTI"
    report = check_types(PlannerTarget.model_validate(graph), {"image0"})
    assert "bbox_requires_single_visual_source" in {issue.code for issue in report.errors}
