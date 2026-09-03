from __future__ import annotations

import re

import pytest

from taskgraph_lab.taskgraph.dsl import compile_taskgraph_to_dsl
from taskgraph_lab.taskgraph.dsl.constraint import CanonicalDSLPrefixGrammar
from taskgraph_lab.taskgraph.enums import (
    AnswerType,
    ExtremeDirection,
    GroupMode,
    IntentLabel,
    RegionPosition,
    SortOrder,
    SpatialRelation,
    SubregionType,
)
from taskgraph_lab.tests.dsl_fixtures import representative_graphs


def _grammar(*images: str, **kwargs: object) -> CanonicalDSLPrefixGrammar:
    return CanonicalDSLPrefixGrammar(images or ("image0",), **kwargs)


def _program(node: str, *, intent: str = "OTHER", answer_type: str = "CHOICE_SINGLE") -> str:
    return f"INTENT({intent})\n{node}\nFINAL($n1,{answer_type})"


def test_every_compiler_fixture_and_every_prefix_is_accepted() -> None:
    for graph in representative_graphs().values():
        dsl = compile_taskgraph_to_dsl(graph)
        images = sorted(set(re.findall(r"\$image[0-9]+", dsl)))
        grammar = _grammar(*(images or ["$image0"]))
        assert grammar.accepts(dsl), dsl
        for offset in range(len(dsl) + 1):
            assert grammar.analyze(dsl[:offset]).valid_prefix, (offset, dsl)


def test_compiler_fixtures_cover_every_surface_operator() -> None:
    surfaces = {
        match.group(1)
        for graph in representative_graphs().values()
        for match in re.finditer(
            r"^n[1-9][0-9]*=([A-Z_]+)\(",
            compile_taskgraph_to_dsl(graph),
            re.MULTILINE,
        )
    }
    assert surfaces == {
        "REGION",
        "REGION_BBOX",
        "FIND_MARKER",
        "LOCATE",
        "SELECT_REL",
        "SELECT_RANK",
        "SELECT_ORD",
        "SELECT_EXTREME",
        "SELECT_SUBREGION",
        "GROUP",
        "COUNT",
        "COUNT_ENTITIES",
        "ATTRIBUTE",
        "CLASSIFY",
        "MULTILABEL_CLASSIFY",
        "MOTION",
        "RELATION",
        "ABS_DIFF",
        "VLM_REASON",
        "BUILD_ROUTE_CONTEXT",
        "ROUTE_REASON",
        "MATCH_CHOICE",
    }


@pytest.mark.parametrize("intent", list(IntentLabel))
def test_every_intent_enum_is_accepted(intent: IntentLabel) -> None:
    dsl = _program(
        'n1=COUNT_IMAGE($image0,T("ship"),true)',
        intent=intent.value,
    )
    assert _grammar().accepts(dsl)


@pytest.mark.parametrize("position", list(RegionPosition))
def test_every_region_enum_is_accepted(position: RegionPosition) -> None:
    assert _grammar().accepts(_program(f"n1=REGION($image0,{position.value})"))


@pytest.mark.parametrize("relation", list(SpatialRelation))
def test_every_relation_enum_is_accepted(relation: SpatialRelation) -> None:
    dsl = (
        'n1=LOCATE($image0,T("ship"))\n'
        f"n2=SELECT_REL($n1,$n1,{relation.value})\n"
        "FINAL_QUESTION($n2,CHOICE_SINGLE,\"Which one?\")"
    )
    assert _grammar().accepts(dsl)


@pytest.mark.parametrize("order", list(SortOrder))
def test_every_sort_order_enum_is_accepted(order: SortOrder) -> None:
    if order in {SortOrder.ASCENDING, SortOrder.DESCENDING}:
        selection = f'SELECT_RANK($n1,null,"area",1,{order.value})'
    else:
        selection = f"SELECT_ORD($n1,null,1,{order.value})"
    dsl = (
        'n1=LOCATE($image0,T("ship"))\n'
        f"n2={selection}\n"
        "FINAL_QUESTION($n2,CHOICE_SINGLE,\"Which one?\")"
    )
    assert _grammar().accepts(dsl)


@pytest.mark.parametrize(
    "criterion",
    ["bbox_area", "bboxarea", "area", "size", "score"],
)
def test_supported_rank_criteria_are_accepted(criterion: str) -> None:
    dsl = (
        'n1=LOCATE($image0,T("ship"))\n'
        f'n2=SELECT_RANK($n1,null,"{criterion}",1,DESCENDING)\n'
        "FINAL_QUESTION($n2,CHOICE_SINGLE,\"Which one?\")"
    )
    assert _grammar().accepts(dsl)


@pytest.mark.parametrize(
    "criterion",
    ["height", "width", "distance", "distance to center", "length", "cluster_size"],
)
def test_unsupported_rank_criteria_are_rejected(criterion: str) -> None:
    dsl = (
        'n1=LOCATE($image0,T("ship"))\n'
        f'n2=SELECT_RANK($n1,null,"{criterion}",1,DESCENDING)\n'
        "FINAL_QUESTION($n2,CHOICE_SINGLE,\"Which one?\")"
    )
    assert not _grammar().accepts(dsl)


@pytest.mark.parametrize("direction", list(ExtremeDirection))
def test_every_extreme_enum_is_accepted(direction: ExtremeDirection) -> None:
    dsl = (
        'n1=LOCATE($image0,T("ship"))\n'
        f"n2=SELECT_EXTREME($n1,null,{direction.value})\n"
        "FINAL_QUESTION($n2,CHOICE_SINGLE,\"Which one?\")"
    )
    assert _grammar().accepts(dsl)


@pytest.mark.parametrize("subregion", list(SubregionType))
def test_every_subregion_enum_is_accepted(subregion: SubregionType) -> None:
    dsl = (
        "n1=REGION($image0,CENTER)\n"
        f"n2=SELECT_SUBREGION($n1,null,{subregion.value})\n"
        "FINAL_QUESTION($n2,CHOICE_SINGLE,\"What is visible?\")"
    )
    assert _grammar().accepts(dsl)


@pytest.mark.parametrize("mode", list(GroupMode))
def test_every_group_enum_is_accepted(mode: GroupMode) -> None:
    dsl = (
        'n1=LOCATE($image0,T("ship"))\n'
        f"n2=GROUP($n1,{mode.value})\n"
        "FINAL_QUESTION($n2,CHOICE_SINGLE,\"Which group?\")"
    )
    assert _grammar().accepts(dsl)


@pytest.mark.parametrize("answer_type", list(AnswerType))
def test_every_final_answer_enum_is_accepted(answer_type: AnswerType) -> None:
    assert _grammar().accepts(
        _program('n1=COUNT_IMAGE($image0,T("ship"),true)', answer_type=answer_type.value)
    )


def test_target_final_question_multisource_escaping_and_multiple_images() -> None:
    dsl = (
        'INTENT(COMPLEX_REASONING)\n'
        'n1=LOCATE($image0,T("cargo ship",color="deep red",has_part="deck\\\"A"))\n'
        'n2=LOCATE($image1,T("U-shaped road"))\n'
        'FINAL_QUESTION([$n1,$n2],TEXT,"Which route\\nconnects them?")'
    )
    assert _grammar("image0", "image1").accepts(dsl)


@pytest.mark.parametrize(
    "text",
    [
        'n1=LOCATE($image2,T("ship"))\nFINAL($n1,CHOICE_SINGLE)',
        'n1=LOCATE($n2,T("ship"))\nFINAL($n1,CHOICE_SINGLE)',
        'n2=LOCATE($image0,T("ship"))\nFINAL($n2,CHOICE_SINGLE)',
        "n1=REGION($image0,UPPER)\nFINAL($n1,CHOICE_SINGLE)",
        "n1=UNKNOWN($image0)\nFINAL($n1,CHOICE_SINGLE)",
        'n1=LOCATE($image0,T("ship"),true)\nFINAL($n1,CHOICE_SINGLE)',
        'n1=LOCATE($image0,T("ship))\nFINAL($n1,CHOICE_SINGLE)',
        'n1=LOCATE($image0,T("ship"))\nFINAL($n1,CHOICE_SINGLE)\n'
        'n2=LOCATE($image0,T("car"))',
    ],
)
def test_invalid_structure_is_not_an_accepted_prefix(text: str) -> None:
    assert not _grammar().analyze(text).valid_prefix


def test_truncated_or_incomplete_final_is_not_complete() -> None:
    text = 'n1=COUNT_IMAGE($image0,T("ship"),true)\nFINAL($n1,CHOICE_'
    analysis = _grammar().analyze(text)
    assert analysis.valid_prefix
    assert not analysis.complete


def test_dynamic_count_role_requires_explicit_surface_for_ambiguous_select() -> None:
    prefix = (
        'n1=LOCATE($image0,T("ship"))\n'
        "n2=SELECT_EXTREME($n1,null,LEFTMOST)\n"
    )
    ambiguous = prefix + 'n3=COUNT($n2,T("ship"),false)\nFINAL($n3,INTEGER)'
    explicit = prefix + 'n3=COUNT_ENTITIES($n2,T("ship"),false)\nFINAL($n3,INTEGER)'
    assert not _grammar().analyze(ambiguous).valid_prefix
    assert _grammar().accepts(explicit)


def test_repeat_guard_and_node_budget_are_operational_not_language_constraints() -> None:
    repeated = (
        'n1=LOCATE($image0,T("ship"))\n'
        "n2=SELECT_REL($n1,$n1,NEAR)\n"
        "n3=SELECT_REL($n2,$n1,NEAR)\n"
        "n4=SELECT_REL($n3,$n1,NEAR)\n"
        "n5=SELECT_REL($n4,$n1,NEAR)"
    )
    assert _grammar().analyze(repeated).valid_prefix
    guarded = _grammar(repeat_guard_repetitions=4)
    assert guarded.analyze(repeated).reason == "repeat_guard"

    two_nodes = 'n1=LOCATE($image0,T("ship"))\nn2=ATTRIBUTE($n1,"color")\n'
    budgeted = _grammar(max_nodes=2)
    assert budgeted.analyze(two_nodes + "F").valid_prefix
    assert not budgeted.analyze(two_nodes + "n3").valid_prefix


def test_target_attributes_accept_arbitrary_order_and_compile_canonically() -> None:
    color_first = _program(
        'n1=LOCATE($image0,T("building",color="white",shape="L-shaped"))'
    )
    shape_first = _program(
        'n1=LOCATE($image0,T("building",shape="L-shaped",color="white"))'
    )
    grammar = _grammar()
    assert grammar.accepts(color_first)
    assert grammar.accepts(shape_first)


def test_duplicate_target_attribute_is_rejected() -> None:
    duplicated = _program(
        'n1=LOCATE($image0,T("building",color="white",color="gray"))'
    )
    analysis = _grammar().analyze(duplicated)
    assert analysis.valid_prefix is False
    assert analysis.reason in {"duplicate_target_attribute", "invalid_completed_line"}


def test_repeat_guard_forces_only_final_surface() -> None:
    repeated = (
        'n1=LOCATE($image0,T("ship"))\n'
        "n2=SELECT_REL($n1,$n1,NEAR)\n"
        "n3=SELECT_REL($n2,$n1,NEAR)\n"
        "n4=SELECT_REL($n3,$n1,NEAR)\n"
        "n5=SELECT_REL($n4,$n1,NEAR)"
    )
    grammar = _grammar(repeat_guard_repetitions=4)
    guarded = grammar.analyze(repeated)
    assert guarded.valid_prefix and guarded.force_final and guarded.current_node_complete
    assert grammar.analyze(repeated + "\nF").valid_prefix
    assert grammar.accepts(repeated + "\nFINAL($n5,CHOICE_SINGLE)")
    assert grammar.accepts(
        repeated + '\nFINAL_QUESTION($n5,CHOICE_SINGLE,"Which object?")'
    )
    continued = grammar.analyze(repeated + '\nn6=LOCATE($image0,T("car"))')
    assert continued.valid_prefix is False
    assert continued.reason in {"forced_final_required", "node_after_repeat_guard"}
