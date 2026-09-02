from __future__ import annotations

import pytest

from taskgraph_lab.taskgraph.dsl import (
    DSLCompileError,
    DSLParseError,
    compile_taskgraph_to_dsl,
    parse_taskgraph_dsl,
)
from taskgraph_lab.tests.dsl_fixtures import representative_graphs


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("n1=FOO($image0)\nFINAL($n1,TEXT)", "unknown DSL operator"),
        ("n1=REGION($image0,top_right)\nFINAL($n1,TEXT)", "invalid TaskGraph"),
        ('n1=LOCATE($foo,T("ship"))\nFINAL($n1,TEXT)', "invalid reference"),
        (
            'n1=LOCATE($n2,T("ship"))\nn2=LOCATE($image0,T("car"))\n'
            'FINAL_QUESTION($n1,TEXT,"What is visible?")',
            "forward_reference",
        ),
        (
            'n1=LOCATE($image0,T("ship"))\nn1=LOCATE($image0,T("car"))\n'
            'FINAL_QUESTION($n1,TEXT,"What is visible?")',
            "duplicate node id",
        ),
        ('n1=LOCATE($image0,T("ship"))', "requires exactly one FINAL"),
        (
            'n1=LOCATE($image0,T("ship"))\nFINAL_QUESTION($n1,TEXT,"A?")\n'
            'FINAL_QUESTION($n1,TEXT,"B?")',
            "after FINAL",
        ),
        (
            'n1=LOCATE($image0,T("ship"))\nFINAL_QUESTION($n1,TEXT,"A?")\n'
            'n2=LOCATE($image0,T("car"))',
            "after FINAL",
        ),
        ('n1=LOCATE($image0,T())\nFINAL($n1,TEXT)', "T requires exactly one"),
        ('n1=LOCATE($image0,T("ship))\nFINAL($n1,TEXT)', "malformed JSON string"),
        ('n1=CLASSIFY($image0,["farm")\nFINAL($n1,LABEL)', "expected ]"),
        ('n1=COUNT($image0,T("ship"),TRUE)\nFINAL($n1,INTEGER)', "expected true or false"),
        ('n1=__import__("os")\nFINAL($n1,TEXT)', "unknown DSL operator"),
        ('n1=eval("1+1")\nFINAL($n1,TEXT)', "unknown DSL operator"),
        ('n1=system("whoami")\nFINAL($n1,TEXT)', "unknown DSL operator"),
    ],
)
def test_malformed_or_injection_like_dsl_is_rejected(text: str, message: str) -> None:
    with pytest.raises(DSLParseError, match=message):
        parse_taskgraph_dsl(text)


def test_count_shorthand_rejects_static_role_ambiguity() -> None:
    text = (
        "n1=LOCATE($image0,T(\"ship\"))\n"
        "n2=SELECT_EXTREME($n1,null,LEFTMOST)\n"
        "n3=COUNT($n2,T(\"ship\"),false)\n"
        "FINAL($n3,INTEGER)"
    )
    with pytest.raises(DSLParseError, match="not statically unique"):
        parse_taskgraph_dsl(text)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda graph: graph["nodes"][0].update(op="UNKNOWN"),
        lambda graph: graph["nodes"][0]["inputs"].update(image="$n99"),
        lambda graph: graph["nodes"][0]["params"].update(
            target={"category": "", "attributes": {}}
        ),
    ],
)
def test_compiler_rejects_invalid_canonical_graph(mutation) -> None:
    graph = representative_graphs()["whole_image_count"]
    mutation(graph)
    with pytest.raises(DSLCompileError):
        compile_taskgraph_to_dsl(graph)
