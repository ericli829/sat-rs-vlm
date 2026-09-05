from __future__ import annotations

from sat_rs_vlm.taskgraph.schema import GraphNode
from sat_rs_vlm.taskgraph.semantic_prompts import (
    semantic_question,
    semantic_reasoning_instruction,
)


def _node(operator: str, inputs: dict[str, object], params: dict[str, object]) -> GraphNode:
    return GraphNode.model_validate(
        {"id": "n1", "op": operator, "inputs": inputs, "params": params}
    )


def test_vlm_reason_prompt_limits_work_to_supplied_residual_evidence() -> None:
    node = _node(
        "VLM_REASON",
        {"evidence": ["$n1", "$n2"]},
        {"question": "$question", "choices": "$choices"},
    )
    question = semantic_question(
        node,
        "Which supplied candidate satisfies the remaining condition?",
        final_choice_fusion=False,
    )

    assert "authoritative" in question
    assert "Residual semantic question" in question
    assert "do not search for a replacement target" in question
    assert "redo COUNT" in question
    assert "rescan the original whole image" in question


def test_motion_prompt_freezes_temporal_roles() -> None:
    node = _node("MOTION", {"before": "$image1", "after": "$image0"}, {})
    question = semantic_question(node, "Did it move?", final_choice_fusion=False)
    instruction = semantic_reasoning_instruction(node)

    assert "BEFORE observation" in question
    assert "AFTER observation" in question
    assert "do not swap" in question
    assert "BEFORE as time t0" in instruction
    assert "AFTER as time t1" in instruction
    assert "filenames" in question


def test_route_prompt_preserves_navigation_constraints_without_endpoint_research() -> None:
    node = _node(
        "ROUTE_REASON",
        {"context": "$n5"},
        {"question": "$question", "choices": "$choices"},
    )
    residual = "Which walking route takes the second left after a U-turn?"
    question = semantic_question(node, residual, final_choice_fusion=True)

    assert "START and GOAL have already been resolved" in question
    assert "do not search for replacement endpoints" in question
    for required in (
        "driving",
        "walking",
        "boat",
        "shortest",
        "north/south/east/west",
        "left/right",
        "first or second intersections",
        "U-turns",
        "forks",
        "T-junctions",
        "traversability constraints",
        residual,
    ):
        assert required in question
