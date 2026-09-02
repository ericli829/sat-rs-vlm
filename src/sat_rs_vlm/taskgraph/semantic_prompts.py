"""Runtime-owned prompts for semantic operators."""

from __future__ import annotations

from .schema import GraphNode, OperatorName

_INTERMEDIATE_QUESTIONS = {
    OperatorName.CLASSIFY: (
        "Classify the already selected visual source. Do not search for or replace the target."
    ),
    OperatorName.MULTILABEL_CLASSIFY: (
        "Determine every applicable label for the already selected visual source."
    ),
    OperatorName.MOTION: (
        "Determine whether the already selected object is moving from the supplied evidence."
    ),
    OperatorName.RELATION: (
        "Analyze the spatial relation between the marked SUBJECT and REFERENCE using only the "
        "supplied visual evidence."
    ),
    OperatorName.MATCH_CHOICE: "Match the resolved value to the supplied benchmark options.",
}

_INTERMEDIATE_INSTRUCTIONS = {
    OperatorName.ATTRIBUTE: (
        "Reason freely about the requested attribute of the already selected target. "
        "Do not re-identify the target. A separate canonical decision step will select the value."
    ),
    OperatorName.CLASSIFY: (
        "Reason freely about the selected visual source. A separate canonical decision step "
        "will select the label."
    ),
    OperatorName.MULTILABEL_CLASSIFY: (
        "Reason once about all applicable labels. Separate cached verification steps will test "
        "each canonical label independently."
    ),
    OperatorName.MOTION: (
        "Reason freely from the visual evidence. A separate binary decision step will determine "
        "YES or NO."
    ),
    OperatorName.RELATION: (
        "Reason freely about the marked subject and reference. A separate canonical decision "
        "step will select the spatial relation."
    ),
    OperatorName.VLM_REASON: (
        "Treat every supplied crop, candidate set, label, count, and localized region as "
        "authoritative upstream evidence. Resolve only the residual question. Do not search for "
        "a replacement target, redo COUNT, or rescan the original whole image."
    ),
}

_FINAL_FUSION = (
    "The supplied answer options are the original benchmark options. Compare them carefully "
    "against the visual evidence. Do not finalize an option in free-form text. A separate "
    "constrained continuation will select the answer."
)

_ROUTE_V1_CONTEXT = (
    "START and GOAL have already been resolved by upstream localization and are marked in the "
    "supplied route crop. Treat referring descriptions of those endpoints as identity context "
    "only; do not search for replacement endpoints. Preserve and apply all navigation, direction, "
    "obstacle, and shortest-route constraints. This includes driving, walking, or boat mode; "
    "shortest or best route; north/south/east/west and left/right directions; first or second "
    "intersections; U-turns; forks; T-junctions; and traversability constraints. "
    "Prompt version: route-v1."
)


def semantic_question(
    node: GraphNode,
    original_question: str,
    *,
    final_choice_fusion: bool,
) -> str:
    if node.op is OperatorName.ATTRIBUTE:
        part = f" for part {node.params['part']}" if node.params.get("part") else ""
        base = (
            f"Determine the {node.params['attribute']}{part} of the already selected target. "
            "The upstream target selection is authoritative; do not localize a replacement."
        )
    elif node.op is OperatorName.MOTION and set(node.inputs) == {"before", "after"}:
        base = (
            "Compare the supplied BEFORE observation to the supplied AFTER observation in that "
            "temporal order. Determine whether the target moved between them; do not swap the "
            "roles or infer order from filenames."
        )
    elif node.op in {OperatorName.VLM_REASON, OperatorName.ROUTE_REASON}:
        configured = str(node.params["question"])
        base = original_question if configured == "$question" else configured
        if node.op is OperatorName.VLM_REASON:
            residual = base or "Resolve the remaining semantic question from the supplied evidence."
            base = (
                "Upstream localization, selection, and structured results are authoritative. "
                "Use only the supplied evidence; do not search for a replacement target, redo "
                "COUNT, or rescan the original whole image.\n\n"
                f"Residual semantic question:\n{residual}"
            )
    else:
        base = _INTERMEDIATE_QUESTIONS.get(node.op, original_question)
    if node.op is OperatorName.ROUTE_REASON:
        residual = base or "Determine which supplied route option satisfies the route constraints."
        return f"{_ROUTE_V1_CONTEXT}\n\nResidual route question:\n{residual}"
    if final_choice_fusion:
        residual = original_question or "Match the resolved visual evidence to the options."
        return f"{base}\n\nResidual final question:\n{residual}\n\n{_FINAL_FUSION}"
    return base


def semantic_reasoning_instruction(node: GraphNode) -> str:
    if node.op is OperatorName.MOTION and set(node.inputs) == {"before", "after"}:
        return (
            "Use BEFORE as time t0 and AFTER as time t1. Compare the same supplied target across "
            "those observations. A separate binary decision step will determine YES or NO."
        )
    return _INTERMEDIATE_INSTRUCTIONS.get(
        node.op,
        "Analyze the supplied evidence freely. A separate finite decision step will select the "
        "canonical result.",
    )
