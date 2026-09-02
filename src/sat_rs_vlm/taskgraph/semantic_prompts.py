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
}

_FINAL_FUSION = (
    "The supplied answer options are the original benchmark options. Compare them carefully "
    "against the visual evidence. Do not finalize an option in free-form text. A separate "
    "constrained continuation will select the answer."
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
    elif node.op in {OperatorName.VLM_REASON, OperatorName.ROUTE_REASON}:
        configured = str(node.params["question"])
        base = original_question if configured == "$question" else configured
    else:
        base = _INTERMEDIATE_QUESTIONS.get(node.op, original_question)
    if final_choice_fusion and node.op is not OperatorName.ROUTE_REASON:
        return f"{base}\n\nOriginal benchmark question:\n{original_question}\n\n{_FINAL_FUSION}"
    return base


def semantic_reasoning_instruction(node: GraphNode) -> str:
    return _INTERMEDIATE_INSTRUCTIONS.get(
        node.op,
        "Analyze the supplied evidence freely. A separate finite decision step will select the "
        "canonical result.",
    )
