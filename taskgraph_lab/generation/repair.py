from __future__ import annotations

import json
import re

from taskgraph_lab.datasets.base import NormalizedSample
from taskgraph_lab.taskgraph.validator import ValidationResult

SEMANTIC_ERROR_CODES = {
    "authoritative_count_visual_reintroduction",
    "choice_answer_type_mismatch",
    "count_entities_requires_non_entire",
    "count_input_xor",
    "dag_cycle",
    "dead_node",
    "dedicated_operator_bypass",
    "final_type_mismatch",
    "forward_reference",
    "input_type_mismatch",
    "invalid_reference",
    "missing_final_ref",
    "missing_input_ref",
    "missing_node_ref",
    "missing_residual_final_question",
    "original_choices_copied_into_target",
    "relation_query_should_use_relation",
    "relation_result_not_consumed",
    "runtime_prediction_in_final_question",
}


def classify_repair(validation: ValidationResult) -> str:
    """Route only syntactic/schema defects to the LLM repair pass."""
    if validation.valid:
        return "AUTO_NORMALIZED" if validation.normalized_fields else "NOT_NEEDED"
    for issue in validation.errors:
        if issue.code in SEMANTIC_ERROR_CODES:
            return "SEMANTIC_ERROR"
        if (
            issue.stage == "schema"
            and issue.code == "enum"
            and re.search(r"(?:^|\.)relation:", issue.message)
        ):
            return "SEMANTIC_ERROR"
    return "LLM_REPAIRABLE"


def render_repair_prompt(
    template: str,
    sample: NormalizedSample,
    invalid_graph: str,
    validation: ValidationResult,
) -> str:
    safe_sample = {
        "question": sample.question,
        "question_type": sample.question_type.value,
        "choices": sample.choices,
        "inputs": {key: value.model_dump(mode="json") for key, value in sample.inputs.items()},
        "metadata": sample.metadata,
    }
    errors = [item.model_dump(mode="json") for item in validation.errors]
    return (
        template.replace("{{SAMPLE}}", json.dumps(safe_sample, ensure_ascii=False, indent=2))
        .replace("{{INVALID_GRAPH}}", invalid_graph)
        .replace("{{VALIDATOR_ERRORS}}", json.dumps(errors, ensure_ascii=False, indent=2))
    )
