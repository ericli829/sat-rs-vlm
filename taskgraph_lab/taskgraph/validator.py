from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from .canonicalize import normalize_candidate_payload
from .enums import AnswerType, IntentLabel, OperatorName, QuestionType, RuntimeType
from .schema import PlannerTarget
from .type_checker import check_types

PHYSICAL_KEYS = {
    "model",
    "detector",
    "retriever",
    "route_model",
    "threshold",
    "score_threshold",
    "pred_score_thr",
    "tile_size",
    "overlap",
    "nms",
    "gpu",
    "cpu",
    "beam_width",
    "zoom_depth",
}
EXTERNAL_RELATIONS = re.compile(
    r"\b(?:left|right|above|below|near|next\s+to|inside|outside|between|around|both\s+sides)\b",
    re.IGNORECASE,
)
BBOX_HINT = re.compile(r"\b(?:bounding\s*box|bbox)\b", re.IGNORECASE)
MARKER_HINT = re.compile(
    r"\b(?:red|blue|green|yellow|colored|coloured|light\s+blue)\s+"
    r"(?:circle|rectangle|box|border|outline|marker)\b",
    re.IGNORECASE,
)
RELATION_QUERY_HINT = re.compile(
    r"\b(?:where\s+is\b.+\brelative\s+to|position\s+of\b.+\brelative\s+to|"
    r"located\b.+\bin\s+relation\s+to)\b",
    re.IGNORECASE,
)
RUNTIME_RESULT_HINT = re.compile(
    r"\b(?:(?:detected|predicted|computed|resolved)\s+)?"
    r"(?:count|number|result|value|label)\s+(?:is|equals|=)\s+"
    r"(?:-?\d+(?:\.\d+)?|true|false|[A-Z][\w-]*)\b",
    re.IGNORECASE,
)
NUMERIC_VALUE_HINT = re.compile(r"(?<![\w$])-?\d+(?:\.\d+)?(?!\w)")
GENERIC_STRUCTURED_FINAL_QUESTION = re.compile(
    r"\bwhich\s+options?\b.*\b(?:match|matches)\b.*"
    r"\b(?:count|result|value|label|relation)\b",
    re.IGNORECASE,
)
EXPLICIT_ALTERNATIVE = re.compile(
    r"\b(?:either\s+)?(?P<a>[A-Za-z][A-Za-z0-9_-]*)\s*"
    r"(?:or|/)\s*(?P<b>[A-Za-z][A-Za-z0-9_-]*)\b",
    re.IGNORECASE,
)
VISUAL_RUNTIME_TYPES = {
    RuntimeType.REGION,
    RuntimeType.REGION_SET,
    RuntimeType.ENTITY,
    RuntimeType.ENTITY_SET,
    RuntimeType.ROUTE_CONTEXT,
    RuntimeType.EVIDENCE_SET,
}
STRUCTURED_RUNTIME_TYPES = {
    RuntimeType.SCALAR_INT,
    RuntimeType.SCALAR_FLOAT,
    RuntimeType.BOOLEAN,
    RuntimeType.LABEL,
    RuntimeType.LABEL_SET,
}


def _semantic_token(value: str) -> str:
    token = value.casefold().replace("_", "-").strip(" -")
    if token.endswith("ies") and len(token) > 3:
        return token[:-3] + "y"
    if token.endswith("ses") and len(token) > 3:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 2:
        return token[:-1]
    return token


def _explicit_alternative_loss_warnings(
    target: PlannerTarget, question: str
) -> list[ValidationIssue]:
    category_tokens: set[str] = set()
    for node in target.nodes:
        raw_target = node.params.get("target")
        if not isinstance(raw_target, Mapping):
            continue
        category = str(raw_target.get("category", ""))
        category_tokens.update(
            _semantic_token(token) for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]*", category)
        )
    warnings: list[ValidationIssue] = []
    seen: set[tuple[str, str]] = set()
    for match in EXPLICIT_ALTERNATIVE.finditer(question):
        alternatives = (_semantic_token(match.group("a")), _semantic_token(match.group("b")))
        if alternatives in seen:
            continue
        seen.add(alternatives)
        present = [value in category_tokens for value in alternatives]
        if any(present) and not all(present):
            warnings.append(
                ValidationIssue(
                    stage="semantic",
                    code="explicit_alternative_loss",
                    message=(
                        "source category alternatives must be preserved; found "
                        f"'{match.group('a')} or {match.group('b')}' but target categories "
                        "retain only one branch"
                    ),
                )
            )
    return warnings


class ValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stage: str
    code: str
    message: str
    node_id: str | None = None


class ValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    valid: bool
    schema_valid: bool
    graph_valid: bool
    type_valid: bool
    semantic_valid: bool
    errors: list[ValidationIssue]
    warnings: list[ValidationIssue]
    inferred_types: dict[str, list[str]]
    resolved_input_types: dict[str, dict[str, list[dict[str, Any]]]]
    normalized_fields: list[dict[str, Any]]


def _input_refs(node_inputs: Mapping[str, str | list[str]]) -> list[str]:
    return [
        item for raw in node_inputs.values() for item in (raw if isinstance(raw, list) else [raw])
    ]


def _graph_issues(target: PlannerTarget, input_names: set[str]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    ids = [node.id for node in target.nodes]
    if len(ids) != len(set(ids)):
        issues.append(
            ValidationIssue(
                stage="graph", code="duplicate_node_id", message="node ids must be unique"
            )
        )
    all_ids = set(ids)
    order = {node_id: index for index, node_id in enumerate(ids)}
    dependencies: dict[str, set[str]] = {node_id: set() for node_id in all_ids}
    for index, node in enumerate(target.nodes):
        for ref in _input_refs(node.inputs):
            if ref.startswith("$image"):
                if ref[1:] not in input_names:
                    issues.append(
                        ValidationIssue(
                            stage="graph",
                            code="missing_input_ref",
                            message=f"unknown input reference {ref}",
                            node_id=node.id,
                        )
                    )
            elif ref.startswith("$n"):
                ref_id = ref[1:]
                if ref_id not in all_ids:
                    issues.append(
                        ValidationIssue(
                            stage="graph",
                            code="missing_node_ref",
                            message=f"unknown node reference {ref}",
                            node_id=node.id,
                        )
                    )
                else:
                    dependencies.setdefault(node.id, set()).add(ref_id)
                    if order.get(ref_id, -1) >= index:
                        issues.append(
                            ValidationIssue(
                                stage="graph",
                                code="forward_reference",
                                message=f"{node.id} may only reference preceding nodes, got {ref}",
                                node_id=node.id,
                            )
                        )
            else:
                issues.append(
                    ValidationIssue(
                        stage="graph",
                        code="invalid_reference",
                        message=f"unsupported reference {ref}",
                        node_id=node.id,
                    )
                )

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> bool:
        if node_id in visiting:
            return True
        if node_id in visited:
            return False
        visiting.add(node_id)
        for dependency in dependencies.get(node_id, set()):
            if visit(dependency):
                return True
        visiting.remove(node_id)
        visited.add(node_id)
        return False

    if any(visit(node_id) for node_id in ids if node_id not in visited):
        issues.append(
            ValidationIssue(stage="graph", code="dag_cycle", message="graph contains a cycle")
        )

    existing_final_ids: list[str] = []
    for final_ref in target.final.sources:
        final_id = final_ref[1:]
        if final_id not in all_ids:
            issues.append(
                ValidationIssue(
                    stage="graph",
                    code="missing_final_ref",
                    message=f"unknown final source {final_ref}",
                )
            )
        else:
            existing_final_ids.append(final_id)

    contributing: set[str] = set()

    def mark_contributing(node_id: str) -> None:
        if node_id in contributing:
            return
        contributing.add(node_id)
        for dependency in dependencies.get(node_id, set()):
            mark_contributing(dependency)

    for final_id in existing_final_ids:
        mark_contributing(final_id)
    for node_id in ids:
        if node_id not in contributing:
            issues.append(
                ValidationIssue(
                    stage="graph",
                    code="dead_node",
                    message=f"{node_id} does not contribute to any final source",
                    node_id=node_id,
                )
            )
    return issues


def _walk_keys(value: Any) -> list[str]:
    keys: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            keys.append(str(key).lower())
            keys.extend(_walk_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.extend(_walk_keys(child))
    return keys


def _semantic_issues(
    target: PlannerTarget,
    *,
    question: str,
    question_type: str,
    inferred: dict[str, set[RuntimeType]],
) -> tuple[list[ValidationIssue], list[ValidationIssue]]:
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []
    ops = [node.op for node in target.nodes]
    nodes_by_id = {node.id: node for node in target.nodes}
    dependencies = {
        node.id: {
            ref[1:]
            for ref in _input_refs(node.inputs)
            if ref.startswith("$n") and ref[1:] in nodes_by_id
        }
        for node in target.nodes
    }

    def ancestors(node_id: str) -> set[str]:
        found: set[str] = set()
        pending = list(dependencies.get(node_id, set()))
        while pending:
            current = pending.pop()
            if current in found:
                continue
            found.add(current)
            pending.extend(dependencies.get(current, set()))
        return found

    final_ids = {ref[1:] for ref in target.final.sources}
    contributing_ids = set(final_ids)
    for final_id in final_ids:
        contributing_ids.update(ancestors(final_id))

    for node in target.nodes:
        forbidden = sorted(set(_walk_keys(node.params)).intersection(PHYSICAL_KEYS))
        if forbidden:
            errors.append(
                ValidationIssue(
                    stage="semantic",
                    code="physical_execution_key",
                    message=f"physical execution keys are forbidden: {forbidden}",
                    node_id=node.id,
                )
            )
        if (
            node.op is OperatorName.COUNT
            and "entities" in node.inputs
            and node.params.get("entire") is not False
        ):
            errors.append(
                ValidationIssue(
                    stage="semantic",
                    code="count_entities_requires_non_entire",
                    message="COUNT over EntitySet requires params.entire=false",
                    node_id=node.id,
                )
            )
        raw_target = node.params.get("target")
        if isinstance(raw_target, Mapping) and EXTERNAL_RELATIONS.search(
            str(raw_target.get("category", ""))
        ):
            warnings.append(
                ValidationIssue(
                    stage="semantic",
                    code="external_relation_in_target",
                    message="TargetSpec category appears to hide an external spatial relation",
                    node_id=node.id,
                )
            )
        if isinstance(node.params.get("choices"), list):
            errors.append(
                ValidationIssue(
                    stage="semantic",
                    code="original_choices_copied_into_target",
                    message="Teacher targets must not copy or rewrite original choice options",
                    node_id=node.id,
                )
            )

    if BBOX_HINT.search(question) and OperatorName.REGION_FROM_BBOX not in ops:
        warnings.append(
            ValidationIssue(
                stage="semantic",
                code="bbox_without_region_from_bbox",
                message="question mentions a bbox but graph lacks REGION_FROM_BBOX",
            )
        )
    if MARKER_HINT.search(question) and OperatorName.FIND_MARKER not in ops:
        warnings.append(
            ValidationIssue(
                stage="semantic",
                code="marker_without_find_marker",
                message="question mentions a visual marker but graph lacks FIND_MARKER",
            )
        )
    if OperatorName.FIND_MARKER in ops and not MARKER_HINT.search(question):
        warnings.append(
            ValidationIssue(
                stage="semantic",
                code="find_marker_without_artificial_marker",
                message="FIND_MARKER is reserved for explicit artificial benchmark markers",
            )
        )

    warnings.extend(_explicit_alternative_loss_warnings(target, question))

    required_ops: dict[IntentLabel, set[OperatorName]] = {
        IntentLabel.SIMPLE_COUNT: {OperatorName.COUNT},
        IntentLabel.RELATIONAL_COUNT: {OperatorName.COUNT},
        IntentLabel.OBJECT_RELATION: {OperatorName.RELATION},
        IntentLabel.ROUTE_PLANNING: {OperatorName.BUILD_ROUTE_CONTEXT},
        IntentLabel.MOTION_QUERY: {OperatorName.MOTION},
    }
    if target.intent in required_ops:
        missing_ops = required_ops[target.intent] - set(ops)
        if missing_ops:
            errors.append(
                ValidationIssue(
                    stage="semantic",
                    code="dedicated_operator_bypass",
                    message="intent requires dedicated operators: "
                    + ", ".join(sorted(op.value for op in missing_ops)),
                )
            )

    if target.intent is IntentLabel.ATTRIBUTE_QUERY and OperatorName.ATTRIBUTE not in ops:
        final_runtime_types = set().union(
            *(inferred.get(ref, set()) for ref in target.final.sources), set()
        )
        if not final_runtime_types.intersection(VISUAL_RUNTIME_TYPES):
            errors.append(
                ValidationIssue(
                    stage="semantic",
                    code="dedicated_operator_bypass",
                    message=(
                        "ATTRIBUTE_QUERY requires ATTRIBUTE or selected visual evidence in "
                        "final.sources"
                    ),
                )
            )

    if RELATION_QUERY_HINT.search(question) and OperatorName.RELATION not in ops:
        errors.append(
            ValidationIssue(
                stage="semantic",
                code="relation_query_should_use_relation",
                message="explicit relative-position questions must use RELATION",
            )
        )

    if target.intent is IntentLabel.RELATIONAL_COUNT:
        select_ids = {node.id for node in target.nodes if node.op is OperatorName.SELECT}
        if select_ids:
            final_counts = [
                node
                for node in target.nodes
                if node.op is OperatorName.COUNT and node.id in contributing_ids
            ]

            def count_consumes_select(count_id: str) -> bool:
                count = nodes_by_id[count_id]
                selected_ancestors = ancestors(count_id).intersection(select_ids)
                if "entities" in count.inputs:
                    return bool(selected_ancestors)
                if "image" in count.inputs:
                    return any(
                        nodes_by_id[node_id].params.get("mode") == "SUBREGION"
                        for node_id in selected_ancestors
                    )
                return False

            select_is_consumed = any(count_consumes_select(count.id) for count in final_counts)
            if not select_is_consumed:
                errors.append(
                    ValidationIssue(
                        stage="semantic",
                        code="relation_result_not_consumed",
                        message=(
                            "RELATIONAL_COUNT SELECT result must reach a contributing final "
                            "COUNT dependency through COUNT.image or COUNT.entities"
                        ),
                    )
                )

    choice_question_types = {
        QuestionType.MULTIPLE_CHOICE.value,
        QuestionType.MULTIPLE_CHOICE_SINGLE.value,
        QuestionType.MULTIPLE_CHOICE_MULTI.value,
    }
    choice_answer_types = {AnswerType.CHOICE_SINGLE, AnswerType.CHOICE_MULTI}
    if (
        question_type in choice_question_types
        and target.final.answer_type not in choice_answer_types
    ):
        errors.append(
            ValidationIssue(
                stage="semantic",
                code="choice_answer_type_mismatch",
                message=(
                    "multiple-choice questions require final.answer_type to be "
                    "CHOICE_SINGLE or CHOICE_MULTI; cardinality is determined by the "
                    "question semantics, not source question_type metadata"
                ),
            )
        )

    count_is_authoritative = any(
        node_id in nodes_by_id and nodes_by_id[node_id].op is OperatorName.COUNT
        for node_id in contributing_ids
    )
    final_question = target.final.question
    if final_question is not None and (
        RUNTIME_RESULT_HINT.search(final_question)
        or (count_is_authoritative and NUMERIC_VALUE_HINT.search(final_question))
    ):
        errors.append(
            ValidationIssue(
                stage="semantic",
                code="runtime_prediction_in_final_question",
                message=(
                    "final.question must be static and must not contain predicted runtime values"
                ),
            )
        )

    final_runtime_types = {ref: inferred.get(ref, set()) for ref in target.final.sources}
    combined_final_types = set().union(*final_runtime_types.values(), set())
    if combined_final_types.intersection(VISUAL_RUNTIME_TYPES) and final_question is None:
        errors.append(
            ValidationIssue(
                stage="semantic",
                code="missing_residual_final_question",
                message=("visual or RouteContext final sources require a residual final.question"),
            )
        )
    if (
        final_question is not None
        and combined_final_types
        and combined_final_types.issubset(STRUCTURED_RUNTIME_TYPES)
        and GENERIC_STRUCTURED_FINAL_QUESTION.search(final_question)
    ):
        warnings.append(
            ValidationIssue(
                stage="semantic",
                code="non_minimal_structured_final_question",
                message=(
                    "authoritative structured final sources normally omit generic choice questions"
                ),
            )
        )
    if count_is_authoritative and any(
        runtime_types.intersection(VISUAL_RUNTIME_TYPES)
        for runtime_types in final_runtime_types.values()
    ):
        errors.append(
            ValidationIssue(
                stage="semantic",
                code="authoritative_count_visual_reintroduction",
                message=(
                    "authoritative COUNT results must not reintroduce visual evidence into "
                    "final.sources"
                ),
            )
        )
    return errors, warnings


def validate_candidate(
    candidate: str | Mapping[str, Any] | PlannerTarget,
    *,
    inputs: Mapping[str, Any],
    question: str = "",
    question_type: str = QuestionType.FREE_FORM.value,
) -> tuple[PlannerTarget | None, ValidationResult]:
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []
    normalized_fields: list[dict[str, Any]] = []
    payload: Any = candidate
    if isinstance(candidate, str):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            errors.append(ValidationIssue(stage="schema", code="json_parse", message=str(exc)))
            return None, ValidationResult(
                valid=False,
                schema_valid=False,
                graph_valid=False,
                type_valid=False,
                semantic_valid=False,
                errors=errors,
                warnings=[],
                inferred_types={},
                resolved_input_types={},
                normalized_fields=[],
            )
    if isinstance(payload, Mapping) and not isinstance(payload, PlannerTarget):
        payload, normalized_fields = normalize_candidate_payload(payload)
    try:
        target = (
            payload if isinstance(payload, PlannerTarget) else PlannerTarget.model_validate(payload)
        )
    except ValidationError as exc:
        for item in exc.errors(include_url=False):
            location = ".".join(str(value) for value in item["loc"])
            errors.append(
                ValidationIssue(
                    stage="schema", code=str(item["type"]), message=f"{location}: {item['msg']}"
                )
            )
        return None, ValidationResult(
            valid=False,
            schema_valid=False,
            graph_valid=False,
            type_valid=False,
            semantic_valid=False,
            errors=errors,
            warnings=[],
            inferred_types={},
            resolved_input_types={},
            normalized_fields=normalized_fields,
        )

    graph_errors = _graph_issues(target, set(inputs))
    errors.extend(graph_errors)
    type_result = check_types(target, set(inputs))
    errors.extend(
        ValidationIssue(stage="type", code=item.code, message=item.message, node_id=item.node_id)
        for item in type_result.errors
    )
    semantic_errors, semantic_warnings = _semantic_issues(
        target, question=question, question_type=str(question_type), inferred=type_result.inferred
    )
    errors.extend(semantic_errors)
    warnings.extend(semantic_warnings)
    inferred_types = {
        key: sorted(item.value for item in value)
        for key, value in sorted(type_result.inferred.items())
    }
    resolved_input_types = {
        node_id: {
            name: [
                {"ref": ref, "types": sorted(runtime_type.value for runtime_type in types)}
                for ref, types in values
            ]
            for name, values in inputs_by_name.items()
        }
        for node_id, inputs_by_name in type_result.resolved_inputs.items()
    }
    graph_valid = not graph_errors
    type_valid = not type_result.errors
    semantic_valid = not semantic_errors
    return target, ValidationResult(
        valid=not errors,
        schema_valid=True,
        graph_valid=graph_valid,
        type_valid=type_valid,
        semantic_valid=semantic_valid,
        errors=errors,
        warnings=warnings,
        inferred_types=inferred_types,
        resolved_input_types=resolved_input_types,
        normalized_fields=normalized_fields,
    )
