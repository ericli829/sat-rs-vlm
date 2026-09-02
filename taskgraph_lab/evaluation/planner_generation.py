"""Strict generation metrics for the text-only TaskGraph Planner."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from taskgraph_lab.taskgraph.canonicalize import canonicalize_target
from taskgraph_lab.taskgraph.dsl import (
    CanonicalDSLPrefixGrammar,
    compile_taskgraph_to_dsl,
    parse_taskgraph_dsl,
    parse_taskgraph_dsl_payload,
)
from taskgraph_lab.taskgraph.validator import validate_candidate

from .relational_metrics import relational_metrics


def _messages(row: Mapping[str, Any]) -> tuple[list[dict[str, Any]], str]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise TypeError("Planner evaluation row must contain messages")
    prompt = [dict(message) for message in messages if message.get("role") != "assistant"]
    answers = [message for message in messages if message.get("role") == "assistant"]
    if len(answers) != 1:
        raise ValueError("Planner evaluation row must contain exactly one assistant target")
    return prompt, str(answers[0].get("content", "")).strip()


def _user_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    messages = row.get("messages")
    users = [message for message in messages or [] if message.get("role") == "user"]
    if len(users) != 1:
        raise ValueError("Planner evaluation row must contain exactly one user message")
    payload = json.loads(str(users[0].get("content", "")))
    if not isinstance(payload, dict):
        raise TypeError("Planner user content must decode to an object")
    return payload


def prompt_messages(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the exact system/user prompt used for generation."""

    return _messages(row)[0]


def evaluate_prediction(row: Mapping[str, Any], prediction: str) -> dict[str, Any]:
    """Compare one raw model generation with its canonical Planner target.

    The prediction is deliberately not repaired or stripped of Markdown fences.
    Only surrounding whitespace is ignored.
    """

    sample_id = str(row.get("id", ""))
    _, expected_dsl = _messages(row)
    user = _user_payload(row)
    expected = canonicalize_target(parse_taskgraph_dsl(expected_dsl))
    raw = str(prediction)
    stripped = raw.strip()
    result: dict[str, Any] = {
        "sample_id": sample_id,
        "dataset": (row.get("metadata") or {}).get("dataset"),
        "expected_intent": expected.get("intent"),
        "expected_dsl": expected_dsl,
        "prediction": raw,
        "nonempty": bool(stripped),
        "text_exact": stripped == expected_dsl,
        "surface_grammar_valid": False,
        "dsl_parse_valid": False,
        "schema_valid": False,
        "graph_valid": False,
        "type_valid": False,
        "semantic_valid": False,
        "graph_runtime_valid": False,
        "runtime_valid": False,
        "canonical_exact": False,
        "canonical_dsl_exact": False,
        "intent_exact": False,
        "operator_sequence_exact": False,
        "node_count_exact": False,
        "final_answer_type_exact": False,
        "final_question_presence_exact": False,
        "parse_error": None,
        "validation_error_codes": [],
        "predicted_intent": None,
        "predicted_dsl": None,
        "canonical_compile_error": None,
    }
    result.update(relational_metrics(expected, None))
    inputs = user.get("inputs") or {}
    if isinstance(inputs, Mapping) and inputs:
        try:
            grammar = CanonicalDSLPrefixGrammar(inputs.keys())
            result["surface_grammar_valid"] = grammar.accepts(stripped)
        except (TypeError, ValueError):
            result["surface_grammar_valid"] = False
    try:
        predicted_payload = parse_taskgraph_dsl_payload(stripped)
    except Exception as exc:  # Parser error type can wrap schema/type failures.
        result["parse_error"] = f"{type(exc).__name__}: {exc}"
        return result

    result["dsl_parse_valid"] = True
    predicted_target, validation = validate_candidate(
        predicted_payload,
        inputs=inputs,
        question=str(user.get("question", "")),
        question_type=str(user.get("question_type", "FREE_FORM")),
    )
    result["schema_valid"] = bool(validation.schema_valid)
    result["graph_valid"] = bool(validation.graph_valid)
    result["type_valid"] = bool(validation.type_valid)
    result["semantic_valid"] = bool(validation.semantic_valid)
    result["graph_runtime_valid"] = bool(validation.valid)
    result["runtime_valid"] = result["graph_runtime_valid"]
    result["validation_error_codes"] = [issue.code for issue in validation.errors]
    if predicted_target is None:
        return result

    try:
        predicted = canonicalize_target(predicted_target)
    except ValueError as exc:
        result["parse_error"] = f"CanonicalizationError: {exc}"
        return result
    result["predicted_intent"] = predicted.get("intent")
    try:
        result["predicted_dsl"] = compile_taskgraph_to_dsl(predicted)
    except Exception as exc:
        result["canonical_compile_error"] = f"{type(exc).__name__}: {exc}"
    result["canonical_exact"] = predicted == expected
    result["canonical_dsl_exact"] = result["predicted_dsl"] == expected_dsl
    result["intent_exact"] = predicted.get("intent") == expected.get("intent")
    expected_ops = [node["op"] for node in expected["nodes"]]
    predicted_ops = [node["op"] for node in predicted["nodes"]]
    result["operator_sequence_exact"] = predicted_ops == expected_ops
    result["node_count_exact"] = len(predicted_ops) == len(expected_ops)
    result["final_answer_type_exact"] = (
        predicted["final"]["answer_type"] == expected["final"]["answer_type"]
    )
    result["final_question_presence_exact"] = (
        ("question" in predicted["final"]) == ("question" in expected["final"])
    )
    result.update(
        relational_metrics(
            expected,
            predicted,
            validation_error_codes=result["validation_error_codes"],
        )
    )
    return result


_BOOLEAN_METRICS = (
    "nonempty",
    "text_exact",
    "surface_grammar_valid",
    "dsl_parse_valid",
    "schema_valid",
    "graph_valid",
    "type_valid",
    "semantic_valid",
    "graph_runtime_valid",
    "runtime_valid",
    "canonical_exact",
    "canonical_dsl_exact",
    "intent_exact",
    "operator_sequence_exact",
    "node_count_exact",
    "final_answer_type_exact",
    "final_question_presence_exact",
)


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def summarize_predictions(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate strict metrics with all samples retained in the denominator."""

    total = len(records)
    counts = {name: sum(bool(row.get(name)) for row in records) for name in _BOOLEAN_METRICS}
    rates = {name: (counts[name] / total if total else 0.0) for name in _BOOLEAN_METRICS}
    per_intent_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        per_intent_rows[str(row.get("expected_intent") or "UNKNOWN")].append(row)
    per_intent: dict[str, Any] = {}
    for intent, intent_rows in sorted(per_intent_rows.items()):
        intent_counts = {
            name: sum(bool(row.get(name)) for row in intent_rows)
            for name in _BOOLEAN_METRICS
        }
        intent_rates = {
            name: value / len(intent_rows) for name, value in intent_counts.items()
        }
        relation_summary: dict[str, Any] = {}
        for name in (
            "relation_direction_accuracy",
            "reference_attachment_accuracy",
            "select_rel_vs_relation_accuracy",
            "count_filtered_source_accuracy",
            "count_scope_entire_accuracy",
        ):
            values = [row[name] for row in intent_rows if row.get(name) is not None]
            relation_summary[name] = {
                "applicable_count": len(values),
                "correct_count": sum(bool(value) for value in values),
                "rate": (
                    sum(bool(value) for value in values) / len(values) if values else None
                ),
            }
        per_intent[intent] = {
            "count": len(intent_rows),
            "counts": intent_counts,
            "rates": intent_rates,
            "relational_metrics": relation_summary,
            # Compatibility fields retained for existing report readers.
            "surface_grammar_valid": intent_counts["surface_grammar_valid"],
            "dsl_parse_valid": intent_counts["dsl_parse_valid"],
            "graph_runtime_valid": intent_counts["graph_runtime_valid"],
            "runtime_valid": intent_counts["runtime_valid"],
            "canonical_exact": intent_counts["canonical_exact"],
        }
    parse_errors = Counter(
        str(row.get("parse_error", "")).split(":", 1)[0]
        for row in records
        if row.get("parse_error")
    )
    validation_errors = Counter(
        str(code)
        for row in records
        for code in (row.get("validation_error_codes") or [])
    )
    latencies = [float(row["latency_seconds"]) for row in records if "latency_seconds" in row]
    total_latencies = [
        float(row["total_planner_latency"])
        for row in records
        if row.get("total_planner_latency") is not None
    ]
    retrieval_latencies = [
        float(row["retrieval_latency_ms"])
        for row in records
        if row.get("retrieval_latency_ms") is not None
    ]
    prompt_tokens = [int(row["prompt_tokens"]) for row in records if "prompt_tokens" in row]
    generated_tokens = [
        int(row["generated_tokens"]) for row in records if "generated_tokens" in row
    ]
    generation_attempts = [
        int(row["generation_attempts"])
        for row in records
        if row.get("generation_attempts") is not None
    ]
    termination_reasons = Counter(
        str(row.get("termination_reason"))
        for row in records
        if row.get("termination_reason") is not None
    )
    constraint_failures = Counter(
        str(row.get("constraint_failure"))
        for row in records
        if row.get("constraint_failure") is not None
    )
    relational: dict[str, Any] = {}
    for name in (
        "relation_direction_accuracy",
        "reference_attachment_accuracy",
        "select_rel_vs_relation_accuracy",
        "count_filtered_source_accuracy",
        "count_scope_entire_accuracy",
    ):
        applicable = [row[name] for row in records if row.get(name) is not None]
        relational[name] = {
            "applicable_count": len(applicable),
            "correct_count": sum(bool(value) for value in applicable),
            "rate": (
                sum(bool(value) for value in applicable) / len(applicable)
                if applicable
                else None
            ),
        }
    relation_rows = [row for row in records if row.get("relational_metric_applicable")]
    broken_depths = [
        int(row["first_broken_relation_chain_depth"])
        for row in relation_rows
        if row.get("first_broken_relation_chain_depth") is not None
    ]
    relational["dead_node_rate"] = (
        sum(bool(row.get("dead_node")) for row in relation_rows) / len(relation_rows)
        if relation_rows
        else None
    )
    relational["mean_first_broken_relation_chain_depth"] = (
        sum(broken_depths) / len(broken_depths) if broken_depths else None
    )
    return {
        "sample_count": total,
        "counts": counts,
        "rates": rates,
        "per_intent": per_intent,
        "parse_error_types": dict(parse_errors.most_common()),
        "validation_error_codes": dict(validation_errors.most_common()),
        "mean_latency_seconds": sum(latencies) / len(latencies) if latencies else None,
        "p50_latency_seconds": _percentile(latencies, 0.50),
        "p95_latency_seconds": _percentile(latencies, 0.95),
        "p50_total_planner_latency_seconds": _percentile(total_latencies, 0.50),
        "p95_total_planner_latency_seconds": _percentile(total_latencies, 0.95),
        "mean_total_planner_latency_seconds": (
            sum(total_latencies) / len(total_latencies) if total_latencies else None
        ),
        "mean_retrieval_latency_ms": (
            sum(retrieval_latencies) / len(retrieval_latencies)
            if retrieval_latencies
            else None
        ),
        "p95_retrieval_latency_ms": _percentile(retrieval_latencies, 0.95),
        "mean_prompt_tokens": sum(prompt_tokens) / len(prompt_tokens) if prompt_tokens else None,
        "total_prompt_tokens": sum(prompt_tokens),
        "mean_generated_tokens": (
            sum(generated_tokens) / len(generated_tokens) if generated_tokens else None
        ),
        "total_generated_tokens": sum(generated_tokens),
        "mean_generation_attempts": (
            sum(generation_attempts) / len(generation_attempts)
            if generation_attempts
            else None
        ),
        "rag_used_count": sum(bool(row.get("rag_used")) for row in records),
        "planner_failed_count": sum(
            row.get("termination_reason") == "planner_failed" for row in records
        ),
        "termination_reason_counts": dict(termination_reasons.most_common()),
        "constraint_failure_counts": dict(constraint_failures.most_common()),
        "grammar_dead_end_count": sum(
            int(row.get("grammar_dead_end_count", 0)) for row in records
        ),
        "max_token_truncation_count": sum(
            row.get("termination_reason") == "max_tokens" for row in records
        ),
        "relational_metrics": relational,
    }
