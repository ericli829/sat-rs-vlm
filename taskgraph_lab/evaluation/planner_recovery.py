"""Bounded, fail-closed recovery policy for Small Planner inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Any


class RecoveryLevel(IntEnum):
    NORMALIZATION = 0
    GENERATION = 1
    VALIDATION = 2
    HARD_RAG = 3


class RecoveryAction(StrEnum):
    RETURN = "return"
    DIAGNOSTIC_RETRY = "diagnostic_retry"
    RAG_RETRY = "rag_retry"
    FAIL = "planner_failed"


@dataclass(frozen=True)
class RecoveryDecision:
    action: RecoveryAction
    level: RecoveryLevel
    termination_reason: str


_ERROR_HINTS = {
    "dead_node": "A node is not connected to the final dependency chain.",
    "relation_result_not_consumed": (
        "The relation-filtered result is not consumed by the final COUNT operation."
    ),
    "input_type_mismatch": "An operator receives the wrong input role or runtime type.",
    "missing_final_ref": "The final result points to a missing node.",
    "missing_node_ref": "A node points to a missing dependency.",
    "forward_reference": "A node references a dependency that has not been produced yet.",
    "dedicated_operator_bypass": "The plan bypasses an operator required by the intent.",
    "relation_query_should_use_relation": (
        "A question asking for an unknown relation must use RELATION, not SELECT_REL."
    ),
    "count_entities_requires_non_entire": (
        "COUNT over a filtered EntitySet must use entire=false."
    ),
}


def is_runtime_valid(record: Mapping[str, Any]) -> bool:
    return bool(record.get("graph_runtime_valid"))


def concise_validator_diagnostic(record: Mapping[str, Any], *, max_codes: int = 4) -> str:
    """Build a short model-facing diagnostic without exceptions or stack traces."""

    codes = [str(value) for value in (record.get("validation_error_codes") or [])]
    if not bool(record.get("surface_grammar_valid")):
        failure = str(record.get("constraint_failure") or record.get("termination_reason") or "")
        detail = {
            "repeat_guard_forced_final": "A repeated semantic loop was stopped.",
            "constraint_abort": "The constrained decoder could not complete a legal program.",
            "max_tokens": "The program was incomplete at the generation limit.",
        }.get(failure, "The output is not a complete legal TaskGraph DSL program.")
        return (
            "Previous plan is invalid.\n\n"
            f"Error:\n{detail}\n\n"
            "Regenerate the complete plan as DSL only. Include exactly one FINAL or "
            "FINAL_QUESTION and do not repeat branches."
        )

    hints: list[str] = []
    for code in dict.fromkeys(codes):
        hints.append(_ERROR_HINTS.get(code, f"Validator error: {code}."))
        if len(hints) >= max_codes:
            break
    if not hints:
        hints.append("The graph, type, or semantic validator rejected the plan.")
    return (
        "Previous plan is invalid.\n\nError:\n"
        + "\n".join(hints)
        + "\n\nRegenerate the complete plan. Preserve every relation required by the "
        "original question. Return DSL only."
    )


def retry_prompt_messages(
    original_messages: Sequence[Mapping[str, Any]],
    *,
    previous_prediction: str,
    diagnostic: str,
) -> list[dict[str, Any]]:
    """Append one compact correction turn while preserving the original request."""

    messages = [dict(message) for message in original_messages]
    messages.append({"role": "assistant", "content": str(previous_prediction).strip()})
    messages.append({"role": "user", "content": diagnostic})
    return messages


class PlannerRetryPolicy:
    """Maximum 1 normal + 1 diagnostic + 1 hard-case RAG generation."""

    def __init__(self, *, max_attempts: int = 3) -> None:
        if not 1 <= max_attempts <= 3:
            raise ValueError("max_attempts must be between 1 and 3")
        self.max_attempts = max_attempts

    def decide(
        self,
        record: Mapping[str, Any],
        *,
        attempts: int,
        hard_case: bool,
        rag_available: bool,
    ) -> RecoveryDecision:
        if attempts < 1:
            raise ValueError("attempts must be positive")
        if is_runtime_valid(record):
            return RecoveryDecision(RecoveryAction.RETURN, RecoveryLevel.NORMALIZATION, "final")
        if attempts >= self.max_attempts:
            return RecoveryDecision(RecoveryAction.FAIL, RecoveryLevel.VALIDATION, "planner_failed")
        if attempts == 1:
            level = (
                RecoveryLevel.GENERATION
                if not bool(record.get("surface_grammar_valid"))
                else RecoveryLevel.VALIDATION
            )
            return RecoveryDecision(
                RecoveryAction.DIAGNOSTIC_RETRY,
                level,
                "validation_retry",
            )
        if attempts == 2 and hard_case and rag_available and self.max_attempts >= 3:
            return RecoveryDecision(RecoveryAction.RAG_RETRY, RecoveryLevel.HARD_RAG, "rag_retry")
        return RecoveryDecision(RecoveryAction.FAIL, RecoveryLevel.VALIDATION, "planner_failed")

