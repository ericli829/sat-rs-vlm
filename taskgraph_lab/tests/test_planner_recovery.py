from __future__ import annotations

import argparse

import pytest

from taskgraph_lab.evaluation.planner_recovery import (
    PlannerRetryPolicy,
    RecoveryAction,
    RecoveryLevel,
    concise_validator_diagnostic,
    retry_prompt_messages,
)
from taskgraph_lab.tools.evaluate_qwen3vl_planner import _validate_args


def _invalid_record(*, surface: bool = True) -> dict:
    return {
        "surface_grammar_valid": surface,
        "graph_runtime_valid": False,
        "validation_error_codes": ["dead_node", "relation_result_not_consumed"],
        "constraint_failure": None if surface else "constraint_abort",
    }


def test_validator_retry_receives_concise_diagnostic() -> None:
    diagnostic = concise_validator_diagnostic(_invalid_record())
    assert "not connected to the final dependency chain" in diagnostic
    assert "filtered result is not consumed" in diagnostic
    assert "Traceback" not in diagnostic
    assert len(diagnostic) < 600
    messages = retry_prompt_messages(
        [{"role": "system", "content": "DSL only"}, {"role": "user", "content": "q"}],
        previous_prediction="bad dsl",
        diagnostic=diagnostic,
    )
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert messages[-1]["content"] == diagnostic


def test_retry_budget_is_one_diagnostic_plus_one_hard_rag() -> None:
    policy = PlannerRetryPolicy(max_attempts=3)
    first = policy.decide(_invalid_record(), attempts=1, hard_case=True, rag_available=True)
    second = policy.decide(_invalid_record(), attempts=2, hard_case=True, rag_available=True)
    third = policy.decide(_invalid_record(), attempts=3, hard_case=True, rag_available=True)
    assert first.action is RecoveryAction.DIAGNOSTIC_RETRY
    assert first.level is RecoveryLevel.VALIDATION
    assert second.action is RecoveryAction.RAG_RETRY
    assert second.level is RecoveryLevel.HARD_RAG
    assert third.action is RecoveryAction.FAIL


def test_simple_failure_never_spends_third_rag_attempt() -> None:
    decision = PlannerRetryPolicy().decide(
        _invalid_record(),
        attempts=2,
        hard_case=False,
        rag_available=True,
    )
    assert decision.action is RecoveryAction.FAIL


def test_grammar_failure_uses_level_one() -> None:
    decision = PlannerRetryPolicy().decide(
        _invalid_record(surface=False),
        attempts=1,
        hard_case=True,
        rag_available=True,
    )
    assert decision.level is RecoveryLevel.GENERATION


def test_recovery_has_no_unconstrained_fallback() -> None:
    args = argparse.Namespace(
        batch_size=1,
        constraint_top_k=64,
        constraint_max_candidate_checks=256,
        rag_mode="off",
        constrained=False,
        enable_recovery=True,
        rag_top_k=0,
        rag_bank=None,
        rag_rules=False,
        rag_rule_cards=None,
        rag_router="heuristic",
        intent_filter=[],
    )
    with pytest.raises(ValueError, match="unconstrained fallback is forbidden"):
        _validate_args(args)
