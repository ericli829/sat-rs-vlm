from __future__ import annotations

from scripts.evaluation.evaluate_change_judge_hybrid import evaluate_hybrid_rows


def test_hybrid_uses_rules_then_falls_back_to_local_judge() -> None:
    gold = [
        {
            "audit_id": "positive",
            "caption": "No buildings changed, but a new road appeared.",
            "human_gold_label": "1",
        },
        {
            "audit_id": "non-target",
            "caption": "The forest became lighter and the field changed color.",
            "human_gold_label": "0",
        },
        {
            "audit_id": "fallback",
            "caption": "The overall area looks different.",
            "human_gold_label": "1",
        },
    ]
    answers = {
        "positive": {"old_parser_decision": 1, "local_judge_decision": 0},
        "non-target": {"old_parser_decision": 1, "local_judge_decision": 1},
        "fallback": {"old_parser_decision": 1, "local_judge_decision": 1},
    }

    summary, rows = evaluate_hybrid_rows(gold, answers)

    assert [row["hybrid_v2_decision"] for row in rows] == [1, 0, 1]
    assert summary["implementation_version"] == "levir-local-text-judge-v2.3-hybrid"
    assert summary["decision_profile"] == "local_text_judge_priority_v1.3"
    assert summary["compatibility"]["legacy_decision_field"] == "hybrid_v2_decision"
    assert summary["hybrid_v2"]["accuracy"] == 1.0
    assert summary["source_distribution"] == {
        "local_llm_judge": 1,
        "local_semantic_non_target_rule": 1,
        "local_semantic_positive_rule": 1,
    }
