from taskgraph_lab.tools.compare_planner_constraint_ab import (
    _backfill_legacy_termination,
)


def test_legacy_termination_backfill_preserves_and_recovers_labels() -> None:
    rows = [
        {"dsl_parse_valid": True, "generated_tokens": 40},
        {"dsl_parse_valid": False, "generated_tokens": 512},
        {"dsl_parse_valid": False, "generated_tokens": 20},
        {"termination_reason": "repeat_guard", "generated_tokens": 12},
    ]

    _backfill_legacy_termination(rows, max_new_tokens=512)

    assert [row["termination_reason"] for row in rows] == [
        "final",
        "max_tokens",
        "error",
        "repeat_guard",
    ]
