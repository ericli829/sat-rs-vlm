"""Convert measured sensitivity results into an auditable tiered protection policy."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal

RiskTier = Literal["low", "medium", "high", "critical"]

TIER_ACTIONS: dict[RiskTier, list[str]] = {
    "low": ["monitor", "record_fault_metadata"],
    "medium": [
        "monitor",
        "record_fault_metadata",
        "output_validation",
        "retry",
        "recompute_current_step",
    ],
    "high": [
        "monitor",
        "record_fault_metadata",
        "output_validation",
        "retry",
        "recompute_current_step",
        "sha256_manifest",
        "warm_replica",
        "golden_replica",
        "scheduled_scrub",
    ],
    "critical": [
        "monitor",
        "record_fault_metadata",
        "output_validation",
        "retry",
        "recompute_current_step",
        "sha256_manifest",
        "warm_replica",
        "golden_replica",
        "scheduled_scrub",
        "selective_ecc",
        "dual_execution_detection",
        "tmr",
        "rollback",
    ],
}


def classify_risk(
    *,
    changed_rate: float | None,
    invalid_rate: float | None,
    exact_match_drop: float | None = None,
    task_degradations: Iterable[Mapping[str, Any]] = (),
) -> RiskTier:
    """Classify risk from diagnostics and Evaluation task-metric degradation.

    ``exact_match_drop`` remains a compatibility diagnostic for historical reports.
    New sensitivity groups should provide ``task_degradations`` whose values already
    follow the Evaluation comparison convention (positive means degradation here).
    """

    changed = float(changed_rate or 0.0)
    invalid = float(invalid_rate or 0.0)
    measured = [max(0.0, float(row.get("degradation_mean") or 0.0)) for row in task_degradations]
    degradation = max([float(exact_match_drop or 0.0), *measured])
    if invalid >= 0.20 or degradation >= 0.20 or changed >= 0.80:
        return "critical"
    if invalid >= 0.05 or degradation >= 0.08 or changed >= 0.40:
        return "high"
    if invalid > 0.0 or degradation >= 0.02 or changed >= 0.10:
        return "medium"
    return "low"


def build_protection_policy(groups: list[dict[str, Any]]) -> dict[str, Any]:
    """Map sensitivity groups to recommended software and hardware actions."""

    decisions: list[dict[str, Any]] = []
    implemented = {
        "monitor",
        "record_fault_metadata",
        "output_validation",
        "sha256_manifest",
        "warm_replica",
        "golden_replica",
        "scheduled_scrub",
    }
    for group in groups:
        tier = classify_risk(
            changed_rate=group.get("changed_rate_mean"),
            invalid_rate=group.get("invalid_rate_mean"),
            exact_match_drop=group.get("exact_match_drop_mean"),
            task_degradations=group.get("task_degradations", []),
        )
        actions = TIER_ACTIONS[tier]
        decisions.append(
            {
                "target": group.get("target"),
                "layers": group.get("layers", []),
                "bit_plane": group.get("bit_plane"),
                "intensity": group.get("intensity"),
                "repeats": group.get("repeats"),
                "risk_tier": tier,
                "evidence": {
                    "changed_rate_mean": group.get("changed_rate_mean"),
                    "invalid_rate_mean": group.get("invalid_rate_mean"),
                    "exact_match_drop_mean": group.get("exact_match_drop_mean"),
                    "changed_rate_ci95": group.get("changed_rate_ci95"),
                    "invalid_rate_ci95": group.get("invalid_rate_ci95"),
                    "exact_match_drop_ci95": group.get("exact_match_drop_ci95"),
                    "task_degradations": group.get("task_degradations", []),
                },
                "recommended_actions": actions,
                "implemented_actions": [action for action in actions if action in implemented],
                "platform_integration_required": [
                    action for action in actions if action not in implemented
                ],
            }
        )
    return {
        "schema_version": "2.0",
        "method": "evaluation_task_metric_tiered_protection",
        "thresholds": {
            "critical": "invalid>=0.20 OR exact_match_drop>=0.20 OR changed>=0.80",
            "high": "invalid>=0.05 OR exact_match_drop>=0.08 OR changed>=0.40",
            "medium": "invalid>0 OR exact_match_drop>=0.02 OR changed>=0.10",
        },
        "decisions": decisions,
    }
