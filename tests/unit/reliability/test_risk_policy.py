from sat_rs_vlm.models.reliability.risk_policy import build_protection_policy, classify_risk


def test_risk_tier_thresholds_are_conservative() -> None:
    assert classify_risk(changed_rate=0.01, invalid_rate=0.0, exact_match_drop=0.0) == "low"
    assert classify_risk(changed_rate=0.12, invalid_rate=0.0, exact_match_drop=0.0) == "medium"
    assert classify_risk(changed_rate=0.5, invalid_rate=0.0, exact_match_drop=0.0) == "high"
    assert classify_risk(changed_rate=0.1, invalid_rate=0.3, exact_match_drop=0.0) == "critical"


def test_policy_distinguishes_implemented_and_platform_actions() -> None:
    policy = build_protection_policy([
        {
            "target": "attention", "layers": [14], "bit_plane": "exponent",
            "intensity": 10, "repeats": 20, "changed_rate_mean": 0.9,
            "invalid_rate_mean": 0.25, "exact_match_drop_mean": 0.3,
        }
    ])
    decision = policy["decisions"][0]
    assert decision["risk_tier"] == "critical"
    assert "golden_replica" in decision["implemented_actions"]
    assert "tmr" in decision["platform_integration_required"]
