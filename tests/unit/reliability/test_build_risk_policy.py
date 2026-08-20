import json
from pathlib import Path


def test_build_risk_policy_from_sensitivity_run(tmp_path: Path) -> None:
    from scripts.reliability.build_risk_policy import main
    import sys

    root = tmp_path / "run"
    condition = root / "conditions" / "attention_layer_0_exponent_bits_1_repeat_0" / "comparison"
    condition.mkdir(parents=True)
    (root / "condition_plan.json").write_text(json.dumps({"conditions": [{
        "id": "attention_layer_0_exponent_bits_1_repeat_0", "target": "attention",
        "layers": [0], "bit_plane": "exponent", "num_bits": 1,
    }]}), encoding="utf-8")
    (condition / "comparison.json").write_text(json.dumps({"overall": {"prediction_changed_rate": 0.9}}), encoding="utf-8")
    output = tmp_path / "policy.json"
    argv = sys.argv
    sys.argv = ["build_risk_policy", "--sensitivity-root", str(root), "--output", str(output)]
    try:
        assert main() == 0
    finally:
        sys.argv = argv
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["decisions"][0]["risk_tier"] == "critical"
