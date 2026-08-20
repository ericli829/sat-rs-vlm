from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_v15_conditions_expand_layered_targets_and_unique_seeds() -> None:
    from scripts.reliability.run_v15_sensitivity import build_conditions, validate_condition_plan

    conditions = build_conditions(
        {
            "experiment": {"seed": 10},
            "fault": {
                "sensitivity_targets": ["lora_adapter", "attention"],
                "layer_indices": [0, 1],
                "bit_flip_counts": [1, 10],
                "repeats": 2,
                "bit_planes": ["all", "exponent"],
            },
        }
    )

    assert len(conditions) == 32
    assert len({condition["seed"] for condition in conditions}) == len(conditions)
    assert {
        tuple(condition["layers"]) for condition in conditions if condition["target"] == "attention"
    } == {
        (0,),
        (1,),
    }
    assert {
        tuple(condition["layers"])
        for condition in conditions
        if condition["target"] == "lora_adapter"
    } == {
        (0,),
        (1,),
    }
    assert validate_condition_plan(conditions)["valid"]


def test_condition_plan_rejects_duplicate_ids_and_seeds() -> None:
    from scripts.reliability.run_v15_sensitivity import validate_condition_plan

    result = validate_condition_plan(
        [
            {"id": "same", "seed": 7, "num_bits": 1, "bit_plane": "all", "target": "attention", "layers": [], "repeat": 0},
            {"id": "same", "seed": 7, "num_bits": 1, "bit_plane": "all", "target": "attention", "layers": [], "repeat": 1},
        ]
    )

    assert not result["valid"]
    assert result["duplicate_ids"] == ["same"]
    assert result["duplicate_seeds"] == [7]


def test_condition_complete_rejects_inconsistent_fault_records(tmp_path: Path) -> None:
    import json

    from scripts.reliability.run_v15_sensitivity import _condition_complete

    directory = tmp_path / "condition"
    (directory / "comparison").mkdir(parents=True)
    (directory / "fault_injection_summary.json").write_text(
        json.dumps({"schema_version": "2.0", "condition_id": "c1", "planned_bit_flips": 2, "actual_bit_flips": 1, "records": []}),
        encoding="utf-8",
    )
    (directory / "comparison" / "comparison.json").write_text(json.dumps({"overall": {}}), encoding="utf-8")

    assert not _condition_complete(directory, {"id": "c1", "num_bits": 2})
