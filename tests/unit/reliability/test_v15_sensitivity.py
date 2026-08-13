from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_v15_conditions_expand_layered_targets_and_unique_seeds() -> None:
    from scripts.reliability.run_v15_sensitivity import build_conditions

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

    assert len(conditions) == 24
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
        (),
    }
