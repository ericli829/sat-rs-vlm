from __future__ import annotations

import json
from pathlib import Path

from sat_rs_vlm.evaluation.runner import run_evaluation

ROOT = Path(__file__).resolve().parents[3]
CONTRACT = ROOT / "configs" / "eval" / "evaluation_contract_v1.5.yaml"


def test_vrsbench_official_grounding_stratification(tmp_path: Path) -> None:
    rows = []
    for sample_id, is_unique, prediction in (
        ("unique", True, [0.0, 0.0, 1.0, 1.0]),
        ("non-unique", False, [0.0, 0.0, 0.1, 0.1]),
    ):
        rows.append(
            {
                "id": sample_id,
                "task_type": "detection",
                "prediction": json.dumps({"label": "ship", "bbox": prediction}),
                "reference": json.dumps(
                    {"label": "ship", "bbox": [0.0, 0.0, 1.0, 1.0]}
                ),
                "metadata": {
                    "dataset": "VRSBench",
                    "source_task": "referring",
                    "bbox_target_format": "normalized_0_1",
                    "is_unique": is_unique,
                },
            }
        )
    source = tmp_path / "predictions.jsonl"
    source.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    outputs = run_evaluation(
        source,
        tmp_path / "results",
        contract_path=CONTRACT,
        protected_repository=tmp_path / "protected",
        semantic_enabled=False,
    )
    summary = json.loads(outputs["summary"].read_text(encoding="utf-8"))
    metrics = summary["by_protocol"]["vrsbench_visual_grounding"]["metrics"]
    assert metrics["official_acc_at_0_5_unique"]["value"] == 1.0
    assert metrics["official_acc_at_0_5_non_unique"]["value"] == 0.0
    assert metrics["official_acc_at_0_5_all"]["value"] == 0.5
    assert metrics["official_acc_at_0_7_all"]["value"] == 0.5
