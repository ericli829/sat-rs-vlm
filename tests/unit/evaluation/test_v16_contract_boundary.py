from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sat_rs_vlm.evaluation.runner import run_evaluation

ROOT = Path(__file__).resolve().parents[3]
V15_CONTRACT = ROOT / "configs" / "eval" / "evaluation_contract_v1.5.yaml"
V16_CONTRACT = ROOT / "configs" / "eval" / "evaluation_contract_v1.6.yaml"


def write_prediction(path: Path) -> None:
    row = {
        "id": "contract-boundary",
        "task_type": "change_detection",
        "prediction": "A new building appeared.",
        "prediction_changeflag": 0,
        "binary_prediction": "0",
        "reference": "no change has occurred .",
        "metadata": {"dataset": "LEVIR-CC", "changeflag": 0},
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


class ContractBoundaryTests(unittest.TestCase):
    def test_v15_stays_caption_only_and_v16_prioritizes_explicit_binary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = root / "predictions.jsonl"
            write_prediction(predictions)

            v15 = run_evaluation(
                predictions,
                root / "v15",
                contract_path=V15_CONTRACT,
                strict=True,
                protected_repository=root / "protected",
                semantic_enabled=False,
            )
            v16 = run_evaluation(
                predictions,
                root / "v16",
                contract_path=V16_CONTRACT,
                strict=True,
                protected_repository=root / "protected",
                semantic_enabled=False,
            )
            v15_row = json.loads(
                v15["evaluated_predictions"].read_text(encoding="utf-8").splitlines()[0]
            )
            v16_row = json.loads(
                v16["evaluated_predictions"].read_text(encoding="utf-8").splitlines()[0]
            )
            v15_summary = json.loads(v15["summary"].read_text(encoding="utf-8"))
            v16_summary = json.loads(v16["summary"].read_text(encoding="utf-8"))

            self.assertEqual(v15_row["predicted_changeflag"], 1)
            self.assertNotIn("binary_prediction_source", v15_row)
            self.assertEqual(v15_summary["schema_version"], "1.5")
            self.assertNotIn("change_decision_version", v15_summary)
            self.assertNotIn(
                "explicit_binary_decision_rate",
                v15_summary["by_protocol"]["levir_cc_change_caption"]["metrics"],
            )

            self.assertEqual(v16_row["predicted_changeflag"], 0)
            self.assertEqual(v16_row["binary_prediction_source"], "explicit_changeflag")
            self.assertEqual(v16_summary["schema_version"], "1.6")
            self.assertEqual(
                v16_summary["change_decision_version"],
                "explicit_binary_priority_v1",
            )
            self.assertEqual(
                v16_summary["by_protocol"]["levir_cc_change_caption"]["metrics"][
                    "explicit_binary_decision_rate"
                ]["value"],
                1.0,
            )


if __name__ == "__main__":
    unittest.main()
