from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sat_rs_vlm.evaluation.extended_metrics import (
    caption_scores,
    cider_d_single_reference_approx_scores,
    generalized_box_iou,
    normalized_center_distance,
    text_task_scores,
)
from sat_rs_vlm.evaluation.runner import run_evaluation

ROOT = Path(__file__).resolve().parents[3]
CONTRACT = ROOT / "configs" / "eval" / "evaluation_contract_v1.5.yaml"


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class ExtendedMetricUnitTests(unittest.TestCase):
    def test_exact_caption_has_maximal_local_overlap_scores(self) -> None:
        scores = caption_scores("a new building appears", "a new building appears")
        for key in (
            "bleu_1_approx",
            "bleu_2_approx",
            "bleu_3_approx",
            "bleu_4_approx",
            "rouge_l_f1_approx",
            "chrf_approx",
        ):
            self.assertAlmostEqual(float(scores[key]), 1.0)
        self.assertGreater(float(scores["meteor_exact_approx"]), 0.99)
        cider = cider_d_single_reference_approx_scores(
            ["a new building appears"], ["a new building appears"]
        )
        self.assertAlmostEqual(cider[0], 10.0)

    def test_text_partial_credit_and_edit_similarity(self) -> None:
        scores = text_task_scores("large airport", "airport")
        self.assertFalse(scores["normalized_exact_match"])
        self.assertGreater(float(scores["token_f1"]), 0.0)
        self.assertLess(float(scores["token_f1"]), 1.0)
        self.assertGreater(float(scores["normalized_edit_similarity"]), 0.0)

    def test_grounding_geometry_diagnostics(self) -> None:
        box = [0.1, 0.2, 0.4, 0.5]
        self.assertAlmostEqual(generalized_box_iou(box, box), 1.0)
        self.assertAlmostEqual(normalized_center_distance(box, box), 0.0)
        self.assertLess(generalized_box_iou(box, [0.7, 0.7, 0.9, 0.9]), 0.0)


class ExtendedMetricEndToEndTests(unittest.TestCase):
    def test_summary_contains_extended_metrics_and_gap_statuses(self) -> None:
        rows = [
            {
                "id": "caption",
                "task_type": "captioning",
                "prediction": "a ship is visible",
                "reference": "a ship is visible",
                "metadata": {
                    "dataset": "VRSBench",
                    "source_task": "caption",
                },
            },
            {
                "id": "vqa",
                "task_type": "vqa",
                "prediction": "ship",
                "reference": "a ship",
                "metadata": {"dataset": "VRSBench", "qa_type": "object category"},
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = root / "predictions.jsonl"
            write_jsonl(predictions, rows)
            outputs = run_evaluation(
                predictions,
                root / "results",
                contract_path=CONTRACT,
                protected_repository=root / "protected",
                semantic_enabled=False,
            )
            summary = json.loads(outputs["summary"].read_text(encoding="utf-8"))
            caption_metrics = summary["by_task"]["captioning"]["metrics"]
            self.assertIn("bleu_2_approx", caption_metrics)
            self.assertIn("meteor_exact_approx", caption_metrics)
            self.assertIn("cider_d_single_reference_approx", caption_metrics)
            self.assertEqual(
                caption_metrics["corpus_bleu_4_single_reference_approx"]["value"],
                1.0,
            )
            text_metrics = summary["by_task"]["vqa"]["metrics"]
            self.assertIn("token_f1", text_metrics)
            self.assertEqual(
                summary["p0_data_availability"]["vqa_question_coverage"]["status"],
                "data_insufficient",
            )

    def test_levir_agreement_metrics(self) -> None:
        rows = [
            {
                "id": "tn",
                "task_type": "change_detection",
                "prediction": "unchanged",
                "reference": "unchanged",
                "metadata": {"dataset": "LEVIR-CC", "changeflag": 0},
            },
            {
                "id": "tp",
                "task_type": "change_detection",
                "prediction": "a building appeared",
                "reference": "a building appeared",
                "metadata": {"dataset": "LEVIR-CC", "changeflag": 1},
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = root / "predictions.jsonl"
            write_jsonl(predictions, rows)
            outputs = run_evaluation(
                predictions,
                root / "results",
                contract_path=CONTRACT,
                protected_repository=root / "protected",
                semantic_enabled=False,
            )
            summary = json.loads(outputs["summary"].read_text(encoding="utf-8"))
            metrics = summary["by_task"]["change_detection"]["metrics"]
            self.assertEqual(metrics["matthews_correlation_coefficient"]["value"], 1.0)
            self.assertEqual(metrics["cohen_kappa"]["value"], 1.0)
            self.assertEqual(metrics["negative_predictive_value"]["value"], 1.0)


if __name__ == "__main__":
    unittest.main()
