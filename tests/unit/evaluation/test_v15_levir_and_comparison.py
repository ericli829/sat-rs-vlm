from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sat_rs_vlm.evaluation.comparison import ComparisonError, compare_evaluations
from sat_rs_vlm.evaluation.parsers import parse_change_prediction
from sat_rs_vlm.evaluation.protocols import _protocol_name
from sat_rs_vlm.evaluation.records import InputValidationError, PredictionRecord
from sat_rs_vlm.evaluation.runner import run_evaluation

ROOT = Path(__file__).resolve().parents[3]
CONTRACT = ROOT / "configs" / "eval" / "evaluation_contract_v1.5.yaml"


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class LevirParserAndRoutingTests(unittest.TestCase):
    def test_no_change_expressions_and_compound_change(self) -> None:
        expressions = (
            "No change has occurred.",
            "No change occurred!",
            "No changes have occurred.",
            "There is no change.",
            "There are no changes.",
            "There is no difference.",
            "No difference.",
            "The two scenes seem identical.",
            "The two scenes are identical.",
            "The scene is the same as before.",
            "The scene remains the same as before.",
            "Almost nothing has changed.",
            "Nothing has changed.",
            "Unchanged.",
        )
        for expression in expressions:
            with self.subTest(expression=expression):
                self.assertEqual(parse_change_prediction(expression).value, 0)
        self.assertEqual(
            parse_change_prediction("No building changed, but a road appeared.").value,
            1,
        )
        self.assertIsNone(parse_change_prediction("  ").value)

    def test_levir_dataset_spelling_routes_to_change_protocol(self) -> None:
        for dataset in ("LEVIR-CC", "levir-cc", "LEVIR_CC", "levircc"):
            with self.subTest(dataset=dataset):
                record = PredictionRecord.from_mapping(
                    {
                        "id": dataset,
                        "task_type": "change_detection",
                        "prediction": "unchanged",
                        "reference": "no change has occurred",
                        "metadata": {"dataset": dataset, "changeflag": 0},
                    },
                    1,
                )
                self.assertEqual(_protocol_name(record)[0], "levir_cc_change_caption")


class LevirEndToEndTests(unittest.TestCase):
    def test_disabled_visual_semantic_skips_gold_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = root / "predictions.jsonl"
            write_jsonl(
                predictions,
                [
                    {
                        "id": "sample-1",
                        "task_type": "change_detection",
                        "prediction": "unchanged",
                        "reference": "no change has occurred",
                        "metadata": {"dataset": "LEVIR-CC", "changeflag": 0},
                    }
                ],
            )
            with patch("sat_rs_vlm.evaluation.runner.evaluate_visual_semantics") as evaluate:
                outputs = run_evaluation(
                    predictions,
                    root / "results",
                    contract_path=CONTRACT,
                    strict=True,
                    protected_repository=root / "protected",
                    semantic_enabled=False,
                    visual_semantic_enabled=False,
                    visual_semantic_gold_path=root / "missing-gold.csv",
                )

            evaluate.assert_not_called()
            manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
            self.assertFalse(manifest["visual_semantic_evaluation_enabled"])

    def test_visual_semantic_profile_runs_through_unified_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = root / "predictions.jsonl"
            write_jsonl(
                predictions,
                [
                    {
                        "id": "sample-1",
                        "task_type": "change_detection",
                        "prediction": "A new building was constructed.",
                        "reference": "A building appeared.",
                        "metadata": {"dataset": "LEVIR-CC", "changeflag": 1},
                    }
                ],
            )
            gold = root / "visual_semantic_gold.csv"
            with gold.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "sample_id",
                        "image_t1_path",
                        "image_t2_path",
                        "gold_change_label",
                        "gold_changed_objects",
                        "gold_change_directions",
                        "gold_change_events",
                        "annotation_confidence",
                        "label_source",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "sample_id": "sample-1",
                        "image_t1_path": "missing-t1.png",
                        "image_t2_path": "missing-t2.png",
                        "gold_change_label": "1",
                        "gold_changed_objects": "building",
                        "gold_change_directions": "appearance_construction",
                        "gold_change_events": "building:appearance_construction",
                        "annotation_confidence": "high",
                        "label_source": "image_audited_annotation",
                    }
                )
            outputs = run_evaluation(
                predictions,
                root / "results",
                contract_path=CONTRACT,
                strict=True,
                protected_repository=root / "protected",
                semantic_enabled=False,
                visual_semantic_enabled=True,
                visual_semantic_gold_path=gold,
                visual_semantic_verify_image_paths=False,
            )

            summary = json.loads(outputs["summary"].read_text(encoding="utf-8"))
            auxiliary = summary["visual_semantic_auxiliary"]
            assert auxiliary["metric_label"] == "research_auxiliary_visual_semantic_profile"
            assert auxiliary["official_benchmark_metric"] is False
            assert auxiliary["gold_source"] == "image_audited_annotation"
            assert auxiliary["metrics"]["binary"]["accuracy"] == 1.0
            assert outputs["visual_semantic_summary"].is_file()
            manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
            assert manifest["visual_semantic_evaluation_enabled"] is True
            assert "visual_semantic_summary" in manifest["visual_semantic_output_files"]

    def test_confusion_matrix_and_positive_description_metrics(self) -> None:
        rows = [
            {
                "id": "tn",
                "task_type": "change_detection",
                "prediction": "no change has occurred .",
                "reference": "no change has occurred .",
                "metadata": {"dataset": "LEVIR-CC", "changeflag": 0},
            },
            {
                "id": "fp",
                "task_type": "change_detection",
                "prediction": "a building appears .",
                "reference": "there is no difference .",
                "metadata": {"dataset": "LEVIR-CC", "changeflag": 0},
            },
            {
                "id": "fn",
                "task_type": "change_detection",
                "prediction": "the scene is the same as before .",
                "reference": "a road appears .",
                "metadata": {"dataset": "LEVIR-CC", "changeflag": 1},
            },
            {
                "id": "tp",
                "task_type": "change_detection",
                "prediction": "no building changed, but a road appeared .",
                "reference": "a road appeared .",
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
                strict=True,
                protected_repository=root / "protected",
                semantic_enabled=False,
            )
            summary = json.loads(outputs["summary"].read_text(encoding="utf-8"))
            metrics = summary["by_protocol"]["levir_cc_change_caption"]["metrics"]
            self.assertEqual(metrics["true_positives"]["value"], 1)
            self.assertEqual(metrics["true_negatives"]["value"], 1)
            self.assertEqual(metrics["false_positives"]["value"], 1)
            self.assertEqual(metrics["false_negatives"]["value"], 1)
            for name in (
                "binary_accuracy",
                "balanced_accuracy",
                "change_precision",
                "change_recall",
                "change_f1",
                "false_positive_rate",
                "false_negative_rate",
            ):
                self.assertEqual(metrics[name]["value"], 0.5)
            self.assertEqual(metrics["positive_change_bleu_1_approx"]["num_samples"], 2)
            evaluated = [
                json.loads(line)
                for line in outputs["evaluated_predictions"]
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(evaluated[0]["reference_changeflag"], 0)
            self.assertEqual(evaluated[0]["predicted_changeflag"], 0)
            self.assertTrue(evaluated[0]["binary_correct"])

    def test_invalid_changeflag_fails_in_strict_mode(self) -> None:
        invalid_values = (None, True, -1, 2, "1")
        for index, value in enumerate(invalid_values):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                predictions = root / "predictions.jsonl"
                metadata = {"dataset": "LEVIR-CC"}
                if value is not None:
                    metadata["changeflag"] = value
                write_jsonl(
                    predictions,
                    [
                        {
                            "id": f"invalid-{index}",
                            "task_type": "change_detection",
                            "prediction": "unchanged",
                            "reference": "unchanged",
                            "metadata": metadata,
                        }
                    ],
                )
                with self.assertRaises(InputValidationError):
                    run_evaluation(
                        predictions,
                        root / "results",
                        contract_path=CONTRACT,
                        strict=True,
                        protected_repository=root / "protected",
                        semantic_enabled=False,
                    )


class ComparisonTests(unittest.TestCase):
    def _make_evaluation(
        self,
        directory: Path,
        rows: list[dict[str, object]],
    ) -> None:
        directory.mkdir()
        write_json(directory / "evaluation_manifest.json", {"contract_version": "1.5"})
        write_json(directory / "summary.json", {"contract_version": "1.5"})
        write_jsonl(directory / "evaluated_predictions.jsonl", rows)

    def _rows(self, candidate: bool) -> list[dict[str, object]]:
        def row(
            sample_id: str,
            task: str,
            prediction: str,
            metrics: dict[str, object],
        ) -> dict[str, object]:
            return {
                "id": sample_id,
                "task_type": task,
                "prediction": prediction,
                "reference": "same reference",
                "metadata": {"dataset": "VRSBench"},
                "sample_metrics": metrics,
            }

        return [
            row(
                "det",
                "detection",
                "new" if candidate else "old",
                {
                    "iou": 0.6 if candidate else 0.4,
                    "correct_at_0_5": candidate,
                    "correct_at_0_7": False,
                    "label_and_iou_correct_at_0_5": candidate,
                },
            ),
            row(
                "count",
                "counting",
                "1" if candidate else "2",
                {
                    "absolute_error": 1 if candidate else 2,
                    "exact_count_correct": False,
                    "within_1_correct": candidate,
                },
            ),
            row(
                "vqa",
                "vqa",
                "yes" if candidate else "no",
                {"exact_match": candidate, "normalized_exact_match": candidate},
            ),
            row(
                "caption",
                "captioning",
                "better" if candidate else "worse",
                {
                    "bleu_1_approx": 0.4 if candidate else 0.2,
                    "bleu_4_approx": 0.2 if candidate else 0.1,
                    "rouge_l_f1_approx": 0.5 if candidate else 0.25,
                },
            ),
        ]

    def test_complete_comparison_is_deterministic_and_directional(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline"
            candidate = root / "candidate"
            self._make_evaluation(baseline, self._rows(False))
            self._make_evaluation(candidate, self._rows(True))
            first = compare_evaluations(
                baseline,
                candidate,
                root / "comparison-1",
                protected_repository=root / "protected",
                bootstrap_resamples=20,
                seed=7,
            )
            second = compare_evaluations(
                baseline,
                candidate,
                root / "comparison-2",
                protected_repository=root / "protected",
                bootstrap_resamples=20,
                seed=7,
            )
            first_summary = json.loads(first["comparison_summary"].read_text(encoding="utf-8"))
            second_summary = json.loads(second["comparison_summary"].read_text(encoding="utf-8"))
            self.assertEqual(first_summary, second_summary)
            self.assertEqual(first_summary["overall"]["improvements"], 4)
            self.assertEqual(first_summary["overall"]["regressions"], 0)
            self.assertGreater(
                first_summary["by_task"]["detection"]["metrics"]["iou"]["improvement_mean"],
                0,
            )
            reversed_outputs = compare_evaluations(
                candidate,
                baseline,
                root / "comparison-reversed",
                protected_repository=root / "protected",
                bootstrap_resamples=20,
                seed=7,
            )
            reversed_summary = json.loads(
                reversed_outputs["comparison_summary"].read_text(encoding="utf-8")
            )
            self.assertLess(
                reversed_summary["by_task"]["detection"]["metrics"]["iou"]["improvement_mean"],
                0,
            )
            self.assertEqual(reversed_summary["overall"]["regressions"], 4)

    def test_mismatched_reference_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline"
            candidate = root / "candidate"
            baseline_rows = self._rows(False)
            candidate_rows = self._rows(True)
            candidate_rows[0]["reference"] = "different"
            self._make_evaluation(baseline, baseline_rows)
            self._make_evaluation(candidate, candidate_rows)
            with self.assertRaises(ComparisonError):
                compare_evaluations(
                    baseline,
                    candidate,
                    root / "comparison",
                    protected_repository=root / "protected",
                    bootstrap_resamples=5,
                )

    def test_legacy_and_unified_tier_versions_cannot_be_paired(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline"
            candidate = root / "candidate"
            self._make_evaluation(baseline, self._rows(False))
            self._make_evaluation(candidate, self._rows(True))
            write_json(
                baseline / "evaluation_manifest.json",
                {
                    "contract_version": "1.5",
                    "evaluation_tier": "E2",
                    "evaluation_tier_version": "legacy-vrs-v1",
                    "evaluation_tier_sha256": "legacy-sha",
                },
            )
            write_json(
                candidate / "evaluation_manifest.json",
                {
                    "contract_version": "1.5",
                    "evaluation_tier": "E2",
                    "evaluation_tier_version": "unified-v2",
                    "evaluation_tier_sha256": "unified-sha",
                },
            )

            with self.assertRaisesRegex(ComparisonError, "tier versions"):
                compare_evaluations(
                    baseline,
                    candidate,
                    root / "comparison",
                    protected_repository=root / "protected",
                    bootstrap_resamples=5,
                )


if __name__ == "__main__":
    unittest.main()
