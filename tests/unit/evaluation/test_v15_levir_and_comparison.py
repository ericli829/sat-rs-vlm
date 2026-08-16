from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sat_rs_vlm.evaluation.comparison import ComparisonError, compare_evaluations
from sat_rs_vlm.evaluation.parsers import (
    parse_change_prediction,
    parse_explicit_change_prediction,
)
from sat_rs_vlm.evaluation.protocols import _protocol_name
from sat_rs_vlm.evaluation.records import InputValidationError, PredictionRecord
from sat_rs_vlm.evaluation.runner import run_evaluation

ROOT = Path(__file__).resolve().parents[3]
CONTRACT = ROOT / "configs" / "eval" / "evaluation_contract_v1.6.yaml"


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class LevirParserAndRoutingTests(unittest.TestCase):
    def test_explicit_binary_parser_does_not_guess_from_caption(self) -> None:
        for expression, expected in (
            ("0", 0),
            ("Answer: 1", 1),
            ('{"changeflag": 0}', 0),
            ("unchanged", 0),
        ):
            with self.subTest(expression=expression):
                self.assertEqual(parse_explicit_change_prediction(expression).value, expected)
        self.assertIsNone(
            parse_explicit_change_prediction(
                "The two scenes appear visually similar with no obvious differences."
            ).value
        )

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

    def test_relaxed_no_change_variants_and_binary_outputs(self) -> None:
        expressions = (
            "No changes were observed between the two images.",
            "No visible differences can be seen between the images.",
            "There were no significant changes.",
            "No changes detected.",
            "Both images appear unchanged.",
            "The two images look the same.",
            "The scene has not changed.",
            "Answer: no obvious change was detected.",
        )
        for expression in expressions:
            with self.subTest(expression=expression):
                result = parse_change_prediction(expression)
                self.assertEqual(result.value, 0)
                self.assertIn(result.match_type, {"exact_no_change", "pattern_no_change"})

        binary_cases = {
            "0": 0,
            "1": 1,
            '{"changeflag": 0}': 0,
            '{"change_flag": "1"}': 1,
            '{"changed": false}': 0,
            '{"has_change": true}': 1,
        }
        for expression, expected in binary_cases.items():
            with self.subTest(expression=expression):
                self.assertEqual(parse_change_prediction(expression).value, expected)

        composite = parse_change_prediction(
            "The second image is identical to the first image. "
            "There are no visible changes between the two images."
        )
        self.assertEqual(composite.value, 0)
        self.assertEqual(composite.match_type, "composite_no_change")

        contextual = parse_change_prediction(
            "The images depict a dense forest with varying shades of green. "
            "The second image is a zoomed-in view of the same area. "
            "There are no visible changes in the forest's appearance between the two images."
        )
        self.assertEqual(contextual.value, 0)
        self.assertEqual(contextual.match_type, "contextual_no_change")

    def test_partial_no_change_statements_remain_positive(self) -> None:
        expressions = (
            "No building changed, but a road appeared.",
            "No change in buildings; however vegetation was removed.",
            "No major change except that a new house was built.",
            "The images are not identical.",
            "There is no building change while a road is newly constructed.",
            (
                "The second image shows a change in the landscape. A new road appeared. "
                "The surrounding forest remains unchanged."
            ),
        )
        for expression in expressions:
            with self.subTest(expression=expression):
                result = parse_change_prediction(expression)
                self.assertEqual(result.value, 1)
                self.assertEqual(result.match_type, "default_change")

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
    def test_confusion_matrix_and_positive_description_metrics(self) -> None:
        rows = [
            {
                "id": "tn",
                "task_type": "change_detection",
                "prediction": "No changes were observed between the two images.",
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
            self.assertEqual(
                evaluated[0]["change_parser_version"],
                "levir_contextual_no_change_v3",
            )
            self.assertEqual(evaluated[0]["change_parse_mode"], "pattern_no_change")
            self.assertTrue(evaluated[0]["binary_correct"])
            self.assertEqual(metrics["pattern_no_change_match_rate"]["value"], 0.25)
            self.assertEqual(metrics["caption_fallback_decision_rate"]["value"], 1.0)
            self.assertEqual(metrics["explicit_binary_decision_rate"]["value"], 0.0)
            self.assertEqual(
                summary["change_parser_version"],
                "levir_contextual_no_change_v3",
            )

    def test_explicit_binary_fields_override_caption_fallback(self) -> None:
        rows = [
            {
                "id": "explicit-flag",
                "task_type": "change_detection",
                "prediction": "A new building appeared.",
                "prediction_changeflag": 0,
                "binary_prediction": "0",
                "reference": "no change has occurred .",
                "metadata": {"dataset": "LEVIR-CC", "changeflag": 0},
            },
            {
                "id": "explicit-text",
                "task_type": "change_detection",
                "prediction": "The scene is unchanged.",
                "prediction_changeflag": None,
                "binary_prediction": "Answer: 1",
                "reference": "a building appeared .",
                "metadata": {"dataset": "LEVIR-CC", "changeflag": 1},
            },
            {
                "id": "legacy",
                "task_type": "change_detection",
                "prediction": "No changes were observed between the images.",
                "reference": "no change has occurred .",
                "metadata": {"dataset": "LEVIR-CC", "changeflag": 0},
            },
            {
                "id": "invalid-explicit",
                "task_type": "change_detection",
                "prediction": "No changes were observed between the images.",
                "prediction_changeflag": "0",
                "reference": "no change has occurred .",
                "metadata": {"dataset": "LEVIR-CC", "changeflag": 0},
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
            evaluated = [
                json.loads(line)
                for line in outputs["evaluated_predictions"]
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(evaluated[0]["predicted_changeflag"], 0)
            self.assertEqual(evaluated[0]["binary_prediction_source"], "explicit_changeflag")
            self.assertEqual(evaluated[1]["predicted_changeflag"], 1)
            self.assertEqual(
                evaluated[1]["binary_prediction_source"], "binary_prediction_text"
            )
            self.assertEqual(evaluated[2]["binary_prediction_source"], "caption_fallback")
            self.assertIsNone(evaluated[3]["predicted_changeflag"])
            self.assertEqual(
                evaluated[3]["binary_prediction_source"],
                "invalid_explicit_changeflag",
            )
            self.assertEqual(evaluated[3]["parse_error"], "invalid_prediction_changeflag")
            metrics = json.loads(outputs["summary"].read_text(encoding="utf-8"))[
                "by_protocol"
            ]["levir_cc_change_caption"]["metrics"]
            self.assertAlmostEqual(metrics["explicit_binary_decision_rate"]["value"], 0.5)
            self.assertAlmostEqual(metrics["caption_fallback_decision_rate"]["value"], 0.25)

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
        *,
        contract_version: str = "1.5",
    ) -> None:
        directory.mkdir()
        write_json(
            directory / "evaluation_manifest.json",
            {"contract_version": contract_version},
        )
        write_json(directory / "summary.json", {"contract_version": contract_version})
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

    def test_v16_comparison_and_cross_contract_rejection(self) -> None:
        rows = [
            {
                "id": "change",
                "task_type": "change_detection",
                "prediction": "No change.",
                "reference": "No change.",
                "metadata": {"dataset": "LEVIR-CC", "changeflag": 0},
                "sample_metrics": {
                    "binary_correct": True,
                    "bleu_1_approx": 1.0,
                    "bleu_4_approx": 1.0,
                    "rouge_l_f1_approx": 1.0,
                    "meteor_exact_approx": 1.0,
                    "chrf_approx": 1.0,
                    "cider_d_single_reference_approx": 1.0,
                },
            }
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline"
            candidate = root / "candidate"
            self._make_evaluation(baseline, rows, contract_version="1.6")
            self._make_evaluation(candidate, rows, contract_version="1.6")
            outputs = compare_evaluations(
                baseline,
                candidate,
                root / "comparison",
                protected_repository=root / "protected",
                bootstrap_resamples=5,
            )
            summary = json.loads(outputs["comparison_summary"].read_text(encoding="utf-8"))
            self.assertEqual(summary["required_contract_version"], "1.6")
            self.assertEqual(
                summary["by_task"]["change_detection"]["primary_metric"],
                "binary_accuracy",
            )

            cross_candidate = root / "cross-candidate"
            self._make_evaluation(cross_candidate, rows, contract_version="1.5")
            with self.assertRaisesRegex(ComparisonError, "same contract_version"):
                compare_evaluations(
                    baseline,
                    cross_candidate,
                    root / "cross-comparison",
                    protected_repository=root / "protected",
                    bootstrap_resamples=5,
                )

    def test_v17_change_detection_comparison_is_supported(self) -> None:
        rows = [
            {
                "id": "change-v17",
                "task_type": "change_detection",
                "prediction": "No change.",
                "reference": "No change.",
                "metadata": {"dataset": "LEVIR-CC", "changeflag": 0},
                "sample_metrics": {
                    "binary_correct": True,
                    "bleu_1_approx": 1.0,
                    "bleu_4_approx": 1.0,
                    "rouge_l_f1_approx": 1.0,
                    "meteor_exact_approx": 1.0,
                    "chrf_approx": 1.0,
                    "cider_d_single_reference_approx": 1.0,
                },
            }
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline"
            candidate = root / "candidate"
            self._make_evaluation(baseline, rows, contract_version="1.7")
            self._make_evaluation(candidate, rows, contract_version="1.7")
            outputs = compare_evaluations(
                baseline,
                candidate,
                root / "comparison",
                protected_repository=root / "protected",
                bootstrap_resamples=5,
            )
            summary = json.loads(outputs["comparison_summary"].read_text(encoding="utf-8"))
            self.assertEqual(summary["required_contract_version"], "1.7")
            self.assertEqual(
                summary["by_task"]["change_detection"]["primary_metric"],
                "binary_accuracy",
            )


if __name__ == "__main__":
    unittest.main()
