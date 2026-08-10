from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from sat_rs_vlm.evaluation.plotting import (
    PlottingError,
    _paired_improvement,
    _prepare_matplotlib,
    load_evaluations,
    parse_named_path,
    plot_evaluation_results,
)


def metric(value: float | int, *, samples: int = 10) -> dict[str, object]:
    return {
        "value": value,
        "label": "internal",
        "status": "ok",
        "num_samples": samples,
        "note": None,
    }


def vrs_summary(scale: float = 1.0) -> dict[str, object]:
    return {
        "contract_version": "1.5",
        "overall": {
            "metrics": {
                "latency_ms_mean": metric(10.0 * scale),
                "latency_ms_p50": metric(8.0 * scale),
                "latency_ms_p95": metric(15.0 * scale),
            },
            "latency_context": {
                "semantics": "batch_amortized_per_sample",
                "eval_batch_size": 16,
                "group_by_task": True,
                "status": "resolved",
            },
            "task_distribution": {
                "captioning": 10,
                "counting": 10,
                "detection": 10,
                "scene_classification": 10,
                "vqa": 10,
            },
        },
        "by_task": {
            "detection": {
                "metrics": {
                    "continuous_mean_iou": metric(0.60 * scale),
                    "continuous_mean_generalized_iou": metric(0.50 * scale),
                    "continuous_acc_at_0_5": metric(0.70 * scale),
                    "continuous_acc_at_0_7": metric(0.55 * scale),
                }
            },
            "counting": {
                "metrics": {
                    "exact_count_accuracy": metric(0.60 * scale),
                    "accuracy_within_1": metric(0.80 * scale),
                    "mae_on_parsed": metric(0.7 / scale),
                    "rmse_on_parsed": metric(1.0 / scale),
                }
            },
            "vqa": {
                "metrics": {
                    "micro_normalized_accuracy": metric(0.70 * scale),
                    "token_f1": metric(0.75 * scale),
                }
            },
            "scene_classification": {
                "metrics": {
                    "micro_normalized_accuracy": metric(0.80 * scale),
                    "token_f1": metric(0.82 * scale),
                }
            },
            "captioning": {
                "metrics": {
                    "bleu_1_approx": metric(0.50 * scale),
                    "bleu_4_approx": metric(0.10 * scale),
                    "rouge_l_f1_approx": metric(0.40 * scale),
                    "meteor_exact_approx": metric(0.35 * scale),
                    "chrf_approx": metric(0.45 * scale),
                    "cider_d_single_reference_approx": metric(0.52 * scale),
                    "length_ratio": metric(1.05),
                }
            },
        },
        "by_qa_type": {
            "object quantity": {
                "metrics": {"micro_normalized_accuracy": metric(0.65 * scale, samples=7)}
            },
            "reasoning": {
                "metrics": {"micro_normalized_accuracy": metric(0.55 * scale, samples=3)}
            },
        },
        "semantic": {
            "overall": {
                "metrics": {
                    "object_precision": metric(0.70 * scale),
                    "object_recall": metric(0.68 * scale),
                    "object_f1": metric(0.69 * scale),
                    "object_omission_rate": metric(0.22 / scale),
                    "reference_unsupported_object_rate": metric(0.18 / scale),
                    "count_consistency_accuracy": metric(0.40 * scale),
                    "spatial_relation_f1": metric(0.25 * scale),
                }
            }
        },
    }


def levir_summary() -> dict[str, object]:
    metrics = {
        "true_negatives": metric(8),
        "false_positives": metric(2),
        "false_negatives": metric(3),
        "true_positives": metric(7),
        "binary_accuracy": metric(0.75),
        "balanced_accuracy": metric(0.75),
        "change_precision": metric(7 / 9),
        "change_recall": metric(0.70),
        "change_f1": metric(0.7368),
        "matthews_correlation_coefficient": metric(0.51),
        "cohen_kappa": metric(0.50),
        "false_positive_rate": metric(0.20),
        "false_negative_rate": metric(0.30),
    }
    for name, value in {
        "bleu_1_approx": 0.50,
        "bleu_4_approx": 0.10,
        "rouge_l_f1_approx": 0.40,
        "meteor_exact_approx": 0.35,
        "chrf_approx": 0.45,
        "cider_d_single_reference_approx": 0.55,
    }.items():
        metrics[name] = metric(value)
        metrics[f"positive_change_{name}"] = metric(value - 0.05)
    return {
        "contract_version": "1.5",
        "overall": {
            "metrics": {},
            "latency_context": {
                "semantics": "batch_amortized_per_sample",
                "eval_batch_size": 8,
                "group_by_task": True,
                "status": "resolved",
            },
            "task_distribution": {"change_detection": 20},
        },
        "by_task": {"change_detection": {"metrics": metrics}},
        "by_qa_type": {},
        "semantic": {},
    }


def comparison_summary() -> dict[str, object]:
    metrics: dict[str, dict[str, object]] = {}
    task_metrics = {
        "detection": ("iou", "generalized_iou", "acc_at_0_5"),
        "counting": ("absolute_error", "exact_count_accuracy"),
        "vqa": ("normalized_accuracy",),
        "scene_classification": ("normalized_accuracy",),
        "captioning": (
            "rouge_l_f1_approx",
            "chrf_approx",
            "cider_d_single_reference_approx",
        ),
    }
    by_task: dict[str, object] = {}
    for task, names in task_metrics.items():
        metrics = {}
        for index, name in enumerate(names):
            improvement = 0.01 * (index + 1)
            metrics[name] = {
                "status": "ok",
                "num_samples": 10,
                "higher_is_better": name != "absolute_error",
                "improvement_mean": improvement,
                "improvement_ci95_paired_bootstrap": [improvement - 0.005, improvement + 0.005],
                "wins": 4,
                "ties": 3,
                "losses": 3,
            }
        by_task[task] = {"metrics": metrics}
    return {
        "required_contract_version": "1.5",
        "overall": {"num_paired_samples": 50},
        "by_task": by_task,
    }


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def write_rows(path: Path) -> None:
    rows = [
        {"id": "d0", "task_type": "detection", "sample_metrics": {"iou": 0.2}},
        {"id": "d1", "task_type": "detection", "sample_metrics": {"iou": 0.8}},
        {
            "id": "c0",
            "task_type": "counting",
            "sample_metrics": {"absolute_error": 0, "signed_error": 0},
        },
        {
            "id": "c1",
            "task_type": "counting",
            "sample_metrics": {"absolute_error": 2, "signed_error": -2},
        },
        {
            "id": "c2",
            "task_type": "counting",
            "sample_metrics": {"absolute_error": 1, "signed_error": 1},
        },
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PlotInputTests(unittest.TestCase):
    def test_named_path_and_duplicate_labels(self) -> None:
        label, path = parse_named_path("baseline=.")
        self.assertEqual(label, "baseline")
        self.assertTrue(path.is_absolute())
        with self.assertRaises(PlottingError):
            parse_named_path("missing-separator")
        with self.assertRaises(PlottingError):
            load_evaluations(("same=.", "same=."))

    def test_contract_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation = root / "evaluation"
            evaluation.mkdir()
            payload = vrs_summary()
            payload["contract_version"] = "1.4"
            write_json(evaluation / "summary.json", payload)
            with self.assertRaisesRegex(PlottingError, "expected 1.5"):
                load_evaluations((f"old={evaluation}",))


class PlotPipelineTests(unittest.TestCase):
    def _prepare_inputs(self, root: Path) -> tuple[list[str], list[str]]:
        baseline = root / "baseline"
        replay = root / "replay"
        levir = root / "levir"
        comparison = root / "comparison"
        for directory in (baseline, replay, levir, comparison):
            directory.mkdir()
        write_json(baseline / "summary.json", vrs_summary())
        write_json(replay / "summary.json", vrs_summary(1.02))
        write_json(levir / "summary.json", levir_summary())
        write_json(comparison / "comparison_summary.json", comparison_summary())
        write_rows(baseline / "evaluated_predictions.jsonl")
        write_rows(replay / "evaluated_predictions.jsonl")
        return (
            [f"baseline={baseline}", f"replay={replay}", f"levir={levir}"],
            [f"vrsbench={comparison}"],
        )

    def test_complete_plot_run_and_hash_preservation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluations, comparisons = self._prepare_inputs(root)
            baseline_summary = root / "baseline" / "summary.json"
            before = sha256(baseline_summary)
            outputs = plot_evaluation_results(
                evaluations,
                comparisons,
                root / "plots",
                formats=("png",),
            )
            self.assertEqual(before, sha256(baseline_summary))
            generated = outputs["generated"]
            self.assertEqual(len(generated), 13)
            for names in generated.values():
                self.assertGreater((root / "plots" / names[0]).stat().st_size, 100)
            manifest = json.loads((root / "plots" / "plot_manifest.json").read_text())
            self.assertEqual(manifest["contract_version"], "1.5")
            self.assertFalse(manifest["remote_write_performed"])
            self.assertTrue(any("omitted 1" in item["reason"] for item in manifest["skipped"]))

    def test_missing_rows_skips_only_distribution_figures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation = root / "evaluation"
            evaluation.mkdir()
            write_json(evaluation / "summary.json", vrs_summary())
            outputs = plot_evaluation_results(
                [f"model={evaluation}"],
                [],
                root / "plots",
                formats=("png",),
            )
            self.assertNotIn("grounding_iou_cdf", outputs["generated"])
            self.assertNotIn("counting_error_distribution", outputs["generated"])
            self.assertIn("vrsbench_core_metrics", outputs["generated"])

    def test_non_empty_output_requires_explicit_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation = root / "evaluation"
            output = root / "plots"
            evaluation.mkdir()
            output.mkdir()
            (output / "keep.txt").write_text("keep", encoding="utf-8")
            write_json(evaluation / "summary.json", vrs_summary())
            with self.assertRaisesRegex(PlottingError, "not empty"):
                plot_evaluation_results([f"model={evaluation}"], [], output, formats=("png",))
            plot_evaluation_results(
                [f"model={evaluation}"],
                [],
                output,
                formats=("png",),
                overwrite=True,
            )
            self.assertEqual((output / "keep.txt").read_text(encoding="utf-8"), "keep")

    def test_ci_plot_uses_direction_normalized_improvement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "comparison"
            directory.mkdir()
            write_json(directory / "comparison_summary.json", comparison_summary())
            from sat_rs_vlm.evaluation.plotting import load_comparisons

            comparison = load_comparisons((f"paired={directory}",))[0]
            _, plt = _prepare_matplotlib()
            figure = _paired_improvement(plt, comparison)
            self.assertIsNotNone(figure)
            assert figure is not None
            widths = [patch.get_width() for patch in figure.axes[0].patches]
            self.assertTrue(widths)
            self.assertTrue(all(width > 0 for width in widths))
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
