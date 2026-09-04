from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from sat_rs_vlm.evaluation.extended_metrics import box_iou, latency_statistics, text_task_scores
from sat_rs_vlm.evaluation.parsers import extract_json_object, parse_count, parse_grounding
from sat_rs_vlm.evaluation.protocols import load_contract, resolve_protocol
from sat_rs_vlm.evaluation.records import EvaluationError, PredictionRecord
from sat_rs_vlm.evaluation.runner import run_evaluation, validate_output_directory
from sat_rs_vlm.evaluation.semantic.extractors import extract_semantic_facts, load_ontology
from sat_rs_vlm.evaluation.semantic.metrics import semantic_sample_metrics
from sat_rs_vlm.evaluation.semantic.runner import SemanticEvaluator

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONTRACT = PROJECT_ROOT / "configs" / "eval" / "evaluation_contract_v1.5.yaml"
FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "evaluation_v1_5" / "predictions.jsonl"
SEMANTIC_FIXTURE = (
    PROJECT_ROOT / "tests" / "fixtures" / "evaluation_v1_5" / "semantic_predictions.jsonl"
)
SEMANTIC_ONTOLOGY = PROJECT_ROOT / "configs" / "eval" / "semantic" / "remote_sensing_ontology.json"
SEMANTIC_CONTRACT = PROJECT_ROOT / "configs" / "eval" / "semantic" / "semantic_contract.json"


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ParserTests(unittest.TestCase):
    def test_json_extraction_modes(self) -> None:
        strict = extract_json_object('{"label":"ship","bbox":[0,0,1,1]}')
        fenced = extract_json_object('```json\n{"label":"ship","bbox":[0,0,1,1]}\n```')
        mixed = extract_json_object('answer: {"label":"ship","bbox":[0,0,1,1]} done')
        self.assertTrue(strict.strict_json)
        self.assertIsNotNone(fenced.payload)
        self.assertIsNotNone(mixed.payload)

    def test_grounding_current_and_legacy_schema(self) -> None:
        current = parse_grounding(
            '{"label":"ship","bbox":[0,0,1,1]}',
            coordinate_format="normalized_0_1",
        )
        legacy = parse_grounding(
            '{"labels":["ship"],"boxes":[[0,0,1,1]]}',
            coordinate_format="normalized_0_1",
        )
        self.assertTrue(current.parse_ok and current.coordinate_valid)
        self.assertEqual(current.bbox, legacy.bbox)

    def test_grounding_rejects_multi_target_legacy_schema(self) -> None:
        parsed = parse_grounding(
            '{"labels":["ship","boat"],"boxes":[[0,0,1,1],[0,0,0.5,0.5]]}',
            coordinate_format="normalized_0_1",
        )
        self.assertFalse(parsed.parse_ok)
        self.assertEqual(parsed.parse_error, "legacy_schema_must_have_one_target")

    def test_grounding_coordinate_validation(self) -> None:
        parsed = parse_grounding(
            '{"label":"ship","bbox":[10,20,80,90]}',
            coordinate_format="percent_0_100",
        )
        invalid = parse_grounding(
            '{"label":"ship","bbox":[2,0,3,1]}',
            coordinate_format="normalized_0_1",
        )
        unresolved = parse_grounding(
            '{"label":"ship","bbox":[0,0,1,1]}',
            coordinate_format=None,
        )
        self.assertEqual(parsed.bbox, (0.1, 0.2, 0.8, 0.9))
        self.assertFalse(invalid.coordinate_valid)
        self.assertEqual(unresolved.coordinate_error, "coordinate_format_unresolved")

    def test_counting_formats_and_failures(self) -> None:
        cases = {
            "2": 2,
            "2 vehicles": 2,
            "There are 2 vehicles": 2,
            "Two": 2,
            "No vehicles": 0,
            '{"count":2}': 2,
            "twenty-one buildings": 21,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(parse_count(text).value, expected)
        self.assertEqual(parse_count("one or two ships").reason, "ambiguous_multiple_counts")
        self.assertEqual(parse_count("-1").reason, "negative_count")
        self.assertIsNone(parse_count("2.5").value)
        self.assertEqual(parse_count("").reason, "empty_count")


class MetricAndProtocolTests(unittest.TestCase):
    def test_iou_threshold_examples(self) -> None:
        full = (0.0, 0.0, 1.0, 1.0)
        half = (0.0, 0.0, 1.0, 0.5)
        seven_tenths = (0.0, 0.0, 1.0, 0.7)
        self.assertAlmostEqual(box_iou(full, half), 0.5)
        self.assertAlmostEqual(box_iou(full, seven_tenths), 0.7)

    def test_text_normalization(self) -> None:
        scores = text_task_scores("Yes, there is!", "yes there is")
        self.assertFalse(scores["exact_match"])
        self.assertTrue(scores["normalized_exact_match"])

    def test_latency_nearest_rank(self) -> None:
        stats = latency_statistics([10, 20, 30, 40, 50])
        self.assertEqual(stats["p50"], 30)
        self.assertEqual(stats["p95"], 50)

    def test_vrsbench_protocol_routing(self) -> None:
        contract = load_contract(CONTRACT)
        cases = [
            (
                "detection",
                {"dataset": "VRSBench", "source_task": "referring"},
                "vrsbench_visual_grounding",
            ),
            ("counting", {"dataset": "VRSBench", "source_task": "vqa"}, "vrsbench_counting"),
            (
                "captioning",
                {"dataset": "VRSBench", "source_task": "caption"},
                "vrsbench_detailed_caption",
            ),
            ("vqa", {"dataset": "VRSBench", "source_task": "vqa"}, "vrsbench_open_vqa"),
        ]
        for task, metadata, expected in cases:
            record = PredictionRecord.from_mapping(
                {
                    "id": expected,
                    "task_type": task,
                    "prediction": "x",
                    "reference": "x",
                    "metadata": metadata,
                },
                1,
            )
            self.assertEqual(resolve_protocol(record, contract).name, expected)


class SemanticEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ontology = load_ontology(SEMANTIC_ONTOLOGY)

    def test_synonyms_counts_and_symmetric_relation(self) -> None:
        prediction = extract_semantic_facts(
            "Two vessels are near the port.",
            self.ontology,
        )
        reference = extract_semantic_facts(
            "There are two ships near the harbor.",
            self.ontology,
        )
        self.assertEqual(prediction.objects, ("harbor", "ship"))
        self.assertEqual(prediction.counts, (("ship", 2),))
        self.assertEqual(prediction.relations, (("harbor", "near", "ship"),))
        self.assertEqual(prediction, reference)

    def test_reference_unsupported_and_omitted_objects(self) -> None:
        prediction = extract_semantic_facts("An aircraft is near the harbor.", self.ontology)
        reference = extract_semantic_facts("A ship is near the harbor.", self.ontology)
        metrics = semantic_sample_metrics(prediction, reference)
        self.assertEqual(metrics["object_precision"], 0.5)
        self.assertEqual(metrics["object_recall"], 0.5)
        self.assertEqual(metrics["object_f1"], 0.5)
        self.assertEqual(metrics["reference_unsupported_object_rate"], 0.5)
        self.assertEqual(metrics["object_omission_rate"], 0.5)

    def test_change_events_and_directional_relation(self) -> None:
        facts = extract_semantic_facts(
            "A new building appeared north of the road.",
            self.ontology,
        )
        self.assertIn(("building", "appearance"), facts.changes)
        self.assertEqual(facts.relations, (("building", "north_of", "road"),))

    def test_relation_does_not_cross_sentence_boundary(self) -> None:
        facts = extract_semantic_facts(
            "A ship is near. The harbor is visible.",
            self.ontology,
        )
        self.assertEqual(facts.relations, ())

    def test_static_caption_does_not_score_change_events(self) -> None:
        output = {
            "task_type": "captioning",
            "prediction": "A new building is visible.",
            "reference": "A new building is visible.",
        }
        evaluator = SemanticEvaluator(SEMANTIC_CONTRACT, SEMANTIC_ONTOLOGY)
        evaluator.extend_output(output)
        self.assertEqual(output["semantic_prediction"]["changes"], [])
        self.assertEqual(output["semantic_reference"]["changes"], [])

    def test_chinese_aliases(self) -> None:
        facts = extract_semantic_facts("两艘船靠近港口。", self.ontology)
        self.assertEqual(facts.objects, ("harbor", "ship"))
        self.assertIn(("ship", 2), facts.counts)
        self.assertEqual(facts.relations, (("harbor", "near", "ship"),))

    def test_semantic_end_to_end_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = run_evaluation(
                SEMANTIC_FIXTURE,
                root / "results",
                contract_path=CONTRACT,
                protected_repository=root / "protected",
            )
            summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
            semantic = summary["semantic"]
            self.assertEqual(semantic["profile"], "semantic_reference_text_v1")
            overall = semantic["overall"]["metrics"]
            self.assertAlmostEqual(overall["object_f1"]["value"], 6 / 7)
            self.assertEqual(overall["count_consistency_accuracy"]["value"], 1.0)
            self.assertAlmostEqual(overall["spatial_relation_f1"]["value"], 2 / 3)
            self.assertEqual(overall["change_event_f1"]["value"], 0.5)
            rows = [
                json.loads(line)
                for line in paths["evaluated_predictions"].read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(all(row["semantic_profile"] for row in rows))
            self.assertEqual(
                rows[0]["semantic_reference_source"],
                "reference_text_rule_based",
            )

    def test_semantic_can_be_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = run_evaluation(
                SEMANTIC_FIXTURE,
                root / "results",
                contract_path=CONTRACT,
                protected_repository=root / "protected",
                semantic_enabled=False,
            )
            summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
            self.assertNotIn("semantic", summary)
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            self.assertFalse(manifest["semantic_evaluation_enabled"])


class EndToEndTests(unittest.TestCase):
    def test_resource_telemetry_is_attached_to_summary_and_manifest(self) -> None:
        resource_benchmark = {
            "scope": "main_evaluation_prediction_loop",
            "timing_ms": {"e2e": 123.0},
            "resources": {"peak_cpu_rss_mb": 456.0},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = run_evaluation(
                FIXTURE,
                root / "results",
                contract_path=CONTRACT,
                protected_repository=root / "protected",
                resource_benchmark=resource_benchmark,
            )

            summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
            attached = summary["p0_data_availability"]["resource_benchmark"]
            self.assertEqual(attached["status"], "ok")
            self.assertEqual(attached["value"], resource_benchmark)
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(manifest["resource_benchmark"], resource_benchmark)

    def test_complete_evaluation_and_denominators(self) -> None:
        input_hash = file_hash(FIXTURE)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "results"
            protected = root / "protected-repository"
            protected.mkdir()
            paths = run_evaluation(
                FIXTURE,
                output,
                contract_path=CONTRACT,
                strict=True,
                protected_repository=protected,
            )
            self.assertEqual(file_hash(FIXTURE), input_hash)
            self.assertEqual(
                set(paths),
                {"evaluated_predictions", "metrics", "summary", "manifest"},
            )
            self.assertEqual(
                paths["metrics"].read_bytes(),
                paths["summary"].read_bytes(),
            )
            summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
            grounding = summary["by_protocol"]["vrsbench_visual_grounding"]["metrics"]
            self.assertAlmostEqual(grounding["continuous_mean_iou"]["value"], 0.5)
            self.assertAlmostEqual(
                grounding["repository_mean_iou_on_valid"]["value"],
                0.75,
            )
            self.assertEqual(grounding["repository_valid_detection_rate"]["value"], 1.0)
            self.assertEqual(
                grounding["json_object_extraction_rate"]["value"],
                grounding["valid_json_rate"]["value"],
            )
            self.assertAlmostEqual(grounding["continuous_acc_at_0_5"]["value"], 2 / 3)
            self.assertAlmostEqual(grounding["continuous_acc_at_0_7"]["value"], 1 / 3)
            counting = summary["by_protocol"]["vrsbench_counting"]["metrics"]
            self.assertEqual(counting["number_parse_success_rate"]["value"], 0.5)
            self.assertEqual(counting["exact_count_accuracy"]["value"], 0.5)
            self.assertEqual(counting["mae_on_parsed"]["value"], 0.0)
            self.assertIn("segmentation_not_implemented", summary["unimplemented_protocols"])
            repository = summary["repository_compatibility"]
            self.assertEqual(repository["profile"], "repository_native_v2")
            self.assertEqual(repository["metrics_version"], "v2_task_metrics")
            self.assertEqual(
                repository["upstream_repository"]["commit"],
                "da9f6d93b3f848afed68403dad3e0ed26344a626",
            )
            self.assertAlmostEqual(repository["by_task"]["detection"]["mean_iou"], 0.75)
            self.assertEqual(
                repository["by_task"]["change_detection"]["normalized_exact_match"],
                0.0,
            )
            self.assertEqual(
                repository["by_task"]["change_detection"]["keyword_hit"],
                1.0,
            )
            self.assertIn("bleu_1_approx", summary["by_task"]["change_detection"]["metrics"])
            rows = [
                json.loads(line)
                for line in paths["evaluated_predictions"].read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(rows[1]["backend"], "bnb_int8")
            self.assertEqual(rows[2]["fault_case"], {"bit": 7})
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(manifest["input_sha256"], input_hash)
            self.assertFalse(manifest["remote_write_performed"])
            self.assertEqual(manifest["contract_version"], "1.5")
            self.assertEqual(manifest["latency_context"]["status"], "unresolved")

    def test_latest_repository_batch_latency_context(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = run_evaluation(
                FIXTURE,
                root / "results",
                contract_path=CONTRACT,
                protected_repository=root / "protected",
                latency_semantics="batch_amortized_per_sample",
                eval_batch_size=16,
                group_by_task=True,
            )
            summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
            context = summary["overall"]["latency_context"]
            self.assertEqual(context["semantics"], "batch_amortized_per_sample")
            self.assertEqual(context["eval_batch_size"], 16)
            self.assertTrue(context["group_by_task"])
            self.assertEqual(context["status"], "resolved")
            self.assertIn("not independent request latency", context["note"])
            self.assertEqual(
                summary["repository_compatibility"]["latency_context"],
                context,
            )

    def test_invalid_single_sample_latency_context_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(EvaluationError):
                run_evaluation(
                    FIXTURE,
                    root / "results",
                    contract_path=CONTRACT,
                    protected_repository=root / "protected",
                    latency_semantics="single_sample",
                    eval_batch_size=16,
                )

    def test_non_empty_output_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "results"
            output.mkdir()
            (output / "existing.txt").write_text("keep", encoding="utf-8")
            with self.assertRaises(EvaluationError):
                run_evaluation(
                    FIXTURE,
                    output,
                    contract_path=CONTRACT,
                    protected_repository=root / "protected",
                )
            self.assertEqual((output / "existing.txt").read_text(encoding="utf-8"), "keep")

    def test_output_inside_protected_repository_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            protected = Path(temporary) / "repo"
            protected.mkdir()
            with self.assertRaises(EvaluationError):
                validate_output_directory(protected / "reports" / "eval", protected)


if __name__ == "__main__":
    unittest.main()
