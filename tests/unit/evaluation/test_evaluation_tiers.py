from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from sat_rs_vlm.evaluation.config import EvaluationTierBuildConfig
from sat_rs_vlm.evaluation.tiers import (
    EvaluationTierError,
    build_evaluation_tiers,
    counting_bucket,
    tier_metadata,
)
from sat_rs_vlm.training.hard_example_mining import load_evaluation_ids

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def sample(
    sample_id: str,
    task: str,
    answer: str,
    *,
    dataset: str,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    values = {"dataset": dataset, "split": "val", **(metadata or {})}
    images = [f"images/{sample_id}.png"]
    if task == "change_detection":
        images.append(f"images/{sample_id}-after.png")
    content: list[dict[str, str]] = [
        {"type": "image", "image": image} for image in images
    ]
    content.append({"type": "text", "text": "question"})
    return {
        "id": sample_id,
        "task_type": task,
        "messages": [
            {"role": "user", "content": content},
            {"role": "assistant", "content": answer},
        ],
        "metadata": values,
    }


class EvaluationTierTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[EvaluationTierBuildConfig, set[str]]:
        vrs_rows = [
            sample(
                "det-small-ship",
                "detection",
                '{"label":"ship","bbox":[0,0,0.05,0.05]}',
                dataset="VRSBench",
            ),
            sample(
                "det-medium-vehicle",
                "detection",
                '{"label":"vehicle","bbox":[0,0,0.2,0.2]}',
                dataset="VRSBench",
            ),
            sample(
                "det-large-aircraft",
                "detection",
                '{"label":"aircraft","bbox":[0,0,0.8,0.8]}',
                dataset="VRSBench",
            ),
            sample("count-0", "counting", "0", dataset="VRSBench"),
            sample("count-6", "counting", "6", dataset="VRSBench"),
            sample("count-12", "counting", "12", dataset="VRSBench"),
            sample(
                "vqa-position",
                "vqa",
                "left",
                dataset="VRSBench",
                metadata={"qa_type": "object position"},
            ),
            sample(
                "scene-type",
                "scene_classification",
                "urban",
                dataset="VRSBench",
                metadata={"qa_type": "scene type"},
            ),
            sample("caption", "captioning", "A harbor.", dataset="VRSBench"),
        ]
        levir_rows = [
            sample(
                "change-0",
                "change_detection",
                "No change has occurred.",
                dataset="LEVIR-CC",
                metadata={"changeflag": 0},
            ),
            sample(
                "change-1",
                "change_detection",
                "A building appeared.",
                dataset="LEVIR-CC",
                metadata={"changeflag": 1},
            ),
        ]
        write_jsonl(root / "vrs_val.jsonl", vrs_rows)
        write_jsonl(root / "levir_val.jsonl", levir_rows)
        write_jsonl(root / "vrs_train.jsonl", [{"id": "train-vrs"}])
        write_jsonl(root / "levir_train.jsonl", [{"id": "train-levir"}])
        fixed = {"det-small-ship", "count-6", "vqa-position", "change-0"}
        (root / "fixed.txt").write_text("\n".join(sorted(fixed)) + "\n", encoding="utf-8")
        by_id = {row["id"]: row for row in [*vrs_rows, *levir_rows]}
        fixed_samples: list[dict[str, object]] = []
        for sample_id in sorted(fixed):
            row = by_id[sample_id]
            user = row["messages"][0]  # type: ignore[index]
            assistant = row["messages"][1]  # type: ignore[index]
            content = user["content"]  # type: ignore[index]
            fixed_samples.append(
                {
                    "id": sample_id,
                    "task_type": row["task_type"],
                    "images": [item["image"] for item in content if item["type"] == "image"],  # type: ignore[index]
                    "question": next(item["text"] for item in content if item["type"] == "text"),  # type: ignore[index]
                    "reference": assistant["content"],  # type: ignore[index]
                }
            )
        write_jsonl(root / "fixed_samples.jsonl", fixed_samples)
        config = EvaluationTierBuildConfig.model_validate(
            {
                "seed": 42,
                "sources": [
                    {
                        "name": "VRSBench",
                        "eval_file": "vrs_val.jsonl",
                        "train_file": "vrs_train.jsonl",
                        "image_prefix": "VRSBench",
                    },
                    {
                        "name": "LEVIR-CC",
                        "eval_file": "levir_val.jsonl",
                        "train_file": "levir_train.jsonl",
                        "image_prefix": "LEVIR-CC",
                    },
                ],
                "tiers": {
                    "E1": {"target_samples": 4},
                    "E2": {"target_samples": 8},
                    "E3": {"mode": "full"},
                },
                "existing_e1": {
                    "ids_file": "fixed.txt",
                    "samples_file": "fixed_samples.jsonl",
                    "required": True,
                    "origin": "test fixture",
                },
                "output": {"directory": "tiers-a"},
            }
        )
        return config, fixed

    def test_nested_reproducible_tiers_and_stratification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, fixed = self._fixture(root)
            first = build_evaluation_tiers(config, project_root=root)
            second_config = config.model_copy(
                update={"output": config.output.model_copy(update={"directory": "tiers-b"})}
            )
            second = build_evaluation_tiers(second_config, project_root=root)
            ids = {
                name: set(first["tiers"][name]["sample_ids"])
                for name in ("E1", "E2", "E3")
            }
            self.assertEqual(ids["E1"], fixed)
            self.assertLess(ids["E1"], ids["E2"])
            self.assertLess(ids["E2"], ids["E3"])
            self.assertEqual(len(ids["E3"]), 11)
            for name in ("E1", "E2", "E3"):
                self.assertEqual(
                    first["tiers"][name]["sha256"], second["tiers"][name]["sha256"]
                )
            e3 = first["tiers"]["E3"]
            self.assertEqual(
                e3["detection_size_distribution"], {"large": 1, "medium": 1, "small": 1}
            )
            self.assertEqual(e3["count_bucket_distribution"], {"0": 1, "10+": 1, "5-9": 1})
            self.assertEqual(
                e3["qa_type_distribution"], {"object position": 1, "scene type": 1}
            )
            self.assertEqual(e3["levir_changeflag_distribution"], {"0": 1, "1": 1})
            self.assertTrue(first["invariants"]["train_eval_disjoint"])
            manifest_path = root / "tiers-a" / "evaluation_tiers_manifest.json"
            self.assertEqual(load_evaluation_ids(manifest_path), ids["E3"])
            tier_path, tier_hash = tier_metadata("E2", manifest_path, project_root=root)
            self.assertEqual(tier_path.name, "e2_standard.jsonl")
            self.assertEqual(tier_hash, first["tiers"]["E2"]["sha256"])

    def test_train_eval_id_leakage_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _ = self._fixture(root)
            write_jsonl(root / "vrs_train.jsonl", [{"id": "caption"}])
            with self.assertRaisesRegex(EvaluationTierError, "leakage"):
                build_evaluation_tiers(config, project_root=root)

    def test_counting_bucket_boundaries(self) -> None:
        self.assertEqual(
            [counting_bucket(value) for value in range(5)], ["0", "1", "2", "3", "4"]
        )
        self.assertEqual(counting_bucket(5), "5-9")
        self.assertEqual(counting_bucket(9), "5-9")
        self.assertEqual(counting_bucket(10), "10+")

    def test_default_evaluation_config_points_to_e2_without_runtime_slice(self) -> None:
        payload = yaml.safe_load(
            (PROJECT_ROOT / "configs/eval/qwen3vl_eval.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(payload["evaluation"]["tier"], "E2")
        self.assertEqual(
            payload["data"]["eval_file"], "data/evaluation/tiers/e2_standard.jsonl"
        )
        self.assertIsNone(payload["data"]["max_eval_samples"])


if __name__ == "__main__":
    unittest.main()
