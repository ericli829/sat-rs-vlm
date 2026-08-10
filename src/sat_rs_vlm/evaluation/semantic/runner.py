"""语义抽取、逐样本扩展字段和微平均汇总。"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

from sat_rs_vlm.evaluation.extended_metrics import metric_value
from sat_rs_vlm.evaluation.semantic.extractors import extract_semantic_facts, load_ontology
from sat_rs_vlm.evaluation.semantic.metrics import semantic_sample_metrics


class SemanticEvaluator:
    def __init__(self, contract_path: Path, ontology_path: Path) -> None:
        self.contract_path = contract_path.resolve()
        self.ontology_path = ontology_path.resolve()
        try:
            contract = json.loads(self.contract_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid semantic contract: {exc}") from exc
        if not isinstance(contract, dict) or not str(contract.get("profile", "")).strip():
            raise ValueError("semantic contract must contain a profile")
        tasks = contract.get("applicable_task_types")
        if not isinstance(tasks, list) or not all(isinstance(item, str) for item in tasks):
            raise ValueError("semantic contract applicable_task_types must be a string list")
        self.contract = contract
        self.profile = str(contract["profile"])
        self.applicable_tasks = {item.strip().lower() for item in tasks}
        self.ontology = load_ontology(self.ontology_path)

    def applies_to(self, output: dict[str, Any]) -> bool:
        return str(output.get("task_type", "")).strip().lower() in self.applicable_tasks

    def extend_output(self, output: dict[str, Any]) -> None:
        if not self.applies_to(output):
            return
        task_type = str(output.get("task_type", "")).strip().lower()
        prediction = extract_semantic_facts(str(output.get("prediction", "")), self.ontology)
        reference = extract_semantic_facts(str(output.get("reference", "")), self.ontology)
        if task_type != "change_detection":
            prediction = replace(prediction, changes=())
            reference = replace(reference, changes=())
        output.update(
            {
                "semantic_profile": self.profile,
                "semantic_reference_source": str(self.contract["reference_basis"]),
                "semantic_prediction": prediction.to_dict(),
                "semantic_reference": reference.to_dict(),
                "semantic_metrics": semantic_sample_metrics(prediction, reference),
            }
        )

    @staticmethod
    def _micro_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        sample_metrics = [row["semantic_metrics"] for row in rows]

        def total(name: str) -> int:
            return sum(int(sample.get(name) or 0) for sample in sample_metrics)

        def ratio(numerator: int, denominator: int) -> float | None:
            return numerator / denominator if denominator else None

        def prf(prefix: str) -> tuple[float | None, float | None, float | None, int, int]:
            tp = total(f"{prefix}_tp")
            fp = total(f"{prefix}_fp")
            fn = total(f"{prefix}_fn")
            predicted = tp + fp
            reference = tp + fn
            precision = ratio(tp, predicted)
            recall = ratio(tp, reference)
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision is not None and recall is not None and precision + recall
                else (0.0 if predicted or reference else None)
            )
            return precision, recall, f1, predicted, reference

        object_precision, object_recall, object_f1, object_predicted, object_reference = prf(
            "object"
        )
        relation_precision, relation_recall, relation_f1, relation_predicted, relation_reference = (
            prf("spatial_relation")
        )
        change_precision, change_recall, change_f1, change_predicted, change_reference = prf(
            "change_event"
        )
        count_correct = total("count_correct")
        count_reference = total("count_reference_facts")

        def packaged(
            value: float | None,
            denominator: int,
            note: str | None = None,
        ) -> dict[str, Any]:
            return metric_value(
                value,
                num_samples=denominator,
                status="ok" if value is not None else "not_available",
                note=note,
            )

        return {
            "status": "ok",
            "metrics": {
                "num_semantic_samples": metric_value(len(rows), num_samples=len(rows)),
                "object_precision": packaged(object_precision, object_predicted),
                "object_recall": packaged(object_recall, object_reference),
                "object_f1": packaged(object_f1, object_reference),
                "reference_unsupported_object_rate": packaged(
                    ratio(total("object_fp"), object_predicted),
                    object_predicted,
                    "Reference-text unsupported mentions; not image-grounded hallucination.",
                ),
                "object_omission_rate": packaged(
                    ratio(total("object_fn"), object_reference),
                    object_reference,
                    "Omissions relative to reference text, not exhaustive image annotations.",
                ),
                "count_consistency_accuracy": packaged(
                    ratio(count_correct, count_reference), count_reference
                ),
                "spatial_relation_precision": packaged(relation_precision, relation_predicted),
                "spatial_relation_recall": packaged(relation_recall, relation_reference),
                "spatial_relation_f1": packaged(relation_f1, relation_reference),
                "spatial_relation_accuracy": packaged(
                    ratio(total("spatial_relation_tp"), relation_reference),
                    relation_reference,
                    "Reference relation coverage; equal to micro relation recall.",
                ),
                "change_event_precision": packaged(change_precision, change_predicted),
                "change_event_recall": packaged(change_recall, change_reference),
                "change_event_f1": packaged(change_f1, change_reference),
            },
        }

    def build_summary(self, outputs: list[dict[str, Any]]) -> dict[str, Any]:
        rows = [row for row in outputs if row.get("semantic_profile") == self.profile]
        by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_task[str(row.get("task_type", "unknown"))].append(row)
        return {
            "profile": self.profile,
            "contract_version": str(self.contract.get("contract_version", "unknown")),
            "status": str(self.contract.get("status", "unknown")),
            "reference_basis": str(self.contract.get("reference_basis", "unknown")),
            "ontology_version": str(self.ontology.get("ontology_version", "unknown")),
            "overall": (
                self._micro_summary(rows) if rows else {"status": "not_available", "metrics": {}}
            ),
            "by_task": {
                task: self._micro_summary(task_rows) for task, task_rows in sorted(by_task.items())
            },
            "registered_not_implemented": list(self.contract.get("registered_not_implemented", [])),
        }
