"""Summarize human LEVIR caption audits and compare automatic judges."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotator-a", type=Path, required=True)
    parser.add_argument(
        "--annotator-b",
        type=Path,
        help="Second independent audit. Omit for a preliminary single-annotator report.",
    )
    parser.add_argument("--answer-key", type=Path, required=True)
    parser.add_argument("--adjudicated", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load_rows(path: Path) -> dict[str, dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
    else:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        rows = payload.get("rows", [])
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        audit_id = str(row.get("audit_id", ""))
        if not audit_id or audit_id in result:
            raise ValueError(f"missing or duplicate audit_id in {path}: {audit_id!r}")
        result[audit_id] = row
    return result


def _cohen_kappa(a: list[str], b: list[str]) -> float | None:
    if not a:
        return None
    observed = sum(left == right for left, right in zip(a, b, strict=True)) / len(a)
    labels = {"0", "1", "U"}
    counts_a, counts_b = Counter(a), Counter(b)
    expected = sum(counts_a[label] * counts_b[label] for label in labels) / len(a) ** 2
    return (observed - expected) / (1 - expected) if not math.isclose(expected, 1.0) else 1.0


def _binary_metrics(
    gold: dict[str, str], key: dict[str, dict[str, Any]], field: str
) -> dict[str, Any]:
    tp = tn = fp = fn = unresolved = 0
    for audit_id, label in gold.items():
        if label not in {"0", "1"}:
            continue
        prediction = key[audit_id].get(field)
        if prediction not in {0, 1}:
            unresolved += 1
            continue
        expected = int(label)
        if expected == 1 and prediction == 1:
            tp += 1
        elif expected == 0 and prediction == 0:
            tn += 1
        elif expected == 0:
            fp += 1
        else:
            fn += 1
    resolved = tp + tn + fp + fn
    total = resolved + unresolved
    recall_0 = tn / (tn + fp) if tn + fp else None
    recall_1 = tp / (tp + fn) if tp + fn else None
    return {
        "num_human_binary_labels": total,
        "coverage": resolved / total if total else None,
        "accuracy_on_resolved": (tp + tn) / resolved if resolved else None,
        "balanced_accuracy_on_resolved": (
            (recall_0 + recall_1) / 2 if recall_0 is not None and recall_1 is not None else None
        ),
        "change_precision": tp / (tp + fp) if tp + fp else None,
        "change_recall": recall_1,
        "change_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "unresolved": unresolved,
    }


def _paired_judge_metrics(
    gold: dict[str, str], answer_key: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    both_correct = old_only = local_only = both_wrong = unresolved = 0
    for audit_id, label in gold.items():
        if label not in {"0", "1"}:
            continue
        expected = int(label)
        old = answer_key[audit_id].get("old_parser_decision")
        local = answer_key[audit_id].get("local_judge_decision")
        if old not in {0, 1} or local not in {0, 1}:
            unresolved += 1
            continue
        old_correct = old == expected
        local_correct = local == expected
        if old_correct and local_correct:
            both_correct += 1
        elif old_correct:
            old_only += 1
        elif local_correct:
            local_only += 1
        else:
            both_wrong += 1
    comparable = both_correct + old_only + local_only + both_wrong
    return {
        "num_comparable": comparable,
        "both_correct": both_correct,
        "old_only_correct": old_only,
        "local_only_correct": local_only,
        "both_wrong": both_wrong,
        "unresolved": unresolved,
        "local_minus_old_accuracy": ((local_only - old_only) / comparable if comparable else None),
        "note": "The audit sample is risk-stratified, so this is a diagnostic paired comparison.",
    }


def evaluate_single_annotator(
    annotator: dict[str, dict[str, Any]],
    answer_key: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Produce a preliminary report when only one human audit is available."""

    if set(annotator) != set(answer_key):
        raise ValueError("annotator and answer-key audit ID sets must match exactly")
    gold: dict[str, str] = {}
    incomplete: list[str] = []
    for audit_id, row in annotator.items():
        label = str(row.get("human_caption_semantic_label") or "")
        if label in {"0", "1", "U"}:
            gold[audit_id] = label
        else:
            incomplete.append(audit_id)
    summary = {
        "schema_version": "1.0",
        "protocol": "levir_cc_permanent_structure_change_v1",
        "audit_mode": "single_annotator_preliminary",
        "human_reference_status": "preliminary_not_adjudicated",
        "num_rows": len(annotator),
        "num_singly_labeled": len(gold),
        "num_incomplete": len(incomplete),
        "incomplete_ids": sorted(incomplete),
        "raw_agreement": None,
        "cohen_kappa": None,
        "num_disagreements": None,
        "num_gold_labels": len(gold),
        "num_unadjudicated_disagreements": None,
        "old_contextual_parser": _binary_metrics(gold, answer_key, "old_parser_decision"),
        "local_small_llm_judge": _binary_metrics(gold, answer_key, "local_judge_decision"),
        "paired_judge_comparison": _paired_judge_metrics(gold, answer_key),
        "interpretation_note": (
            "Preliminary comparison against one human annotator. A second independent audit "
            "and adjudication are required before this becomes a human gold standard."
        ),
    }
    return summary, []


def evaluate_audit(
    annotator_a: dict[str, dict[str, Any]],
    annotator_b: dict[str, dict[str, Any]],
    answer_key: dict[str, dict[str, Any]],
    adjudicated: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if set(annotator_a) != set(annotator_b) or set(annotator_a) != set(answer_key):
        raise ValueError("annotator and answer-key audit ID sets must match exactly")
    labels_a: list[str] = []
    labels_b: list[str] = []
    disagreements: list[dict[str, Any]] = []
    gold: dict[str, str] = {}
    incomplete: list[str] = []
    for audit_id in sorted(annotator_a):
        left = str(annotator_a[audit_id].get("human_caption_semantic_label") or "")
        right = str(annotator_b[audit_id].get("human_caption_semantic_label") or "")
        if left not in {"0", "1", "U"} or right not in {"0", "1", "U"}:
            incomplete.append(audit_id)
            continue
        labels_a.append(left)
        labels_b.append(right)
        if left == right:
            gold[audit_id] = left
            continue
        final = None
        if adjudicated and audit_id in adjudicated:
            candidate = str(adjudicated[audit_id].get("human_caption_semantic_label") or "")
            if candidate in {"0", "1", "U"}:
                final = candidate
                gold[audit_id] = candidate
        disagreements.append(
            {
                "audit_id": audit_id,
                "caption": annotator_a[audit_id].get("caption"),
                "annotator_a": left,
                "annotator_b": right,
                "adjudicated": final,
            }
        )
    summary = {
        "schema_version": "1.0",
        "protocol": "levir_cc_permanent_structure_change_v1",
        "num_rows": len(annotator_a),
        "num_doubly_labeled": len(labels_a),
        "num_incomplete": len(incomplete),
        "incomplete_ids": incomplete,
        "raw_agreement": (
            sum(a == b for a, b in zip(labels_a, labels_b, strict=True)) / len(labels_a)
            if labels_a
            else None
        ),
        "cohen_kappa": _cohen_kappa(labels_a, labels_b),
        "num_disagreements": len(disagreements),
        "num_gold_labels": len(gold),
        "num_unadjudicated_disagreements": sum(
            item["adjudicated"] is None for item in disagreements
        ),
        "old_contextual_parser": _binary_metrics(gold, answer_key, "old_parser_decision"),
        "local_small_llm_judge": _binary_metrics(gold, answer_key, "local_judge_decision"),
        "paired_judge_comparison": _paired_judge_metrics(gold, answer_key),
        "interpretation_note": (
            "These metrics compare each text judge with blind human labels of caption meaning; "
            "they do not measure agreement with image-level changeflag."
        ),
    }
    return summary, disagreements


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"output directory must be empty or absent: {output_dir}")
    a = _load_rows(args.annotator_a.resolve())
    key = _load_rows(args.answer_key.resolve())
    try:
        if args.annotator_b:
            b = _load_rows(args.annotator_b.resolve())
            adjudicated = _load_rows(args.adjudicated.resolve()) if args.adjudicated else None
            summary, disagreements = evaluate_audit(a, b, key, adjudicated)
        else:
            if args.adjudicated:
                raise ValueError("--adjudicated requires --annotator-b")
            summary, disagreements = evaluate_single_annotator(a, key)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "audit_disagreements.jsonl").open("w", encoding="utf-8") as file:
        for row in disagreements:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved audit summary: {output_dir / 'audit_summary.json'}")
    if summary["num_unadjudicated_disagreements"] is not None:
        print(
            "Human disagreements requiring adjudication: "
            f"{summary['num_unadjudicated_disagreements']}"
        )
    else:
        print("Preliminary single-annotator report; second audit still required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
