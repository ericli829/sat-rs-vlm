"""Measure local binary-judge errors by adjudicated LEVIR caption semantics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-csv", type=Path, required=True)
    parser.add_argument("--judge-results-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _read_rows(path: Path) -> list[dict[str, str]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        raw_rows = payload.get("rows", []) if isinstance(payload, dict) else []
        if not isinstance(raw_rows, list):
            raise ValueError("judge result JSON must contain a rows list")
        return [
            {str(key): str(value) for key, value in row.items()}
            for row in raw_rows
            if isinstance(row, dict)
        ]
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _judge_value(raw: str) -> int | None:
    value = raw.strip()
    return int(value) if value in {"0", "1"} else None


def _binary_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = tn = fp = fn = unresolved = 0
    for row in rows:
        prediction = row["judge"]
        expected = int(row["human_change_label"])
        if prediction is None:
            unresolved += 1
        elif expected == 1 and prediction == 1:
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
        "num_samples": total,
        "num_resolved": resolved,
        "unresolved_rate": unresolved / total if total else None,
        "accuracy_on_resolved": (tp + tn) / resolved if resolved else None,
        "balanced_accuracy_on_resolved": (
            (recall_0 + recall_1) / 2 if recall_0 is not None and recall_1 is not None else None
        ),
        "change_recall": recall_1,
        "no_change_recall": recall_0,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def build_slice_report(
    gold_rows: list[dict[str, str]], judge_rows: list[dict[str, str]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    judges = {
        row.get("audit_id", "").strip(): _judge_value(row.get("local_judge_decision", ""))
        for row in judge_rows
    }
    gold_ids = {row.get("audit_id", "").strip() for row in gold_rows}
    if not gold_ids or gold_ids != set(judges):
        raise ValueError("gold and judge-result audit ID sets must match exactly")
    rows: list[dict[str, Any]] = []
    for item in gold_rows:
        label = item["human_change_label"].strip()
        if label not in {"0", "1"}:
            continue
        rows.append({**item, "judge": judges[item["audit_id"].strip()]})
    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_direction: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["human_change_label"] != "1":
            continue
        for value in row["human_changed_objects"].split("|"):
            by_object[value].append(row)
        for value in row["human_change_directions"].split("|"):
            by_direction[value].append(row)
    report = {
        "schema_version": "1.0",
        "protocol": "levir_cc_caption_semantics_v1",
        "evaluation_scope": "caption_semantic_binary_judge_slices",
        "limitation": (
            "Slices evaluate 0/1 caption-meaning decisions, not image-grounded "
            "object/direction extraction."
        ),
        "overall": _binary_summary(rows),
        "by_changed_object": {
            key: _binary_summary(value) for key, value in sorted(by_object.items())
        },
        "by_change_direction": {
            key: _binary_summary(value) for key, value in sorted(by_direction.items())
        },
    }
    hard_cases = [
        {
            "audit_id": row["audit_id"],
            "caption": row["caption"],
            "gold_label": row["human_change_label"],
            "gold_objects": row["human_changed_objects"],
            "gold_directions": row["human_change_directions"],
            "judge_prediction": row["judge"],
            "error_type": "false_negative" if row["judge"] == 0 else "unresolved",
        }
        for row in rows
        if row["human_change_label"] == "1" and row["judge"] != 1
    ]
    return report, hard_cases


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"output directory must be empty or absent: {output_dir}")
    try:
        report, hard_cases = build_slice_report(
            _read_rows(args.gold_csv.resolve()), _read_rows(args.judge_results_csv.resolve())
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "semantic_slice_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fields = [
        "audit_id",
        "caption",
        "gold_label",
        "gold_objects",
        "gold_directions",
        "judge_prediction",
        "error_type",
    ]
    with (output_dir / "local_judge_semantic_hard_cases.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(hard_cases)
    print(f"Saved semantic slice report to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
