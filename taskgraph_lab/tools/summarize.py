from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize(
    raw: list[dict[str, Any]],
    valid: list[dict[str, Any]],
    repaired: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
) -> dict[str, Any]:
    validation_records = [*valid, *repaired, *rejected]
    validations = [
        item.get("validation", {})
        for item in raw
        if item.get("status") == "generated" and item.get("validation")
    ]
    repair_attempts = len(repaired) + sum(int(item.get("repair_count", 0)) > 0 for item in rejected)
    warnings: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    operators: Counter[str] = Counter()
    intents: Counter[str] = Counter()
    node_counts: Counter[str] = Counter()
    datasets: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    for record in validation_records:
        validation = record.get("validation", {})
        warnings.update(str(item.get("code")) for item in validation.get("warnings", []))
        errors.update(str(item.get("code")) for item in validation.get("errors", []))
        target = record.get("target") or {}
        if target:
            intents[str(target.get("intent", "OTHER"))] += 1
            nodes = target.get("nodes", [])
            node_counts[str(len(nodes))] += 1
            operators.update(str(node.get("op")) for node in nodes)
        metadata = record.get("metadata") or (record.get("sample") or {}).get("metadata") or {}
        datasets[str(metadata.get("dataset", "unknown"))] += 1
        categories[str(metadata.get("source_category", "unknown"))] += 1
    review_distribution = Counter(
        str(item.get("review", {}).get("verdict", item.get("status", "unknown")))
        for item in reviews
    )
    total = len(raw)
    generated = sum(item.get("status") == "generated" for item in raw)
    repair_classifications = Counter(
        str(item.get("repair_classification", "UNKNOWN"))
        for item in raw
        if item.get("status") == "generated"
    )
    return {
        "total": total,
        "generated": generated,
        "api_failed": sum(item.get("status") == "api_failed" for item in raw),
        "processing_failed": sum(item.get("status") == "processing_failed" for item in raw),
        "schema_valid_rate": _rate(
            sum(bool(v.get("schema_valid")) for v in validations), len(validations)
        ),
        "graph_valid_rate": _rate(
            sum(bool(v.get("graph_valid")) for v in validations), len(validations)
        ),
        "type_valid_rate": _rate(
            sum(bool(v.get("type_valid")) for v in validations), len(validations)
        ),
        "semantic_valid_rate": _rate(
            sum(bool(v.get("semantic_valid")) for v in validations), len(validations)
        ),
        "auto_normalized_count": sum(bool(v.get("normalized_fields")) for v in validations),
        "repair_classification_counts": dict(repair_classifications.most_common()),
        "repaired_count": len(repaired),
        "repair_success_rate": _rate(len(repaired), repair_attempts),
        "rejected_count": len(rejected),
        "warning_counts": dict(warnings.most_common()),
        "operator_frequency": dict(operators.most_common()),
        "intent_distribution": dict(intents.most_common()),
        "node_count_distribution": dict(sorted(node_counts.items(), key=lambda item: int(item[0]))),
        "dataset_distribution": dict(datasets.most_common()),
        "category_distribution": dict(categories.most_common()),
        "semantic_review_distribution": dict(review_distribution.most_common()),
        "top_validator_errors": dict(errors.most_common(20)),
        "top_warnings": dict(warnings.most_common(20)),
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# TaskGraph generation summary",
        "",
        f"- Total: {report['total']}",
        f"- Generated: {report['generated']}",
        f"- API failed: {report['api_failed']}",
        f"- Processing failed: {report['processing_failed']}",
        f"- Repaired: {report['repaired_count']}",
        f"- Rejected: {report['rejected_count']}",
        f"- Schema valid rate: {report['schema_valid_rate']}",
        f"- Graph valid rate: {report['graph_valid_rate']}",
        f"- Type valid rate: {report['type_valid_rate']}",
        f"- Semantic valid rate: {report['semantic_valid_rate']}",
        f"- Auto-normalized: {report['auto_normalized_count']}",
        f"- Repair success rate: {report['repair_success_rate']}",
        "",
    ]
    for key in (
        "operator_frequency",
        "intent_distribution",
        "node_count_distribution",
        "dataset_distribution",
        "category_distribution",
        "semantic_review_distribution",
        "repair_classification_counts",
        "top_validator_errors",
        "top_warnings",
    ):
        lines.extend([f"## {key.replace('_', ' ').title()}", ""])
        values = report[key]
        lines.extend([f"- {name}: {count}" for name, count in values.items()] or ["- None"])
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize TaskGraph generation outputs")
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--valid", type=Path)
    parser.add_argument("--repaired", type=Path)
    parser.add_argument("--rejected", type=Path)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(
        read_jsonl(args.raw),
        read_jsonl(args.valid),
        read_jsonl(args.repaired),
        read_jsonl(args.rejected),
        read_jsonl(args.reviews),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "summary.md").write_text(markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {"output_dir": str(args.output_dir.resolve()), **report}, ensure_ascii=False, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
