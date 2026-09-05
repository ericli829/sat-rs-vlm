"""Audit the coarse counting train population and exact-cardinality aux subset."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from sat_rs_vlm.data.object_adapter_v0 import (
    _cardinality_prompt_target,
    count_bin,
    extract_answer,
    extract_prompt,
)
from sat_rs_vlm.data.task_protocol import parse_count
from sat_rs_vlm.evaluation.counting_protocol import classify_counting_predictions
from sat_rs_vlm.models.reliability.checksum import file_sha256
from sat_rs_vlm.utils.jsonl import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", default="data/processed/rs_count_merger_v1/train.jsonl")
    parser.add_argument(
        "--train-manifest", default="data/processed/rs_count_merger_v1/manifest.json"
    )
    parser.add_argument("--output-dir", default="reports/rs_merger_expert")
    parser.add_argument("--max-count", type=int, default=15)
    return parser.parse_args()


def _subtype(question: str, answer: Any, *, exact: bool, valid: bool) -> str:
    normalized_question = question.strip().lower()
    normalized_answer = str(answer).strip().lower()
    if exact and valid:
        return "exact_cardinality_valid"
    if exact:
        if normalized_answer in {"multiple", "several", "many", "few"}:
            return "exact_cardinality_vague_quantifier_reference"
        if normalized_answer in {"yes", "no", "true", "false"}:
            return "exact_cardinality_binary_reference"
        return "exact_cardinality_invalid_reference"
    if normalized_question.startswith(("is ", "are ", "do ", "does ", "can ", "has ", "have ")):
        return "non_cardinality_binary_question"
    return "non_cardinality_other"


def build_audit(train_file: Path, manifest_path: Path, *, max_count: int) -> dict[str, Any]:
    rows = [dict(row) for row in read_jsonl(train_file)]
    protocol_rows = [
        {
            "id": str(row.get("id", "")),
            "task_type": str(row.get("task_type", "")),
            "question": extract_prompt(row),
            "reference": extract_answer(row),
            "prediction": extract_answer(row),
        }
        for row in rows
    ]
    classified = classify_counting_predictions(protocol_rows)
    valid_by_id = {str(row["id"]): int(row["parsed_reference"]) for row in classified["valid_rows"]}
    subtypes: Counter[str] = Counter()
    invalid_reasons: Counter[str] = Counter()
    invalid_examples: list[dict[str, str]] = []
    values: list[int] = []
    for row in rows:
        sample_id = str(row.get("id", ""))
        question = extract_prompt(row)
        answer = extract_answer(row)
        valid = sample_id in valid_by_id
        exact = _cardinality_prompt_target(question) is not None
        subtypes[_subtype(question, answer, exact=exact, valid=valid)] += 1
        parsed = parse_count(answer)
        if valid:
            values.append(valid_by_id[sample_id])
        elif exact:
            invalid_reasons[str(parsed.reason or "non_cardinality")] += 1
            if len(invalid_examples) < 50:
                invalid_examples.append(
                    {
                        "id": sample_id,
                        "question": question,
                        "reference": str(answer),
                        "reason": str(parsed.reason),
                    }
                )
    bins = Counter({"0-2": 0, "3-5": 0, "6-10": 0, "11+": 0})
    for value in values:
        bins[count_bin(value)] += 1
    clipped = Counter(min(value, max_count) for value in values)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "schema_version": "1.0",
        "train_file": str(train_file.resolve()),
        "train_file_sha256": file_sha256(train_file),
        "train_manifest": str(manifest_path.resolve()),
        "train_manifest_sha256": file_sha256(manifest_path),
        "source_train": manifest.get("source_train"),
        "source_manifest": manifest.get("source_manifest"),
        "raw_counting_rows_lm_supervision": len(rows),
        "exact_cardinality_aux_rows": len(values),
        "aux_excluded_rows": len(rows) - len(values),
        "formal_protocol_diagnostics": classified["diagnostics"],
        "count_statistics": {
            "minimum": min(values),
            "maximum": max(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
        },
        "count_bins": dict(bins),
        "question_subtypes": dict(sorted(subtypes.items())),
        "invalid_reference_reasons": dict(sorted(invalid_reasons.items())),
        "invalid_reference_examples": invalid_examples,
        "auxiliary_class_contract": {
            "K": max_count,
            "classes": list(range(max_count + 1)),
            "tail_policy": "clip_to_K",
            "tail_rows": sum(value > max_count for value in values),
            "class_frequency_after_tail_policy": {
                str(index): clipped[index] for index in range(max_count + 1)
            },
            "non_eligible_policy": "LM CE retained; auxiliary target=-100",
        },
    }


def _markdown(audit: dict[str, Any]) -> str:
    stats = audit["count_statistics"]
    contract = audit["auxiliary_class_contract"]
    lines = [
        "# Counting training population audit",
        "",
        f"- raw counting rows (LM CE): {audit['raw_counting_rows_lm_supervision']}",
        f"- exact-cardinality aux rows: {audit['exact_cardinality_aux_rows']}",
        f"- aux-excluded rows: {audit['aux_excluded_rows']}",
        (
            f"- count min/max/mean/median: {stats['minimum']} / {stats['maximum']} / "
            f"{stats['mean']:.4f} / {stats['median']}"
        ),
        f"- K: {contract['K']}",
        f"- tail policy: {contract['tail_policy']} ({contract['tail_rows']} rows)",
        "",
        "## Count bins",
        "",
    ]
    lines.extend(f"- {name}: {value}" for name, value in audit["count_bins"].items())
    lines.extend(["", "## Question/reference subtypes", ""])
    lines.extend(f"- {name}: {value}" for name, value in audit["question_subtypes"].items())
    lines.extend(["", "## Invalid reference reasons", ""])
    lines.extend(f"- {name}: {value}" for name, value in audit["invalid_reference_reasons"].items())
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    audit = build_audit(Path(args.train_file), Path(args.train_manifest), max_count=args.max_count)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "count_train_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "count_train_audit.md").write_text(_markdown(audit), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
