from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from taskgraph_lab.taskgraph.validator import validate_candidate


def _context(record: dict[str, Any]) -> tuple[Any, dict[str, Any], str, str, str]:
    sample = record.get("sample") or record.get("input") or {}
    candidate = record.get("target", record.get("candidate", record.get("candidate_text")))
    if candidate is None:
        candidate = record
    inputs = sample.get("inputs") or record.get("inputs") or {}
    question = str(sample.get("question", record.get("question", "")))
    question_type = str(sample.get("question_type", record.get("question_type", "FREE_FORM")))
    sample_id = str(record.get("sample_id", sample.get("sample_id", "unknown")))
    return candidate, inputs, question, question_type, sample_id


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate TaskGraph candidates in JSONL")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts = {"total": 0, "valid": 0, "invalid": 0}
    with (
        args.input.open("r", encoding="utf-8") as source,
        args.output.open("w", encoding="utf-8", newline="\n") as destination,
    ):
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"row {line_number} must be an object")
            candidate, inputs, question, question_type, sample_id = _context(record)
            _, validation = validate_candidate(
                candidate, inputs=inputs, question=question, question_type=question_type
            )
            counts["total"] += 1
            counts["valid" if validation.valid else "invalid"] += 1
            destination.write(
                json.dumps(
                    {"sample_id": sample_id, "validation": validation.model_dump(mode="json")},
                    ensure_ascii=False,
                )
                + "\n"
            )
            destination.flush()
    print(json.dumps({"output": str(args.output.resolve()), **counts}, ensure_ascii=False))
    return 0 if counts["invalid"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
