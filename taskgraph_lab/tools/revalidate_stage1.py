from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from taskgraph_lab.datasets.base import NormalizedSample
from taskgraph_lab.quality.answer_audit import (
    audit_choice_answer,
    final_answer_type,
    load_answer_index,
)
from taskgraph_lab.taskgraph.validator import validate_candidate


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _accepted_graphs(run_dir: Path) -> dict[str, dict[str, Any]]:
    graphs: dict[str, dict[str, Any]] = {}
    for name in ("valid.jsonl", "repaired.jsonl"):
        for record in _read_jsonl(run_dir / name):
            graph = record.get("accepted_taskgraph")
            if isinstance(graph, dict):
                graphs[str(record["sample_id"])] = graph
    return graphs


def _run_records(run_dir: Path) -> dict[str, dict[str, Any]]:
    accepted = _accepted_graphs(run_dir)
    records: dict[str, dict[str, Any]] = {}
    for raw in _read_jsonl(run_dir / "raw.jsonl"):
        sample_id = str(raw["sample_id"])
        teacher_item = raw.get("repair_teacher_raw_item") or raw.get("teacher_raw_item") or {}
        graph = accepted.get(sample_id) or teacher_item.get("taskgraph")
        records[sample_id] = {
            "sample": raw.get("sample"),
            "taskgraph": graph,
            "provider_trace": raw.get("provider_trace"),
        }
    return records


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def revalidate(
    *,
    base_run_dir: Path,
    output_dir: Path,
    answer_index: dict[str, dict[str, Any]],
    override_run_dirs: list[Path] | None = None,
) -> dict[str, Any]:
    base = _run_records(base_run_dir)
    override_ids: set[str] = set()
    for override_dir in override_run_dirs or []:
        for sample_id, record in _run_records(override_dir).items():
            if sample_id not in base:
                raise ValueError(f"override sample is absent from base run: {sample_id}")
            base[sample_id] = record
            override_ids.add(sample_id)

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    buckets: dict[str, list[dict[str, Any]]] = {
        "accepted": [],
        "runtime_rejected": [],
        "answer_rejected": [],
        "answer_unknown": [],
    }
    runtime_errors: Counter[str] = Counter()
    answer_reasons: Counter[str] = Counter()
    for sample_id, source in base.items():
        sample = NormalizedSample.model_validate(source["sample"])
        graph = source.get("taskgraph")
        _, runtime = validate_candidate(
            graph or {},
            inputs=sample.inputs,
            question=sample.question,
            question_type=sample.question_type.value,
        )
        runtime_errors.update(issue.code for issue in runtime.errors)
        answer_source = answer_index.get(sample_id)
        if answer_source is None:
            answer_audit = {
                "status": "unknown",
                "valid": None,
                "reason": "answer source is missing",
            }
        else:
            answer_audit = audit_choice_answer(
                answer=answer_source["answer"],
                choices=answer_source["choices"],
                final_answer_type=final_answer_type(graph),
            )
            answer_audit["source"] = answer_source["source"]
        if answer_audit.get("reason"):
            answer_reasons[str(answer_audit["reason"])] += 1

        if not runtime.valid:
            bucket = "runtime_rejected"
        elif answer_audit["status"] == "invalid":
            bucket = "answer_rejected"
        elif answer_audit["status"] == "unknown":
            bucket = "answer_unknown"
        else:
            bucket = "accepted"
        record = {
            "sample_id": sample_id,
            "bucket": bucket,
            "used_override": sample_id in override_ids,
            "sample": sample.model_dump(mode="json"),
            "taskgraph": graph,
            "runtime_validation": runtime.model_dump(mode="json"),
            "answer_audit": answer_audit,
            "provider_trace": source.get("provider_trace"),
        }
        records.append(record)
        buckets[bucket].append(record)

    _write_jsonl(output_dir / "records.jsonl", records)
    for bucket, bucket_records in buckets.items():
        _write_jsonl(output_dir / f"{bucket}.jsonl", bucket_records)
    report = {
        "version": "taskgraph-stage1-revalidation-v2",
        "base_run_dir": str(base_run_dir.resolve()),
        "override_run_dirs": [str(path.resolve()) for path in override_run_dirs or []],
        "sample_count": len(records),
        "override_count": len(override_ids),
        "bucket_counts": {name: len(value) for name, value in buckets.items()},
        "runtime_error_counts": dict(runtime_errors),
        "answer_reason_counts": dict(answer_reasons),
        "override_sample_ids": sorted(override_ids),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Revalidate a Stage-1 run with separate runtime and answer quality gates"
    )
    parser.add_argument("--base-run-dir", type=Path, required=True)
    parser.add_argument("--override-run-dir", type=Path, action="append", default=[])
    parser.add_argument("--xlrs-json", type=Path)
    parser.add_argument("--mme-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    answer_index = load_answer_index(xlrs_json=args.xlrs_json, mme_json=args.mme_json)
    report = revalidate(
        base_run_dir=args.base_run_dir,
        output_dir=args.output_dir,
        answer_index=answer_index,
        override_run_dirs=args.override_run_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
