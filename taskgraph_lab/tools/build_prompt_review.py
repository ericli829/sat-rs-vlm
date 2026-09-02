from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _output_path(raw_path: Path, kind: str) -> Path:
    if raw_path.parent.name != "raw":
        raise ValueError(f"run raw path must be inside an outputs/raw directory: {raw_path}")
    return raw_path.parent.parent / kind / raw_path.name


def _trace_totals(traces: list[dict[str, Any]]) -> dict[str, int | float]:
    totals: dict[str, int | float] = {
        "logical_calls": len(traces),
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "latency_ms": 0.0,
    }
    for trace in traces:
        usage = trace.get("usage") or {}
        completion_details = usage.get("completion_tokens_details") or {}
        totals["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        totals["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        totals["reasoning_tokens"] += int(completion_details.get("reasoning_tokens") or 0)
        totals["latency_ms"] += float(trace.get("latency_ms") or 0.0)
    return totals


def _initial_validation_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ("schema_valid", "graph_valid", "type_valid", "semantic_valid")
    validations = [record.get("validation") for record in records]
    validations = [item for item in validations if isinstance(item, dict)]
    denominator = len(validations)
    counts = {field: sum(bool(item.get(field)) for item in validations) for field in fields}
    return {
        "validated_samples": denominator,
        "counts": counts,
        "rates": {
            field: (counts[field] / denominator if denominator else 0.0) for field in fields
        },
        "repair_classification_counts": dict(
            Counter(str(record.get("repair_classification") or "UNKNOWN") for record in records)
        ),
    }


def _load_run(name: str, raw_path: Path) -> dict[str, Any]:
    raw_records = _read_jsonl(raw_path)
    terminal_records = {
        kind: _read_jsonl(_output_path(raw_path, kind))
        for kind in ("valid", "repaired", "rejected")
    }
    terminal_by_id: dict[str, tuple[str, dict[str, Any]]] = {}
    for kind, records in terminal_records.items():
        for record in records:
            terminal_by_id[str(record["sample_id"])] = (kind, record)

    outcomes: dict[str, dict[str, Any]] = {}
    traces: list[dict[str, Any]] = []
    statuses: Counter[str] = Counter()
    for raw in raw_records:
        sample_id = str(raw["sample_id"])
        initial_trace = raw.get("provider_trace") or {}
        if initial_trace:
            traces.append(initial_trace)
        terminal_status, terminal = terminal_by_id.get(sample_id, (str(raw.get("status")), {}))
        statuses[terminal_status] += 1
        repair_trace: dict[str, Any] | None = None
        if terminal_status == "repaired":
            repair_trace = terminal.get("provider_trace")
        elif terminal_status == "rejected":
            repair_trace = terminal.get("repair_provider_trace")
        if repair_trace:
            traces.append(repair_trace)
        outcomes[sample_id] = {
            "terminal_status": terminal_status,
            "initial_candidate_text": raw.get("candidate_text"),
            "initial_validation": raw.get("validation"),
            "initial_provider_trace": initial_trace or None,
            "final_target": terminal.get("target"),
            "final_validation": terminal.get("validation"),
            "repaired_candidate_text": terminal.get("repaired_candidate_text"),
            "repair_provider_trace": repair_trace,
            "repair_api_error": terminal.get("repair_api_error"),
        }

    return {
        "name": name,
        "raw_path": str(raw_path.resolve()),
        "summary": {
            "samples": len(raw_records),
            "terminal_status_counts": dict(statuses),
            "initial_validation": _initial_validation_summary(raw_records),
            **_trace_totals(traces),
        },
        "inputs": {
            str(record["sample_id"]): record.get("sample")
            for record in raw_records
            if record.get("sample") is not None
        },
        "outcomes": outcomes,
    }


def build_review(prompt_path: Path, runs: list[tuple[str, Path]]) -> dict[str, Any]:
    prompt_text = prompt_path.read_text(encoding="utf-8")
    loaded = [_load_run(name, path) for name, path in runs]
    samples: dict[str, dict[str, Any]] = {}
    for run in loaded:
        for sample_id, outcome in run["outcomes"].items():
            sample = samples.setdefault(
                sample_id,
                {"sample_id": sample_id, "input": run["inputs"].get(sample_id), "outcomes": {}},
            )
            if sample["input"] is None and run["inputs"].get(sample_id) is not None:
                sample["input"] = run["inputs"][sample_id]
            sample["outcomes"][run["name"]] = outcome
    return {
        "prompt": {
            "path": str(prompt_path.resolve()),
            "sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
            "text": prompt_text,
        },
        "runs": {
            run["name"]: {"raw_path": run["raw_path"], "summary": run["summary"]} for run in loaded
        },
        "samples": list(samples.values()),
    }


def _errors(validation: dict[str, Any] | None) -> str:
    if not validation:
        return "None"
    errors = validation.get("errors") or []
    if not errors:
        return "None"
    return "; ".join(
        f"{item.get('stage')}:{item.get('code')} - {item.get('message')}" for item in errors
    )


def markdown(review: dict[str, Any]) -> str:
    lines = [
        "# TaskGraph Teacher prompt review",
        "",
        "## Prompt (single copy)",
        "",
        f"- Path: `{review['prompt']['path']}`",
        f"- SHA256: `{review['prompt']['sha256']}`",
        "",
        "```text",
        review["prompt"]["text"].rstrip(),
        "```",
        "",
        "## Run summary",
        "",
        "| Run | Valid | Repaired | Rejected | Initial schema | Initial graph | Initial type | "
        "Initial semantic | Calls | Prompt tokens | Completion tokens | Reasoning tokens | "
        "Latency ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, run in review["runs"].items():
        summary = run["summary"]
        counts = summary["terminal_status_counts"]
        rates = summary["initial_validation"]["rates"]
        lines.append(
            f"| {name} | {counts.get('valid', 0)} | {counts.get('repaired', 0)} | "
            f"{counts.get('rejected', 0)} | {rates['schema_valid']:.0%} | "
            f"{rates['graph_valid']:.0%} | {rates['type_valid']:.0%} | "
            f"{rates['semantic_valid']:.0%} | {summary['logical_calls']} | "
            f"{summary['prompt_tokens']} | {summary['completion_tokens']} | "
            f"{summary['reasoning_tokens']} | {summary['latency_ms']:.1f} |"
        )
    for sample in review["samples"]:
        source = sample.get("input") or {}
        lines.extend(
            [
                "",
                f"## {sample['sample_id']}",
                "",
                f"- Question: {source.get('question')}",
                f"- Question type: {source.get('question_type')}",
                f"- Choices: `{json.dumps(source.get('choices'), ensure_ascii=False)}`",
            ]
        )
        for run_name, outcome in sample["outcomes"].items():
            lines.extend(
                [
                    "",
                    f"### {run_name}: {outcome['terminal_status']}",
                    "",
                    f"Initial errors: {_errors(outcome.get('initial_validation'))}",
                    "",
                    "Initial candidate:",
                    "",
                    "```json",
                    outcome.get("initial_candidate_text") or "",
                    "```",
                ]
            )
            if outcome.get("final_target") is not None:
                lines.extend(
                    [
                        "",
                        "Final accepted target:",
                        "",
                        "```json",
                        json.dumps(outcome["final_target"], ensure_ascii=False, indent=2),
                        "```",
                    ]
                )
            elif outcome.get("repaired_candidate_text") is not None:
                lines.extend(
                    [
                        "",
                        f"Final errors: {_errors(outcome.get('final_validation'))}",
                        "",
                        "Rejected repair candidate:",
                        "",
                        "```json",
                        outcome["repaired_candidate_text"],
                        "```",
                    ]
                )
    return "\n".join(lines) + "\n"


def _run_arg(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("--run must use NAME=RAW_JSONL")
    return name, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Combine Teacher outputs with one prompt copy")
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--run", type=_run_arg, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    review = build_review(args.prompt, args.run)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_md.write_text(markdown(review), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_json": str(args.output_json.resolve()),
                "output_md": str(args.output_md.resolve()),
                "runs": list(review["runs"]),
                "samples": len(review["samples"]),
                "prompt_copies": 1,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
