"""Compare unconstrained and grammar-constrained Planner evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from taskgraph_lab.evaluation.planner_generation import summarize_predictions


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _backfill_legacy_termination(
    rows: list[dict[str, Any]], *, max_new_tokens: int
) -> None:
    """Recover termination labels absent from evaluations made before this field."""

    for row in rows:
        if row.get("termination_reason") is not None:
            continue
        if bool(row.get("dsl_parse_valid")):
            reason = "final"
        elif int(row.get("generated_tokens", 0)) >= max_new_tokens:
            reason = "max_tokens"
        else:
            reason = "error"
        row["termination_reason"] = reason


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_predictions(rows)
    rates = summary["rates"]
    return {
        "dsl_parse_valid": rates["dsl_parse_valid"],
        "runtime_valid": rates["runtime_valid"],
        "intent_exact": rates["intent_exact"],
        "operator_sequence_exact": rates["operator_sequence_exact"],
        "node_count_exact": rates["node_count_exact"],
        "canonical_exact": rates["canonical_exact"],
        "mean_generated_tokens": summary["mean_generated_tokens"],
        "mean_latency_seconds": summary["mean_latency_seconds"],
        "p50_latency_seconds": summary["p50_latency_seconds"],
        "p95_latency_seconds": summary["p95_latency_seconds"],
        "grammar_dead_end_count": summary["grammar_dead_end_count"],
        "max_token_truncation_count": summary["max_token_truncation_count"],
        "termination_reason_counts": summary["termination_reason_counts"],
        "constraint_failure_counts": summary["constraint_failure_counts"],
    }


def _per_intent(rows: list[dict[str, Any]]) -> dict[str, Any]:
    intents = sorted({str(row.get("expected_intent") or "UNKNOWN") for row in rows})
    report: dict[str, Any] = {}
    for intent in intents:
        selected = [row for row in rows if str(row.get("expected_intent") or "UNKNOWN") == intent]
        report[intent] = {"count": len(selected), **_metrics(selected)}
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--constrained-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline_preflight = _json(args.baseline_dir / "preflight.json")
    constrained_preflight = _json(args.constrained_dir / "preflight.json")
    invariant_fields = (
        "base_model",
        "adapter",
        "adapter_weights_sha256",
        "validation_file",
        "validation_sha256",
        "sample_count",
    )
    mismatches = {
        field: {
            "baseline": baseline_preflight.get(field),
            "constrained": constrained_preflight.get(field),
        }
        for field in invariant_fields
        if baseline_preflight.get(field) != constrained_preflight.get(field)
    }
    generation_fields = (
        "do_sample",
        "num_beams",
        "batch_size",
        "max_new_tokens",
        "max_prompt_tokens",
        "repair",
    )
    baseline_generation = baseline_preflight.get("generation") or {}
    constrained_generation = constrained_preflight.get("generation") or {}
    for field in generation_fields:
        if baseline_generation.get(field) != constrained_generation.get(field):
            mismatches[f"generation.{field}"] = {
                "baseline": baseline_generation.get(field),
                "constrained": constrained_generation.get(field),
            }
    if mismatches:
        raise ValueError(f"A/B provenance mismatch: {json.dumps(mismatches, ensure_ascii=False)}")
    if constrained_generation.get("constrained") is not True:
        raise ValueError("B run is not marked as grammar constrained")

    baseline_rows = _jsonl(args.baseline_dir / "predictions.jsonl")
    constrained_rows = _jsonl(args.constrained_dir / "predictions.jsonl")
    _backfill_legacy_termination(
        baseline_rows,
        max_new_tokens=int(baseline_generation["max_new_tokens"]),
    )
    _backfill_legacy_termination(
        constrained_rows,
        max_new_tokens=int(constrained_generation["max_new_tokens"]),
    )
    baseline_ids = [str(row["sample_id"]) for row in baseline_rows]
    constrained_ids = [str(row["sample_id"]) for row in constrained_rows]
    if baseline_ids != constrained_ids:
        raise ValueError("A/B sample IDs or order do not match")

    baseline_metrics = _metrics(baseline_rows)
    constrained_metrics = _metrics(constrained_rows)
    numeric_delta = {
        key: constrained_metrics[key] - baseline_metrics[key]
        for key in baseline_metrics
        if isinstance(baseline_metrics[key], (int, float))
        and isinstance(constrained_metrics[key], (int, float))
    }
    report = {
        "schema_version": "taskgraph-planner-constraint-ab-v1",
        "sample_count": len(baseline_rows),
        "provenance_match": True,
        "baseline_dir": str(args.baseline_dir.resolve()),
        "constrained_dir": str(args.constrained_dir.resolve()),
        "baseline": baseline_metrics,
        "constrained": constrained_metrics,
        "delta_constrained_minus_baseline": numeric_delta,
        "per_intent": {
            "baseline": _per_intent(baseline_rows),
            "constrained": _per_intent(constrained_rows),
        },
        "constraint": constrained_preflight.get("constraint"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
