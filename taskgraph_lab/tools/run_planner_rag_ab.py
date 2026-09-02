"""Run the four frozen hard-subset Planner RAG ablations in serial subprocesses."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HARD_INTENTS = (
    "RELATIONAL_COUNT",
    "OBJECT_RELATION",
    "ROUTE_PLANNING",
    "COMPLEX_REASONING",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--rag-bank", type=Path, required=True)
    parser.add_argument("--rule-cards", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--sample-id", action="append", default=[])
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _run(command: list[str], log_path: Path) -> int:
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUTF8"] = "1"
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[2],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=environment,
        )
        if process.stdout is None:
            raise RuntimeError("failed to capture benchmark subprocess output")
        for line in process.stdout:
            log.write(line)
            log.flush()
        return process.wait()


def _metric(summary: dict[str, Any], name: str) -> float | None:
    value = (summary.get("rates") or {}).get(name)
    return float(value) if value is not None else None


def _comparison_row(label: str, summary: dict[str, Any]) -> dict[str, Any]:
    relation = (summary.get("per_intent") or {}).get("RELATIONAL_COUNT") or {}
    relation_metrics = relation.get("relational_metrics") or {}

    def relation_rate(name: str) -> float | None:
        value = (relation_metrics.get(name) or {}).get("rate")
        return float(value) if value is not None else None

    return {
        "mode": label,
        "sample_count": summary.get("sample_count"),
        "surface_grammar_valid": _metric(summary, "surface_grammar_valid"),
        "graph_runtime_valid": _metric(summary, "graph_runtime_valid"),
        "intent_exact": _metric(summary, "intent_exact"),
        "operator_sequence_exact": _metric(summary, "operator_sequence_exact"),
        "node_count_exact": _metric(summary, "node_count_exact"),
        "canonical_exact": _metric(summary, "canonical_exact"),
        "mean_prompt_tokens": summary.get("mean_prompt_tokens"),
        "mean_generated_tokens": summary.get("mean_generated_tokens"),
        "p50_total_planner_latency_seconds": summary.get(
            "p50_total_planner_latency_seconds"
        ),
        "p95_total_planner_latency_seconds": summary.get(
            "p95_total_planner_latency_seconds"
        ),
        "mean_retrieval_latency_ms": summary.get("mean_retrieval_latency_ms"),
        "relational_count_canonical_exact": (relation.get("rates") or {}).get(
            "canonical_exact"
        ),
        "relational_count_attachment_accuracy": relation_rate(
            "reference_attachment_accuracy"
        ),
        "relational_count_filtered_source_accuracy": relation_rate(
            "count_filtered_source_accuracy"
        ),
    }


def _markdown(rows: list[dict[str, Any]]) -> str:
    columns = (
        "mode",
        "surface_grammar_valid",
        "graph_runtime_valid",
        "intent_exact",
        "operator_sequence_exact",
        "node_count_exact",
        "canonical_exact",
        "mean_prompt_tokens",
        "p50_total_planner_latency_seconds",
        "p95_total_planner_latency_seconds",
        "mean_retrieval_latency_ms",
        "relational_count_canonical_exact",
        "relational_count_attachment_accuracy",
        "relational_count_filtered_source_accuracy",
    )
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        values = []
        for name in columns:
            value = row.get(name)
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    modes = (
        ("no_rag", "off", 0, False),
        ("rules_only", "hard_intent", 0, True),
        ("top2_only", "hard_intent", 2, False),
        ("rules_top2", "hard_intent", 2, True),
    )
    state: dict[str, Any] = {
        "started_at": datetime.now(UTC).isoformat(),
        "status": "running",
        "modes": [],
    }
    state_path = args.output_root / "launcher_state.json"
    _write_json(state_path, state)
    comparisons = []
    for label, rag_mode, top_k, rules in modes:
        output_dir = args.output_root / label
        command = [
            sys.executable,
            "-m",
            "taskgraph_lab.tools.evaluate_qwen3vl_planner",
            "--base-model",
            str(args.base_model),
            "--adapter",
            str(args.adapter),
            "--validation-file",
            str(args.validation_file),
            "--output-dir",
            str(output_dir),
            "--batch-size",
            "1",
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--max-prompt-tokens",
            str(args.max_prompt_tokens),
            "--constrained",
            "--enable-recovery",
            "--max-attempts",
            "3",
            "--rag-mode",
            rag_mode,
            "--rag-router",
            "benchmark_metadata_intent",
            "--rag-top-k",
            str(top_k),
        ]
        for intent in HARD_INTENTS:
            command.extend(("--intent-filter", intent))
        for sample_id in args.sample_id:
            command.extend(("--sample-id", sample_id))
        if top_k:
            command.extend(("--rag-bank", str(args.rag_bank)))
        if rules:
            command.extend(("--rag-rules", "--rag-rule-cards", str(args.rule_cards)))
        mode_state = {"mode": label, "command": command, "status": "running"}
        state["modes"].append(mode_state)
        _write_json(state_path, state)
        try:
            return_code = _run(command, args.output_root / f"{label}.log")
        except Exception as exc:
            mode_state["status"] = "failed"
            mode_state["launcher_error"] = f"{type(exc).__name__}: {exc}"
            state["status"] = "failed"
            state["finished_at"] = datetime.now(UTC).isoformat()
            _write_json(state_path, state)
            raise
        mode_state["return_code"] = return_code
        mode_state["status"] = "complete" if return_code == 0 else "failed"
        _write_json(state_path, state)
        if return_code:
            state["status"] = "failed"
            state["finished_at"] = datetime.now(UTC).isoformat()
            _write_json(state_path, state)
            return return_code
        summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
        comparisons.append(_comparison_row(label, summary))
        _write_json(args.output_root / "comparison.json", comparisons)
        (args.output_root / "comparison.md").write_text(
            _markdown(comparisons), encoding="utf-8"
        )
    state["status"] = "complete"
    state["finished_at"] = datetime.now(UTC).isoformat()
    _write_json(state_path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
