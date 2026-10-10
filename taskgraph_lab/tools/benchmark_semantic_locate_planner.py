"""Run unlabeled English questions through the production text-only Planner."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
import traceback
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sat_rs_vlm.taskgraph.providers import (
    PlannerFailedError,
    PlannerRequest,
    Qwen3VLPlannerProvider,
)
from sat_rs_vlm.taskgraph.runtime_types import ImageRef
from taskgraph_lab.taskgraph.dsl import parse_taskgraph_dsl


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def request_for(row: dict) -> PlannerRequest:
    # This is an opaque input binding, never a file opened or sent to a visual model.
    return PlannerRequest(
        question=row["question"],
        question_type="FREE_FORM",
        choices=(),
        inputs={"$image0": ImageRef("text_only_evaluation_image0")},
        sample_id=row["id"],
    )


def latency_summary(values: list[float]) -> dict:
    ordered = sorted(values)
    if not ordered:
        return {}
    return {
        "mean_ms": statistics.mean(ordered),
        "median_ms": statistics.median(ordered),
        "p95_ms": ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def summarize(results: list[dict], total: int) -> dict:
    families: dict[str, list[dict]] = defaultdict(list)
    operators: Counter = Counter()
    semantic_plans = 0
    for row in results:
        families[row["family"]].append(row)
        nodes = (row.get("taskgraph") or {}).get("nodes", [])
        operators.update(node["op"] for node in nodes)
        semantic_plans += any(
            node["op"] == "SELECT" and node.get("params", {}).get("mode") == "SEMANTIC"
            for node in nodes
        )
    valid = sum(row["status"] == "valid_plan" for row in results)
    attempts = [attempt for row in results for attempt in row.get("attempts", [])]
    return {
        "requested_count": total,
        "completed_count": len(results),
        "valid_plan_count": valid,
        "valid_plan_rate": valid / len(results) if results else None,
        "first_attempt_dsl_valid_count": sum(
            bool(row.get("attempts")) and row["attempts"][0]["dsl_valid"] for row in results
        ),
        "retried_question_count": sum(len(row.get("attempts", [])) > 1 for row in results),
        "status_counts": dict(Counter(row["status"] for row in results)),
        "semantic_select_plan_count": semantic_plans,
        "operator_counts": dict(operators),
        "latency": latency_summary([row["elapsed_ms"] for row in results]),
        "attempt_generation_ms": sum(attempt.get("latency_ms", 0) for attempt in attempts),
        "generated_tokens": sum(attempt.get("generated_tokens", 0) for attempt in attempts),
        "families": {
            family: {
                "count": len(rows),
                "valid_plan_count": sum(row["status"] == "valid_plan" for row in rows),
                "semantic_select_plan_count": sum(row["uses_select_sem"] for row in rows),
                "latency": latency_summary([row["elapsed_ms"] for row in rows]),
            }
            for family, rows in families.items()
        },
        "semantic_accuracy": None,
        "limitations": [
            "Questions have no gold TaskGraph labels; validity is not semantic accuracy.",
            "No image inference or Executor was run; image0 is a symbolic binding only.",
            "Latency reflects this machine and the recorded automatic device placement.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.questions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("questions must be nonempty and have unique IDs")
    if any(row.get("language") != "en" or not row["question"].strip() for row in rows):
        raise ValueError("questions must be nonempty English questions")
    if not (args.base_model / "model.safetensors.index.json").is_file():
        raise FileNotFoundError("missing base model weight index")
    if not (args.adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError("missing Planner adapter weights")
    config = {
        "model_id": str(args.base_model.resolve()),
        "processor_id": str(args.base_model.resolve()),
        "adapter_path": str(args.adapter.resolve()),
        "device": "auto",
        "dtype": "bfloat16",
        "local_files_only": True,
        "trust_remote_code": True,
        "max_new_tokens": 512,
        "max_prompt_tokens": 2048,
        "max_attempts": 2,
        "constrained_decoding": True,
        "constraint_top_k": 64,
        "constraint_max_candidate_checks": 256,
        "constraint_max_nodes": 24,
        "repeat_guard_repetitions": 4,
        "max_finish_node_tokens": 32,
    }
    provider = Qwen3VLPlannerProvider(config)
    system_prompt = provider._system_prompt()
    if args.check_only:
        print(json.dumps({"status": "preflight_passed", "count": len(rows), "config": config}))
        return
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob("*.json*")):
        raise FileExistsError(f"refusing to overwrite an existing run: {output}")
    started_at = datetime.now(UTC).isoformat()
    write_json(output / "status.json", {"status": "loading_model", "started_at": started_at})
    write_json(
        output / "preflight.json",
        {
            "started_at": started_at,
            "config": config,
            "questions_sha256": sha256(args.questions),
            "adapter_config_sha256": sha256(args.adapter / "adapter_config.json"),
            "system_prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
            "runner_sha256": sha256(Path(__file__)),
            "sample_count": len(rows),
            "gold_graphs": False,
            "image_input": "opaque symbolic image0 binding; no real image",
        },
    )
    with (output / "prompts.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "messages": provider._messages(request_for(row), system_prompt),
                    }
                )
                + "\n"
            )
    results: list[dict] = []
    fatal: str | None = None
    try:
        print(json.dumps({"status": "loading_model", "sample_count": len(rows)}), flush=True)
        provider.preload()
        torch = provider._torch
        torch.manual_seed(20261010)
        cuda = torch.cuda.is_available()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        write_json(
            output / "model_load.json",
            {
                **provider.load_info,
                "hf_device_map": {
                    str(key): str(value)
                    for key, value in getattr(provider._model, "hf_device_map", {}).items()
                },
                "cuda_device": torch.cuda.get_device_name(0) if cuda else None,
            },
        )
        with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for index, row in enumerate(rows, 1):
                request = request_for(row)
                messages = provider._messages(request, system_prompt)
                rendered = provider._processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                prompt_tokens = len(provider._tokenizer.encode(rendered))
                if prompt_tokens > provider.max_prompt_tokens:
                    raise RuntimeError("prompt would be truncated; aborting without modifying it")
                if cuda:
                    torch.cuda.synchronize()
                started = time.perf_counter()
                result = {**row, "prompt_tokens_before_truncation": prompt_tokens}
                inference_error = None
                try:
                    graph = provider.plan(request)
                    result.update(status="valid_plan", taskgraph=graph.model_dump(mode="json"))
                except PlannerFailedError as exc:
                    result.update(status="planner_failed", error=str(exc), taskgraph=None)
                except Exception as exc:
                    result.update(status="inference_error", error=repr(exc), taskgraph=None)
                    inference_error = exc
                if cuda:
                    torch.cuda.synchronize()
                result["elapsed_ms"] = (time.perf_counter() - started) * 1000
                metadata = provider.last_metadata
                attempts = []
                for attempt in metadata.get("attempts", []):
                    attempt = dict(attempt)
                    try:
                        parse_taskgraph_dsl(attempt.get("prediction", ""))
                        attempt["dsl_valid"] = True
                    except Exception as exc:
                        attempt.update(dsl_valid=False, dsl_validation_error=str(exc))
                    attempts.append(attempt)
                result["attempts"] = attempts
                result["generated_output"] = metadata.get("planner_output")
                result["uses_select_sem"] = any(
                    node["op"] == "SELECT" and node.get("params", {}).get("mode") == "SEMANTIC"
                    for node in (result.get("taskgraph") or {}).get("nodes", [])
                )
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                results.append(result)
                progress = {
                    "status": "running",
                    "completed": index,
                    "total": len(rows),
                    "last_id": row["id"],
                    "last_status": result["status"],
                    "last_elapsed_ms": result["elapsed_ms"],
                }
                write_json(output / "status.json", progress)
                print(json.dumps(progress), flush=True)
                if inference_error is not None:
                    raise inference_error
        if cuda:
            write_json(
                output / "gpu_memory.json",
                {"peak_allocated_bytes": torch.cuda.max_memory_allocated()},
            )
    except BaseException:
        fatal = traceback.format_exc()
        raise
    finally:
        write_json(output / "summary.json", summarize(results, len(rows)))
        write_json(
            output / "status.json",
            {
                "status": "failed" if fatal else "completed",
                "completed": len(results),
                "total": len(rows),
                "finished_at": datetime.now(UTC).isoformat(),
                "fatal_error": fatal,
            },
        )
        provider.close()


if __name__ == "__main__":
    main()
