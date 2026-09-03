"""Replay selected planner failures against the production planner boundary."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SAMPLE_IDS = (
    "Complex_reasoning_Environmental_condition_reasoning_18",
    "Land_use_classification_Regional_Land_use_classification_0",
    "Land_use_classification_Overall_Land_use_classification_0",
    "Object_properties_Object_motion_state_0",
    "Object_spatial_relationship_Object_spatial_relationship_2",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _user_payload(row: dict[str, Any]) -> dict[str, Any]:
    users = [message for message in row.get("messages", []) if message.get("role") == "user"]
    if len(users) != 1:
        raise ValueError(f"planner row {row.get('id')!r} must contain one user message")
    payload = json.loads(str(users[0].get("content", "")))
    if not isinstance(payload, dict):
        raise TypeError(f"planner row {row.get('id')!r} user payload must be an object")
    return payload


def _source_basename(value: str) -> str:
    return Path(value.replace("\\", "/")).name


def _find_planner_row(
    failure_row: dict[str, Any], planner_rows: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    failure_question = str(
        (failure_row.get("reasoning_chain") or {}).get("final", {}).get("question", "")
    )
    failure_images = failure_row.get("input_image_paths") or []
    failure_basename = _source_basename(str(failure_images[0])) if failure_images else ""
    question_matches: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for planner_row in planner_rows:
        payload = _user_payload(planner_row)
        if str(payload.get("question", "")) != failure_question:
            continue
        question_matches.append((planner_row, payload))
        planner_inputs = payload.get("inputs") or {}
        planner_uris = [
            _source_basename(str(value.get("uri_or_key", "")))
            for value in planner_inputs.values()
            if isinstance(value, dict)
        ]
        if failure_basename and any(
            failure_basename == uri or failure_basename.startswith(uri + "_")
            for uri in planner_uris
        ):
            return planner_row, payload
    if len(question_matches) == 1:
        return question_matches[0]
    candidates = [str(row.get("id")) for row, _ in question_matches]
    raise LookupError(
        f"cannot uniquely map failure {failure_row.get('sample_id')!r}; candidates={candidates}"
    )


def _request(sample_id: str, payload: dict[str, Any]) -> Any:
    from sat_rs_vlm.taskgraph.providers import PlannerRequest
    from sat_rs_vlm.taskgraph.runtime_types import ImageRef

    raw_inputs = payload.get("inputs")
    if not isinstance(raw_inputs, dict) or not raw_inputs:
        raise ValueError(f"sample {sample_id!r} has no planner image input")
    inputs = {
        f"${str(key).removeprefix('$')}": ImageRef(str(value.get("uri_or_key", "")))
        for key, value in raw_inputs.items()
        if isinstance(value, dict)
    }
    choices = payload.get("choices")
    return PlannerRequest(
        question=str(payload.get("question", "")),
        question_type=str(payload.get("question_type", "FREE_FORM")),
        choices=tuple(str(choice) for choice in choices) if choices else (),
        inputs=inputs,
        sample_id=sample_id,
    )


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failures", type=Path, required=True)
    parser.add_argument("--planner-data", type=Path, action="append", required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-id", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    from sat_rs_vlm.taskgraph.providers import Qwen3VLPlannerProvider

    args = _parse_args()
    requested_ids = tuple(args.sample_id) if args.sample_id else DEFAULT_SAMPLE_IDS
    failures = {
        str(row.get("sample_id")): row
        for row in _read_jsonl(args.failures)
        if str(row.get("sample_id")) in requested_ids
    }
    missing = [sample_id for sample_id in requested_ids if sample_id not in failures]
    if missing:
        raise ValueError(f"requested failure ids are absent: {missing}")
    planner_rows = [
        row
        for planner_data in args.planner_data
        for row in _read_jsonl(planner_data)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    provider = Qwen3VLPlannerProvider(
        {
            "model_id": str(args.base_model),
            "adapter_path": str(args.adapter),
            "processor_id": str(args.base_model),
            "local_files_only": True,
            "max_attempts": 2,
        }
    )
    started_at = datetime.now(UTC)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for sample_id in requested_ids:
            failure_row = failures[sample_id]
            planner_row, payload = _find_planner_row(failure_row, planner_rows)
            request = _request(sample_id, payload)
            result: dict[str, Any] = {
                "sample_id": sample_id,
                "planner_sample_id": planner_row.get("id"),
                "reference_answer": (failure_row.get("answer_judgment") or {}).get(
                    "reference_answer"
                ),
                "question": request.question,
                "choices": list(request.choices),
                "planner_inputs": {
                    key: value.uri_or_key for key, value in request.inputs.items()
                },
                "started_at": datetime.now(UTC).isoformat(),
            }
            sample_started = time.perf_counter()
            try:
                graph = provider.plan(request)
                result.update(
                    {
                        "status": "success",
                        "generated_output": provider.last_metadata.get("planner_output"),
                        "planner_metadata": provider.last_metadata,
                        "taskgraph": _jsonable(graph),
                    }
                )
            except Exception as exc:
                result.update(
                    {
                        "status": "planner_failed",
                        "exception_type": type(exc).__name__,
                        "error": str(exc),
                        "generated_output": provider.last_metadata.get("planner_output"),
                        "planner_metadata": provider.last_metadata,
                        "taskgraph": None,
                    }
                )
            result["elapsed_ms"] = (time.perf_counter() - sample_started) * 1000.0
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "status": result["status"],
                        "elapsed_ms": result["elapsed_ms"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            gc.collect()
    provider.close()
    print(
        json.dumps(
            {
                "status": "completed",
                "sample_count": len(requested_ids),
                "started_at": started_at.isoformat(),
                "finished_at": datetime.now(UTC).isoformat(),
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
