"""Fault-tolerant, resumable evaluation for the production TaskGraph runtime."""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .runtime import RuntimeRequest, RuntimeResult, TaskGraphRuntime
from .runtime_types import ChoiceResult, runtime_summary
from .schema import QuestionType, parse_taskgraph


@dataclass(frozen=True)
class TaskGraphEvaluationConfig:
    """Output and fault policy for a long-running TaskGraph evaluation."""

    output_path: Path
    continue_on_sample_error: bool = True
    fail_fast: bool = False
    resume: bool = True
    image_root: Path | None = None


def _metadata(sample: Mapping[str, Any]) -> Mapping[str, Any]:
    value = sample.get("metadata", {})
    return value if isinstance(value, Mapping) else {}


def sample_id_for(sample: RuntimeRequest | Mapping[str, Any]) -> str:
    if isinstance(sample, RuntimeRequest):
        return sample.sample_id
    for key in ("sample_id", "id", "Question_id"):
        value = sample.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise ValueError("sample is missing a non-empty sample_id or id")


def _sample_id_or_placeholder(sample: Any, ordinal: int) -> str:
    try:
        return sample_id_for(sample)
    except (AttributeError, TypeError, ValueError):
        return f"<sample-{ordinal}>"


def _first_value(
    sample: Mapping[str, Any],
    metadata: Mapping[str, Any],
    keys: tuple[str, ...],
) -> Any:
    for key in keys:
        if key in sample and sample[key] is not None:
            return sample[key]
        if key in metadata and metadata[key] is not None:
            return metadata[key]
    return None


def _question_type(value: Any, choices: tuple[str, ...]) -> QuestionType:
    normalized = str(value or "").strip().upper().replace("-", "_")
    if normalized in {"MULTIPLE_CHOICE_MULTI", "MULTI"}:
        return QuestionType.MULTIPLE_CHOICE_MULTI
    if normalized in {"MULTIPLE_CHOICE_SINGLE", "MULTIPLE_CHOICE", "SINGLE"}:
        return QuestionType.MULTIPLE_CHOICE_SINGLE
    if normalized in {item.value for item in QuestionType}:
        return QuestionType(normalized)
    return QuestionType.MULTIPLE_CHOICE_SINGLE if choices else QuestionType.FREE_FORM


def _options(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable) and not isinstance(value, (bytes, Mapping)):
        return tuple(str(item) for item in value)
    raise TypeError("sample choices must be a sequence or string")


def _image_values(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        ordered = sorted(value.items(), key=lambda item: str(item[0]))
        result: list[str] = []
        for _, item in ordered:
            if isinstance(item, Mapping):
                item = item.get("uri_or_key", item.get("path", item.get("image")))
            if item is not None:
                result.append(str(item))
        return tuple(result)
    if isinstance(value, Iterable) and not isinstance(value, bytes):
        result = []
        for item in value:
            if isinstance(item, Mapping):
                item = item.get("uri_or_key", item.get("path", item.get("image")))
            if item is not None:
                result.append(str(item))
        return tuple(result)
    raise TypeError("sample images must be a sequence, mapping, or string")


def _resolve_images(images: tuple[str, ...], image_root: Path | None) -> tuple[str, ...]:
    if image_root is None:
        return images
    root = image_root.expanduser().resolve()
    return tuple(
        str((root / image).resolve()) if not Path(image).is_absolute() else image
        for image in images
    )


def runtime_request_from_sample(
    sample: RuntimeRequest | Mapping[str, Any],
    *,
    image_root: str | Path | None = None,
) -> RuntimeRequest:
    """Convert common MME/XLRS normalized rows to ``RuntimeRequest``."""

    if isinstance(sample, RuntimeRequest):
        return sample
    metadata = _metadata(sample)
    sample_id = sample_id_for(sample)
    question = _first_value(sample, metadata, ("question", "instruction", "Text"))
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"sample {sample_id}: question is required")
    choices = _options(
        _first_value(
            sample,
            metadata,
            ("choices", "options", "multi_choice_options", "Answer choices"),
        )
    )
    images = _image_values(
        _first_value(sample, metadata, ("image_paths", "images", "image", "path", "Image"))
    )
    root = Path(image_root).expanduser() if image_root is not None else None
    dataset = str(
        _first_value(sample, metadata, ("dataset", "Dataset")) or "MME_RealWorld_RS"
    )
    task_category = str(
        _first_value(
            sample,
            metadata,
            ("task_category", "task", "category", "Category", "l2_category"),
        )
        or "default"
    )
    raw_question_type = _first_value(sample, metadata, ("question_type", "Question Type"))
    graph_value = sample.get("graph", sample.get("taskgraph"))
    graph = (
        parse_taskgraph(graph_value)
        if isinstance(graph_value, (str, bytes, Mapping))
        else None
    )
    target = _first_value(sample, metadata, ("target_category", "target"))
    if isinstance(target, Mapping):
        target = target.get("category")
    return RuntimeRequest(
        sample_id=sample_id,
        dataset=dataset,
        task_category=task_category,
        question=question,
        image_paths=_resolve_images(images, root),
        options=choices,
        question_type=_question_type(raw_question_type, choices),
        target_category=str(target) if target is not None else None,
        graph=graph,
    )


def _serialized_output(result: RuntimeResult) -> dict[str, Any]:
    if isinstance(result.output, tuple):
        output: Any = [runtime_summary(item) for item in result.output]
    else:
        output = runtime_summary(result.output)
    if isinstance(result.output, ChoiceResult):
        answer: Any = result.output.choice_id
        if answer is None:
            answer = list(result.output.selected_ids)
    elif isinstance(output, Mapping) and "text" in output:
        answer = output["text"]
    elif isinstance(output, Mapping) and "value" in output:
        answer = output["value"]
    else:
        answer = output
    return {"answer": answer, "output": output}


def _stage_for_exception(exc: BaseException) -> str:
    explicit_stage = getattr(exc, "stage", None)
    if isinstance(explicit_stage, str) and explicit_stage:
        return explicit_stage
    details = getattr(exc, "details", None)
    operator = str(details.get("operator", "")) if isinstance(details, Mapping) else ""
    provider = str(details.get("provider", "")) if isinstance(details, Mapping) else ""
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and len(chain) < 8:
        chain.append(current)
        current = current.__cause__ or current.__context__
    names = " ".join(type(item).__name__.casefold() for item in chain)
    if "planner" in names:
        return "planner"
    if "retrieval" in names or "locator" in names:
        return "retriever"
    if "proposal" in names or "detector" in names:
        return "detector"
    if "choice" in names:
        return "choice"
    if "semantic" in names:
        return "semantic_vlm"
    if operator in {
        "ATTRIBUTE",
        "CLASSIFY",
        "MULTILABEL_CLASSIFY",
        "MOTION",
        "RELATION",
    }:
        return "semantic_vlm"
    if operator in {"ROUTE_REASON", "BUILD_ROUTE_CONTEXT"} or "route" in provider.casefold():
        return "route_vlm"
    if operator == "LOCATE":
        return "provider"
    if operator:
        return "taskgraph"
    if "graph" in str(exc).casefold() or "validation" in str(exc).casefold():
        return "taskgraph_validation"
    if isinstance(exc, (ValueError, TypeError, KeyError)):
        return "dataset_adapter"
    return "runtime"


def _error_row(
    request_like: Any,
    exc: BaseException,
    elapsed_ms: float,
    *,
    fallback_sample_id: str = "<unknown>",
) -> dict[str, Any]:
    if isinstance(request_like, RuntimeRequest):
        sample_id = request_like.sample_id
        dataset = request_like.dataset
        task_category = request_like.task_category
    elif isinstance(request_like, Mapping):
        sample_id = fallback_sample_id
        metadata = _metadata(request_like)
        dataset = str(
            _first_value(request_like, metadata, ("dataset", "Dataset")) or "unknown"
        )
        task_category = str(
            _first_value(
                request_like,
                metadata,
                ("task_category", "task", "category", "Category"),
            )
            or "unknown"
        )
    else:
        sample_id = fallback_sample_id
        dataset = "unknown"
        task_category = "unknown"
    error_type = str(getattr(exc, "error_type", type(exc).__name__))
    return {
        "sample_id": sample_id,
        "dataset": dataset,
        "task_category": task_category,
        "status": "failure",
        "result_status": "sample_failure",
        "prediction": None,
        "error_type": error_type,
        "exception_type": type(exc).__name__,
        "message": str(exc),
        "stage": _stage_for_exception(exc),
        "elapsed_ms": elapsed_ms,
    }


def _completed_success_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid evaluation output JSON at line {line_number}") from exc
            if isinstance(row, Mapping) and row.get("status") == "success":
                sample_id = str(row.get("sample_id", "")).strip()
                if sample_id:
                    completed.add(sample_id)
    return completed


def run_taskgraph_evaluation(
    runtime: TaskGraphRuntime,
    samples: Iterable[RuntimeRequest | Mapping[str, Any]],
    output_path: str | Path,
    *,
    continue_on_sample_error: bool = True,
    fail_fast: bool = False,
    resume: bool = True,
    image_root: str | Path | None = None,
    request_factory: Callable[
        [RuntimeRequest | Mapping[str, Any]], RuntimeRequest
    ] | None = None,
) -> dict[str, Any]:
    """Run samples with a durable JSONL row for every attempted sample."""

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed_success_ids(destination) if resume else set()
    config = TaskGraphEvaluationConfig(
        destination,
        continue_on_sample_error=continue_on_sample_error,
        fail_fast=fail_fast,
        resume=resume,
        image_root=Path(image_root).expanduser() if image_root is not None else None,
    )
    mode = "a" if resume and destination.is_file() else "w"
    processed = skipped = successes = failures = 0
    failure_stages: Counter[str] = Counter()
    failure_types: Counter[str] = Counter()
    started = time.perf_counter()
    with destination.open(mode, encoding="utf-8", newline="\n") as output:
        for ordinal, sample in enumerate(samples, start=1):
            sample_id = _sample_id_or_placeholder(sample, ordinal)
            if sample_id in completed:
                skipped += 1
                continue
            processed += 1
            sample_started = time.perf_counter()
            request_like: RuntimeRequest | Mapping[str, Any] = sample
            try:
                request = (
                    request_factory(sample)
                    if request_factory is not None
                    else runtime_request_from_sample(sample, image_root=config.image_root)
                )
                result = runtime.run(request)
                elapsed_ms = (time.perf_counter() - sample_started) * 1000.0
                row = {
                    "sample_id": request.sample_id,
                    "dataset": request.dataset,
                    "task_category": request.task_category,
                    "status": "success",
                    "result_status": "success",
                    "execution_mode": result.execution_mode.value,
                    "prediction": _serialized_output(result),
                    "trace": result.trace.to_dict(),
                    "elapsed_ms": elapsed_ms,
                }
                successes += 1
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - sample_started) * 1000.0
                row = _error_row(
                    request_like,
                    exc,
                    elapsed_ms,
                    fallback_sample_id=sample_id,
                )
                failures += 1
                failure_stages[str(row["stage"])] += 1
                failure_types[str(row["error_type"])] += 1
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            output.flush()
            if row["status"] == "failure" and (fail_fast or not continue_on_sample_error):
                raise RuntimeError(
                    f"TaskGraph evaluation stopped after sample failure: {row['sample_id']}"
                ) from None
    return {
        "output_path": str(config.output_path),
        "processed_samples": processed,
        "skipped_completed_samples": skipped,
        "success_count": successes,
        "failure_count": failures,
        "failure_by_stage": dict(sorted(failure_stages.items())),
        "failure_by_type": dict(sorted(failure_types.items())),
        "continue_on_sample_error": continue_on_sample_error,
        "fail_fast": fail_fast,
        "resume": resume,
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
    }


__all__ = [
    "TaskGraphEvaluationConfig",
    "run_taskgraph_evaluation",
    "runtime_request_from_sample",
    "sample_id_for",
]
