"""Fault-tolerant, resumable evaluation for the production TaskGraph runtime."""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from .runtime import RuntimeRequest, RuntimeResult, TaskGraphRuntime
from .runtime_types import ChoiceResult, runtime_summary
from .schema import QuestionType, parse_taskgraph
from .tracing import ExecutionTrace


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
    dataset = str(_first_value(sample, metadata, ("dataset", "Dataset")) or "MME_RealWorld_RS")
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
    graph = parse_taskgraph(graph_value) if isinstance(graph_value, (str, bytes, Mapping)) else None
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


_COMPACT_TRACE_DROP_KEYS = frozenset(
    {
        "adapter_config",
        "all_tiles",
        "bert",
        "bert_manifest",
        "checkpoint",
        "checkpoint_identity",
        "checkpoint_manifest",
        "config_dump",
        "environment",
        "environment_info",
        "full_config",
        "full_environment",
        "manifest",
        "model_config",
        "model_configuration",
        "raw_proposal_records",
        "raw_proposals",
        "raw_response",
        "proposal_records",
        "tile_records",
        "tiles",
    }
)
_COMPACT_TRACE_COUNT_KEYS = frozenset({"boxes", "candidates", "detections", "entities", "regions"})
_TRACE_OMIT = object()


def _compact_trace_value(value: Any, *, key: str | None = None, depth: int = 0) -> Any:
    normalized_key = key.casefold() if key is not None else None
    if normalized_key in _COMPACT_TRACE_DROP_KEYS:
        return _TRACE_OMIT
    if depth > 8:
        return "<trace_depth_limit>"
    if isinstance(value, Mapping):
        if normalized_key in {"candidate_mapping", "role_mapping"}:
            return {"count": len(value)}
        compact: dict[str, Any] = {}
        for raw_key, item in value.items():
            child_key = str(raw_key)
            child = _compact_trace_value(item, key=child_key, depth=depth + 1)
            if child is not _TRACE_OMIT:
                compact[child_key] = child
        return compact
    if isinstance(value, (list, tuple)):
        if normalized_key in _COMPACT_TRACE_COUNT_KEYS:
            return {"count": len(value)}
        compact_items = [
            item
            for item in (
                _compact_trace_value(item, depth=depth + 1) for item in value
            )
            if item is not _TRACE_OMIT
        ]
        return compact_items
    return value


def _serialized_trace(trace: ExecutionTrace, *, compact: bool) -> dict[str, Any]:
    if not compact:
        return trace.to_dict()
    serialized = {
        field.name: getattr(trace, field.name)
        for field in fields(trace)
        if field.name != "nodes"
    }
    serialized["nodes"] = [
        {field.name: getattr(node, field.name) for field in fields(node)}
        for node in trace.nodes
    ]
    compacted = _compact_trace_value(serialized)
    return compacted if isinstance(compacted, dict) else {}


_REFERENCE_ANSWER_KEYS = ("ground_truth", "Ground truth", "reference_answer", "answer")


def _reference_answer(sample: Any) -> Any:
    if not isinstance(sample, Mapping):
        return None
    metadata = _metadata(sample)
    value = _first_value(sample, metadata, _REFERENCE_ANSWER_KEYS)
    if isinstance(value, str) and not value.strip():
        return None
    if isinstance(value, (list, tuple, set, frozenset)) and not value:
        return None
    return value


def _choice_ids(options: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(chr(ord("A") + index) for index in range(len(options)))


def _normalize_choice_atom(value: Any, options: tuple[str, ...]) -> str:
    text = " ".join(str(value).strip().split())
    folded = text.casefold()
    ids = _choice_ids(options)
    for index, option in enumerate(options):
        choice_id = ids[index]
        option_text = " ".join(str(option).strip().split())
        if folded in {choice_id.casefold(), option_text.casefold()}:
            return choice_id
        option_body = re.sub(
            r"^\s*[\(\[\{]?\s*[A-Za-z]+(?:\s*[\)\]\}:.\-]|\s+)\s*",
            "",
            option_text,
        )
        if option_body and folded == option_body.casefold():
            return choice_id
    match = re.match(r"^\s*[\(\[\{]?\s*([A-Za-z]+)\s*[\)\]\}:.\-]\s*", text)
    if match and match.group(1).upper() in ids:
        return match.group(1).upper()
    if text.upper() in ids:
        return text.upper()
    return folded


def _normalize_answer(value: Any, options: tuple[str, ...]) -> Any:
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(_normalize_answer(item, options) for item in value)
    text = " ".join(str(value).strip().split())
    if options and re.fullmatch(r"[A-Za-z](?:\s*[,;/]\s*[A-Za-z])+", text):
        return tuple(_normalize_choice_atom(item, options) for item in re.split(r"[,;/]", text))
    return _normalize_choice_atom(text, options) if options else text.casefold()


def _answer_judgment(
    sample: Any,
    request: RuntimeRequest | None,
    predicted_answer: Any,
    *,
    options: tuple[str, ...] = (),
) -> dict[str, Any]:
    reference_answer = _reference_answer(sample)
    options = request.options if request is not None else options
    comparison = "normalized_choice" if options else "normalized_text"
    if reference_answer is None:
        return {
            "predicted_answer": predicted_answer,
            "reference_answer": None,
            "status": "unavailable",
            "exact_match": None,
            "comparison": comparison,
        }
    normalized_reference = _normalize_answer(reference_answer, options)
    if predicted_answer is None:
        return {
            "predicted_answer": None,
            "reference_answer": reference_answer,
            "normalized_reference_answer": normalized_reference,
            "status": "not_predicted",
            "exact_match": False,
            "comparison": comparison,
        }
    normalized_predicted = _normalize_answer(predicted_answer, options)
    exact_match = normalized_predicted == normalized_reference
    return {
        "predicted_answer": predicted_answer,
        "reference_answer": reference_answer,
        "normalized_predicted_answer": normalized_predicted,
        "normalized_reference_answer": normalized_reference,
        "status": "correct" if exact_match else "incorrect",
        "exact_match": exact_match,
        "comparison": comparison,
    }


def _planner_chain(trace: ExecutionTrace, *, compact: bool = False) -> dict[str, Any]:
    telemetry = trace.telemetry
    raw_metadata = telemetry.get("planner_metadata")
    metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
    raw_attempts = metadata.get("attempts")
    attempts = list(raw_attempts) if isinstance(raw_attempts, list) else []
    generated_output = metadata.get("planner_output")
    if generated_output is None:
        for attempt in reversed(attempts):
            if isinstance(attempt, Mapping) and attempt.get("termination_reason") == "final":
                generated_output = attempt.get("planner_output", attempt.get("prediction"))
                break
    metadata.pop("attempts", None)
    metadata.pop("planner_output", None)
    if compact:
        attempts = [
            {
                key: attempt[key]
                for key in ("attempt", "termination_reason", "error_type", "error")
                if key in attempt
            }
            for attempt in attempts
            if isinstance(attempt, Mapping)
        ]
        compact_metadata = _compact_trace_value(metadata)
        metadata = compact_metadata if isinstance(compact_metadata, dict) else {}
    status = str(telemetry.get("planner_status", "not_used"))
    if trace.taskgraph is not None and status == "deferred":
        status = "provided_graph"
    return {
        "status": status,
        "generated_output": generated_output,
        "latency_ms": trace.planner_ms
        if trace.planner_ms is not None
        else telemetry.get("planner_ms"),
        "taskgraph": trace.taskgraph,
        "attempts": attempts,
        "metadata": metadata,
    }


def _module_chain(trace: ExecutionTrace, *, compact: bool = False) -> list[dict[str, Any]]:
    if trace.nodes:
        modules: list[dict[str, Any]] = []
        for node in trace.nodes:
            trace_metadata: Any = dict(node.trace_metadata)
            output: Any = dict(node.output_summary)
            if compact:
                trace_metadata = _compact_trace_value(trace_metadata)
                output = _compact_trace_value(output)
            modules.append(
                {
                    "node_id": node.node_id,
                    "operator": node.operator,
                    "inputs": dict(node.input_refs),
                    "resolved_input_types": dict(node.resolved_input_types),
                    "provider": node.provider,
                    "output_runtime_type": node.output_runtime_type,
                    "latency_ms": node.latency_ms,
                    "fallback": node.fallback,
                    "output": output,
                    "trace_metadata": trace_metadata,
                    "error": dict(node.error) if node.error is not None else None,
                }
            )
        return modules
    telemetry = trace.telemetry
    modules: list[dict[str, Any]] = []
    direct_modules = (
        ("detector", "DETECTOR", "detector_metadata", trace.result),
        ("counting", "COUNT", "counting_metadata", trace.result),
        ("semantic_vlm", "SEMANTIC_VLM", "semantic_metadata", trace.result),
        ("choice", "CHOICE", "choice_metadata", trace.choice_result),
    )
    for stage, operator, metadata_key, output in direct_modules:
        metadata = telemetry.get(metadata_key)
        if metadata is None and output is None:
            continue
        if compact:
            metadata = _compact_trace_value(metadata)
            output = _compact_trace_value(output)
        modules.append(
            {
                "node_id": None,
                "stage": stage,
                "operator": operator,
                "provider": trace.choice_provider if stage == "choice" else None,
                "latency_ms": telemetry.get(stage + "_ms"),
                "output": output,
                "trace_metadata": metadata if isinstance(metadata, Mapping) else {},
                "error": None,
            }
        )
    return modules


def _reasoning_chain(
    trace: ExecutionTrace,
    *,
    prediction: dict[str, Any] | None,
    answer_judgment: dict[str, Any],
    failure: Mapping[str, Any] | None = None,
    compact: bool = False,
) -> dict[str, Any]:
    answer = prediction.get("answer") if prediction is not None else None
    output = prediction.get("output") if prediction is not None else None
    chain: dict[str, Any] = {
        "execution_mode": trace.execution_mode,
        "input_image_paths": list(trace.input_image_paths),
        "intermediate_output_paths": list(trace.intermediate_output_paths),
        "planner": _planner_chain(trace, compact=compact),
        "modules": _module_chain(trace, compact=compact),
        "final": {
            "answer": answer,
            "output": output,
            "source_refs": list(trace.final_sources),
            "question": trace.final_question,
            "choice_result": trace.choice_result,
            "answer_judgment": answer_judgment,
        },
    }
    if failure is not None:
        chain["failure"] = dict(failure)
    return chain


def _sample_options(sample: Any) -> tuple[str, ...]:
    if not isinstance(sample, Mapping):
        return ()
    metadata = _metadata(sample)
    try:
        return _options(
            _first_value(
                sample,
                metadata,
                ("choices", "options", "multi_choice_options", "Answer choices"),
            )
        )
    except (TypeError, ValueError):
        return ()


def _sample_image_paths(sample: Any, image_root: Path | None) -> list[str]:
    if not isinstance(sample, Mapping):
        return []
    metadata = _metadata(sample)
    try:
        images = _image_values(
            _first_value(sample, metadata, ("image_paths", "images", "image", "path", "Image"))
        )
        return list(_resolve_images(images, image_root))
    except (TypeError, ValueError):
        return []


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
    sample: Any = None,
    request: RuntimeRequest | None = None,
    image_root: Path | None = None,
    compact_trace: bool = False,
) -> dict[str, Any]:
    if isinstance(request_like, RuntimeRequest):
        sample_id = request_like.sample_id
        dataset = request_like.dataset
        task_category = request_like.task_category
    elif isinstance(request_like, Mapping):
        sample_id = fallback_sample_id
        metadata = _metadata(request_like)
        dataset = str(_first_value(request_like, metadata, ("dataset", "Dataset")) or "unknown")
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
    trace = getattr(exc, "execution_trace", None)
    trace = trace if isinstance(trace, ExecutionTrace) else None
    input_image_paths = (
        list(trace.input_image_paths)
        if trace is not None and trace.input_image_paths
        else list(request.image_paths)
        if request is not None
        else _sample_image_paths(sample, image_root)
    )
    intermediate_output_paths = list(trace.intermediate_output_paths) if trace is not None else []
    answer_judgment = _answer_judgment(
        sample,
        request,
        None,
        options=_sample_options(sample),
    )
    failure = {
        "stage": _stage_for_exception(exc),
        "error_type": error_type,
        "exception_type": type(exc).__name__,
        "message": str(exc),
    }
    if trace is not None:
        reasoning_chain = _reasoning_chain(
            trace,
            prediction=None,
            answer_judgment=answer_judgment,
            failure=failure,
            compact=compact_trace,
        )
    else:
        reasoning_chain = {
            "execution_mode": None,
            "input_image_paths": input_image_paths,
            "intermediate_output_paths": intermediate_output_paths,
            "planner": {
                "status": "failed" if failure["stage"] == "planner" else "not_reached",
                "generated_output": None,
                "taskgraph": None,
                "attempts": [],
                "metadata": {},
            },
            "modules": [],
            "final": {
                "answer": None,
                "output": None,
                "source_refs": [],
                "question": request.question if request is not None else None,
                "answer_judgment": answer_judgment,
            },
            "failure": failure,
        }
    row: dict[str, Any] = {
        "sample_id": sample_id,
        "dataset": dataset,
        "task_category": task_category,
        "status": "failure",
        "result_status": "sample_failure",
        "prediction": None,
        "answer": None,
        "answer_judgment": answer_judgment,
        "input_image_paths": input_image_paths,
        "intermediate_output_paths": intermediate_output_paths,
        "reasoning_chain": reasoning_chain,
        "error_type": error_type,
        "exception_type": type(exc).__name__,
        "message": str(exc),
        "stage": _stage_for_exception(exc),
        "elapsed_ms": elapsed_ms,
    }
    if trace is not None:
        row["trace"] = _serialized_trace(trace, compact=compact_trace)
    return row


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
    request_factory: Callable[[RuntimeRequest | Mapping[str, Any]], RuntimeRequest] | None = None,
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
    compact_trace = bool(getattr(getattr(runtime, "composer", None), "compact_trace", False))
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
            request: RuntimeRequest | None = None
            try:
                request = (
                    request_factory(sample)
                    if request_factory is not None
                    else runtime_request_from_sample(sample, image_root=config.image_root)
                )
                request_like = request
                result = runtime.run(request)
                elapsed_ms = (time.perf_counter() - sample_started) * 1000.0
                prediction = _serialized_output(result)
                if not result.trace.input_image_paths:
                    result.trace.input_image_paths = list(request.image_paths)
                answer_judgment = _answer_judgment(sample, request, prediction["answer"])
                row = {
                    "sample_id": request.sample_id,
                    "dataset": request.dataset,
                    "task_category": request.task_category,
                    "status": "success",
                    "result_status": "success",
                    "execution_mode": result.execution_mode.value,
                    "prediction": prediction,
                    "answer": prediction["answer"],
                    "answer_judgment": answer_judgment,
                    "input_image_paths": list(request.image_paths),
                    "intermediate_output_paths": list(result.trace.intermediate_output_paths),
                    "reasoning_chain": _reasoning_chain(
                        result.trace,
                        prediction=prediction,
                        answer_judgment=answer_judgment,
                        compact=compact_trace,
                    ),
                    "trace": _serialized_trace(result.trace, compact=compact_trace),
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
                    sample=sample,
                    request=request,
                    image_root=config.image_root,
                    compact_trace=compact_trace,
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
