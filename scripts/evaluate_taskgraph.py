"""Run the complete TaskGraph system and emit submission-grade artifacts."""

# The source path is inserted below before importing the local package.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, cast

import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sat_rs_vlm.evaluation.extended_metrics import latency_statistics
from sat_rs_vlm.evaluation.runner import run_evaluation, validate_output_directory
from sat_rs_vlm.infrastructure.telemetry import (
    canonical_json_sha256,
    collect_prompt_provenance,
    collect_provider_inventory,
    collect_repository_provenance,
    collect_runtime_environment,
    preload_provider_models,
    visual_input_telemetry,
)
from sat_rs_vlm.taskgraph.runtime import RuntimeRequest, runtime_from_config
from sat_rs_vlm.taskgraph.runtime_types import (
    Answer,
    Boolean,
    ChoiceResult,
    Label,
    LabelSet,
    RuntimeObject,
    ScalarFloat,
    ScalarInt,
    runtime_summary,
)
from sat_rs_vlm.taskgraph.schema import QuestionType
from sat_rs_vlm.taskgraph.tracing import ExecutionTrace


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number}: expected an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"evaluation input contains no rows: {path}")
    return rows


def _message_fields(row: dict[str, Any]) -> tuple[list[str], str, str]:
    images: list[str] = []
    question = str(row.get("question", "")).strip()
    reference = str(row.get("reference", row.get("answer", "")))
    for message in row.get("messages", []):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            if message.get("role") == "user" and not question:
                question = content.strip()
            if message.get("role") == "assistant" and not reference:
                reference = content
        elif isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "image":
                    images.append(str(item.get("image", "")))
                elif item.get("type") == "text" and message.get("role") == "user" and not question:
                    question = str(item.get("text", "")).strip()
    return images, question, reference


def _sample_id(row: dict[str, Any]) -> str:
    """Return the row sample id, tolerating both ``sample_id`` and ``id`` keys."""
    value = row.get("sample_id", row.get("id", ""))
    return str(value)


def _row_request(row: dict[str, Any], image_root: Path) -> RuntimeRequest:
    message_images, message_question, message_reference = _message_fields(row)
    raw_images = row.get("image_paths", row.get("images", message_images))
    if isinstance(raw_images, str):
        raw_images = [raw_images]
    images = tuple(
        str((image_root / Path(str(item))).resolve())
        if not Path(str(item)).expanduser().is_absolute()
        else str(Path(str(item)).expanduser())
        for item in list(raw_images or [])
    )
    metadata = row.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    raw_options = row.get("options", metadata.get("options", metadata.get("answer_choices", [])))
    graph = row.get("graph", row.get("taskgraph"))
    raw_question_type = str(row.get("question_type", "free_form"))
    try:
        question_type = QuestionType(raw_question_type)
    except ValueError:
        question_type = QuestionType.FREE_FORM
    return RuntimeRequest(
        sample_id=_sample_id(row),
        dataset=str(row.get("dataset", metadata.get("dataset", "unknown"))),
        task_category=str(row.get("task_category", row.get("task_type", "default"))),
        question=str(row.get("question", message_question)),
        image_paths=images,
        options=tuple(str(item) for item in list(raw_options or [])),
        question_type=question_type,
        target_category=(
            str(row.get("target_category", metadata.get("target_category", ""))).strip()
            or None
        ),
        graph=graph,
    )


def _grounding_prediction(
    output: RuntimeObject,
    image_paths: tuple[str, ...],
    metadata: dict[str, Any],
) -> dict[str, Any] | None:
    entity = output
    if hasattr(output, "entities"):
        entities = output.entities  # type: ignore[union-attr]
        if not entities:
            return None
        entity = max(
            entities,
            key=lambda item: float(item.score) if item.score is not None else float("-inf"),
        )
    if not hasattr(entity, "region") or not hasattr(entity, "label"):
        return None
    bbox = tuple(float(value) for value in entity.region.bbox_xyxy_global)
    if not image_paths:
        return None
    try:
        with Image.open(image_paths[0]) as image:
            width, height = image.size
    except (OSError, ValueError):
        return None
    normalized = [bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height]
    target_format = str(metadata.get("bbox_target_format", "normalized_0_1"))
    if target_format == "percent_0_100":
        values = [value * 100.0 for value in normalized]
    elif target_format == "scaled_0_1000":
        values = [value * 1000.0 for value in normalized]
    elif target_format == "pixel_xyxy":
        values = list(bbox)
    else:
        values = normalized
    return {"label": str(entity.label), "bbox": values}


def _prediction_text(
    output: RuntimeObject | ChoiceResult | tuple[RuntimeObject, ...],
    *,
    image_paths: tuple[str, ...] = (),
    metadata: dict[str, Any] | None = None,
    task_category: str = "",
) -> str:
    if isinstance(output, ChoiceResult):
        return str(output.choice_id)
    if task_category.casefold() in {"detection", "grounding", "visual_grounding", "referring"}:
        grounding = _grounding_prediction(output, image_paths, metadata or {})
        if grounding is not None:
            return json.dumps(grounding, ensure_ascii=False, separators=(",", ":"))
    if isinstance(output, Answer):
        return str(output.text)
    if isinstance(output, ScalarInt | ScalarFloat | Boolean | Label):
        return str(output.value)
    if isinstance(output, LabelSet):
        return json.dumps(list(output.values), ensure_ascii=False, separators=(",", ":"))
    if isinstance(output, tuple):
        value = [runtime_summary(item) for item in output]
    else:
        value = runtime_summary(output)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _activated_providers(trace: ExecutionTrace) -> list[str]:
    executor = trace.telemetry.get("executor", {})
    values = executor.get("activated_providers", []) if isinstance(executor, dict) else []
    result = {str(item) for item in values}
    if trace.choice_provider:
        result.add(trace.choice_provider)
    if isinstance(trace.result, dict):
        provenance = trace.result.get("provenance")
        if isinstance(provenance, dict) and provenance.get("provider"):
            result.add(str(provenance["provider"]))
    provider_metadata = trace.telemetry.get("provider_metadata", {})
    if isinstance(provider_metadata, dict):
        provider = provider_metadata.get("base_provider") or provider_metadata.get("provider")
        if provider:
            result.add(str(provider))
    for node in trace.nodes:
        if isinstance(node.telemetry, dict) and node.telemetry.get("base_provider"):
            result.add(str(node.telemetry["base_provider"]))
    planner = trace.telemetry.get("planner", {})
    if isinstance(planner, dict) and planner.get("provider"):
        result.add(str(planner["provider"]))
    return sorted(result)


def _provider_models(inventory: dict[str, Any], providers: list[str]) -> list[dict[str, Any]]:
    rows = []
    seen: set[str] = set()
    for model in inventory.get("models", []):
        identity = str(model.get("identity", model.get("provider", "")))
        name = str(model.get("provider", ""))
        matches = any(identity == provider for provider in providers)
        if not matches and not model.get("role"):
            matches = any(name == provider for provider in providers)
        if matches and identity not in seen:
            rows.append(dict(model))
            seen.add(identity)
    return rows


def _path_summary(
    signature: tuple[str, ...],
    inventory: dict[str, Any],
    count: int,
) -> dict[str, Any]:
    models = _provider_models(inventory, list(signature))
    known_parameters = sum(
        int(model["parameter_count"])
        for model in models
        if model.get("parameter_count") is not None
    )
    known_storage = sum(
        int(model["local_model_storage_bytes"])
        for model in models
        if model.get("local_model_storage_bytes") is not None
    )
    return {
        "providers": list(signature),
        "models": models,
        "known_parameter_count": known_parameters,
        "known_model_storage_bytes": known_storage,
        "parameter_accounting_status": (
            "complete"
            if bool(models) and all(model.get("parameter_count") is not None for model in models)
            else "partial"
        ),
        "storage_accounting_status": (
            "complete"
            if bool(models)
            and all(model.get("local_model_storage_bytes") is not None for model in models)
            else "partial"
        ),
        "sample_count": count,
    }


def _generation_summary(events: list[Any]) -> dict[str, Any]:
    valid = [event for event in events if isinstance(event, dict)]
    ttft_values = [
        float(event["timing_ms"]["ttft"])
        for event in valid
        if isinstance(event.get("timing_ms"), dict)
        and event["timing_ms"].get("ttft") is not None
    ]
    decode_ms = sum(
        float(event["timing_ms"]["decode_generation"])
        for event in valid
        if isinstance(event.get("timing_ms"), dict)
        and event["timing_ms"].get("decode_generation") is not None
    )
    generated = sum(
        int(event["tokens"]["generated"])
        for event in valid
        if isinstance(event.get("tokens"), dict)
        and event["tokens"].get("generated") is not None
    )
    output_tokens = [
        int(value)
        for event in valid
        if isinstance(event.get("tokens"), dict)
        for value in event["tokens"].get("output", [])
        if value is not None
    ]
    visual_counts = [
        int(event["vision_input"]["visual_token_count"])
        for event in valid
        if isinstance(event.get("vision_input"), dict)
        and event["vision_input"].get("visual_token_count") is not None
    ]
    return {
        "timing_ms": {
            "ttft": sum(ttft_values) if ttft_values else None,
            "decode_generation": decode_ms or None,
        },
        "tokens": {
            "generated": generated or None,
            "output": sum(output_tokens) if output_tokens else None,
            "decode_tokens_per_second": (
                generated / (decode_ms / 1000.0) if generated and decode_ms > 0 else None
            ),
            "events": len(valid),
        },
        "vision_input": {
            "visual_token_count": sum(visual_counts) if visual_counts else None,
            "events": [
                event.get("vision_input")
                for event in valid
                if isinstance(event.get("vision_input"), dict)
            ],
        },
    }


def _repeat_telemetry(results: list[Any], latencies: list[float]) -> dict[str, Any]:
    measurements: list[dict[str, Any]] = []
    for index, (result, wall_latency_ms) in enumerate(zip(results, latencies, strict=True)):
        trace = result.trace
        generation = _generation_summary(trace.telemetry.get("generation_events", []))
        system = dict(trace.telemetry.get("system", {}))
        measurements.append(
            {
                "repeat_index": index,
                "route": result.execution_mode.value,
                "wall_e2e_ms": wall_latency_ms,
                "system_e2e_ms": system.get("timing_ms", {}).get("e2e"),
                "timing_ms": generation["timing_ms"],
                "tokens": generation["tokens"],
                "resources": system.get("resources", {}),
                "prediction": _prediction_text(result.output),
            }
        )

    ttft_values = [
        float(item["timing_ms"]["ttft"])
        for item in measurements
        if item["timing_ms"].get("ttft") is not None
    ]
    decode_values = [
        float(item["timing_ms"]["decode_generation"])
        for item in measurements
        if item["timing_ms"].get("decode_generation") is not None
    ]
    generated_values = [
        int(item["tokens"]["generated"])
        for item in measurements
        if item["tokens"].get("generated") is not None
    ]
    output_values = [
        int(item["tokens"]["output"])
        for item in measurements
        if item["tokens"].get("output") is not None
    ]
    resource_keys = (
        "peak_cpu_rss_mb",
        "peak_gpu_allocated_mb",
        "peak_gpu_reserved_mb",
    )
    peak_resources = {
        key: max(
            (
                float(item["resources"][key])
                for item in measurements
                if item["resources"].get(key) is not None
            ),
            default=None,
        )
        for key in resource_keys
    }
    predictions = [str(item["prediction"]) for item in measurements]
    return {
        "measurements": measurements,
        "timing_ms": {
            "ttft": latency_statistics(ttft_values)["mean"],
            "decode_generation": latency_statistics(decode_values)["mean"],
        },
        "tokens": {
            "generated": generated_values[0] if generated_values else None,
            "output": output_values[0] if output_values else None,
            "generated_mean_per_repeat": (
                sum(generated_values) / len(generated_values) if generated_values else None
            ),
            "output_mean_per_repeat": (
                sum(output_values) / len(output_values) if output_values else None
            ),
            "decode_tokens_per_second": (
                sum(generated_values) / (sum(decode_values) / 1000.0)
                if generated_values and decode_values and sum(decode_values) > 0
                else None
            ),
            "events": sum(int(item["tokens"].get("events", 0)) for item in measurements),
        },
        "resources": peak_resources,
        "output_consistent": len(set(predictions)) <= 1,
    }


def _performance_summary(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [
        row for row in predictions if bool(row.get("telemetry", {}).get("success"))
    ]

    def values(path: tuple[str, ...]) -> list[float]:
        result: list[float] = []
        for row in successful:
            value: Any = row
            for key in path:
                value = value.get(key) if isinstance(value, dict) else None
            if isinstance(value, int | float) and not isinstance(value, bool):
                result.append(float(value))
        return result

    resources = {
        key: max(values(("telemetry", "resources", key)), default=None)
        for key in (
            "peak_cpu_rss_mb",
            "peak_gpu_allocated_mb",
            "peak_gpu_reserved_mb",
        )
    }
    return {
        "latency_semantics": "complete_system_single_sample_e2e_repeat_mean",
        "e2e_ms": latency_statistics(values(("inference_latency_ms",))),
        "ttft_ms": latency_statistics(values(("telemetry", "timing_ms", "ttft"))),
        "decode_tokens_per_second": latency_statistics(
            values(("telemetry", "tokens", "decode_tokens_per_second"))
        ),
        "generated_tokens": latency_statistics(
            values(("telemetry", "tokens", "generated"))
        ),
        "output_tokens": latency_statistics(values(("telemetry", "tokens", "output"))),
        "visual_tokens": latency_statistics(
            values(("telemetry", "vision_input", "visual_token_count"))
        ),
        "resources": resources,
        "coverage": {
            "successful_samples": len(successful),
            "ttft_samples": len(values(("telemetry", "timing_ms", "ttft"))),
            "decode_rate_samples": len(
                values(("telemetry", "tokens", "decode_tokens_per_second"))
            ),
            "output_token_samples": len(values(("telemetry", "tokens", "output"))),
            "visual_token_samples": len(
                values(("telemetry", "vision_input", "visual_token_count"))
            ),
            "gpu_memory_samples": len(
                values(("telemetry", "resources", "peak_gpu_allocated_mb"))
            ),
            "cpu_memory_samples": len(
                values(("telemetry", "resources", "peak_cpu_rss_mb"))
            ),
        },
    }


def _source_visual_metadata(image_paths: tuple[str, ...]) -> dict[str, Any]:
    images: list[Any] = []
    for path in image_paths:
        try:
            with Image.open(path) as image:
                images.append(image.copy())
        except (OSError, ValueError):
            continue
    return cast(
        dict[str, Any],
        visual_input_telemetry([images], None, resize_policy="runtime_provider"),
    )


def _execution_visual_metadata(
    trace: ExecutionTrace, image_paths: tuple[str, ...]
) -> dict[str, Any]:
    """Combine source geometry with provider-reported tiling and crop work."""

    result = _source_visual_metadata(image_paths)
    tile_records: list[Any] = []
    crop_records: list[Any] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if isinstance(value.get("tile_count"), int | float):
                tile_records.append(
                    {
                        key: value.get(key)
                        for key in ("tile_count", "tile_size", "overlap_ratio")
                    }
                )
            if isinstance(value.get("crop_count"), int | float):
                crop_records.append({key: value.get(key) for key in ("crop_count", "model_id")})
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(trace.to_dict())
    result["tile_count"] = int(sum(int(item.get("tile_count", 0)) for item in tile_records))
    result["crop_count"] = int(sum(int(item.get("crop_count", 0)) for item in crop_records))
    result["provider_tile_records"] = tile_records
    result["provider_crop_records"] = crop_records
    return result


def _prompt_provenance(row: dict[str, Any], request: RuntimeRequest) -> dict[str, Any]:
    metadata = row.get("metadata", {})
    return cast(
        dict[str, Any],
        collect_prompt_provenance(
            dataset=request.dataset,
            task_type=request.task_category,
            question=request.question,
            options=request.options,
            metadata=metadata if isinstance(metadata, dict) else None,
            graph=request.graph,
        ),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="TaskGraph runtime YAML")
    parser.add_argument("--input", required=True, help="Complete-system evaluation JSONL")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--contract",
        default="configs/eval/evaluation_contract_v1.8_local_complete.yaml",
    )
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--repeat-runs", type=int, default=1)
    parser.add_argument("--strict", action="store_true", default=False)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.warmup_runs < 0 or args.repeat_runs < 1:
        raise SystemExit("warmup-runs must be non-negative and repeat-runs must be positive")
    config_path = Path(args.config).expanduser().resolve()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    validate_output_directory(output_dir)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    image_root = Path(args.image_root or config.get("data", {}).get("image_root", "."))
    if not image_root.is_absolute():
        image_root = (PROJECT_ROOT / image_root).resolve()
    rows = _read_jsonl(input_path)
    contract_path = (
        (PROJECT_ROOT / args.contract).resolve()
        if not Path(args.contract).is_absolute()
        else Path(args.contract).resolve()
    )
    runtime_started = time.perf_counter()
    runtime = runtime_from_config(config)
    runtime_init_ms = (time.perf_counter() - runtime_started) * 1000.0
    configured_providers = [
        runtime.providers.detection,
        runtime.providers.semantic_2b,
        runtime.providers.route_4b,
        runtime.providers.retriever,
        runtime.providers.choice,
        runtime.providers.planner,
    ]
    cold_start: dict[str, Any] | None = None
    predictions: list[dict[str, Any]] = []
    path_counts: Counter[tuple[str, ...]] = Counter()
    path_rows: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    prompt_records: list[dict[str, Any]] = []
    failed_samples = 0
    warmup_attempts = 0
    warmup_failures: list[dict[str, str]] = []
    try:
        cold_start = preload_provider_models(
            [provider for provider in configured_providers if provider is not None]
        )
        warmup_rows = rows[:1]
        for _ in range(args.warmup_runs):
            for row in warmup_rows:
                warmup_attempts += 1
                try:
                    runtime.run(_row_request(row, image_root))
                except Exception as exc:
                    warmup_failures.append(
                        {
                            "sample_id": _sample_id(row),
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        }
                    )
        for row in rows:
            sample_id = _sample_id(row)
            _, _, reference = _message_fields(row)
            if not reference:
                reference = str(row.get("reference", row.get("answer", "")))
            started = time.perf_counter()
            result = None
            error: Exception | None = None
            prompt_provenance: dict[str, Any] | None = None
            try:
                request = _row_request(row, image_root)
                prompt_provenance = _prompt_provenance(row, request)
                prompt_records.append({"sample_id": sample_id, **prompt_provenance})
                repeat_results = []
                repeat_latencies: list[float] = []
                for _ in range(args.repeat_runs):
                    repeat_started = time.perf_counter()
                    repeat_results.append(runtime.run(request))
                    repeat_latencies.append((time.perf_counter() - repeat_started) * 1000.0)
                result = repeat_results[0]
            except Exception as exc:
                error = exc
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if error is not None:
                failed_samples += 1
                predictions.append(
                    {
                        "id": sample_id,
                        "task_type": str(row.get("task_type", row.get("task_category", "unknown"))),
                        "prediction": "",
                        "reference": reference,
                        "metadata": dict(row.get("metadata", {})),
                        "inference_latency_ms": None,
                        "latency_semantics": "complete_system_e2e_failed",
                        "telemetry": {
                            "schema_version": "1.0",
                            "success": False,
                            "timing_ms": {"e2e": elapsed_ms, "ttft": None},
                            "error_type": type(error).__name__,
                            "error_message": str(error),
                            "prompt_provenance": prompt_provenance,
                        },
                    }
                )
                continue
            assert result is not None
            trace = result.trace
            system = dict(trace.telemetry.get("system", {}))
            providers = _activated_providers(trace)
            signature = tuple(providers)
            inventory = dict(system.get("provider_inventory", {}))
            path_counts[signature] += 1
            path_rows[signature].append(row)
            measured_latency_ms = (
                sum(repeat_latencies) / len(repeat_latencies)
                if repeat_latencies
                else elapsed_ms
            )
            repeat_telemetry = _repeat_telemetry(repeat_results, repeat_latencies)
            generation_events = trace.telemetry.get("generation_events", [])
            generation = _generation_summary(generation_events)
            telemetry = {
                "schema_version": "1.0",
                "success": True,
                "route": result.execution_mode.value,
                "activated_models": providers,
                "activated_parameters": _path_summary(signature, inventory, 1)[
                    "known_parameter_count"
                ],
                "timing_ms": {
                    **dict(trace.telemetry.get("phase_timing_ms", {})),
                    "e2e": measured_latency_ms,
                    "repeat_batch_e2e": elapsed_ms,
                },
                "resources": repeat_telemetry["resources"],
                "provider_inventory": inventory,
                "generation_events": generation_events,
                "repeat_runs": args.repeat_runs,
                "repeat_measurements": repeat_telemetry["measurements"],
                "repeat_output_consistent": repeat_telemetry["output_consistent"],
                "prompt_provenance": prompt_provenance,
            }
            telemetry["timing_ms"].update(repeat_telemetry["timing_ms"])
            telemetry["tokens"] = repeat_telemetry["tokens"]
            telemetry["vision_input"] = generation["vision_input"]
            provider_visual = _execution_visual_metadata(trace, request.image_paths)
            if telemetry["vision_input"]["events"] == []:
                telemetry["vision_input"] = provider_visual
            else:
                telemetry["vision_input"].update(
                    {
                        "tile_count": provider_visual["tile_count"],
                        "crop_count": provider_visual["crop_count"],
                        "provider_tile_records": provider_visual["provider_tile_records"],
                        "provider_crop_records": provider_visual["provider_crop_records"],
                    }
                )
            predictions.append(
                {
                    "id": sample_id,
                    "task_type": str(row.get("task_type", row.get("task_category", "unknown"))),
                    "prediction": _prediction_text(
                        result.output,
                        image_paths=request.image_paths,
                        metadata=dict(row.get("metadata", {})),
                        task_category=str(
                            row.get("task_type", row.get("task_category", ""))
                        ),
                    ),
                    "reference": reference,
                    "metadata": dict(row.get("metadata", {})),
                    "inference_latency_ms": measured_latency_ms,
                    "latency_semantics": "complete_system_single_sample_e2e",
                    "telemetry": telemetry,
                    "execution_trace": trace.to_dict(),
                }
            )
    finally:
        try:
            final_inventory = collect_provider_inventory(
                [
                    runtime.providers.detection,
                    runtime.providers.counting,
                    runtime.providers.semantic_2b,
                    runtime.providers.route_4b,
                    runtime.providers.retriever,
                    runtime.providers.choice,
                    runtime.providers.planner,
                ]
            )
        finally:
            runtime.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as file:
        for prediction in predictions:
            file.write(json.dumps(prediction, ensure_ascii=False, separators=(",", ":")) + "\n")

    evaluation_dir = output_dir / "evaluation"
    evaluation_outputs = run_evaluation(
        prediction_path,
        evaluation_dir,
        contract_path=contract_path,
        strict=args.strict,
        semantic_enabled=False,
        latency_semantics="single_sample",
        eval_batch_size=1,
        group_by_task=False,
        resource_benchmark={
            "scope": "complete_taskgraph_system",
            "runtime_init_ms": runtime_init_ms,
            "cold_start": cold_start or {
                "scope": "all_configured_model_providers",
                "latency_ms": None,
                "providers": [],
                "all_supported": False,
            },
            "warmup_runs": args.warmup_runs,
            "repeat_runs": args.repeat_runs,
            "failed_samples": failed_samples,
            "system_inventory": final_inventory,
        },
    )
    ordered_paths = sorted(path_counts.items(), key=lambda item: (-item[1], item[0]))
    typical = (
        _path_summary(ordered_paths[0][0], final_inventory, ordered_paths[0][1])
        if ordered_paths
        else None
    )
    heaviest = None
    if ordered_paths:
        heaviest = max(
            (
                _path_summary(signature, final_inventory, count)
                for signature, count in ordered_paths
            ),
            key=lambda item: item["known_parameter_count"],
        )
    prompt_records_sorted = sorted(prompt_records, key=lambda item: str(item["sample_id"]))
    prompt_summary = {
        "sample_count": len(prompt_records_sorted),
        "unique_profiles": sorted({str(item["profile"]) for item in prompt_records_sorted}),
        "unique_versions": sorted({str(item["version"]) for item in prompt_records_sorted}),
        "unique_prompt_hash_count": len(
            {str(item["sha256"]) for item in prompt_records_sorted}
        ),
        "aggregate_sha256": canonical_json_sha256(prompt_records_sorted)
        if prompt_records_sorted
        else None,
        "records": prompt_records_sorted,
    }
    performance_summary = _performance_summary(predictions)
    evaluation_config = config.get("evaluation", {})
    if not isinstance(evaluation_config, dict):
        evaluation_config = {}
    system_manifest = {
        "schema_version": "1.0",
        "system": {
            "scope": "complete_taskgraph_system",
            "total_parameter_count": final_inventory.get("total_parameter_count"),
            "known_parameter_count": final_inventory.get("known_parameter_count"),
            "total_model_storage_bytes": final_inventory.get("total_model_storage_bytes"),
            "known_model_storage_bytes": final_inventory.get("known_model_storage_bytes"),
            "parameter_accounting_status": final_inventory.get("parameter_accounting_status"),
            "storage_accounting_status": final_inventory.get("storage_accounting_status"),
        },
        "models": final_inventory.get("models", []),
        "paths": {
            "typical": typical,
            "heaviest": heaviest,
            "distribution": [
                _path_summary(signature, final_inventory, count)
                for signature, count in ordered_paths
            ],
        },
        "performance": performance_summary,
        "runtime": collect_runtime_environment(),
        "benchmark": {
            "sample_count": len(predictions),
            "failed_samples": failed_samples,
            "warmup_runs": args.warmup_runs,
            "warmup_scope": "first_evaluation_sample",
            "warmup_attempts": warmup_attempts,
            "warmup_failures": warmup_failures,
            "repeat_runs": args.repeat_runs,
            "repeat_output_policy": "first_repeat_used_for_scoring_all_repeats_profiled",
            "runtime_init_ms": runtime_init_ms,
            "cold_start": cold_start or {
                "scope": "all_configured_model_providers",
                "latency_ms": None,
                "providers": [],
                "all_supported": False,
            },
            "latency_semantics": "complete_system_single_sample_e2e_repeat_mean",
            "cache_policy": str(evaluation_config.get("cache_policy", "unspecified")),
            "timing_boundaries": {
                "runtime_init": "runtime provider construction before warmup",
                "single_sample_e2e": "TaskGraphRuntime.run including routing and providers",
            },
            "input_file": str(input_path),
            "input_sha256": _sha256(input_path),
        },
        "prompt": prompt_summary,
        "configuration": {
            "runtime_config_file": str(config_path),
            "runtime_config_sha256": _sha256(config_path),
            "evaluation_contract_file": str(contract_path),
            "evaluation_contract_sha256": _sha256(contract_path),
        },
        "repository": collect_repository_provenance(PROJECT_ROOT),
    }
    (output_dir / "system_manifest.json").write_text(
        json.dumps(system_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "evaluation_metadata.json").write_text(
        json.dumps(
            {
                "sample_count": len(predictions),
                "failed_samples": failed_samples,
                "successful_samples": len(predictions) - failed_samples,
                "warmup_runs": args.warmup_runs,
                "repeat_runs": args.repeat_runs,
                "warmup_attempts": warmup_attempts,
                "warmup_failures": warmup_failures,
                "performance": performance_summary,
                "prompt": prompt_summary,
                "system_manifest_file": str(output_dir / "system_manifest.json"),
                "evaluation_outputs": {
                    name: str(path) for name, path in evaluation_outputs.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "predictions": str(prediction_path),
                "system_manifest": str(output_dir / "system_manifest.json"),
            },
            indent=2,
        )
    )
    return 0


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
