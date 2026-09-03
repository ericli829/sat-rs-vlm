"""Validate complete-system performance artifacts before they are reported."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number}: expected a JSON object")
            rows.append(value)
    return rows


def _get(value: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _number(value: Any, *, positive: bool = False) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    return math.isfinite(float(value)) and (float(value) > 0 if positive else float(value) >= 0)


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256.fullmatch(value.lower()))


def audit_taskgraph_performance(
    run_directory: str | Path,
    *,
    submission: bool = False,
    require_official: bool = False,
) -> dict[str, Any]:
    """Audit TaskGraph artifacts without loading a model or changing the run."""

    run_dir = Path(run_directory).expanduser().resolve()
    checks: list[dict[str, Any]] = []

    def check(
        code: str,
        condition: bool,
        message: str,
        *,
        submission_only: bool = False,
        details: Any = None,
    ) -> None:
        if condition:
            severity = "pass"
        elif submission_only and not submission:
            severity = "warning"
        else:
            severity = "blocker"
        item: dict[str, Any] = {
            "code": code,
            "status": severity,
            "message": message,
        }
        if details is not None:
            item["details"] = details
        checks.append(item)

    paths = {
        "predictions": run_dir / "predictions.jsonl",
        "manifest": run_dir / "system_manifest.json",
        "metadata": run_dir / "evaluation_metadata.json",
        "metrics": run_dir / "evaluation" / "metrics.json",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    check(
        "artifacts.required_files",
        not missing,
        "Complete-system prediction, manifest, metadata, and metric files are required.",
        details={"missing": missing},
    )
    if missing:
        return _finish(run_dir, submission, require_official, checks, {})

    try:
        predictions = _read_jsonl(paths["predictions"])
        manifest = _read_json(paths["manifest"])
        metadata = _read_json(paths["metadata"])
        _read_json(paths["metrics"])
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        check("artifacts.parseable", False, f"Artifacts must contain valid JSON: {exc}")
        return _finish(run_dir, submission, require_official, checks, {})
    check("artifacts.parseable", True, "All required artifacts contain valid JSON.")

    ids = [str(row.get("id", "")) for row in predictions]
    duplicate_ids = sorted({sample_id for sample_id in ids if ids.count(sample_id) > 1})
    check(
        "samples.identifiers",
        bool(ids) and all(ids) and not duplicate_ids,
        "Prediction IDs must be present and unique.",
        details={"duplicate_ids": duplicate_ids},
    )
    expected_counts = {
        "predictions": len(predictions),
        "manifest": _get(manifest, "benchmark", "sample_count"),
        "metadata": metadata.get("sample_count"),
    }
    check(
        "samples.count_consistency",
        len(set(expected_counts.values())) == 1,
        "Sample counts must agree across predictions, manifest, and metadata.",
        details=expected_counts,
    )

    successful = [row for row in predictions if _get(row, "telemetry", "success") is True]
    failed = [row for row in predictions if _get(row, "telemetry", "success") is not True]
    reported_failures = {
        "predictions": len(failed),
        "manifest": _get(manifest, "benchmark", "failed_samples"),
        "metadata": metadata.get("failed_samples"),
    }
    check(
        "samples.failure_consistency",
        len(set(reported_failures.values())) == 1,
        "Failure counts must agree across artifacts.",
        details=reported_failures,
    )
    check(
        "samples.no_failures",
        not failed,
        "Submission runs must not silently omit or retain failed samples.",
        submission_only=True,
        details={"failed_ids": [row.get("id") for row in failed]},
    )

    warmup_runs = _get(manifest, "benchmark", "warmup_runs")
    repeat_runs = _get(manifest, "benchmark", "repeat_runs")
    warmup_failures = _get(manifest, "benchmark", "warmup_failures")
    check(
        "benchmark.warmup",
        isinstance(warmup_runs, int)
        and warmup_runs >= 1
        and isinstance(warmup_failures, list)
        and not warmup_failures,
        "Submission profiling requires at least one successful warmup.",
        submission_only=True,
        details={"warmup_runs": warmup_runs, "warmup_failures": warmup_failures},
    )
    repeat_lengths = [
        len(_get(row, "telemetry", "repeat_measurements") or []) for row in successful
    ]
    check(
        "benchmark.repeats",
        isinstance(repeat_runs, int)
        and repeat_runs >= 3
        and all(length == repeat_runs for length in repeat_lengths),
        "Submission profiling requires at least three recorded repeats per successful sample.",
        submission_only=True,
        details={"repeat_runs": repeat_runs, "recorded_repeats": repeat_lengths},
    )
    inconsistent = [
        row.get("id")
        for row in successful
        if _get(row, "telemetry", "repeat_output_consistent") is not True
    ]
    check(
        "benchmark.repeat_output_consistency",
        not inconsistent,
        "Deterministic submission repeats must produce the same scored output.",
        submission_only=True,
        details={"inconsistent_ids": inconsistent},
    )

    bad_latency = [
        row.get("id")
        for row in successful
        if row.get("latency_semantics") != "complete_system_single_sample_e2e"
        or not _number(row.get("inference_latency_ms"))
    ]
    check(
        "performance.e2e_latency",
        bool(successful) and not bad_latency,
        "Every successful sample needs complete-system single-sample E2E latency.",
        details={"invalid_ids": bad_latency},
    )

    generation_rows = [
        row
        for row in successful
        if int(_get(row, "telemetry", "tokens", "events") or 0) > 0
    ]

    def missing_generation_field(*keys: str) -> list[Any]:
        return [
            row.get("id")
            for row in generation_rows
            if not _number(_get(row, "telemetry", *keys), positive=True)
        ]

    check(
        "generation.present",
        bool(generation_rows),
        "At least one model-generation path is needed to report generation performance.",
        submission_only=True,
    )
    for code, keys, label in (
        ("generation.ttft", ("timing_ms", "ttft"), "TTFT"),
        (
            "generation.decode_tokens_per_second",
            ("tokens", "decode_tokens_per_second"),
            "decode-only Token/s",
        ),
        ("generation.generated_tokens", ("tokens", "generated"), "generated tokens"),
        ("generation.output_tokens", ("tokens", "output"), "output tokens"),
        ("vision.visual_tokens", ("vision_input", "visual_token_count"), "visual tokens"),
    ):
        missing_ids = missing_generation_field(*keys)
        check(
            code,
            bool(generation_rows) and not missing_ids,
            f"Every generation sample must report {label}.",
            submission_only=True,
            details={"missing_ids": missing_ids},
        )

    missing_vision_geometry = [
        row.get("id")
        for row in successful
        if not isinstance(_get(row, "telemetry", "vision_input", "original_size"), list)
        or not _number(_get(row, "telemetry", "vision_input", "tile_count"))
        or not _number(_get(row, "telemetry", "vision_input", "crop_count"))
    ]
    check(
        "vision.geometry",
        not missing_vision_geometry,
        "Every successful sample must retain original size and tile/crop counts.",
        details={"missing_ids": missing_vision_geometry},
    )

    missing_cpu = [
        row.get("id")
        for row in successful
        if not _number(_get(row, "telemetry", "resources", "peak_cpu_rss_mb"), positive=True)
    ]
    missing_gpu = [
        row.get("id")
        for row in successful
        if not _number(
            _get(row, "telemetry", "resources", "peak_gpu_allocated_mb"), positive=True
        )
        or not _number(
            _get(row, "telemetry", "resources", "peak_gpu_reserved_mb"), positive=True
        )
    ]
    check(
        "resources.cpu_memory",
        not missing_cpu,
        "Every successful sample must report peak process CPU RSS.",
        details={"missing_ids": missing_cpu},
    )
    check(
        "resources.gpu_memory",
        bool(successful) and not missing_gpu,
        "Every successful submission sample must report allocated and reserved peak GPU memory.",
        submission_only=True,
        details={"missing_ids": missing_gpu},
    )

    runtime = manifest.get("runtime", {})
    base_environment_ok = all(
        (
            _get(runtime, "os", "system"),
            _get(runtime, "cpu", "logical_cores"),
            _get(runtime, "software", "python"),
            _get(runtime, "software", "torch"),
        )
    )
    check(
        "environment.base",
        bool(base_environment_ok),
        "OS, CPU, Python, and framework versions must be recorded.",
    )
    gpu_environment_ok = (
        _get(runtime, "gpu", "cuda_available") is True
        and int(_get(runtime, "gpu", "count") or 0) > 0
        and bool(_get(runtime, "gpu", "devices"))
        and bool(_get(runtime, "gpu", "driver_version"))
        and bool(_get(runtime, "gpu", "cuda_runtime"))
    )
    check(
        "environment.gpu",
        gpu_environment_ok,
        "CUDA device, driver, and runtime must be present in a submission run.",
        submission_only=True,
    )

    system = manifest.get("system", {})
    inventory_ok = (
        _number(system.get("total_parameter_count"), positive=True)
        and _number(system.get("total_model_storage_bytes"), positive=True)
        and system.get("parameter_accounting_status") == "complete"
        and system.get("storage_accounting_status") == "complete"
    )
    check(
        "system.inventory",
        inventory_ok,
        "Complete parameter and actual local model-storage totals are required.",
        submission_only=True,
        details={
            "total_parameter_count": system.get("total_parameter_count"),
            "total_model_storage_bytes": system.get("total_model_storage_bytes"),
            "parameter_accounting_status": system.get("parameter_accounting_status"),
            "storage_accounting_status": system.get("storage_accounting_status"),
        },
    )
    paths_section = manifest.get("paths", {})
    path_ok = all(
        (
            isinstance(paths_section.get("typical"), dict),
            isinstance(paths_section.get("heaviest"), dict),
            bool(paths_section.get("distribution")),
        )
    )
    check(
        "system.paths",
        path_ok,
        "Typical, heaviest, and path-distribution summaries are required.",
    )
    path_accounting_ok = bool(
        path_ok
        and all(
            path.get("parameter_accounting_status") == "complete"
            and path.get("storage_accounting_status") == "complete"
            for path in [paths_section.get("typical"), paths_section.get("heaviest")]
            if isinstance(path, dict)
        )
    )
    check(
        "system.path_accounting",
        path_accounting_ok,
        "Typical and heaviest paths must have complete parameter and storage accounting.",
        submission_only=True,
    )

    prompt = manifest.get("prompt", {})
    prompt_ok = (
        prompt.get("sample_count") == len(predictions)
        and _sha256(prompt.get("aggregate_sha256"))
        and all(
            _sha256(_get(row, "telemetry", "prompt_provenance", "sha256"))
            for row in predictions
        )
    )
    check(
        "provenance.prompts",
        prompt_ok,
        "Every sample and the run aggregate must retain prompt hashes.",
    )
    configuration = manifest.get("configuration", {})
    hashes_ok = all(
        _sha256(value)
        for value in (
            _get(manifest, "benchmark", "input_sha256"),
            configuration.get("runtime_config_sha256"),
            configuration.get("evaluation_contract_sha256"),
        )
    )
    check(
        "provenance.hashes",
        hashes_ok,
        "Input, runtime config, and evaluation contract SHA256 values are required.",
    )
    repository = manifest.get("repository", {})
    repository_ok = bool(repository.get("commit")) and repository.get("dirty") is False
    check(
        "provenance.repository",
        repository_ok,
        "Submission runs must use a recorded clean repository commit.",
        submission_only=True,
        details={"commit": repository.get("commit"), "dirty": repository.get("dirty")},
    )

    if require_official:
        incomplete = [
            row.get("id")
            for row in predictions
            if any(
                not str(_get(row, "metadata", key) or "").strip()
                for key in ("dataset_version", "split", "language", "prompt_profile")
            )
            or _get(row, "metadata", "evaluation_scope") != "official_full_split"
        ]
        check(
            "official.sample_provenance",
            not incomplete,
            "Official runs require full-split scope and dataset/prompt provenance per sample.",
            details={"invalid_ids": incomplete},
        )
        input_file = Path(str(_get(manifest, "benchmark", "input_file") or ""))
        source_manifest_path = input_file.with_suffix(input_file.suffix + ".manifest.json")
        source_ok = False
        source_details: dict[str, Any] = {"path": str(source_manifest_path)}
        if source_manifest_path.is_file():
            try:
                source_manifest = _read_json(source_manifest_path)
                source_details.update(source_manifest)
                source_ok = (
                    source_manifest.get("evaluation_scope") == "official_full_split"
                    and _get(source_manifest, "count_check", "status") == "passed"
                    and bool(_get(source_manifest, "official_source", "repository"))
                    and bool(_get(source_manifest, "official_source", "commit"))
                    and source_manifest.get("output_sha256")
                    == _get(manifest, "benchmark", "input_sha256")
                )
            except (OSError, ValueError, json.JSONDecodeError):
                source_ok = False
        check(
            "official.source_manifest",
            source_ok,
            "Official input needs a matching certified converter manifest.",
            details=source_details,
        )

    summary = {
        "sample_count": len(predictions),
        "successful_samples": len(successful),
        "failed_samples": len(failed),
        "generation_samples": len(generation_rows),
        "warmup_runs": warmup_runs,
        "repeat_runs": repeat_runs,
    }
    return _finish(run_dir, submission, require_official, checks, summary)


def _finish(
    run_dir: Path,
    submission: bool,
    require_official: bool,
    checks: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    blockers = [item for item in checks if item["status"] == "blocker"]
    warnings = [item for item in checks if item["status"] == "warning"]
    return {
        "schema_version": "performance-audit-v1",
        "run_directory": str(run_dir),
        "mode": "submission" if submission else "development",
        "require_official": require_official,
        "status": "blocked" if blockers else ("pass_with_warnings" if warnings else "pass"),
        "summary": summary,
        "blocker_count": len(blockers),
        "warning_count": len(warnings),
        "checks": checks,
    }
