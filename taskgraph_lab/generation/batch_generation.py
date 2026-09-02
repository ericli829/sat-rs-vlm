from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from taskgraph_lab import PROMPT_VERSION, SCHEMA_VERSION
from taskgraph_lab.datasets.base import NormalizedSample
from taskgraph_lab.generation.batch_parser import (
    BatchParseResult,
    BatchTransportItem,
    parse_teacher_batch,
)
from taskgraph_lab.generation.batch_prompt import (
    TRANSPORT_REPAIR_SYSTEM_PROMPT,
    build_batch_user_prompt,
    build_partial_repair_prompt,
    build_transport_repair_prompt,
    chunk_teacher_samples,
    compose_batch_system_prompt,
)
from taskgraph_lab.generation.generate import (
    ProcessOutcome,
    RateLimiter,
    RuntimeSettings,
    _accepted_record,
    _provider_call,
    _trace,
)
from taskgraph_lab.generation.provider import ProviderResponse, TeacherProvider
from taskgraph_lab.generation.repair import classify_repair
from taskgraph_lab.taskgraph.canonicalize import canonicalize_target
from taskgraph_lab.taskgraph.dsl import compile_taskgraph_to_dsl, parse_taskgraph_dsl
from taskgraph_lab.taskgraph.validator import ValidationIssue, ValidationResult, validate_candidate


@dataclass
class BatchCallRecord:
    call_id: str
    kind: str
    batch_id: str
    sample_ids: list[str]
    trace: dict[str, Any] | None
    transport: dict[str, Any] | None
    error: str | None = None


@dataclass
class BatchGenerationResult:
    outcomes: list[ProcessOutcome]
    calls: list[BatchCallRecord]
    requested_batch_size: int
    effective_chunks: list[list[str]]
    transport_parse_failures: int = 0

    def metrics(self) -> dict[str, Any]:
        usages = [call.trace.get("usage", {}) for call in self.calls if call.trace]

        def usage_total(*names: str) -> int:
            total = 0
            for usage in usages:
                for name in names:
                    value = usage.get(name)
                    if isinstance(value, (int, float)):
                        total += int(value)
                        break
            return total

        latencies = [
            float(call.trace["latency_ms"])
            for call in self.calls
            if call.trace and call.trace.get("latency_ms") is not None
        ]
        status_counts = {"valid": 0, "repaired": 0, "rejected": 0, "api_failed": 0}
        dsl_compile = 0
        dsl_roundtrip = 0
        validation_totals = {
            "schema_valid": 0,
            "type_valid": 0,
            "semantic_valid": 0,
        }
        for outcome in self.outcomes:
            if outcome.destination in status_counts:
                status_counts[outcome.destination] += 1
            elif outcome.raw.get("status") == "api_failed":
                status_counts["api_failed"] += 1
            validation = outcome.raw.get("validation")
            if isinstance(validation, dict):
                for key in validation_totals:
                    validation_totals[key] += int(bool(validation.get(key)))
            if outcome.record and outcome.destination in {"valid", "repaired"}:
                dsl_compile += int(bool(outcome.record.get("planner_dsl")))
                dsl_roundtrip += int(bool(outcome.record.get("dsl_roundtrip_valid")))
        sample_count = len(self.outcomes)
        api_calls = len(self.calls)
        prompt_tokens = usage_total("prompt_tokens", "input_tokens")
        completion_tokens = usage_total("completion_tokens", "output_tokens")
        reasoning_tokens = sum(
            int(
                ((usage.get("completion_tokens_details") or {}).get("reasoning_tokens"))
                or usage.get("reasoning_tokens")
                or 0
            )
            for usage in usages
        )
        repair_calls = sum(call.kind == "partial_repair" for call in self.calls)
        successful_transport = sum(
            bool(call.transport) and not bool(call.transport.get("catastrophic"))
            for call in self.calls
        )
        accepted = status_counts["valid"] + status_counts["repaired"]
        denominator = max(1, sample_count)
        return {
            "batch_size": self.requested_batch_size,
            "sample_count": sample_count,
            "api_calls": api_calls,
            "successful_transport_calls": successful_transport,
            "transport_parse_failures": self.transport_parse_failures,
            "initial_valid": status_counts["valid"],
            "repaired": status_counts["repaired"],
            "rejected": status_counts["rejected"],
            "api_failed": status_counts["api_failed"],
            "schema_valid_rate": validation_totals["schema_valid"] / denominator,
            "type_valid_rate": validation_totals["type_valid"] / denominator,
            "semantic_valid_rate": validation_totals["semantic_valid"] / denominator,
            "dsl_compile_success": dsl_compile,
            "dsl_roundtrip_success": dsl_roundtrip,
            "prompt_tokens_total": prompt_tokens,
            "prompt_tokens_per_sample": prompt_tokens / denominator,
            "completion_tokens_total": completion_tokens,
            "completion_tokens_per_sample": completion_tokens / denominator,
            "reasoning_tokens": reasoning_tokens,
            "latency_total_ms": sum(latencies),
            "latency_per_sample_ms": sum(latencies) / denominator,
            "repair_api_calls": repair_calls,
            "total_api_calls_including_repair": api_calls,
            "calls_per_sample": api_calls / denominator,
            "accepted_per_api_call": accepted / max(1, api_calls),
        }


@dataclass
class _Failure:
    sample: NormalizedSample
    teacher_raw_item: dict[str, Any] | None
    validator_errors: list[dict[str, Any]] = field(default_factory=list)
    transport_errors: list[dict[str, Any]] = field(default_factory=list)
    original_call_id: str = ""
    batch_id: str = ""
    batch_size: int = 0
    request_index: int | None = None
    raw_record: dict[str, Any] = field(default_factory=dict)


def _validated_candidate(
    sample: NormalizedSample,
    taskgraph: dict[str, Any],
) -> tuple[dict[str, Any] | None, ValidationResult, str | None, bool]:
    target, validation = validate_candidate(
        taskgraph,
        inputs=sample.inputs,
        question=sample.question,
        question_type=sample.question_type.value,
    )
    if target is None or not validation.valid:
        return None, validation, None, False
    try:
        canonical = canonicalize_target(target)
        planner_dsl = compile_taskgraph_to_dsl(canonical)
        roundtrip = canonicalize_target(parse_taskgraph_dsl(planner_dsl)) == canonical
    except Exception as exc:
        issue = ValidationIssue(
            stage="dsl",
            code="dsl_compile_or_parse",
            message=f"{type(exc).__name__}: {exc}",
        )
        validation = validation.model_copy(
            update={"valid": False, "errors": [*validation.errors, issue]}
        )
        return None, validation, None, False
    if not roundtrip:
        issue = ValidationIssue(
            stage="dsl",
            code="dsl_roundtrip_mismatch",
            message="compiled DSL did not round-trip to the canonical TaskGraph",
        )
        validation = validation.model_copy(
            update={"valid": False, "errors": [*validation.errors, issue]}
        )
        return None, validation, planner_dsl, False
    return canonical, validation, planner_dsl, True


def _batch_fields(
    *,
    batch_id: str,
    batch_size: int,
    original_call_id: str,
    request_index: int | None,
) -> dict[str, Any]:
    return {
        "batch_version": "taskgraph-batch-v1",
        "batch_id": batch_id,
        "batch_size": batch_size,
        "original_call_id": original_call_id,
        "request_index": request_index,
    }


def _accepted_outcome(
    failure_or_sample: _Failure | NormalizedSample,
    *,
    item: BatchTransportItem,
    canonical: dict[str, Any],
    validation: ValidationResult,
    planner_dsl: str,
    trace: dict[str, Any],
    batch_id: str,
    batch_size: int,
    call_id: str,
    destination: str,
    repair_count: int,
    transport_warnings: list[dict[str, Any]],
) -> ProcessOutcome:
    if isinstance(failure_or_sample, _Failure):
        failure = failure_or_sample
        sample = failure.sample
        teacher_raw_item = failure.teacher_raw_item
        original_call_id = failure.original_call_id
        raw = failure.raw_record
        raw["repair_teacher_raw_item"] = item.raw_item
        raw["repair_validation"] = validation.model_dump(mode="json")
    else:
        sample = failure_or_sample
        teacher_raw_item = item.raw_item
        original_call_id = call_id
        raw = {
            "sample_id": sample.sample_id,
            "status": "generated",
            "sample": sample.model_dump(mode="json"),
            "teacher_raw_item": item.raw_item,
            "candidate_text": json.dumps(item.taskgraph, ensure_ascii=False),
            "validation": validation.model_dump(mode="json"),
            "repair_classification": classify_repair(validation),
            "provider_trace": trace,
        }
    fields = _batch_fields(
        batch_id=batch_id,
        batch_size=batch_size,
        original_call_id=original_call_id,
        request_index=item.request_index,
    )
    raw.update(fields)
    record = _accepted_record(
        sample,
        canonical,
        validation,
        trace,
        repair_count=repair_count,
        planner_dsl=planner_dsl,
    )
    record.update(
        {
            "status": destination,
            "teacher_raw_item": teacher_raw_item,
            "accepted_taskgraph": canonical,
            "warnings": [
                *[warning.model_dump(mode="json") for warning in validation.warnings],
                *transport_warnings,
            ],
            "dsl_roundtrip_valid": True,
            **fields,
        }
    )
    if repair_count:
        record["repair_teacher_raw_item"] = item.raw_item
    return ProcessOutcome(raw=raw, destination=destination, record=record)


def _rejected_outcome(
    failure: _Failure,
    *,
    validation: ValidationResult | None,
    repaired_item: BatchTransportItem | None = None,
    repair_trace: dict[str, Any] | None = None,
    repair_count: int = 0,
) -> ProcessOutcome:
    record = {
        "sample_id": failure.sample.sample_id,
        "sample": failure.sample.model_dump(mode="json"),
        "status": "rejected",
        "teacher_raw_item": failure.teacher_raw_item,
        "accepted_taskgraph": None,
        "repair_count": repair_count,
        "batch_version": "taskgraph-batch-v1",
        "batch_id": failure.batch_id,
        "batch_size": failure.batch_size,
        "original_call_id": failure.original_call_id,
        "request_index": failure.request_index,
        "transport_errors": failure.transport_errors,
        "validator_errors": failure.validator_errors,
        "provider_trace": failure.raw_record.get("provider_trace"),
        "repair_provider_trace": repair_trace,
    }
    if repaired_item is not None:
        record["repair_teacher_raw_item"] = repaired_item.raw_item
    if validation is not None:
        record["validation"] = validation.model_dump(mode="json")
    failure.raw_record["final_status"] = "rejected"
    return ProcessOutcome(raw=failure.raw_record, destination="rejected", record=record)


def generate_teacher_batch(
    samples: list[NormalizedSample],
    *,
    provider: TeacherProvider,
    limiter: RateLimiter,
    settings: RuntimeSettings,
    system_prompt: str,
    batch_transport_contract: str,
    batch_size: int = 4,
    teacher_batch_max_input_tokens: int = 24000,
    teacher_batch_max_samples: int = 8,
    max_transport_retries: int = 1,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> BatchGenerationResult:
    """Generate independent TaskGraphs using a shared batch transport envelope."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if max_transport_retries < 0:
        raise ValueError("max_transport_retries must be >= 0")
    effective_max = min(batch_size, teacher_batch_max_samples)
    chunks = chunk_teacher_samples(
        samples,
        max_samples=effective_max,
        max_input_tokens=teacher_batch_max_input_tokens,
    )
    batch_system_prompt = compose_batch_system_prompt(system_prompt, batch_transport_contract)
    calls: list[BatchCallRecord] = []
    outcomes_by_id: dict[str, ProcessOutcome] = {}
    transport_parse_failures = 0

    def emit(event: str, **fields: Any) -> None:
        if progress is not None:
            progress({"event": event, **fields})

    def call_provider(
        subset: list[NormalizedSample],
        *,
        batch_id: str,
        kind: str,
        user_prompt: str,
        call_system_prompt: str,
    ) -> tuple[ProviderResponse | None, dict[str, Any] | None, str, str | None]:
        call_id = f"{batch_id}:{kind}:{len(calls) + 1}"
        try:
            response, attempt = _provider_call(
                provider,
                limiter,
                settings,
                call_system_prompt,
                user_prompt,
                call_id,
            )
            trace = _trace(response, attempt)
            emit(
                "api_call_completed",
                call_id=call_id,
                kind=kind,
                batch_id=batch_id,
                sample_ids=[sample.sample_id for sample in subset],
                latency_ms=trace["latency_ms"],
                usage=trace["usage"],
            )
            return response, trace, call_id, None
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            emit(
                "api_call_failed",
                call_id=call_id,
                kind=kind,
                batch_id=batch_id,
                sample_ids=[sample.sample_id for sample in subset],
                error=error,
            )
            return None, None, call_id, error

    def request_and_parse(
        subset: list[NormalizedSample],
        *,
        batch_id: str,
        kind: str,
        user_prompt: str,
    ) -> tuple[BatchParseResult | None, dict[str, Any] | None, str, str | None]:
        nonlocal transport_parse_failures
        response, trace, call_id, error = call_provider(
            subset,
            batch_id=batch_id,
            kind=kind,
            user_prompt=user_prompt,
            call_system_prompt=batch_system_prompt,
        )
        if response is None:
            calls.append(
                BatchCallRecord(
                    call_id, kind, batch_id, [s.sample_id for s in subset], None, None, error
                )
            )
            return None, None, call_id, error
        parsed = parse_teacher_batch(response.text, subset)
        calls.append(
            BatchCallRecord(
                call_id,
                kind,
                batch_id,
                [s.sample_id for s in subset],
                trace,
                parsed.model_dump(mode="json"),
            )
        )
        if not parsed.catastrophic:
            return parsed, trace, call_id, None
        transport_parse_failures += 1
        raw_response = response.text
        latest_call_id = call_id
        latest_trace = trace
        for retry_index in range(max_transport_retries):
            repair_kind = f"transport_repair_{retry_index + 1}"
            repair_response, repair_trace, repair_call_id, repair_error = call_provider(
                subset,
                batch_id=batch_id,
                kind=repair_kind,
                user_prompt=build_transport_repair_prompt(
                    raw_response, [sample.sample_id for sample in subset]
                ),
                call_system_prompt=TRANSPORT_REPAIR_SYSTEM_PROMPT,
            )
            latest_call_id = repair_call_id
            latest_trace = repair_trace
            if repair_response is None:
                calls.append(
                    BatchCallRecord(
                        repair_call_id,
                        repair_kind,
                        batch_id,
                        [s.sample_id for s in subset],
                        None,
                        None,
                        repair_error,
                    )
                )
                continue
            repaired_parse = parse_teacher_batch(repair_response.text, subset)
            calls.append(
                BatchCallRecord(
                    repair_call_id,
                    repair_kind,
                    batch_id,
                    [s.sample_id for s in subset],
                    repair_trace,
                    repaired_parse.model_dump(mode="json"),
                )
            )
            if not repaired_parse.catastrophic:
                return repaired_parse, repair_trace, repair_call_id, None
            transport_parse_failures += 1
            raw_response = repair_response.text
        return parsed, latest_trace, latest_call_id, "catastrophic batch transport failure"

    def api_failure_outcomes(
        subset: list[NormalizedSample], batch_id: str, call_id: str, error: str
    ) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        for sample in subset:
            outcomes_by_id[sample.sample_id] = ProcessOutcome(
                raw={
                    "sample_id": sample.sample_id,
                    "status": "api_failed",
                    "error": error,
                    "provider": provider.name,
                    "model": provider.model,
                    "timestamp": timestamp,
                    "prompt_version": PROMPT_VERSION,
                    "schema_version": SCHEMA_VERSION,
                    **_batch_fields(
                        batch_id=batch_id,
                        batch_size=len(subset),
                        original_call_id=call_id,
                        request_index=None,
                    ),
                }
            )

    def process_parsed(
        subset: list[NormalizedSample],
        parsed: BatchParseResult,
        trace: dict[str, Any],
        call_id: str,
        batch_id: str,
    ) -> None:
        items = {item.sample_id: item for item in parsed.valid_results}
        transport_errors = [issue.model_dump(mode="json") for issue in parsed.transport_errors]
        failures: dict[str, _Failure] = {}
        for sample in subset:
            item = items.get(sample.sample_id)
            raw_record = {
                "sample_id": sample.sample_id,
                "status": "generated",
                "sample": sample.model_dump(mode="json"),
                "teacher_raw_item": item.raw_item if item else None,
                "candidate_text": (
                    json.dumps(item.taskgraph, ensure_ascii=False) if item else None
                ),
                "provider_trace": trace,
                "transport_errors": transport_errors,
                **_batch_fields(
                    batch_id=batch_id,
                    batch_size=len(subset),
                    original_call_id=call_id,
                    request_index=item.request_index if item else None,
                ),
            }
            if item is None:
                failure = _Failure(
                    sample=sample,
                    teacher_raw_item=None,
                    transport_errors=transport_errors,
                    original_call_id=call_id,
                    batch_id=batch_id,
                    batch_size=len(subset),
                    raw_record=raw_record,
                )
                failures[sample.sample_id] = failure
                continue
            canonical, validation, planner_dsl, roundtrip = _validated_candidate(
                sample, item.taskgraph
            )
            raw_record["validation"] = validation.model_dump(mode="json")
            repair_classification = classify_repair(validation)
            raw_record["repair_classification"] = repair_classification
            if canonical is not None and planner_dsl is not None and roundtrip:
                outcomes_by_id[sample.sample_id] = _accepted_outcome(
                    sample,
                    item=item,
                    canonical=canonical,
                    validation=validation,
                    planner_dsl=planner_dsl,
                    trace=trace,
                    batch_id=batch_id,
                    batch_size=len(subset),
                    call_id=call_id,
                    destination="valid",
                    repair_count=0,
                    transport_warnings=transport_errors,
                )
                continue
            failure = _Failure(
                sample=sample,
                teacher_raw_item=item.raw_item,
                validator_errors=[issue.model_dump(mode="json") for issue in validation.errors],
                transport_errors=transport_errors,
                original_call_id=call_id,
                batch_id=batch_id,
                batch_size=len(subset),
                request_index=item.request_index,
                raw_record=raw_record,
            )
            if repair_classification == "LLM_REPAIRABLE":
                failures[sample.sample_id] = failure
            else:
                outcomes_by_id[sample.sample_id] = _rejected_outcome(failure, validation=validation)

        if not failures:
            return
        failed_samples = [sample for sample in subset if sample.sample_id in failures]
        failure_payloads = {
            sample_id: {
                "teacher_raw_item": failure.teacher_raw_item,
                "validator_errors": failure.validator_errors,
                "transport_errors": failure.transport_errors,
            }
            for sample_id, failure in failures.items()
        }
        repair_batch_id = f"{batch_id}:partial-repair"
        repaired_parse, repair_trace, repair_call_id, repair_error = request_and_parse(
            failed_samples,
            batch_id=repair_batch_id,
            kind="partial_repair",
            user_prompt=build_partial_repair_prompt(failed_samples, failure_payloads),
        )
        if repaired_parse is None or repair_trace is None or repaired_parse.catastrophic:
            for failure in failures.values():
                if repair_error:
                    failure.transport_errors.append(
                        {"code": "repair_transport_failure", "message": repair_error}
                    )
                outcomes_by_id[failure.sample.sample_id] = _rejected_outcome(
                    failure, validation=None, repair_count=1
                )
            return
        repaired_items = {item.sample_id: item for item in repaired_parse.valid_results}
        for sample in failed_samples:
            failure = failures[sample.sample_id]
            repaired_item = repaired_items.get(sample.sample_id)
            if repaired_item is None:
                outcomes_by_id[sample.sample_id] = _rejected_outcome(
                    failure,
                    validation=None,
                    repair_trace=repair_trace,
                    repair_count=1,
                )
                continue
            canonical, validation, planner_dsl, roundtrip = _validated_candidate(
                sample, repaired_item.taskgraph
            )
            if canonical is not None and planner_dsl is not None and roundtrip:
                outcomes_by_id[sample.sample_id] = _accepted_outcome(
                    failure,
                    item=repaired_item,
                    canonical=canonical,
                    validation=validation,
                    planner_dsl=planner_dsl,
                    trace=repair_trace,
                    batch_id=failure.batch_id,
                    batch_size=failure.batch_size,
                    call_id=repair_call_id,
                    destination="repaired",
                    repair_count=1,
                    transport_warnings=[
                        issue.model_dump(mode="json") for issue in repaired_parse.transport_errors
                    ],
                )
            else:
                outcomes_by_id[sample.sample_id] = _rejected_outcome(
                    failure,
                    validation=validation,
                    repaired_item=repaired_item,
                    repair_trace=repair_trace,
                    repair_count=1,
                )

    def process_initial(subset: list[NormalizedSample], label: str) -> None:
        batch_id = f"batch-{label}-{uuid4().hex[:8]}"
        parsed, trace, call_id, error = request_and_parse(
            subset,
            batch_id=batch_id,
            kind="teacher",
            user_prompt=build_batch_user_prompt(subset),
        )
        if parsed is None:
            api_failure_outcomes(subset, batch_id, call_id, error or "batch API failure")
            return
        if parsed.catastrophic:
            if len(subset) > 1:
                midpoint = len(subset) // 2
                process_initial(subset[:midpoint], f"{label}a")
                process_initial(subset[midpoint:], f"{label}b")
                return
            failure = _Failure(
                sample=subset[0],
                teacher_raw_item=None,
                transport_errors=[
                    issue.model_dump(mode="json") for issue in parsed.transport_errors
                ],
                original_call_id=call_id,
                batch_id=batch_id,
                batch_size=1,
                raw_record={
                    "sample_id": subset[0].sample_id,
                    "status": "generated",
                    "sample": subset[0].model_dump(mode="json"),
                    "transport_errors": [
                        issue.model_dump(mode="json") for issue in parsed.transport_errors
                    ],
                    "provider_trace": trace,
                },
            )
            outcomes_by_id[subset[0].sample_id] = _rejected_outcome(failure, validation=None)
            return
        assert trace is not None
        process_parsed(subset, parsed, trace, call_id, batch_id)

    for chunk_index, chunk in enumerate(chunks, 1):
        emit(
            "chunk_started",
            chunk_index=chunk_index,
            chunk_count=len(chunks),
            sample_ids=[sample.sample_id for sample in chunk],
        )
        process_initial(chunk, str(chunk_index))
        emit(
            "chunk_completed",
            chunk_index=chunk_index,
            chunk_count=len(chunks),
            completed_samples=sum(sample.sample_id in outcomes_by_id for sample in samples),
            total_samples=len(samples),
        )

    outcomes = [outcomes_by_id[sample.sample_id] for sample in samples]
    return BatchGenerationResult(
        outcomes=outcomes,
        calls=calls,
        requested_batch_size=batch_size,
        effective_chunks=[[sample.sample_id for sample in chunk] for chunk in chunks],
        transport_parse_failures=transport_parse_failures,
    )
