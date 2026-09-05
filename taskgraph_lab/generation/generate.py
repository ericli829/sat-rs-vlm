from __future__ import annotations

import argparse
import json
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from taskgraph_lab import PROMPT_VERSION, SCHEMA_VERSION
from taskgraph_lab.datasets.base import NormalizedSample
from taskgraph_lab.generation.provider import (
    ProviderResponse,
    TeacherProvider,
    provider_from_config,
)
from taskgraph_lab.generation.repair import classify_repair, render_repair_prompt
from taskgraph_lab.generation.review import parse_review, render_review_prompt
from taskgraph_lab.taskgraph.canonicalize import canonicalize_target
from taskgraph_lab.taskgraph.dsl import DSL_VERSION, compile_taskgraph_to_dsl
from taskgraph_lab.taskgraph.validator import ValidationResult, validate_candidate

LAB_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RuntimeSettings:
    concurrency: int = 1
    requests_per_minute: float = 30.0
    timeout_seconds: float = 60.0
    max_retries: int = 3
    backoff_base_seconds: float = 1.0
    temperature: float = 0.1
    max_output_tokens: int = 4096
    repair_enabled: bool = True
    semantic_review: bool = False
    emit_planner_dsl: bool = True

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> RuntimeSettings:
        unknown = sorted(set(value) - set(cls.__annotations__))
        if unknown:
            raise ValueError(f"unknown runtime config keys: {unknown}")
        settings = cls(**value)
        if settings.concurrency < 1:
            raise ValueError("runtime.concurrency must be >= 1")
        if settings.requests_per_minute < 0:
            raise ValueError("runtime.requests_per_minute must be >= 0")
        if settings.timeout_seconds <= 0 or settings.max_retries < 1:
            raise ValueError("timeout_seconds must be > 0 and max_retries must be >= 1")
        if not 0.0 <= settings.temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        return settings


class RateLimiter:
    def __init__(self, requests_per_minute: float) -> None:
        self.limit = requests_per_minute
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        if self.limit <= 0:
            return
        while True:
            delay = 0.0
            with self._lock:
                now = time.monotonic()
                while self._calls and now - self._calls[0] >= 60.0:
                    self._calls.popleft()
                if len(self._calls) < self.limit:
                    self._calls.append(now)
                    return
                delay = max(0.01, 60.0 - (now - self._calls[0]))
            time.sleep(delay)


@dataclass
class ProcessOutcome:
    raw: dict[str, Any]
    destination: str | None = None
    record: dict[str, Any] | None = None
    review: dict[str, Any] | None = None


def load_completed_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid resume JSONL at {path}:{line_number}: {exc}") from exc
            sample_id = payload.get("sample_id")
            if not sample_id and isinstance(payload.get("sample"), dict):
                sample_id = payload["sample"].get("sample_id")
            if sample_id:
                completed.add(str(sample_id))
    return completed


def iter_samples(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield NormalizedSample.model_validate_json(line)
            except Exception as exc:
                raise ValueError(
                    f"invalid normalized sample at {path}:{line_number}: {exc}"
                ) from exc


def build_user_prompt(sample: NormalizedSample, few_shot: str | None = None) -> str:
    payload = {
        "question": sample.question,
        "question_type": sample.question_type.value,
        "choices": sample.choices,
        "inputs": {key: value.model_dump(mode="json") for key, value in sample.inputs.items()},
        "metadata": sample.metadata,
    }
    text = (
        "Compile the following sample into TaskGraph v1.1.\n\n"
        f"Question:\n{payload['question']}\n\n"
        f"Question type:\n{payload['question_type']}\n\n"
        f"Choices:\n{json.dumps(payload['choices'], ensure_ascii=False)}\n\n"
        f"Inputs:\n{json.dumps(payload['inputs'], ensure_ascii=False, indent=2)}\n\n"
        f"Metadata:\n{json.dumps(payload['metadata'], ensure_ascii=False, indent=2)}"
    )
    return f"{few_shot.rstrip()}\n\n{text}" if few_shot else text


def _provider_call(
    provider: TeacherProvider,
    limiter: RateLimiter,
    settings: RuntimeSettings,
    system_prompt: str,
    user_prompt: str,
    request_id: str,
) -> tuple[ProviderResponse, int]:
    last_error: Exception | None = None
    for attempt in range(1, settings.max_retries + 1):
        limiter.acquire()
        try:
            response = provider.generate(
                system_prompt,
                user_prompt,
                request_id=request_id,
                temperature=settings.temperature,
                max_output_tokens=settings.max_output_tokens,
                timeout_seconds=settings.timeout_seconds,
                json_output=True,
            )
            return response, attempt
        except Exception as exc:
            last_error = exc
            if attempt < settings.max_retries:
                time.sleep(settings.backoff_base_seconds * (2 ** (attempt - 1)))
    assert last_error is not None
    raise last_error


def _trace(response: ProviderResponse, attempt: int) -> dict[str, Any]:
    return {
        "provider": response.provider,
        "model": response.model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "latency_ms": response.latency_ms,
        "usage": response.usage,
        "attempt": attempt,
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "response_metadata": response.raw_metadata,
    }


def _accepted_record(
    sample: NormalizedSample,
    target: dict[str, Any],
    validation: ValidationResult,
    trace: dict[str, Any],
    *,
    repair_count: int,
    planner_dsl: str | None,
) -> dict[str, Any]:
    metadata = {
        **sample.metadata,
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "teacher_provider": trace["provider"],
        "teacher_model": trace["model"],
        "repair_count": repair_count,
        "normalized_fields": validation.normalized_fields,
    }
    if planner_dsl is not None:
        metadata["planner_dsl_version"] = DSL_VERSION
    record = {
        "sample_id": sample.sample_id,
        "input": {
            "question": sample.question,
            "question_type": sample.question_type.value,
            "choices": sample.choices,
            "inputs": {key: value.model_dump(mode="json") for key, value in sample.inputs.items()},
        },
        "target": target,
        "metadata": metadata,
        "validation": validation.model_dump(mode="json"),
        "provider_trace": trace,
    }
    if planner_dsl is not None:
        record["planner_dsl"] = planner_dsl
    return record


def process_sample(
    sample: NormalizedSample,
    *,
    provider: TeacherProvider,
    limiter: RateLimiter,
    settings: RuntimeSettings,
    system_prompt: str,
    repair_template: str,
    review_template: str,
    few_shot: str | None,
) -> ProcessOutcome:
    user_prompt = build_user_prompt(sample, few_shot)
    try:
        response, attempt = _provider_call(
            provider, limiter, settings, system_prompt, user_prompt, sample.sample_id
        )
    except Exception as exc:
        return ProcessOutcome(
            raw={
                "sample_id": sample.sample_id,
                "status": "api_failed",
                "error": f"{type(exc).__name__}: {exc}",
                "attempts": settings.max_retries,
                "provider": provider.name,
                "model": provider.model,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "prompt_version": PROMPT_VERSION,
                "schema_version": SCHEMA_VERSION,
            }
        )

    trace = _trace(response, attempt)
    raw = {
        "sample_id": sample.sample_id,
        "status": "generated",
        "sample": sample.model_dump(mode="json"),
        "candidate_text": response.text,
        "provider_trace": trace,
    }
    target, validation = validate_candidate(
        response.text,
        inputs=sample.inputs,
        question=sample.question,
        question_type=sample.question_type.value,
    )
    raw["validation"] = validation.model_dump(mode="json")
    repair_classification = classify_repair(validation)
    raw["repair_classification"] = repair_classification
    repaired_text: str | None = None
    repair_trace: dict[str, Any] | None = None
    repair_count = 0
    destination = "valid"
    if target is None or not validation.valid:
        if settings.repair_enabled and repair_classification == "LLM_REPAIRABLE":
            repair_count = 1
            repair_prompt = render_repair_prompt(repair_template, sample, response.text, validation)
            try:
                repair_response, repair_attempt = _provider_call(
                    provider,
                    limiter,
                    settings,
                    system_prompt,
                    repair_prompt,
                    f"{sample.sample_id}:repair",
                )
                repaired_text = repair_response.text
                repair_trace = _trace(repair_response, repair_attempt)
                target, validation = validate_candidate(
                    repaired_text,
                    inputs=sample.inputs,
                    question=sample.question,
                    question_type=sample.question_type.value,
                )
            except Exception as exc:
                raw["repair_api_error"] = f"{type(exc).__name__}: {exc}"
        if target is None or not validation.valid:
            return ProcessOutcome(
                raw=raw,
                destination="rejected",
                record={
                    "sample_id": sample.sample_id,
                    "sample": sample.model_dump(mode="json"),
                    "invalid_candidate_text": response.text,
                    "repaired_candidate_text": repaired_text,
                    "repair_count": repair_count,
                    "repair_classification": repair_classification,
                    "validation": validation.model_dump(mode="json"),
                    "provider_trace": trace,
                    "repair_provider_trace": repair_trace,
                },
            )
        destination = "repaired"
        trace = repair_trace or trace

    assert target is not None
    canonical = canonicalize_target(target)
    planner_dsl = compile_taskgraph_to_dsl(canonical) if settings.emit_planner_dsl else None
    accepted = _accepted_record(
        sample,
        canonical,
        validation,
        trace,
        repair_count=repair_count,
        planner_dsl=planner_dsl,
    )
    review_record: dict[str, Any] | None = None
    if settings.semantic_review:
        review_prompt = render_review_prompt(review_template, sample, canonical)
        try:
            review_response, review_attempt = _provider_call(
                provider,
                limiter,
                settings,
                review_template,
                review_prompt,
                f"{sample.sample_id}:review",
            )
            review = parse_review(review_response.text)
            review_record = {
                "sample_id": sample.sample_id,
                "status": "reviewed",
                "review": review.model_dump(mode="json"),
                "provider_trace": _trace(review_response, review_attempt),
            }
        except Exception as exc:
            review_record = {
                "sample_id": sample.sample_id,
                "status": "review_failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
    return ProcessOutcome(raw=raw, destination=destination, record=accepted, review=review_record)


def process_sample_safely(
    sample: NormalizedSample,
    *,
    provider: TeacherProvider,
    limiter: RateLimiter,
    settings: RuntimeSettings,
    system_prompt: str,
    repair_template: str,
    review_template: str,
    few_shot: str | None,
) -> ProcessOutcome:
    """Keep one unexpected worker failure from terminating the whole batch."""
    try:
        return process_sample(
            sample,
            provider=provider,
            limiter=limiter,
            settings=settings,
            system_prompt=system_prompt,
            repair_template=repair_template,
            review_template=review_template,
            few_shot=few_shot,
        )
    except Exception as exc:
        return ProcessOutcome(
            raw={
                "sample_id": sample.sample_id,
                "status": "processing_failed",
                "error": f"{type(exc).__name__}: {exc}",
                "provider": provider.name,
                "model": provider.model,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "prompt_version": PROMPT_VERSION,
                "schema_version": SCHEMA_VERSION,
            }
        )


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()


def _default_destination(raw: Path, kind: str) -> Path:
    if raw.parent.name == "raw":
        return raw.parent.parent / kind / raw.name
    return raw.with_name(f"{raw.stem}.{kind}{raw.suffix}")


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("generation config must be a YAML mapping")
    return payload


def run_generation(
    *,
    input_path: Path,
    raw_output: Path,
    config: dict[str, Any],
    system_prompt: str,
    repair_template: str,
    review_template: str,
    few_shot: str | None = None,
    semantic_review: bool | None = None,
) -> dict[str, Any]:
    settings = RuntimeSettings.from_mapping(dict(config.get("runtime") or {}))
    if semantic_review is not None:
        settings = RuntimeSettings(**{**settings.__dict__, "semantic_review": semantic_review})
    provider = provider_from_config(dict(config.get("provider") or {}))
    limiter = RateLimiter(settings.requests_per_minute)
    completed = load_completed_ids(raw_output)
    paths = {
        kind: _default_destination(raw_output, kind)
        for kind in ("valid", "repaired", "rejected", "reviews")
    }
    counters = {
        "submitted": 0,
        "skipped": 0,
        "generated": 0,
        "api_failed": 0,
        "processing_failed": 0,
        "valid": 0,
        "repaired": 0,
        "rejected": 0,
        "reviewed": 0,
    }

    def consume(outcome: ProcessOutcome) -> None:
        append_jsonl(raw_output, outcome.raw)
        status = str(outcome.raw.get("status"))
        if status in {"api_failed", "processing_failed"}:
            counters[status] += 1
        else:
            counters["generated"] += 1
        if outcome.destination and outcome.record is not None:
            append_jsonl(paths[outcome.destination], outcome.record)
            counters[outcome.destination] += 1
        if outcome.review is not None:
            append_jsonl(paths["reviews"], outcome.review)
            counters["reviewed"] += 1

    executor = ThreadPoolExecutor(max_workers=settings.concurrency)
    pending: set[Future[ProcessOutcome]] = set()
    interrupted = False
    try:
        for sample in iter_samples(input_path):
            if sample.sample_id in completed:
                counters["skipped"] += 1
                continue
            future = executor.submit(
                process_sample_safely,
                sample,
                provider=provider,
                limiter=limiter,
                settings=settings,
                system_prompt=system_prompt,
                repair_template=repair_template,
                review_template=review_template,
                few_shot=few_shot,
            )
            pending.add(future)
            counters["submitted"] += 1
            if len(pending) >= settings.concurrency * 2:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for item in done:
                    consume(item.result())
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for item in done:
                consume(item.result())
    except KeyboardInterrupt:
        interrupted = True
        for future in pending:
            future.cancel()
    finally:
        executor.shutdown(wait=not interrupted, cancel_futures=interrupted)
    return {
        "raw_output": str(raw_output.resolve()),
        "outputs": {key: str(value.resolve()) for key, value in paths.items()},
        "interrupted": interrupted,
        **counters,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate and validate TaskGraph teacher labels")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--system-prompt", type=Path, default=LAB_ROOT / "prompts/system_prompt.txt"
    )
    parser.add_argument(
        "--repair-prompt", type=Path, default=LAB_ROOT / "prompts/repair_prompt.txt"
    )
    parser.add_argument(
        "--review-prompt", type=Path, default=LAB_ROOT / "prompts/review_prompt.txt"
    )
    parser.add_argument("--few-shot-file", type=Path)
    parser.add_argument("--semantic-review", action="store_true", default=None)
    args = parser.parse_args()
    config = _load_config(args.config)
    report = run_generation(
        input_path=args.input,
        raw_output=args.output,
        config=config,
        system_prompt=args.system_prompt.read_text(encoding="utf-8"),
        repair_template=args.repair_prompt.read_text(encoding="utf-8"),
        review_template=args.review_prompt.read_text(encoding="utf-8"),
        few_shot=args.few_shot_file.read_text(encoding="utf-8") if args.few_shot_file else None,
        semantic_review=args.semantic_review,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 130 if report["interrupted"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
