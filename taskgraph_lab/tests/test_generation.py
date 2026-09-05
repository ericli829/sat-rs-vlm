from __future__ import annotations

import json
from pathlib import Path

from taskgraph_lab.datasets.base import NormalizedSample
from taskgraph_lab.generation.generate import (
    RateLimiter,
    RuntimeSettings,
    load_completed_ids,
    process_sample,
    process_sample_safely,
    run_generation,
)
from taskgraph_lab.generation.provider import (
    DryRunProvider,
    OpenAICompatibleProvider,
    ProviderResponse,
)
from taskgraph_lab.tools.summarize import summarize

FIXTURES = Path(__file__).parent / "fixtures"
SYSTEM = (Path(__file__).parents[1] / "prompts/system_prompt.txt").read_text(encoding="utf-8")
REPAIR = (Path(__file__).parents[1] / "prompts/repair_prompt.txt").read_text(encoding="utf-8")
REVIEW = (Path(__file__).parents[1] / "prompts/review_prompt.txt").read_text(encoding="utf-8")


def first_sample() -> NormalizedSample:
    return NormalizedSample.model_validate_json(
        (FIXTURES / "normalized_samples.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )


def test_dry_run_provider_records_request_and_returns_valid_graph() -> None:
    outcome = process_sample(
        first_sample(),
        provider=DryRunProvider(),
        limiter=RateLimiter(0),
        settings=RuntimeSettings(max_retries=1),
        system_prompt=SYSTEM,
        repair_template=REPAIR,
        review_template=REVIEW,
        few_shot=None,
    )
    assert outcome.destination == "valid"
    metadata = outcome.raw["provider_trace"]["response_metadata"]
    payload = metadata["request_payload"]
    assert payload["request_id"] == "s01_entire_count"
    assert metadata["network_used"] is False


def test_openai_provider_sends_explicit_thinking_settings(monkeypatch) -> None:
    captured: dict = {}

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {
                    "id": "fixture-response",
                    "choices": [
                        {
                            "message": {
                                "content": '{"intent":"OTHER"}',
                                "reasoning_content": "fixture reasoning",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"completion_tokens_details": {"reasoning_tokens": 3}},
                }
            ).encode()

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data)
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    provider = OpenAICompatibleProvider(
        endpoint="https://example.invalid/chat/completions",
        model="fixture-model",
        api_key="fixture-secret",
        thinking="enabled",
        reasoning_effort="high",
    )
    response = provider.generate(
        "system",
        "user",
        request_id="fixture",
        temperature=0.1,
        max_output_tokens=100,
        timeout_seconds=12,
    )
    assert captured["body"]["thinking"] == {"type": "enabled"}
    assert captured["body"]["reasoning_effort"] == "high"
    assert captured["timeout"] == 12
    assert response.raw_metadata["thinking"] == "enabled"
    assert response.raw_metadata["finish_reason"] == "stop"
    assert response.raw_metadata["reasoning_content_chars"] == 17


def test_openai_provider_rejects_empty_content(monkeypatch) -> None:
    class EmptyResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {
                    "id": "empty-response",
                    "choices": [
                        {
                            "message": {"content": "", "reasoning_content": "reasoning only"},
                            "finish_reason": "length",
                        }
                    ],
                    "usage": {"completion_tokens_details": {"reasoning_tokens": 8192}},
                }
            ).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: EmptyResponse())
    provider = OpenAICompatibleProvider(
        endpoint="https://example.invalid/chat/completions",
        model="fixture-model",
        api_key="fixture-secret",
        thinking="enabled",
        reasoning_effort="low",
    )
    try:
        provider.generate(
            "system",
            "user",
            request_id="fixture",
            temperature=0.1,
            max_output_tokens=8192,
            timeout_seconds=12,
        )
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("empty content should fail")
    assert "content is empty" in message
    assert "finish_reason='length'" in message
    assert "reasoning_tokens=8192" in message


def test_resume_logic(tmp_path: Path) -> None:
    path = tmp_path / "raw.jsonl"
    path.write_text(
        '{"sample_id":"a","status":"generated"}\n{"sample":{"sample_id":"b"}}\n', encoding="utf-8"
    )
    assert load_completed_ids(path) == {"a", "b"}


class RepairProvider:
    name = "repair_fixture"
    model = "fixture"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, *args, **kwargs) -> ProviderResponse:
        self.calls += 1
        if self.calls == 1:
            candidate = {
                "intent": "SIMPLE_COUNT",
                "nodes": [
                    {
                        "id": "n1",
                        "op": "COUNT",
                        "inputs": {"image": "$image0"},
                        "params": {
                            "target": {"category": "ship", "attributes": {}},
                            "entire": True,
                            "threshold": 0.2,
                        },
                        "output": "count",
                    }
                ],
                "final": {"source": "$n1", "answer_type": "INTEGER"},
            }
        else:
            candidate = {
                "intent": "SIMPLE_COUNT",
                "nodes": [
                    {
                        "id": "n1",
                        "op": "COUNT",
                        "inputs": {"image": "$image0"},
                        "params": {
                            "target": {"category": "ship", "attributes": {}},
                            "entire": True,
                        },
                        "output": "count",
                    },
                    {
                        "id": "n2",
                        "op": "MATCH_CHOICE",
                        "inputs": {"value": "$n1"},
                        "params": {"choices": "$choices"},
                        "output": "answer",
                    },
                ],
                "final": {"source": "$n2", "answer_type": "CHOICE_SINGLE"},
            }
        return ProviderResponse(
            text=json.dumps(candidate), provider=self.name, model=self.model, latency_ms=1.0
        )


def test_invalid_teacher_output_repairs_once() -> None:
    provider = RepairProvider()
    outcome = process_sample(
        first_sample(),
        provider=provider,
        limiter=RateLimiter(0),
        settings=RuntimeSettings(max_retries=1, repair_enabled=True),
        system_prompt=SYSTEM,
        repair_template=REPAIR,
        review_template=REVIEW,
        few_shot=None,
    )
    assert provider.calls == 2
    assert outcome.destination == "repaired"
    assert outcome.record["metadata"]["repair_count"] == 1


class SemanticErrorProvider:
    name = "semantic_fixture"
    model = "fixture"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, *args, **kwargs) -> ProviderResponse:
        self.calls += 1
        candidate = {
            "intent": "SIMPLE_COUNT",
            "nodes": [
                {
                    "id": "n1",
                    "op": "VLM_REASON",
                    "inputs": {"image": "$image0"},
                    "params": {"question": "$question", "choices": "$choices"},
                }
            ],
            "final": {"source": "$n1", "answer_type": "CHOICE_SINGLE"},
        }
        return ProviderResponse(
            text=json.dumps(candidate), provider=self.name, model=self.model, latency_ms=1.0
        )


def test_semantic_error_is_rejected_without_llm_repair() -> None:
    provider = SemanticErrorProvider()
    outcome = process_sample(
        first_sample(),
        provider=provider,
        limiter=RateLimiter(0),
        settings=RuntimeSettings(max_retries=1, repair_enabled=True),
        system_prompt=SYSTEM,
        repair_template=REPAIR,
        review_template=REVIEW,
        few_shot=None,
    )
    assert provider.calls == 1
    assert outcome.destination == "rejected"
    assert outcome.raw["repair_classification"] == "SEMANTIC_ERROR"
    assert outcome.record["repair_count"] == 0


def test_unexpected_sample_failure_becomes_failure_record(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise RuntimeError("fixture failure")

    monkeypatch.setattr("taskgraph_lab.generation.generate.process_sample", fail)
    outcome = process_sample_safely(
        first_sample(),
        provider=DryRunProvider(),
        limiter=RateLimiter(0),
        settings=RuntimeSettings(max_retries=1),
        system_prompt=SYSTEM,
        repair_template=REPAIR,
        review_template=REVIEW,
        few_shot=None,
    )
    assert outcome.raw["status"] == "processing_failed"
    assert outcome.raw["sample_id"] == "s01_entire_count"
    assert "fixture failure" in outcome.raw["error"]
    report = summarize([outcome.raw], [], [], [], [])
    assert report["processing_failed"] == 1


def test_full_dry_run_pipeline_and_resume(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "smoke.jsonl"
    config = {
        "provider": {"type": "dry_run"},
        "runtime": {"concurrency": 2, "requests_per_minute": 0, "max_retries": 1},
    }
    first = run_generation(
        input_path=FIXTURES / "normalized_samples.jsonl",
        raw_output=raw,
        config=config,
        system_prompt=SYSTEM,
        repair_template=REPAIR,
        review_template=REVIEW,
    )
    assert first["submitted"] == 10
    assert first["valid"] == 10
    second = run_generation(
        input_path=FIXTURES / "normalized_samples.jsonl",
        raw_output=raw,
        config=config,
        system_prompt=SYSTEM,
        repair_template=REPAIR,
        review_template=REVIEW,
    )
    assert second["skipped"] == 10
    assert len(raw.read_text(encoding="utf-8").splitlines()) == 10
