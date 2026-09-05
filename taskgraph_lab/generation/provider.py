from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ProviderResponse:
    text: str
    provider: str
    model: str
    latency_ms: float
    usage: dict[str, Any] = field(default_factory=dict)
    raw_metadata: dict[str, Any] = field(default_factory=dict)


class TeacherProvider(Protocol):
    name: str
    model: str

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        request_id: str,
        temperature: float,
        max_output_tokens: int,
        timeout_seconds: float,
        json_output: bool = True,
    ) -> ProviderResponse: ...


class DryRunProvider:
    name = "dry_run"
    model = "deterministic-dry-run"

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        request_id: str,
        temperature: float,
        max_output_tokens: int,
        timeout_seconds: float,
        json_output: bool = True,
    ) -> ProviderResponse:
        started = time.perf_counter()
        lowered_prompt = user_prompt.lower()
        if " relative to " in lowered_prompt or " in relation to " in lowered_prompt:
            candidate = {
                "intent": "OBJECT_RELATION",
                "nodes": [
                    {
                        "id": "n1",
                        "op": "LOCATE",
                        "inputs": {"image": "$image0"},
                        "params": {
                            "target": {"category": "subject", "attributes": {}},
                        },
                    },
                    {
                        "id": "n2",
                        "op": "LOCATE",
                        "inputs": {"image": "$image0"},
                        "params": {
                            "target": {"category": "reference", "attributes": {}},
                        },
                    },
                    {
                        "id": "n3",
                        "op": "RELATION",
                        "inputs": {"subject": "$n1", "reference": "$n2"},
                        "params": {},
                    },
                ],
                "final": {
                    "sources": ["$n3"],
                    "question": "Which option matches the determined spatial relation?",
                    "answer_type": "CHOICE_SINGLE",
                },
            }
        elif "Question type:\nINTEGER" in user_prompt:
            candidate = {
                "intent": "SIMPLE_COUNT",
                "nodes": [
                    {
                        "id": "n1",
                        "op": "COUNT",
                        "inputs": {"image": "$image0"},
                        "params": {
                            "target": {"category": "object", "attributes": {}},
                            "entire": True,
                        },
                    }
                ],
                "final": {
                    "sources": ["$n1"],
                    "question": "What is this count?",
                    "answer_type": "INTEGER",
                },
            }
        else:
            if "Question type:\nBOOLEAN" in user_prompt:
                answer_type = "BOOLEAN"
                choices: str | None = None
            elif "Question type:\nFREE_FORM" in user_prompt:
                answer_type = "TEXT"
                choices = None
            elif "Question type:\nMULTIPLE_CHOICE_MULTI" in user_prompt:
                answer_type = "CHOICE_MULTI"
                choices = None
            else:
                answer_type = "CHOICE_SINGLE"
                choices = None
            candidate = {
                "intent": "COMPLEX_REASONING",
                "nodes": [
                    {
                        "id": "n1",
                        "op": "VLM_REASON",
                        "inputs": {"image": "$image0"},
                        "params": {"question": "$question", "choices": choices},
                    }
                ],
                "final": {
                    "sources": ["$n1"],
                    "question": "Which option best matches the resolved evidence?",
                    "answer_type": answer_type,
                },
            }
        payload = {
            "request_id": request_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "json_output": json_output,
        }
        return ProviderResponse(
            text=json.dumps(candidate, ensure_ascii=False),
            provider=self.name,
            model=self.model,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            raw_metadata={"request_payload": payload, "network_used": False},
        )


class OpenAICompatibleProvider:
    name = "openai_compatible"

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        api_key: str,
        thinking: str | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        if not endpoint or not model or not api_key:
            raise ValueError("OpenAI-compatible provider requires endpoint, model, and API key")
        if thinking not in {None, "enabled", "disabled"}:
            raise ValueError("provider.thinking must be enabled or disabled")
        if reasoning_effort not in {None, "low", "high", "max"}:
            raise ValueError("provider.reasoning_effort must be low, high, or max")
        self.endpoint = endpoint
        self.model = model
        self._api_key = api_key
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        request_id: str,
        temperature: float,
        max_output_tokens: int,
        timeout_seconds: float,
        json_output: bool = True,
    ) -> ProviderResponse:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_output_tokens,
        }
        if json_output:
            body["response_format"] = {"type": "json_object"}
        if self.thinking is not None:
            body["thinking"] = {"type": self.thinking}
        if self.reasoning_effort is not None:
            body["reasoning_effort"] = self.reasoning_effort
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "X-Request-ID": request_id,
            },
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                status = int(response.status)
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read(2048).decode("utf-8", errors="replace")
            raise RuntimeError(f"teacher HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"teacher request failed: {exc}") from exc
        try:
            choice = payload["choices"][0]
            message = choice["message"]
            text = message["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "OpenAI-compatible response lacks choices[0].message.content"
            ) from exc
        if not isinstance(text, str):
            raise RuntimeError("OpenAI-compatible message content must be a string")
        finish_reason = choice.get("finish_reason")
        reasoning_content = message.get("reasoning_content")
        reasoning_content_chars = (
            len(reasoning_content) if isinstance(reasoning_content, str) else None
        )
        usage = dict(payload.get("usage") or {})
        completion_details = dict(usage.get("completion_tokens_details") or {})
        reasoning_tokens = completion_details.get("reasoning_tokens")
        if not text.strip():
            response_id = payload.get("id")
            raise RuntimeError(
                "OpenAI-compatible response content is empty "
                f"(finish_reason={finish_reason!r}, reasoning_tokens={reasoning_tokens!r}, "
                f"response_id={response_id!r})"
            )
        return ProviderResponse(
            text=text,
            provider=self.name,
            model=self.model,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            usage=usage,
            raw_metadata={
                "http_status": status,
                "response_id": payload.get("id"),
                "finish_reason": finish_reason,
                "reasoning_content_chars": reasoning_content_chars,
                "thinking": self.thinking,
                "reasoning_effort": self.reasoning_effort,
            },
        )


def provider_from_config(config: dict[str, Any]) -> TeacherProvider:
    kind = str(config.get("type", "dry_run"))
    if kind == "dry_run":
        return DryRunProvider()
    if kind != "openai_compatible":
        raise ValueError(f"unsupported provider type: {kind}")
    endpoint = str(
        config.get("endpoint")
        or os.environ.get(str(config.get("endpoint_env", "TASKGRAPH_TEACHER_ENDPOINT")), "")
    )
    model = str(
        config.get("model")
        or os.environ.get(str(config.get("model_env", "TASKGRAPH_TEACHER_MODEL")), "")
    )
    api_key = str(os.environ.get(str(config.get("api_key_env", "TASKGRAPH_TEACHER_API_KEY")), ""))
    thinking = config.get("thinking")
    reasoning_effort = config.get("reasoning_effort")
    return OpenAICompatibleProvider(
        endpoint=endpoint,
        model=model,
        api_key=api_key,
        thinking=None if thinking is None else str(thinking),
        reasoning_effort=None if reasoning_effort is None else str(reasoning_effort),
    )
