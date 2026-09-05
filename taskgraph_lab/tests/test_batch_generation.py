from __future__ import annotations

import json
from pathlib import Path

from taskgraph_lab.datasets.base import NormalizedSample
from taskgraph_lab.generation.batch_generation import generate_teacher_batch
from taskgraph_lab.generation.generate import RateLimiter, RuntimeSettings
from taskgraph_lab.generation.provider import ProviderResponse

ROOT = Path(__file__).parents[1]
SYSTEM = (ROOT / "prompts/system_prompt.txt").read_text(encoding="utf-8")
BATCH_CONTRACT = (ROOT / "prompts/batch_transport_contract.txt").read_text(encoding="utf-8")


def sample(sample_id: str, question: str = "What type of land is shown?") -> NormalizedSample:
    return NormalizedSample.model_validate(
        {
            "sample_id": sample_id,
            "question": question,
            "question_type": "MULTIPLE_CHOICE_SINGLE",
            "choices": ["(A) Urban", "(B) Rural"],
            "inputs": {
                "image0": {"type": "image", "uri_or_key": f"{sample_id}.png"},
            },
            "metadata": {"dataset": "fixture"},
        }
    )


def classify_graph(*, role: str = "input", answer_type: str = "CHOICE_SINGLE") -> dict:
    return {
        "intent": "REGIONAL_CLASSIFICATION",
        "nodes": [
            {
                "id": "n1",
                "op": "CLASSIFY",
                "inputs": {role: "$image0"},
                "params": {"label_space": ["urban", "rural"]},
            }
        ],
        "final": {"sources": ["$n1"], "answer_type": answer_type},
    }


def batch_response(items: list[tuple[str, dict]]) -> str:
    return json.dumps(
        {
            "batch_version": "taskgraph-batch-v1",
            "results": [{"sample_id": sample_id, "taskgraph": graph} for sample_id, graph in items],
        }
    )


class ScriptedProvider:
    name = "scripted"
    model = "fixture"

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def generate(self, system_prompt, user_prompt, **kwargs) -> ProviderResponse:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "request_id": kwargs["request_id"],
            }
        )
        return ProviderResponse(
            text=self.responses.pop(0),
            provider=self.name,
            model=self.model,
            latency_ms=2.0,
            usage={"prompt_tokens": 100, "completion_tokens": 20},
        )


def run(samples: list[NormalizedSample], provider: ScriptedProvider, *, size: int = 4):
    return generate_teacher_batch(
        samples,
        provider=provider,
        limiter=RateLimiter(0),
        settings=RuntimeSettings(max_retries=1),
        system_prompt=SYSTEM,
        batch_transport_contract=BATCH_CONTRACT,
        batch_size=size,
        max_transport_retries=1,
    )


def test_batch_valid_structured_residual_route_and_complex_multi_source() -> None:
    samples = [
        sample("structured"),
        sample("residual", "What colors are visible on the selected house?"),
        sample("route", "What is the best route between the two buildings?"),
        sample("complex", "Why are ponds clustered beside the houses?"),
    ]
    residual = {
        "intent": "ATTRIBUTE_QUERY",
        "nodes": [
            {
                "id": "n1",
                "op": "LOCATE",
                "inputs": {"image": "$image0"},
                "params": {"target": {"category": "house", "attributes": {}}},
            }
        ],
        "final": {
            "sources": ["$n1"],
            "question": "What colors are visible on the selected house?",
            "answer_type": "CHOICE_SINGLE",
        },
    }
    route = {
        "intent": "ROUTE_PLANNING",
        "nodes": [
            {
                "id": "n1",
                "op": "LOCATE",
                "inputs": {"image": "$image0"},
                "params": {"target": {"category": "start building", "attributes": {}}},
            },
            {
                "id": "n2",
                "op": "LOCATE",
                "inputs": {"image": "$image0"},
                "params": {"target": {"category": "goal building", "attributes": {}}},
            },
            {
                "id": "n3",
                "op": "BUILD_ROUTE_CONTEXT",
                "inputs": {"image": "$image0", "start": "$n1", "goal": "$n2"},
                "params": {},
            },
        ],
        "final": {
            "sources": ["$n3"],
            "question": "Which option describes the best route between the selected landmarks?",
            "answer_type": "CHOICE_SINGLE",
        },
    }
    complex_graph = {
        "intent": "COMPLEX_REASONING",
        "nodes": [
            {
                "id": "n1",
                "op": "LOCATE",
                "inputs": {"image": "$image0"},
                "params": {"target": {"category": "pond", "attributes": {}}},
            },
            {
                "id": "n2",
                "op": "LOCATE",
                "inputs": {"image": "$image0"},
                "params": {"target": {"category": "house", "attributes": {}}},
            },
        ],
        "final": {
            "sources": ["$n1", "$n2"],
            "question": "Why are the selected ponds clustered beside this residential area?",
            "answer_type": "CHOICE_SINGLE",
        },
    }
    provider = ScriptedProvider(
        [
            batch_response(
                [
                    ("structured", classify_graph()),
                    ("residual", residual),
                    ("route", route),
                    ("complex", complex_graph),
                ]
            )
        ]
    )
    result = run(samples, provider)
    assert [outcome.destination for outcome in result.outcomes] == ["valid"] * 4
    assert all(outcome.record["dsl_roundtrip_valid"] for outcome in result.outcomes)
    assert all(outcome.record["teacher_raw_item"] for outcome in result.outcomes)


def test_one_invalid_item_repairs_only_failed_subset() -> None:
    samples = [sample("good"), sample("bad")]
    provider = ScriptedProvider(
        [
            batch_response([("good", classify_graph()), ("bad", classify_graph(role="region"))]),
            batch_response([("bad", classify_graph())]),
        ]
    )
    result = run(samples, provider)
    assert [outcome.destination for outcome in result.outcomes] == ["valid", "repaired"]
    assert len(provider.calls) == 2
    repair_payload = json.loads(provider.calls[1]["user_prompt"])
    assert repair_payload["mode"] == "batch_partial_repair"
    assert [item["sample_id"] for item in repair_payload["samples"]] == ["bad"]
    assert result.outcomes[0].record["metadata"]["repair_count"] == 0
    assert result.outcomes[1].record["metadata"]["repair_count"] == 1


def test_teacher_owns_choice_cardinality_when_legacy_source_label_conflicts() -> None:
    samples = [sample("good"), sample("conflict", "Select all land use types shown.")]
    multi = {
        "intent": "MULTILABEL_CLASSIFICATION",
        "nodes": [
            {
                "id": "n1",
                "op": "MULTILABEL_CLASSIFY",
                "inputs": {"input": "$image0"},
                "params": {"label_space": ["urban", "rural"]},
            }
        ],
        "final": {"sources": ["$n1"], "answer_type": "CHOICE_MULTI"},
    }
    provider = ScriptedProvider([batch_response([("good", classify_graph()), ("conflict", multi)])])
    result = run(samples, provider)
    assert result.outcomes[0].destination == "valid"
    assert result.outcomes[1].destination == "valid"
    assert len(provider.calls) == 1
    assert result.outcomes[1].record["accepted_taskgraph"]["final"]["answer_type"] == (
        "CHOICE_MULTI"
    )


def test_cross_sample_reference_is_rejected_without_polluting_peer() -> None:
    bad = classify_graph()
    bad["nodes"][0]["inputs"] = {"input": "$n9"}
    provider = ScriptedProvider([batch_response([("good", classify_graph()), ("bad", bad)])])
    result = run([sample("good"), sample("bad")], provider)
    assert result.outcomes[0].destination == "valid"
    assert result.outcomes[1].destination == "rejected"
    assert "missing_node_ref" in {
        error["code"] for error in result.outcomes[1].record["validation"]["errors"]
    }


def test_catastrophic_transport_repairs_wrapper_before_split() -> None:
    provider = ScriptedProvider(
        [
            "not json",
            batch_response([("a", classify_graph()), ("b", classify_graph())]),
        ]
    )
    result = run([sample("a"), sample("b")], provider)
    assert [outcome.destination for outcome in result.outcomes] == ["valid", "valid"]
    assert [call.kind for call in result.calls] == ["teacher", "transport_repair_1"]
    assert result.transport_parse_failures == 1


def test_progress_callback_reports_chunks_and_api_calls() -> None:
    events: list[dict] = []
    provider = ScriptedProvider([batch_response([("a", classify_graph())])])
    result = generate_teacher_batch(
        [sample("a")],
        provider=provider,
        limiter=RateLimiter(0),
        settings=RuntimeSettings(max_retries=1),
        system_prompt=SYSTEM,
        batch_transport_contract=BATCH_CONTRACT,
        batch_size=4,
        max_transport_retries=1,
        progress=events.append,
    )
    assert result.outcomes[0].destination == "valid"
    assert [event["event"] for event in events] == [
        "chunk_started",
        "api_call_completed",
        "chunk_completed",
    ]
