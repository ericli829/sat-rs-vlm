from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import Any

from taskgraph_lab.datasets.base import NormalizedSample


def sample_transport_payload(sample: NormalizedSample) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "question": sample.question,
        "question_type": sample.question_type.value,
        "choices": sample.choices,
        "inputs": {key: value.model_dump(mode="json") for key, value in sample.inputs.items()},
        "metadata": sample.metadata,
    }


def compose_batch_system_prompt(core_prompt: str, batch_contract: str) -> str:
    return f"{core_prompt.rstrip()}\n\n{batch_contract.strip()}\n"


def build_batch_user_prompt(samples: Sequence[NormalizedSample]) -> str:
    payload = {
        "mode": "batch",
        "samples": [sample_transport_payload(sample) for sample in samples],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def estimate_input_tokens(samples: Sequence[NormalizedSample]) -> int:
    """Conservative tokenizer-free estimate used only for adaptive chunking."""
    text = build_batch_user_prompt(samples)
    return max(1, math.ceil(len(text.encode("utf-8")) / 3.5))


def chunk_teacher_samples(
    samples: Sequence[NormalizedSample],
    *,
    max_samples: int,
    max_input_tokens: int,
) -> list[list[NormalizedSample]]:
    if max_samples < 1:
        raise ValueError("max_samples must be >= 1")
    if max_input_tokens < 1:
        raise ValueError("max_input_tokens must be >= 1")
    chunks: list[list[NormalizedSample]] = []
    current: list[NormalizedSample] = []
    for sample in samples:
        candidate = [*current, sample]
        if current and (
            len(candidate) > max_samples or estimate_input_tokens(candidate) > max_input_tokens
        ):
            chunks.append(current)
            current = [sample]
        else:
            current = candidate
        if len(current) >= max_samples:
            chunks.append(current)
            current = []
    if current:
        chunks.append(current)
    return chunks


def build_partial_repair_prompt(
    samples: Sequence[NormalizedSample],
    failures: dict[str, dict[str, Any]],
) -> str:
    payload = {
        "mode": "batch_partial_repair",
        "instruction": (
            "Return taskgraph-batch-v1 for exactly this failed subset. Repair each item "
            "independently. Do not include or regenerate any sample outside this subset."
        ),
        "samples": [
            {
                **sample_transport_payload(sample),
                "teacher_raw_item": failures[sample.sample_id].get("teacher_raw_item"),
                "validator_errors": failures[sample.sample_id].get("validator_errors", []),
                "transport_errors": failures[sample.sample_id].get("transport_errors", []),
            }
            for sample in samples
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_transport_repair_prompt(raw_response: str, expected_ids: Sequence[str]) -> str:
    payload = {
        "instruction": (
            "Repair only the taskgraph-batch-v1 JSON transport wrapper. Preserve every "
            "TaskGraph object exactly; do not add, remove, or semantically edit TaskGraph "
            "content. Return JSON only."
        ),
        "expected_sample_ids": list(expected_ids),
        "raw_response": raw_response,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


TRANSPORT_REPAIR_SYSTEM_PROMPT = """You repair only a TaskGraph batch JSON envelope.
Recover a JSON object with batch_version=\"taskgraph-batch-v1\" and results items
containing only sample_id and the original taskgraph object. Never modify TaskGraph
semantic content. Never invent a missing TaskGraph. Return JSON only."""
