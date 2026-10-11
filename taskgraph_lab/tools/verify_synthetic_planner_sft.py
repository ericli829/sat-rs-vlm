"""Check actual Planner SFT encoding and masks on CPU, without model weights."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sat_rs_vlm.data.qwen3vl_dataset import Qwen3VLDataset
from sat_rs_vlm.taskgraph.providers import PlannerRequest, Qwen3VLPlannerProvider
from sat_rs_vlm.taskgraph.runtime_types import ImageRef
from taskgraph_lab.taskgraph.canonicalize import canonicalize_target
from taskgraph_lab.taskgraph.dsl import (
    CanonicalDSLPrefixGrammar,
    compile_taskgraph_to_dsl,
    parse_taskgraph_dsl,
)
from taskgraph_lab.taskgraph.validator import validate_candidate
from taskgraph_lab.tools.mix_planner_synthetic import SYNTHETIC_SOURCE, _read_rows
from taskgraph_lab.training.planner_collator import PlannerTextDataCollator
from taskgraph_lab.training.planner_dataset import file_sha256


class TextTokenizerProcessor:
    """Expose local production tokenizer methods through the collator interface."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        return str(self.tokenizer.apply_chat_template(messages, **kwargs))

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        kwargs.pop("images", None)
        kwargs.pop("videos", None)
        return dict(self.tokenizer(**kwargs))


def validate_message_contract(row: dict[str, Any]) -> None:
    messages = row["messages"]
    payload = json.loads(messages[1]["content"])
    if payload.get("question_type") != "FREE_FORM" or payload.get("choices") is not None:
        raise ValueError(f"{row['id']}: bbox planning requires FREE_FORM and choices=null")
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError(f"{row['id']}: input image bindings are missing")
    request = PlannerRequest(
        question=payload["question"],
        question_type=payload["question_type"],
        choices=(),
        inputs={key: ImageRef(value["uri_or_key"]) for key, value in inputs.items()},
        sample_id=row["id"],
    )
    expected = Qwen3VLPlannerProvider._messages(request, messages[0]["content"])
    if expected != messages[:2]:
        raise ValueError(f"{row['id']}: messages differ from production Planner serialization")
    dsl = messages[2]["content"]
    graph = parse_taskgraph_dsl(dsl)
    if graph.final.answer_type.value != "BBOX":
        raise ValueError(f"{row['id']}: synthetic localization target must end with BBOX")
    _, validation = validate_candidate(
        graph, inputs=inputs, question=payload["question"], question_type=payload["question_type"]
    )
    if not validation.valid:
        raise ValueError(f"{row['id']}: current graph validation failed")
    canonical = canonicalize_target(graph)
    if canonicalize_target(parse_taskgraph_dsl(compile_taskgraph_to_dsl(graph))) != canonical:
        raise ValueError(f"{row['id']}: canonical DSL round trip changed the graph")
    if not CanonicalDSLPrefixGrammar(inputs).accepts(dsl):
        raise ValueError(f"{row['id']}: target is rejected by constrained decoding grammar")


def inspect_batch(
    rows: list[dict[str, Any]], processor: Any, *, max_seq_length: int
) -> dict[str, Any]:
    """Exercise the training collator and require exact assistant-only supervision."""

    collator = PlannerTextDataCollator(processor, max_seq_length, ".", include_task_metadata=True)
    encoded = collator(rows)
    if encoded.get("task_types") != ["planner"] * len(rows):
        raise ValueError("Planner task metadata did not survive collation")
    tokenizer = getattr(processor, "tokenizer", None)
    try:
        vocabulary_size = len(tokenizer)
    except TypeError:
        vocabulary_size = None
    unknown_id = getattr(tokenizer, "unk_token_id", None)
    active_input_ids = encoded["input_ids"][encoded["attention_mask"].bool()]
    supervised_ids = encoded["labels"][encoded["labels"] != -100]
    if vocabulary_size is not None:
        for name, values in (("input", active_input_ids), ("supervised label", supervised_ids)):
            if bool(((values < 0) | (values >= vocabulary_size)).any()):
                raise ValueError(
                    f"{name} contains token ids outside vocabulary size {vocabulary_size}"
                )
    if unknown_id is not None and bool((active_input_ids == unknown_id).any()):
        raise ValueError("active Planner input contains unknown tokens")
    if any(key in encoded for key in ("pixel_values", "image_grid_thw", "pixel_values_videos")):
        raise ValueError("symbolic Planner image references unexpectedly produced visual tensors")
    prompt_texts = [
        processor.apply_chat_template(
            row["messages"][:-1], tokenize=False, add_generation_prompt=True
        )
        for row in rows
    ]
    full_texts = [
        processor.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
        for row in rows
    ]
    # Batch independent uncapped tokenization instead of making two calls per
    # individual row. The training collator itself remains unchanged.
    raw_full_batch = processor(
        text=full_texts,
        images=None,
        videos=None,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )
    raw_prompt_batch = processor(
        text=prompt_texts,
        images=None,
        videos=None,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )
    supervised_counts = []
    full_counts = []
    for index, row in enumerate(rows):
        messages = row["messages"]
        prompt, full = prompt_texts[index], full_texts[index]
        if not full.startswith(prompt):
            raise ValueError(f"{row['id']}: chat template prompt is not an exact prefix")
        assistant = messages[-1]["content"]
        if not full[len(prompt) :].startswith(assistant):
            raise ValueError(f"{row['id']}: chat template polluted assistant DSL")
        raw_full = raw_full_batch["input_ids"][index][
            raw_full_batch["attention_mask"][index].bool()
        ]
        raw_prompt = raw_prompt_batch["input_ids"][index][
            raw_prompt_batch["attention_mask"][index].bool()
        ]
        if len(raw_full) > max_seq_length:
            raise ValueError(
                f"{row['id']}: full sequence {len(raw_full)} exceeds "
                f"max_seq_length={max_seq_length}"
            )
        if raw_full[: len(raw_prompt)].tolist() != raw_prompt.tolist():
            raise ValueError(f"{row['id']}: prompt/full token prefix mismatch")
        expected = raw_full[len(raw_prompt) :].tolist()
        labels = encoded["labels"][index]
        actual = labels[labels != -100].tolist()
        if actual != expected or not actual:
            raise ValueError(f"{row['id']}: assistant supervision differs from complete DSL suffix")
        if bool((labels[encoded["attention_mask"][index] == 0] != -100).any()):
            raise ValueError(f"{row['id']}: padding contributes to loss")
        supervised_counts.append(len(actual))
        full_counts.append(len(raw_full))
    return {
        "assistant_tokens": supervised_counts,
        "full_sequence_tokens": full_counts,
        "token_id_range_checked": vocabulary_size is not None,
        "unknown_token_checked": unknown_id is not None,
        "tokenizer_vocabulary_size": vocabulary_size,
        "tokenizer_unknown_id": unknown_id,
    }


def verify_sft(
    paths: list[Path],
    *,
    processor: Any,
    max_seq_length: int = 1088,
    max_output_tokens: int = 512,
    batch_size: int = 8,
    tokenizer_id: str = "injected_test_processor",
) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one SFT input is required")
    if batch_size < 1 or max_seq_length < 1 or max_output_tokens < 1:
        raise ValueError("batch size and token budgets must be positive")
    total = 0
    families: Counter[str] = Counter()
    assistant_lengths = []
    sequence_lengths = []
    token_id_range_checked = True
    unknown_token_checked = True
    tokenizer_vocabulary_size = None
    tokenizer_unknown_id = None
    inputs = []
    for path in paths:
        _read_rows(path, synthetic=True)
        dataset = Qwen3VLDataset(path, skip_bad_samples=False)
        rows = list(dataset)
        if not rows:
            raise ValueError(f"{path}: no SFT rows")
        for row in rows:
            messages = row["messages"]
            if [message.get("role") for message in messages] != ["system", "user", "assistant"]:
                raise ValueError(f"{row['id']}: expected system/user/assistant")
            if any(not isinstance(message.get("content"), str) for message in messages):
                raise ValueError(f"{row['id']}: message content must be text")
            metadata = row["metadata"]
            if metadata.get("source") != SYNTHETIC_SOURCE:
                raise ValueError(f"{row['id']}: not a synthetic programmatic plan")
            if "answer_audit" in metadata:
                raise ValueError(f"{row['id']}: synthetic row cannot claim visual answer audit")
            if row["task_type"] != "planner":
                raise ValueError(f"{row['id']}: wrong training task type")
            validate_message_contract(row)
            families[str(metadata.get("family", "unknown"))] += 1
        for start in range(0, len(rows), batch_size):
            stats = inspect_batch(
                rows[start : start + batch_size], processor, max_seq_length=max_seq_length
            )
            assistant_lengths.extend(stats["assistant_tokens"])
            sequence_lengths.extend(stats["full_sequence_tokens"])
            token_id_range_checked &= stats["token_id_range_checked"]
            unknown_token_checked &= stats["unknown_token_checked"]
            tokenizer_vocabulary_size = stats["tokenizer_vocabulary_size"]
            tokenizer_unknown_id = stats["tokenizer_unknown_id"]
        total += len(rows)
        inputs.append(
            {"path": str(path.resolve()), "count": len(rows), "sha256": file_sha256(path)}
        )
    within = sum(length <= max_output_tokens for length in assistant_lengths)
    if within != total:
        raise ValueError(
            f"{total - within} assistant targets exceed max_output_tokens={max_output_tokens}"
        )
    return {
        "valid": True,
        "verification_scope": (
            "CPU injected processor + training dataset/collator; no model inference"
            if tokenizer_id == "injected_test_processor"
            else "CPU real tokenizer + training dataset/collator; no model inference"
        ),
        "model_weights_loaded": False,
        "tokenizer": tokenizer_id,
        "inputs": inputs,
        "count": total,
        "families": dict(sorted(families.items())),
        "max_seq_length": max_seq_length,
        "max_output_tokens": max_output_tokens,
        "batch_size": batch_size,
        "assistant_only_mask_exact": True,
        "input_and_supervised_token_id_ranges_checked": token_id_range_checked,
        "tokenizer_vocabulary_size": tokenizer_vocabulary_size,
        "unknown_token_check": (
            "checked; zero unknown tokens"
            if unknown_token_checked
            else "tokenizer defines no unk token"
        ),
        "tokenizer_unknown_id": tokenizer_unknown_id,
        "truncated_count": 0,
        "output_budget_pass_count": within,
        "output_budget_pass_fraction": within / total,
        "assistant_tokens_mean": sum(assistant_lengths) / total,
        "assistant_tokens_max": max(assistant_lengths),
        "full_sequence_tokens_mean": sum(sequence_lengths) / total,
        "full_sequence_tokens_max": max(sequence_lengths),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seq-length", type=int, default=1088)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite verification: {args.output}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    processor = TextTokenizerProcessor(tokenizer)
    report = verify_sft(
        args.input,
        processor=processor,
        max_seq_length=args.max_seq_length,
        max_output_tokens=args.max_output_tokens,
        batch_size=args.batch_size,
        tokenizer_id=str(args.tokenizer.resolve()),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
