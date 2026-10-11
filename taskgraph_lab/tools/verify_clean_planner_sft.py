"""Bounded encoding audit for cleaned mixed choice/BBOX Planner supervision."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def verify(path: Path, tokenizer_path: Path, offset: int, limit: int) -> dict:
    from transformers import AutoTokenizer

    from taskgraph_lab.tools.verify_synthetic_planner_sft import (
        TextTokenizerProcessor,
        inspect_batch,
    )

    raw = path.read_bytes()
    population = [json.loads(s) for s in raw.splitlines() if s.strip()]
    rows = population[offset : offset + limit]
    if offset < 0 or not rows or not 1 <= limit <= 1500:
        raise ValueError("require a nonempty slice of at most 1500 rows")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    full, prompts = [], []
    for row in rows:
        messages = row["messages"]
        user = json.loads(messages[1]["content"])
        if user["question_type"] == "MULTIPLE_CHOICE":
            if not user.get("choices"):
                raise ValueError("choice options missing")
        elif user["question_type"] != "FREE_FORM" or user.get("choices"):
            raise ValueError("unexpected Planner input contract")
        prompt = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        if not text.startswith(prompt) or not text[len(prompt) :].startswith(
            messages[2]["content"]
        ):
            raise ValueError(f"{row['id']}: assistant text/prompt changed")
        prompts.append(prompt)
        full.append(text)
    ids = tokenizer(full, padding=False, truncation=False, add_special_tokens=False)["input_ids"]
    prompt_ids = tokenizer(prompts, padding=False, truncation=False, add_special_tokens=False)[
        "input_ids"
    ]
    lengths, output_lengths = [], []
    for row, sequence, prefix in zip(rows, ids, prompt_ids, strict=True):
        if sequence[: len(prefix)] != prefix or len(sequence) <= len(prefix):
            raise ValueError(f"{row['id']}: empty supervision or token prefix mismatch")
        if min(sequence) < 0 or max(sequence) >= len(tokenizer):
            raise ValueError("token outside vocabulary")
        if tokenizer.unk_token_id is not None and tokenizer.unk_token_id in sequence:
            raise ValueError("unknown token")
        lengths.append(len(sequence))
        output_lengths.append(len(sequence) - len(prefix))
    budget = math.ceil(max(lengths) / 256) * 256
    longest = sorted(range(len(rows)), key=lambda i: lengths[i], reverse=True)[:16]
    collator = inspect_batch(
        [rows[i] for i in longest], TextTokenizerProcessor(tokenizer), max_seq_length=budget
    )
    return {
        "valid": True,
        "input": str(path.resolve()),
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "offset": offset,
        "count": len(rows),
        "population": len(population),
        "full_tokens_max": max(lengths),
        "assistant_tokens_max": max(output_lengths),
        "over_seq_1088_count": sum(n > 1088 for n in lengths),
        "over_seq_1536_count": sum(n > 1536 for n in lengths),
        "over_output_512_count": sum(n > 512 for n in output_lengths),
        "minimum_rounded_seq_budget": budget,
        "collator_checked_count": len(longest),
        "collator_assistant_only_mask_exact": True,
        "collator_has_visual_tensors": collator.get("has_visual_tensors", False),
        "model_weights_loaded": False,
        "tokenizer": str(tokenizer_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1500)
    args = parser.parse_args()
    result = verify(args.input, args.tokenizer, args.offset, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
