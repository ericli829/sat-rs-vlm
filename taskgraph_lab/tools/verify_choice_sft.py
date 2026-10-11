"""CPU token-length and real training-collator audit for choice SFT releases.

Use bounded slices for quick local checks. This loads a local tokenizer only;
neither model weights nor images are loaded and no inference is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path


def verify(input_path: Path, tokenizer_path: Path, *, offset: int = 0, limit: int = 1500) -> dict:
    started = time.perf_counter()
    from transformers import AutoTokenizer

    from taskgraph_lab.tools.verify_synthetic_planner_sft import (
        TextTokenizerProcessor,
        inspect_batch,
    )

    data = input_path.read_bytes()
    all_rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    rows = all_rows[offset : offset + limit]
    if not rows or offset < 0 or limit < 1:
        raise ValueError("nonempty, positive audit slice required")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    full_texts, prompt_texts = [], []
    for row in rows:
        messages = row["messages"]
        if [m.get("role") for m in messages] != ["system", "user", "assistant"]:
            raise ValueError(f"{row['id']}: incorrect message roles")
        user = json.loads(messages[1]["content"])
        if user["question_type"] != "MULTIPLE_CHOICE" or not user.get("choices"):
            raise ValueError(f"{row['id']}: choice contract required")
        prompt = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        if not full.startswith(prompt) or not full[len(prompt) :].startswith(
            messages[-1]["content"]
        ):
            raise ValueError(f"{row['id']}: DSL is not an unchanged assistant suffix")
        full_texts.append(full)
        prompt_texts.append(prompt)
    full_ids = tokenizer(full_texts, padding=False, truncation=False, add_special_tokens=False)[
        "input_ids"
    ]
    prompt_ids = tokenizer(prompt_texts, padding=False, truncation=False, add_special_tokens=False)[
        "input_ids"
    ]
    full_lengths, output_lengths = [], []
    over_sequence, over_output = [], []
    vocabulary = len(tokenizer)
    for row, full, prompt in zip(rows, full_ids, prompt_ids, strict=True):
        if full[: len(prompt)] != prompt or not len(full) > len(prompt):
            raise ValueError(f"{row['id']}: token prefix mismatch or empty supervision")
        if min(full) < 0 or max(full) >= vocabulary:
            raise ValueError(f"{row['id']}: token ID outside vocabulary")
        if tokenizer.unk_token_id is not None and tokenizer.unk_token_id in full:
            raise ValueError(f"{row['id']}: unexpected unknown token")
        full_lengths.append(len(full))
        output_lengths.append(len(full) - len(prompt))
        if len(full) > 1088:
            over_sequence.append(row["id"])
        if len(full) - len(prompt) > 512:
            over_output.append(row["id"])
    max_full, max_output = max(full_lengths), max(output_lengths)
    # Exercise the real loss mask for the 16 longest sequences of this slice.
    longest = sorted(range(len(rows)), key=lambda index: full_lengths[index], reverse=True)[:16]
    collator = inspect_batch(
        [rows[index] for index in longest],
        TextTokenizerProcessor(tokenizer),
        max_seq_length=math.ceil(max_full / 256) * 256,
    )
    return {
        "valid": True,
        "scope": "all-slice token/prefix integrity + 16 longest real collator masks; no inference",
        "input": str(input_path.resolve()),
        "input_sha256": hashlib.sha256(data).hexdigest(),
        "tokenizer": str(tokenizer_path),
        "model_weights_loaded": False,
        "offset": offset,
        "count": len(rows),
        "population_count": len(all_rows),
        "assistant_tokens_max": max_output,
        "assistant_tokens_mean": sum(output_lengths) / len(rows),
        "full_tokens_max": max_full,
        "full_tokens_mean": sum(full_lengths) / len(rows),
        "over_existing_seq_1088_count": len(over_sequence),
        "over_existing_seq_1088_ids": over_sequence,
        "over_existing_output_512_count": len(over_output),
        "over_existing_output_512_ids": over_output,
        "minimum_rounded_seq_budget": math.ceil(max_full / 256) * 256,
        "minimum_rounded_output_budget": math.ceil(max_output / 128) * 128,
        "collator_checked_count": len(longest),
        "collator_assistant_only_mask_exact": True,
        "collator_has_visual_tensors": False,
        "tokenizer_vocabulary_size": collator["tokenizer_vocabulary_size"],
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1500)
    args = parser.parse_args()
    report = verify(args.input, args.tokenizer, offset=args.offset, limit=args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if not key.endswith("_ids")},
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
