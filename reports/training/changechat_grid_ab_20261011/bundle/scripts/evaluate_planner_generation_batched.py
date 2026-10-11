#!/usr/bin/env python3
"""Batch-evaluate raw Qwen3-VL Planner generations on a text-only split."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch
import transformers
from peft import PeftModel

from sat_rs_vlm.models.qwen3vl_loader import compatible_model_class
from sat_rs_vlm.training.utils import model_input_device, move_to_device
from taskgraph_lab.evaluation.planner_generation import (
    evaluate_prediction,
    prompt_messages,
    summarize_predictions,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def generate_chunk(
    *,
    model: Any,
    processor: Any,
    rows: list[dict[str, Any]],
    torch_module: Any,
    max_prompt_tokens: int,
    max_new_tokens: int,
) -> list[str]:
    prompts = [
        str(
            processor.apply_chat_template(
                prompt_messages(row),
                tokenize=False,
                add_generation_prompt=True,
            )
        )
        for row in rows
    ]
    encoded = processor(
        text=prompts,
        images=None,
        videos=None,
        padding=True,
        truncation=True,
        max_length=max_prompt_tokens,
        return_tensors="pt",
    )
    prompt_width = int(encoded["input_ids"].shape[-1])
    encoded = move_to_device(encoded, model_input_device(model, torch_module), torch_module)
    with torch_module.inference_mode():
        output_ids = model.generate(
            **encoded,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            pad_token_id=processor.tokenizer.pad_token_id,
            use_cache=True,
        )
    return [
        str(value).strip()
        for value in processor.batch_decode(
            output_ids[:, prompt_width:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    ]


def generate_with_oom_fallback(
    *,
    model: Any,
    processor: Any,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[str]:
    try:
        return generate_chunk(
            model=model,
            processor=processor,
            rows=rows,
            torch_module=torch,
            max_prompt_tokens=args.max_prompt_tokens,
            max_new_tokens=args.max_new_tokens,
        )
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower() or len(rows) <= 1:
            raise
        print(f"CUDA OOM for planner batch={len(rows)}; retrying smaller batches", flush=True)
        del exc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        midpoint = len(rows) // 2
        return generate_with_oom_fallback(
            model=model, processor=processor, rows=rows[:midpoint], args=args
        ) + generate_with_oom_fallback(
            model=model, processor=processor, rows=rows[midpoint:], args=args
        )


def evaluate_adapter(adapter: Path, name: str, args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    if not (adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"adapter weights missing: {adapter}")
    print(f"loading planner={name} adapter={adapter}", flush=True)
    processor = transformers.AutoProcessor.from_pretrained(
        str(args.base_model), trust_remote_code=True, local_files_only=True
    )
    processor.tokenizer.padding_side = "left"
    model = compatible_model_class(transformers).from_pretrained(
        str(args.base_model),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, str(adapter), local_files_only=True)
    model.eval()
    predictions: list[str] = []
    started = time.perf_counter()
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        predictions.extend(
            generate_with_oom_fallback(
                model=model,
                processor=processor,
                rows=batch_rows,
                args=args,
            )
        )
        if (start // args.batch_size + 1) % args.log_every == 0:
            print(
                f"planner={name} done={len(predictions)}/{len(rows)} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    records = [evaluate_prediction(row, output) for row, output in zip(rows, predictions, strict=True)]
    summary = {
        "evaluator": "batched raw greedy generation; same prompt, token limits, and strict validator metrics as evaluate_qwen3vl_planner.py with recovery/RAG/constrained decoding off",
        "adapter": str(adapter),
        "adapter_sha256": sha256(adapter / "adapter_model.safetensors"),
        "base_model": str(args.base_model),
        "validation_file": str(args.validation_file),
        "validation_sha256": sha256(args.validation_file),
        "sample_count": len(rows),
        "requested_batch_size": args.batch_size,
        "max_prompt_tokens": args.max_prompt_tokens,
        "max_new_tokens": args.max_new_tokens,
        "elapsed_seconds": time.perf_counter() - started,
        "metrics": summarize_predictions(records),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{name}.predictions.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / f"{name}.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapter", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-prompt-tokens", type=int, default=1536)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    rows = read_jsonl(args.validation_file)
    if not rows:
        raise ValueError("Planner validation split is empty")
    for item in args.adapter:
        if "=" not in item:
            raise ValueError(f"adapter must be NAME=PATH, got {item!r}")
        name, path = item.split("=", 1)
        evaluate_adapter(Path(path), name, args, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
