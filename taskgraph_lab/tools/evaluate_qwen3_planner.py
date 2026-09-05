"""Strict greedy-generation evaluation for a text-only Qwen3 Planner adapter."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from taskgraph_lab.evaluation.planner_generation import (
    evaluate_prediction,
    prompt_messages,
    summarize_predictions,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-prompt-tokens", type=int, default=1088)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path, label in (
        (args.base_model, "base model"),
        (args.adapter, "adapter"),
        (args.validation_file, "validation file"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    weights = args.adapter / "adapter_model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"adapter weights do not exist: {weights}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = _read_jsonl(args.validation_file)
    if args.limit is not None:
        rows = rows[: args.limit]
    started_at = datetime.now(UTC)
    provenance = {
        "schema_version": "taskgraph-planner-generation-eval-v1",
        "backend": "qwen3_causal_lm",
        "started_at": started_at.isoformat(),
        "base_model": str(args.base_model.resolve()),
        "adapter": str(args.adapter.resolve()),
        "adapter_weights_sha256": _sha256(weights),
        "validation_file": str(args.validation_file.resolve()),
        "validation_sha256": _sha256(args.validation_file),
        "sample_count": len(rows),
        "generation": {
            "do_sample": False,
            "num_beams": 1,
            "enable_thinking": False,
            "batch_size": 1,
            "max_new_tokens": args.max_new_tokens,
            "max_prompt_tokens": args.max_prompt_tokens,
            "repair": False,
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "peft")
        },
    }
    _write_json(args.output_dir / "preflight.json", provenance)
    predictions_path = args.output_dir / "predictions.jsonl"
    results: list[dict[str, Any]] = []
    torch = None
    model = None
    tokenizer = None
    try:
        import peft
        import torch as imported_torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch = imported_torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for local Qwen3 Planner evaluation")
        tokenizer = AutoTokenizer.from_pretrained(
            str(args.base_model), local_files_only=True, trust_remote_code=True
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            str(args.base_model),
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="auto",
        )
        model = peft.PeftModel.from_pretrained(
            model,
            str(args.adapter),
            local_files_only=True,
        )
        model.eval()
        input_device = next(model.parameters()).device
        torch.cuda.reset_peak_memory_stats()
        with predictions_path.open("w", encoding="utf-8", newline="\n") as output:
            for row in rows:
                prompt = str(
                    tokenizer.apply_chat_template(
                        prompt_messages(row),
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                )
                encoded = tokenizer(
                    prompt,
                    truncation=True,
                    max_length=args.max_prompt_tokens,
                    return_tensors="pt",
                )
                encoded = {
                    key: value.to(input_device) if hasattr(value, "to") else value
                    for key, value in encoded.items()
                }
                prompt_width = int(encoded["input_ids"].shape[1])
                begin = time.perf_counter()
                with torch.inference_mode():
                    generated = model.generate(
                        **encoded,
                        do_sample=False,
                        num_beams=1,
                        max_new_tokens=args.max_new_tokens,
                        pad_token_id=int(tokenizer.pad_token_id),
                        eos_token_id=int(tokenizer.eos_token_id),
                        use_cache=True,
                    )
                elapsed = time.perf_counter() - begin
                continuation = generated[0, prompt_width:]
                prediction = tokenizer.decode(
                    continuation,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                record = evaluate_prediction(row, prediction)
                record["latency_seconds"] = elapsed
                record["generated_tokens"] = int(continuation.shape[0])
                results.append(record)
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                print(
                    json.dumps(
                        {
                            "completed": len(results),
                            "total": len(rows),
                            "sample_id": record["sample_id"],
                            "dsl_parse_valid": record["dsl_parse_valid"],
                            "runtime_valid": record["runtime_valid"],
                            "canonical_exact": record["canonical_exact"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                del encoded, generated, continuation
        summary = summarize_predictions(results)
        summary.update(
            {
                "success": True,
                "started_at": started_at.isoformat(),
                "finished_at": datetime.now(UTC).isoformat(),
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / (1024**2),
                "provenance": provenance,
            }
        )
        _write_json(args.output_dir / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception as exc:
        _write_json(
            args.output_dir / "failure.json",
            {
                "success": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "completed_samples": len(results),
            },
        )
        raise
    finally:
        model = None
        tokenizer = None
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


if __name__ == "__main__":
    raise SystemExit(main())

