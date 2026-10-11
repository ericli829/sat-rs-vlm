#!/usr/bin/env python3
"""Measure Qwen3-VL ChangeChat 3x3 grid generation accuracy."""

from __future__ import annotations

import argparse
import gc
import json
import re
import time
from pathlib import Path
from typing import Any

import torch
import transformers
from peft import PeftModel

from sat_rs_vlm.data.qwen3vl_collator import Qwen3VLDataCollator
from sat_rs_vlm.data.qwen3vl_dataset import Qwen3VLDataset
from sat_rs_vlm.models.qwen3vl_loader import compatible_model_class
from sat_rs_vlm.training.utils import model_input_device, move_to_device


CELL_LABELS = {
    f"{vertical}-{horizontal}"
    for vertical in ("top", "center", "bottom")
    for horizontal in ("left", "center", "right")
}
GRID_ANSWER_RE = re.compile(
    r"roads\s*:\s*\[(.*?)\]\s*\.?\s*buildings\s*:\s*\[(.*?)\]",
    re.IGNORECASE | re.DOTALL,
)
GRID_TOKEN_RE = re.compile(r"['\"]([^'\"]+)['\"]")


def parse_answer(text: str) -> tuple[set[str], set[str]]:
    match = GRID_ANSWER_RE.search(text)
    if match is None:
        raise ValueError("missing roads/buildings grid fields")

    def parse_cells(raw: str) -> set[str]:
        if not raw.strip():
            return set()
        values = [item.strip() for item in GRID_TOKEN_RE.findall(raw)]
        residue = GRID_TOKEN_RE.sub("", raw).strip(" \t,\r\n")
        if residue or len(values) != len(set(values)) or set(values) - CELL_LABELS:
            raise ValueError(f"malformed cell list: {raw!r}")
        return set(values)

    return parse_cells(match.group(1)), parse_cells(match.group(2))


def scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    result: dict[str, Any] = {"samples": total}
    for name, expected_key, predicted_key in (
        ("roads", "expected_roads", "predicted_roads"),
        ("buildings", "expected_buildings", "predicted_buildings"),
    ):
        exact = 0
        tp = fp = fn = 0
        for row in rows:
            expected = set(row[expected_key])
            predicted = set(row[predicted_key])
            exact += expected == predicted
            tp += len(expected & predicted)
            fp += len(predicted - expected)
            fn += len(expected - predicted)
        precision = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
        recall = tp / (tp + fn) if tp + fn else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[name] = {
            "exact_match": exact / total if total else 0.0,
            "exact_correct": exact,
            "cell_precision_micro": precision,
            "cell_recall_micro": recall,
            "cell_f1_micro": f1,
            "target_cells": tp + fn,
            "predicted_cells": tp + fp,
        }

    combined_exact = sum(
        set(row["expected_roads"]) == set(row["predicted_roads"])
        and set(row["expected_buildings"]) == set(row["predicted_buildings"])
        for row in rows
    )
    result["both_classes_exact_match"] = {
        "correct": combined_exact,
        "rate": combined_exact / total if total else 0.0,
    }
    result["parse_valid"] = {
        "count": sum(bool(row["parse_valid"]) for row in rows),
        "rate": sum(bool(row["parse_valid"]) for row in rows) / total if total else 0.0,
    }
    per_class: dict[str, Any] = {}
    for label in ("both_empty", "roads_only", "buildings_only", "both_nonempty"):
        subset = [row for row in rows if row["grid_class"] == label]
        if subset:
            per_class[label] = {
                "count": len(subset),
                "both_classes_exact_match": sum(
                    set(row["expected_roads"]) == set(row["predicted_roads"])
                    and set(row["expected_buildings"]) == set(row["predicted_buildings"])
                    for row in subset
                ) / len(subset),
            }
    result["by_label_stratum"] = per_class
    return result


def generate_chunk(
    *,
    model: Any,
    processor: Any,
    collator: Qwen3VLDataCollator,
    samples: list[dict[str, Any]],
    max_new_tokens: int,
) -> list[str]:
    batch = collator(samples)
    prompt_width = int(batch["input_ids"].shape[-1])
    batch = move_to_device(batch, model_input_device(model, torch), torch)
    with torch.inference_mode():
        output_ids = model.generate(
            **batch,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    return [
        str(value)
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
    collator: Qwen3VLDataCollator,
    samples: list[dict[str, Any]],
    max_new_tokens: int,
) -> list[str]:
    try:
        return generate_chunk(
            model=model,
            processor=processor,
            collator=collator,
            samples=samples,
            max_new_tokens=max_new_tokens,
        )
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower() or len(samples) <= 1:
            raise
        print(f"CUDA OOM for batch={len(samples)}; retrying as smaller batches", flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        midpoint = len(samples) // 2
        return generate_with_oom_fallback(
            model=model,
            processor=processor,
            collator=collator,
            samples=samples[:midpoint],
            max_new_tokens=max_new_tokens,
        ) + generate_with_oom_fallback(
            model=model,
            processor=processor,
            collator=collator,
            samples=samples[midpoint:],
            max_new_tokens=max_new_tokens,
        )


def evaluate_route(
    *,
    name: str,
    adapter: Path,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
) -> None:
    if not (adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"adapter weights missing: {adapter}")
    print(f"loading route={name} adapter={adapter}", flush=True)
    processor = transformers.AutoProcessor.from_pretrained(
        str(args.base_model), trust_remote_code=True, local_files_only=True
    )
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.padding_side = "left"
    model_cls = compatible_model_class(transformers)
    model = model_cls.from_pretrained(
        str(args.base_model),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, str(adapter), local_files_only=True)
    model.eval()
    collator = Qwen3VLDataCollator(
        processor,
        max_seq_length=args.max_seq_length,
        image_root=args.image_root,
        for_generation=True,
    )
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for start in range(0, len(rows), args.batch_size):
        samples = rows[start : start + args.batch_size]
        decoded = generate_with_oom_fallback(
            model=model,
            processor=processor,
            collator=collator,
            samples=samples,
            max_new_tokens=args.max_new_tokens,
        )
        for sample, prediction in zip(samples, decoded, strict=True):
            metadata = sample.get("metadata") or {}
            expected_roads = set(metadata.get("road_cells") or [])
            expected_buildings = set(metadata.get("building_cells") or [])
            parse_error = None
            try:
                predicted_roads, predicted_buildings = parse_answer(str(prediction))
                parse_valid = True
            except ValueError as exc:
                predicted_roads, predicted_buildings = set(), set()
                parse_valid = False
                parse_error = str(exc)
            records.append(
                {
                    "id": sample.get("id"),
                    "grid_class": metadata.get("grid_class", "unknown"),
                    "expected_roads": sorted(expected_roads),
                    "expected_buildings": sorted(expected_buildings),
                    "prediction": str(prediction).strip(),
                    "predicted_roads": sorted(predicted_roads),
                    "predicted_buildings": sorted(predicted_buildings),
                    "parse_valid": parse_valid,
                    "parse_error": parse_error,
                    "roads_exact": expected_roads == predicted_roads,
                    "buildings_exact": expected_buildings == predicted_buildings,
                }
            )
        if (start // args.batch_size + 1) % args.log_every == 0:
            print(
                f"route={name} done={len(records)}/{len(rows)} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )

    summary = {
        "route": name,
        "adapter": str(adapter),
        "base_model": str(args.base_model),
        "validation_file": str(args.validation_file),
        "samples": len(records),
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "elapsed_seconds": time.perf_counter() - started,
        "metrics": scores(records),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{name}.predictions.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / f"{name}.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    del model, processor, collator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--route", action="append", required=True, metavar="NAME=ADAPTER")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_new_tokens < 1:
        parser.error("batch size and generation budget must be positive")
    return args


def main() -> int:
    args = parse_args()
    dataset = Qwen3VLDataset(args.validation_file, max_samples=args.limit)
    rows = list(dataset)
    if not rows:
        raise ValueError("validation set is empty")
    for route in args.route:
        if "=" not in route:
            raise ValueError(f"route must be NAME=ADAPTER, got {route!r}")
        name, adapter = route.split("=", 1)
        evaluate_route(name=name, adapter=Path(adapter), args=args, rows=rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
