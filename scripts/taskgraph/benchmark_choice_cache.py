"""Compare recomputed two-pass choice with KV-cached constrained scoring."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from sat_rs_vlm.models.hf_vlm_engine import HuggingFaceVLMEngine


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id")
    parser.add_argument("--legacy-choice-model-id")
    parser.add_argument("--role", choices=("2b", "route_4b"), default="2b")
    parser.add_argument("--image", action="append", default=[])
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--options-json", required=True)
    parser.add_argument("--sample-id", default="choice-cache-benchmark")
    parser.add_argument("--reasoning-max-new-tokens", type=int, default=128)
    parser.add_argument("--output")
    return parser


def _model_id(args: argparse.Namespace) -> str:
    configured = args.model_id
    if configured:
        return str(configured)
    variable = "QWEN3VL_4B_MODEL_DIR" if args.role == "route_4b" else "QWEN3VL_2B_MODEL_DIR"
    value = os.environ.get(variable)
    if not value:
        raise SystemExit(f"--model-id or {variable} is required; models are never downloaded")
    return value


def _engine(model_id: str, max_new_tokens: int) -> HuggingFaceVLMEngine:
    path = Path(model_id).expanduser()
    if not path.exists():
        raise SystemExit(f"local model path does not exist: {path}")
    return HuggingFaceVLMEngine(
        str(path),
        device="auto",
        dtype="auto",
        max_new_tokens=max_new_tokens,
        local_files_only=True,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    options = tuple(str(item) for item in json.loads(args.options_json))
    if not options:
        raise SystemExit("--options-json must contain at least one option")
    choice_ids = tuple(chr(ord("A") + index) for index in range(len(options)))
    model_id = _model_id(args)
    engine = _engine(model_id, args.reasoning_max_new_tokens)
    legacy_choice_engine = (
        _engine(args.legacy_choice_model_id, 16) if args.legacy_choice_model_id else engine
    )
    reasoning_prompt = (
        f"{args.prompt}\n\nOptions:\n" + "\n".join(options) + "\n\n"
        "Analyze the evidence and options carefully. The final option will be selected "
        "in a separate constrained step."
    )

    baseline_started = time.perf_counter()
    baseline_reasoning = engine.generate_text(reasoning_prompt, list(args.image))
    baseline_choice_prompt = (
        f"{reasoning_prompt}\n\nReasoning:\n{baseline_reasoning}\n\n"
        "Return exactly one legal option id.\nFinal choice:"
    )
    baseline_choice = legacy_choice_engine.generate_text(baseline_choice_prompt, list(args.image))
    baseline_total_ms = (time.perf_counter() - baseline_started) * 1000.0

    cached_started = time.perf_counter()
    cached = engine.reason_and_choose(
        reasoning_prompt,
        list(args.image),
        choice_ids=choice_ids,
        option_texts=options,
        answer_type="CHOICE_SINGLE",
        suffix="\n\nFinal choice:",
        reasoning_max_new_tokens=args.reasoning_max_new_tokens,
    )
    cached_total_ms = (time.perf_counter() - cached_started) * 1000.0
    choice_incremental_ms = float(cached.latency_ms.get("choice_suffix_prefill_ms") or 0.0)
    choice_incremental_ms += float(cached.latency_ms.get("choice_scoring_ms") or 0.0)
    return {
        "sample_id": args.sample_id,
        "role": args.role,
        "model": model_id,
        "legacy_choice_model": args.legacy_choice_model_id or model_id,
        "reasoning_tokens": cached.metadata.get("reasoning_tokens"),
        "baseline_total_ms": baseline_total_ms,
        "cached_total_ms": cached_total_ms,
        "choice_incremental_ms": choice_incremental_ms,
        "speedup": baseline_total_ms / cached_total_ms if cached_total_ms else None,
        "cache_reused": cached.cache_reused,
        "peak_vram_mb": cached.metadata.get("peak_vram_mb"),
        "choice_old": baseline_choice,
        "choice_new": list(cached.selected_ids),
        "scores": cached.scores,
        "method": cached.method,
        "latency_ms": cached.latency_ms,
    }


def main() -> None:
    args = _parser().parse_args()
    payload = run(args)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
