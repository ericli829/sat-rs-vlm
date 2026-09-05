"""Benchmark the frozen same-model reasoning-to-choice KV-cache path."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from sat_rs_vlm.models.hf_vlm_engine import HuggingFaceVLMEngine
from sat_rs_vlm.taskgraph.choice_config import ChoiceSystemConfig


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id")
    parser.add_argument("--role", choices=("2b", "route_4b"), default="2b")
    parser.add_argument(
        "--answer-type",
        choices=("CHOICE_SINGLE", "CHOICE_MULTI"),
        default="CHOICE_SINGLE",
    )
    parser.add_argument("--image", action="append", default=[])
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--options-json", required=True)
    parser.add_argument("--sample-id", default="choice-cache-benchmark")
    parser.add_argument("--reasoning-max-new-tokens", type=int, default=128)
    parser.add_argument("--multi-select-threshold", type=float, default=0.0)
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
    if len(options) > 26:
        raise SystemExit("--options-json supports at most 26 options")
    choice_ids = tuple(chr(ord("A") + index) for index in range(len(options)))
    model_id = _model_id(args)
    engine = _engine(model_id, args.reasoning_max_new_tokens)
    config = ChoiceSystemConfig(multi_select_threshold=args.multi_select_threshold)
    role_instruction = (
        "Analyze the route, obstacles, and supplied route options carefully."
        if args.role == "route_4b"
        else "Analyze the evidence and supplied options carefully."
    )
    reasoning_prompt = (
        f"{args.prompt}\n\nOptions:\n"
        + "\n".join(options)
        + f"\n\n{role_instruction} The final decision will use the same model's KV cache."
    )

    wall_started = time.perf_counter()
    try:
        cached = engine.reason_and_choose(
            reasoning_prompt,
            list(args.image),
            choice_ids=choice_ids,
            option_texts=options,
            answer_type=args.answer_type,
            single_choice_suffix=config.single_choice_suffix,
            multi_verify_template=config.multi_verify_template,
            multi_select_threshold=config.multi_select_threshold,
            reasoning_max_new_tokens=args.reasoning_max_new_tokens,
        )
        active_sessions = engine.active_session_count
    finally:
        engine.close()
    overall_wall_ms = (time.perf_counter() - wall_started) * 1000.0
    reasoning_total_ms = cached.latency_ms.get("reasoning_total_ms")
    choice_total_ms = cached.latency_ms.get("choice_total_ms")
    ratio = None
    if reasoning_total_ms and choice_total_ms is not None:
        ratio = float(choice_total_ms) / float(reasoning_total_ms)
    return {
        "sample_id": args.sample_id,
        "role": args.role,
        "architecture": f"{args.role}_reasoning_to_same_model_kv_choice",
        "answer_type": args.answer_type,
        "model": model_id,
        "reasoning_total_ms": reasoning_total_ms,
        "choice_total_ms": choice_total_ms,
        "choice_reasoning_ratio": ratio,
        "overall_wall_ms": overall_wall_ms,
        "cache_reused": cached.cache_reused,
        "reasoning_tokens": cached.metadata.get("reasoning_tokens"),
        "choice_suffix_tokens": cached.metadata.get("choice_suffix_tokens"),
        "choice_scored_tokens": cached.metadata.get("choice_scored_tokens"),
        "peak_vram_mb": cached.metadata.get("peak_vram_mb"),
        "active_session_count_after_inference": active_sessions,
        "selected_ids": list(cached.selected_ids),
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
