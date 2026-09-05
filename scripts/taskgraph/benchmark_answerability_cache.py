"""Compare answerability with and without cross-stage reasoning KV reuse."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from sat_rs_vlm.models.hf_vlm_engine import HuggingFaceVLMEngine
from sat_rs_vlm.taskgraph.answerability import AnswerabilityConfig

STATUSES = ("SUFFICIENT", "NEED_MORE_EVIDENCE", "UNRESOLVED")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id")
    parser.add_argument("--image", action="append", default=[])
    parser.add_argument("--question", required=True)
    parser.add_argument("--sample-id", default="answerability-cache-benchmark")
    parser.add_argument("--reasoning-max-new-tokens", type=int, default=64)
    parser.add_argument("--output")
    return parser


def _model_id(args: argparse.Namespace) -> str:
    value = args.model_id or os.environ.get("QWEN3VL_2B_MODEL_DIR")
    if not value:
        raise SystemExit(
            "--model-id or QWEN3VL_2B_MODEL_DIR is required; models are never downloaded"
        )
    path = Path(str(value)).expanduser()
    if not path.exists():
        raise SystemExit(f"local model path does not exist: {path}")
    return str(path)


def _peak(*values: object) -> float | None:
    numbers = [float(value) for value in values if isinstance(value, (int, float))]
    return max(numbers) if numbers else None


def run(args: argparse.Namespace) -> dict[str, Any]:
    model_id = _model_id(args)
    config = AnswerabilityConfig()
    prompt = (
        "Assess evidence sufficiency only; do not answer the task. Choose one structured status.\n"
        f"Task question: {args.question}\nPrompt version: {config.prompt_version}"
    )
    engine = HuggingFaceVLMEngine(
        model_id,
        device="auto",
        dtype="auto",
        max_new_tokens=args.reasoning_max_new_tokens,
        local_files_only=True,
    )
    choice_args = {
        "choice_ids": STATUSES,
        "option_texts": STATUSES,
        "answer_type": "CHOICE_SINGLE",
        "single_choice_suffix": config.single_decision_suffix,
        "multi_verify_template": config.multi_verify_template,
    }
    try:
        cached_started = time.perf_counter()
        cached = engine.reason_and_choose(
            prompt,
            list(args.image),
            reasoning_max_new_tokens=args.reasoning_max_new_tokens,
            **choice_args,
        )
        cached_wall_ms = (time.perf_counter() - cached_started) * 1000.0

        uncached_started = time.perf_counter()
        first_stage = engine.reason_with_cache(
            prompt,
            list(args.image),
            max_new_tokens=args.reasoning_max_new_tokens,
        )
        first_stage.session.close()
        # The baseline intentionally starts a fresh multimodal session for the
        # status step, reproducing the visual/text prefill that KV reuse avoids.
        second_stage = engine.reason_and_choose(
            prompt + "\n\nNow select the evidence sufficiency status in a fresh session.",
            list(args.image),
            reasoning_max_new_tokens=1,
            **choice_args,
        )
        uncached_wall_ms = (time.perf_counter() - uncached_started) * 1000.0
    finally:
        active_sessions = engine.active_session_count
        engine.close()

    cached_initial = int(cached.metadata.get("initial_prefill_tokens", 0))
    cached_reasoning = int(cached.metadata.get("reasoning_tokens", 0))
    uncached_initial = int(first_stage.metadata.get("initial_prefill_tokens", 0)) + int(
        second_stage.metadata.get("initial_prefill_tokens", 0)
    )
    return {
        "sample_id": args.sample_id,
        "model": model_id,
        "cache_on": {
            "selected_status": cached.selected_ids[0],
            "prefill_tokens": cached_initial,
            "reused_tokens": cached_initial + cached_reasoning,
            "ttft_ms": None,
            "total_latency_ms": cached.latency_ms.get("total_ms", cached_wall_ms),
            "wall_latency_ms": cached_wall_ms,
            "peak_vram_mb": cached.metadata.get("peak_vram_mb"),
            "visual_prefill_count": 1,
            "cache_reused": cached.cache_reused,
        },
        "cache_off": {
            "selected_status": second_stage.selected_ids[0],
            "prefill_tokens": uncached_initial,
            "reused_tokens": 0,
            "ttft_ms": None,
            "total_latency_ms": uncached_wall_ms,
            "wall_latency_ms": uncached_wall_ms,
            "peak_vram_mb": _peak(
                first_stage.metadata.get("peak_vram_mb"),
                second_stage.metadata.get("peak_vram_mb"),
            ),
            "visual_prefill_count": 2,
            "cache_reused": False,
        },
        "metric_notes": {
            "ttft_ms": "backend does not expose first-token timing; null is explicit",
            "cache_off": "reasoning and status stages use separate multimodal sessions",
        },
        "active_session_count_after_inference": active_sessions,
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
