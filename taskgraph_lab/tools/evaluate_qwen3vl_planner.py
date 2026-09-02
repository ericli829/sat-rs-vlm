"""Evaluate a local Qwen3-VL Planner adapter with bounded recovery and optional RAG."""

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

from sat_rs_vlm.models.qwen3vl_loader import compatible_model_class
from taskgraph_lab.evaluation.constrained_decoding import GreedyDSLLogitsProcessor
from taskgraph_lab.evaluation.planner_generation import (
    evaluate_prediction,
    prompt_messages,
    summarize_predictions,
)
from taskgraph_lab.evaluation.planner_recovery import (
    PlannerRetryPolicy,
    RecoveryAction,
    RecoveryLevel,
    concise_validator_diagnostic,
    retry_prompt_messages,
)
from taskgraph_lab.retrieval.hard_cheat_sheet import (
    HARD_INTENTS,
    CheatSheetRetriever,
    compose_cheat_sheet_prompt,
    route_hard_intent,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--intent-filter", action="append", default=[])
    parser.add_argument("--constrained", action="store_true")
    parser.add_argument("--constraint-top-k", type=int, default=64)
    parser.add_argument("--constraint-max-candidate-checks", type=int, default=256)
    parser.add_argument("--constraint-max-nodes", type=int, default=24)
    parser.add_argument("--repeat-guard-repetitions", type=int, default=4)
    parser.add_argument("--max-finish-node-tokens", type=int, default=32)
    parser.add_argument("--enable-recovery", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument(
        "--rag-mode",
        choices=("off", "hard_intent", "fallback_only"),
        default="off",
    )
    parser.add_argument("--rag-bank", type=Path)
    parser.add_argument("--rag-rule-cards", type=Path)
    parser.add_argument("--rag-rules", action="store_true")
    parser.add_argument("--rag-top-k", type=int, choices=(0, 2, 3), default=2)
    parser.add_argument(
        "--rag-router",
        choices=("heuristic", "benchmark_metadata_intent"),
        default="heuristic",
        help=(
            "Production-safe question heuristic, or an evaluation-only oracle for a "
            "pre-filtered hard-intent benchmark."
        ),
    )
    return parser.parse_args()


def _user_payload(row: dict[str, Any]) -> dict[str, Any]:
    messages = row.get("messages") or []
    users = [message for message in messages if message.get("role") == "user"]
    if len(users) != 1:
        raise ValueError("Planner evaluation row must contain exactly one user message")
    payload = json.loads(str(users[0].get("content", "")))
    if not isinstance(payload, dict):
        raise TypeError("Planner user message must decode to a JSON object")
    return payload


def _image_refs(row: dict[str, Any]) -> list[str]:
    inputs = _user_payload(row).get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("Planner evaluation row must provide at least one image input")
    return [f"${name}" for name in inputs]


def _filtered_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.sample_id:
        requested = set(args.sample_id)
        rows = [row for row in rows if str(row.get("id")) in requested]
        missing = sorted(requested - {str(row.get("id")) for row in rows})
        if missing:
            raise ValueError(f"requested sample ids are absent: {missing}")
    if args.intent_filter:
        requested_intents = set(args.intent_filter)
        rows = [
            row
            for row in rows
            if str((row.get("metadata") or {}).get("intent")) in requested_intents
        ]
    if args.limit is not None:
        rows = rows[: args.limit]
    return rows


def _rag_messages(
    messages: list[dict[str, Any]],
    *,
    question: str,
    routed_intent: str | None,
    retriever: CheatSheetRetriever | None,
    rule_cards: str,
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = time.perf_counter()
    retrieved = []
    retrieval_error = None
    try:
        if retriever is not None and top_k:
            retrieved, measured_ms = retriever.retrieve(
                question,
                top_k=top_k,
                intent=routed_intent,
            )
        else:
            measured_ms = (time.perf_counter() - started) * 1000.0
    except Exception as exc:  # RAG is advisory; decoding remains fail-closed.
        measured_ms = (time.perf_counter() - started) * 1000.0
        retrieval_error = f"{type(exc).__name__}: {exc}"
        retrieved = []
    augmented = [dict(message) for message in messages]
    system_indexes = [
        index for index, message in enumerate(augmented) if message.get("role") == "system"
    ]
    if len(system_indexes) != 1:
        raise ValueError("Planner prompt must contain exactly one system message")
    index = system_indexes[0]
    augmented[index]["content"] = compose_cheat_sheet_prompt(
        str(augmented[index].get("content", "")),
        rule_cards=rule_cards,
        retrieved=retrieved,
    )
    return augmented, {
        "rag_used": bool(rule_cards.strip() or retrieved),
        "retrieved_example_ids": [result.example["example_id"] for result in retrieved],
        "retrieval_scores": [result.log_record() for result in retrieved],
        "retrieval_latency_ms": measured_ms,
        "retrieval_error": retrieval_error,
    }


def _generate_attempt(
    *,
    model: Any,
    processor: Any,
    tokenizer: Any,
    torch: Any,
    input_device: Any,
    pad_token_id: int,
    row: dict[str, Any],
    messages: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], str]:
    prompt = str(
        processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    )
    encoded = processor(
        text=[prompt],
        images=None,
        videos=None,
        padding=True,
        truncation=True,
        max_length=args.max_prompt_tokens,
        return_tensors="pt",
    )
    encoded = {
        key: value.to(input_device) if hasattr(value, "to") else value
        for key, value in encoded.items()
    }
    prompt_width = int(encoded["input_ids"].shape[1])
    prompt_tokens = int(encoded.get("attention_mask", encoded["input_ids"]).sum().item())
    constraint_processor = None
    generation_kwargs: dict[str, Any] = {}
    if args.constrained:
        constraint_processor = GreedyDSLLogitsProcessor(
            tokenizer,
            prompt_width=prompt_width,
            image_refs_by_row=[_image_refs(row)],
            initial_top_k=args.constraint_top_k,
            max_candidate_checks=args.constraint_max_candidate_checks,
            max_nodes=args.constraint_max_nodes,
            repeat_guard_repetitions=args.repeat_guard_repetitions,
            max_finish_node_tokens=args.max_finish_node_tokens,
        )
        generation_kwargs["logits_processor"] = [constraint_processor]
    begin = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            do_sample=False,
            num_beams=1,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=pad_token_id,
            use_cache=True,
            **generation_kwargs,
        )
    elapsed = time.perf_counter() - begin
    continuation = generated[0, prompt_width:]
    text = str(
        processor.decode(
            continuation,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    )
    record = evaluate_prediction(row, text)
    generated_tokens = int((continuation != pad_token_id).sum().item())
    record.update(
        {
            "latency_seconds": elapsed,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "constrained": args.constrained,
        }
    )
    if constraint_processor is not None:
        record.update(
            constraint_processor.diagnostics(
                0,
                continuation,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=pad_token_id,
            )
        )
    else:
        record["termination_reason"] = (
            "final"
            if record["dsl_parse_valid"]
            else "max_tokens"
            if generated_tokens >= args.max_new_tokens
            else "error"
        )
        record["constraint_failure"] = None
    del encoded, generated, continuation, constraint_processor
    return record, text


def _evaluate_row(
    *,
    model: Any,
    processor: Any,
    tokenizer: Any,
    torch: Any,
    input_device: Any,
    pad_token_id: int,
    row: dict[str, Any],
    args: argparse.Namespace,
    retriever: CheatSheetRetriever | None,
    rule_cards: str,
) -> dict[str, Any]:
    payload = _user_payload(row)
    question = str(payload.get("question", ""))
    routed_intent = route_hard_intent(question)
    if args.rag_router == "benchmark_metadata_intent":
        metadata_intent = str((row.get("metadata") or {}).get("intent") or "")
        routed_intent = metadata_intent if metadata_intent in HARD_INTENTS else None
    hard_case = routed_intent is not None
    rag_available = bool(rule_cards.strip() or (retriever and retriever.examples))
    policy = PlannerRetryPolicy(max_attempts=args.max_attempts)
    original_messages = prompt_messages(row)
    attempts: list[dict[str, Any]] = []
    next_messages = original_messages
    next_level = RecoveryLevel.NORMALIZATION
    use_rag = args.rag_mode == "hard_intent" and hard_case
    final_record: dict[str, Any] | None = None

    while True:
        rag_log = {
            "rag_used": False,
            "retrieved_example_ids": [],
            "retrieval_scores": [],
            "retrieval_latency_ms": 0.0,
            "retrieval_error": None,
        }
        attempt_messages = next_messages
        if use_rag:
            attempt_messages, rag_log = _rag_messages(
                attempt_messages,
                question=question,
                routed_intent=routed_intent,
                retriever=retriever,
                rule_cards=rule_cards,
                top_k=args.rag_top_k,
            )
        generated_record, prediction = _generate_attempt(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            torch=torch,
            input_device=input_device,
            pad_token_id=pad_token_id,
            row=row,
            messages=attempt_messages,
            args=args,
        )
        attempt_number = len(attempts) + 1
        attempts.append(
            {
                "attempt": attempt_number,
                "recovery_level": int(next_level),
                "recovery_level_name": next_level.name.lower(),
                "termination_reason": generated_record.get("termination_reason"),
                "constraint_failure": generated_record.get("constraint_failure"),
                "validator_error_codes": generated_record.get("validation_error_codes") or [],
                "surface_grammar_valid": generated_record.get("surface_grammar_valid"),
                "dsl_parse_valid": generated_record.get("dsl_parse_valid"),
                "graph_runtime_valid": generated_record.get("graph_runtime_valid"),
                "latency_seconds": generated_record["latency_seconds"],
                "prompt_tokens": generated_record["prompt_tokens"],
                "generated_tokens": generated_record["generated_tokens"],
                **rag_log,
            }
        )
        final_record = generated_record
        if not args.enable_recovery:
            break
        decision = policy.decide(
            generated_record,
            attempts=attempt_number,
            hard_case=hard_case,
            rag_available=rag_available and args.rag_mode != "off",
        )
        if decision.action is RecoveryAction.RETURN:
            break
        if decision.action is RecoveryAction.FAIL:
            final_record["termination_reason"] = "planner_failed"
            break
        diagnostic = concise_validator_diagnostic(generated_record)
        next_messages = retry_prompt_messages(
            original_messages,
            previous_prediction=prediction,
            diagnostic=diagnostic,
        )
        next_level = decision.level
        use_rag = (
            args.rag_mode == "hard_intent" and hard_case
        ) or decision.action is RecoveryAction.RAG_RETRY

    if final_record is None:
        raise AssertionError("Planner produced no attempts")
    retrieved_ids = list(
        dict.fromkeys(
            example_id
            for attempt in attempts
            for example_id in attempt["retrieved_example_ids"]
        )
    )
    retrieval_latency_ms = sum(float(attempt["retrieval_latency_ms"]) for attempt in attempts)
    generation_latency = sum(float(attempt["latency_seconds"]) for attempt in attempts)
    final_record.update(
        {
            "routed_intent": routed_intent,
            "generation_attempts": len(attempts),
            "recovery_level": max(int(attempt["recovery_level"]) for attempt in attempts),
            "rag_mode": args.rag_mode,
            "rag_used": any(bool(attempt["rag_used"]) for attempt in attempts),
            "retrieved_example_ids": retrieved_ids,
            "retrieval_latency_ms": retrieval_latency_ms,
            "latency_per_attempt": [attempt["latency_seconds"] for attempt in attempts],
            "total_planner_latency": generation_latency + retrieval_latency_ms / 1000.0,
            "prompt_tokens": sum(int(attempt["prompt_tokens"]) for attempt in attempts),
            "generated_tokens": sum(int(attempt["generated_tokens"]) for attempt in attempts),
            "validator_error_code": final_record.get("validation_error_codes") or [],
            "attempts": attempts,
        }
    )
    return final_record


def _validate_args(args: argparse.Namespace) -> None:
    if args.batch_size != 1:
        raise ValueError("bounded recovery evaluator currently requires --batch-size 1")
    if args.constraint_top_k < 1:
        raise ValueError("--constraint-top-k must be positive")
    if args.constraint_max_candidate_checks < args.constraint_top_k:
        raise ValueError("candidate checks must be >= constraint top-k")
    if args.rag_mode != "off" and not args.constrained:
        raise ValueError("RAG inference must remain grammar constrained")
    if args.enable_recovery and not args.constrained:
        raise ValueError(
            "recovery must remain grammar constrained; unconstrained fallback is forbidden"
        )
    if args.rag_top_k and args.rag_bank is None and args.rag_mode != "off":
        raise ValueError("--rag-bank is required when RAG top-k is nonzero")
    if args.rag_rules and args.rag_rule_cards is None:
        raise ValueError("--rag-rule-cards is required with --rag-rules")
    if args.rag_router == "benchmark_metadata_intent" and not args.intent_filter:
        raise ValueError("benchmark metadata routing requires an explicit hard intent filter")
    if args.rag_router == "benchmark_metadata_intent" and not set(args.intent_filter).issubset(
        HARD_INTENTS
    ):
        raise ValueError("benchmark metadata routing may only be used on hard intent filters")


def main() -> int:
    args = parse_args()
    _validate_args(args)
    for path, label in (
        (args.base_model, "base model"),
        (args.adapter, "adapter"),
        (args.validation_file, "validation file"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    adapter_weights = args.adapter / "adapter_model.safetensors"
    if not adapter_weights.is_file():
        raise FileNotFoundError(f"adapter weights do not exist: {adapter_weights}")
    if args.rag_bank is not None and not args.rag_bank.is_file():
        raise FileNotFoundError(f"RAG bank does not exist: {args.rag_bank}")
    if args.rag_rule_cards is not None and not args.rag_rule_cards.is_file():
        raise FileNotFoundError(f"RAG rule cards do not exist: {args.rag_rule_cards}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = _filtered_rows(_read_jsonl(args.validation_file), args)
    retriever = CheatSheetRetriever.from_jsonl(args.rag_bank) if args.rag_bank else None
    rule_cards = (
        args.rag_rule_cards.read_text(encoding="utf-8")
        if args.rag_rules and args.rag_rule_cards
        else ""
    )
    started_at = datetime.now(UTC)
    provenance = {
        "schema_version": "taskgraph-planner-generation-eval-v2",
        "started_at": started_at.isoformat(),
        "base_model": str(args.base_model.resolve()),
        "adapter": str(args.adapter.resolve()),
        "adapter_weights_sha256": _sha256(adapter_weights),
        "validation_file": str(args.validation_file.resolve()),
        "validation_sha256": _sha256(args.validation_file),
        "sample_count": len(rows),
        "sample_ids": [str(row.get("id")) for row in rows],
        "generation": {
            "do_sample": False,
            "num_beams": 1,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "max_prompt_tokens": args.max_prompt_tokens,
            "repair": False,
            "constrained": args.constrained,
            "enable_recovery": args.enable_recovery,
            "max_attempts": args.max_attempts,
        },
        "constraint": {
            "implementation": "bounded_canonical_prefix_state_machine",
            "initial_top_k": args.constraint_top_k,
            "max_candidate_checks": args.constraint_max_candidate_checks,
            "max_nodes": args.constraint_max_nodes,
            "repeat_guard_repetitions": args.repeat_guard_repetitions,
            "max_finish_node_tokens": args.max_finish_node_tokens,
            "fallback": "fail_closed_no_unconstrained_fallback",
        },
        "rag": {
            "mode": args.rag_mode,
            "router": args.rag_router,
            "top_k": args.rag_top_k,
            "rules": args.rag_rules,
            "bank": str(args.rag_bank.resolve()) if args.rag_bank else None,
            "bank_sha256": _sha256(args.rag_bank) if args.rag_bank else None,
            "rule_cards": str(args.rag_rule_cards.resolve()) if args.rag_rule_cards else None,
            "rule_cards_sha256": (
                _sha256(args.rag_rule_cards) if args.rag_rule_cards else None
            ),
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
    processor = None
    try:
        import peft
        import torch as imported_torch
        import transformers

        torch = imported_torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this local 2B Planner evaluation")
        processor = transformers.AutoProcessor.from_pretrained(
            str(args.base_model), trust_remote_code=True, local_files_only=True
        )
        tokenizer = processor.tokenizer
        tokenizer.padding_side = "left"
        model_class = compatible_model_class(transformers)
        model = model_class.from_pretrained(
            str(args.base_model),
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="auto",
        )
        model = peft.PeftModel.from_pretrained(model, str(args.adapter), local_files_only=True)
        model.eval()
        input_device = next(model.parameters()).device
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        if pad_token_id is None:
            raise RuntimeError("tokenizer has neither pad_token_id nor eos_token_id")
        torch.cuda.reset_peak_memory_stats()
        with predictions_path.open("w", encoding="utf-8", newline="\n") as output:
            for row in rows:
                record = _evaluate_row(
                    model=model,
                    processor=processor,
                    tokenizer=tokenizer,
                    torch=torch,
                    input_device=input_device,
                    pad_token_id=int(pad_token_id),
                    row=row,
                    args=args,
                    retriever=retriever,
                    rule_cards=rule_cards,
                )
                results.append(record)
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                print(
                    json.dumps(
                        {
                            "completed": len(results),
                            "total": len(rows),
                            "sample_id": record["sample_id"],
                            "graph_runtime_valid": record["graph_runtime_valid"],
                            "canonical_exact": record["canonical_exact"],
                            "attempts": record["generation_attempts"],
                            "rag_used": record["rag_used"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
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
        processor = None
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


if __name__ == "__main__":
    raise SystemExit(main())
