"""Fine-tune a local Qwen text judge with a small LoRA adapter.

This is intentionally a development-only training entry point.  It consumes the
JSONL files made by ``prepare_local_judge_sft_dataset.py`` and writes an adapter,
metrics and a reproducibility manifest to an explicitly named directory.  The
base model is loaded read-only and is never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sat_rs_vlm.evaluation.change_judge import (  # noqa: E402
    LOCAL_JUDGE_PROMPT_VERSION,
    build_judge_messages,
    conservative_rule_decision,
    parse_judge_output,
)

IMPLEMENTATION_VERSION = "levir-local-judge-lora-dev-v1.0"
_VALID_LABELS = {"0", "1"}


@dataclass(frozen=True)
class SftExample:
    """One supervised Caption-to-binary-label example."""

    sample_id: str
    caption: str
    target_label: str
    quality_weight: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--validation-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=1536)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument(
        "--decision-routing",
        choices=("direct", "cascade"),
        default="direct",
        help="Use the model directly or preserve the existing high-confidence rule-first cascade.",
    )
    parser.add_argument("--resume-adapter", type=Path)
    parser.add_argument(
        "--validation-role",
        choices=("development", "locked_holdout"),
        default="development",
        help="Declare whether the validation file is for development or final locked-holdout use.",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Skip optimization and evaluate the base model or adapter on the validation split.",
    )
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_examples(path: Path, *, allow_uncertain: bool = False) -> list[SftExample]:
    """Read and validate deterministic SFT JSONL without model dependencies."""

    examples: list[SftExample] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            sample_id = str(payload.get("id", "")).strip()
            messages = payload.get("messages")
            target = str(payload.get("target_label", "")).strip()
            permitted_labels = _VALID_LABELS | ({"U"} if allow_uncertain else set())
            if not sample_id or sample_id in seen or target not in permitted_labels:
                raise ValueError(f"invalid or duplicate example at {path}:{line_number}")
            if not isinstance(messages, list) or len(messages) < 3:
                raise ValueError(f"messages missing from {path}:{line_number}")
            user_content = str(messages[-2].get("content", ""))
            prefix, suffix = "Caption to classify:\n<caption>\n", "\n</caption>"
            if not user_content.startswith(prefix) or not user_content.endswith(suffix):
                raise ValueError(f"caption wrapper missing from {path}:{line_number}")
            caption = user_content[len(prefix) : -len(suffix)].strip()
            if not caption:
                raise ValueError(f"empty caption at {path}:{line_number}")
            weight = float(payload.get("quality_weight", 1.0))
            if not 0 < weight <= 1:
                raise ValueError(f"quality_weight must be in (0, 1] at {path}:{line_number}")
            examples.append(SftExample(sample_id, caption, target, weight))
            seen.add(sample_id)
    if not examples:
        raise ValueError(f"no examples found in {path}")
    return examples


def build_training_features(
    tokenizer: Any, example: SftExample, max_seq_length: int
) -> dict[str, Any]:
    """Mask the prompt so cross entropy trains only the one-character answer."""

    prompt = tokenizer.apply_chat_template(
        build_judge_messages(example.caption),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(
        example.target_label + tokenizer.eos_token, add_special_tokens=False
    )["input_ids"]
    input_ids = prompt_ids + answer_ids
    if len(input_ids) > max_seq_length:
        raise ValueError(
            f"{example.sample_id} has {len(input_ids)} tokens, above --max-seq-length "
            f"{max_seq_length}; increase the limit rather than truncating its instruction."
        )
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + answer_ids,
        "quality_weight": example.quality_weight,
    }


def accuracy_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return binary classification metrics, treating invalid judge output as wrong."""

    scored_rows = [row for row in rows if str(row["target_label"]) in _VALID_LABELS]
    pairs = [(str(row["target_label"]), row.get("prediction")) for row in scored_rows]
    tp = sum(label == "1" and prediction == 1 for label, prediction in pairs)
    tn = sum(label == "0" and prediction == 0 for label, prediction in pairs)
    fp = sum(label == "0" and prediction != 0 for label, prediction in pairs)
    fn = sum(label == "1" and prediction != 1 for label, prediction in pairs)
    unresolved = sum(prediction not in {0, 1} for _, prediction in pairs)
    total = len(scored_rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "num_input_samples": len(rows),
        "num_scored_binary_samples": total,
        "num_unscored_uncertain_references": len(rows) - total,
        "accuracy": (tp + tn) / total if total else 0.0,
        "balanced_accuracy": (recall + specificity) / 2,
        "change_precision": precision,
        "change_recall": recall,
        "change_f1": f1,
        "no_change_recall": specificity,
        "unresolved_rate_on_scored_binary": unresolved / total if total else 0.0,
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def _seed_everything(torch: Any, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _validate_args(args: argparse.Namespace) -> None:
    if args.gradient_accumulation < 1:
        raise ValueError("gradient-accumulation must be positive")
    if not args.evaluate_only and (args.epochs < 1 or args.max_steps < 1):
        raise ValueError("epochs and max-steps must be positive when training")
    if args.max_seq_length < 64 or args.lora_rank < 1 or args.lora_alpha < 1:
        raise ValueError("max-seq-length, lora-rank and lora-alpha are invalid")
    if args.learning_rate <= 0 or not 0 <= args.lora_dropout < 1:
        raise ValueError("learning-rate or lora-dropout is invalid")


def main() -> int:
    args = parse_args()
    try:
        _validate_args(args)
        output_dir = args.output_dir.resolve()
        if output_dir.exists() and any(output_dir.iterdir()) and not args.allow_overwrite:
            raise ValueError(f"output directory is non-empty: {output_dir}")
        train_path, validation_path = args.train_jsonl.resolve(), args.validation_jsonl.resolve()
        train = read_examples(train_path)
        validation = read_examples(validation_path, allow_uncertain=args.evaluate_only)
        if {row.caption for row in train} & {row.caption for row in validation}:
            raise ValueError("train and validation captions overlap")
        import torch
        import torch.nn.functional as functional
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except (ImportError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"LoRA training setup failed: {exc}", file=sys.stderr)
        return 2

    if not torch.cuda.is_available():
        print("LoRA training requires a CUDA GPU in this development profile.", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)
    _seed_everything(torch, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        train_features = [
            build_training_features(tokenizer, row, args.max_seq_length) for row in train
        ]
    except ValueError as exc:
        print(f"Feature preparation failed: {exc}", file=sys.stderr)
        return 2

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    base_model.config.use_cache = False
    if args.resume_adapter is not None:
        model = PeftModel.from_pretrained(
            base_model,
            args.resume_adapter,
            is_trainable=not args.evaluate_only,
        )
    elif not args.evaluate_only:
        base_model.gradient_checkpointing_enable()
        base_model.enable_input_require_grads()
        model = get_peft_model(
            base_model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
            ),
        )
    else:
        model = base_model
    history: list[dict[str, Any]] = []
    global_step = 0
    if not args.evaluate_only:
        model.train()
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=args.learning_rate,
        )
        optimizer.zero_grad(set_to_none=True)
        for epoch in range(1, args.epochs + 1):
            shuffled = list(train_features)
            random.Random(args.seed + epoch).shuffle(shuffled)
            for index, features in enumerate(shuffled, start=1):
                input_ids = torch.tensor([features["input_ids"]], device="cuda")
                attention_mask = torch.tensor([features["attention_mask"]], device="cuda")
                labels = torch.tensor([features["labels"]], device="cuda")
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                shift_logits = outputs.logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
                (loss * features["quality_weight"] / args.gradient_accumulation).backward()
                if index % args.gradient_accumulation == 0 or index == len(shuffled):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    history.append(
                        {"epoch": epoch, "step": global_step, "loss": float(loss.item())}
                    )
                    if global_step >= args.max_steps:
                        break
            if global_step >= args.max_steps:
                break

    model.eval()
    validation_rows: list[dict[str, Any]] = []
    for example in validation:
        rule_decision = (
            conservative_rule_decision(example.caption)
            if args.decision_routing == "cascade"
            else None
        )
        if rule_decision is None:
            prompt = tokenizer.apply_chat_template(
                build_judge_messages(example.caption),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            encoded = tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                )
            raw_output = tokenizer.batch_decode(
                generated[:, encoded["input_ids"].shape[1] :], skip_special_tokens=True
            )[0]
            decision = parse_judge_output(raw_output)
        else:
            decision = rule_decision
            raw_output = decision.raw_output
        validation_rows.append(
            {
                "id": example.sample_id,
                "target_label": example.target_label,
                "prediction": decision.value,
                "raw_output": raw_output,
                "status": decision.status,
                "source": decision.source,
            }
        )
    adapter_dir = output_dir / "adapter"
    if not args.evaluate_only:
        model.save_pretrained(adapter_dir, safe_serialization=True)
        tokenizer.save_pretrained(adapter_dir)
    metrics = accuracy_summary(validation_rows)
    (output_dir / "validation_predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in validation_rows),
        encoding="utf-8",
    )
    (output_dir / "training_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "implementation_version": IMPLEMENTATION_VERSION,
        "prompt_version": LOCAL_JUDGE_PROMPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": str(args.model.resolve()),
        "base_model_modified": False,
        "adapter_directory": "adapter" if not args.evaluate_only else None,
        "train_jsonl": str(train_path),
        "train_sha256": sha256(train_path),
        "validation_jsonl": str(validation_path),
        "validation_sha256": sha256(validation_path),
        "train_count": len(train),
        "validation_count": len(validation),
        "validation_role": args.validation_role,
        "decision_routing": args.decision_routing,
        "train_label_distribution": dict(Counter(row.target_label for row in train)),
        "validation_label_distribution": dict(Counter(row.target_label for row in validation)),
        "training": {
            "evaluation_only": args.evaluate_only,
            "seed": args.seed,
            "epochs_requested": args.epochs,
            "optimizer_steps": global_step,
            "max_steps": args.max_steps,
            "learning_rate": args.learning_rate,
            "gradient_accumulation": args.gradient_accumulation,
            "max_seq_length": args.max_seq_length,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "cuda_device": torch.cuda.get_device_name(0),
            "peak_cuda_memory_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
        },
        "validation_metrics": metrics,
        "development_only": args.validation_role == "development",
        "notes": [
            (
                "The development validation set is used only for short-run configuration checks."
                if args.validation_role == "development"
                else (
                    "This locked holdout is excluded from optimization and used only for final "
                    "validation."
                )
            ),
            (
                "This adapter is not a final model selection result and requires a separate locked "
                "holdout audit."
                if args.validation_role == "development"
                else "No further tuning may use this locked holdout after this run."
            ),
            "The base model directory remains read-only; only the LoRA adapter is saved.",
        ],
    }
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not args.evaluate_only:
        print(f"Saved LoRA adapter: {adapter_dir}")
    print(json.dumps(metrics, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
