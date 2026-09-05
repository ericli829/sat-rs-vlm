"""Train a text-only Qwen3 LoRA on canonical TaskGraph Planner DSL."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import shutil
import time
from pathlib import Path
from typing import Any

import yaml

from taskgraph_lab.training.text_planner_collator import PlannerCausalLMCollator

LAB_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path: str, *, base: Path) -> Path:
    candidate = Path(path)
    return (candidate if candidate.is_absolute() else base / candidate).resolve()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    ids = [str(row.get("id", "")) for row in rows]
    if any(not sample_id for sample_id in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"Planner dataset has missing or duplicate ids: {path}")
    return rows


class MessageDataset:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("training config must be a mapping")
    return payload


def _paths(config: dict[str, Any], output_override: Path | None) -> dict[str, Path]:
    root = LAB_ROOT.parent
    model = dict(config["model"])
    data = dict(config["data"])
    training = dict(config["training"])
    return {
        "model": _resolve(str(model["model_dir"]), base=root),
        "train": _resolve(str(data["train_file"]), base=root),
        "validation": _resolve(str(data["validation_file"]), base=root),
        "manifest": _resolve(str(data["dataset_manifest"]), base=root),
        "output": (
            output_override.resolve()
            if output_override is not None
            else _resolve(str(training["output_dir"]), base=root)
        ),
    }


def _audit_trainables(model: Any) -> dict[str, Any]:
    trainable = [
        (name, int(value.numel()))
        for name, value in model.named_parameters()
        if value.requires_grad
    ]
    invalid = [name for name, _ in trainable if "lora_" not in name]
    if not trainable or invalid:
        raise RuntimeError(f"unexpected Qwen3 Planner trainables: {invalid[:20]}")
    total = sum(int(value.numel()) for value in model.parameters())
    count = sum(value for _, value in trainable)
    return {
        "scope": "qwen3.causal_lm.lora_only",
        "trainable_parameters": count,
        "total_parameters": total,
        "trainable_ratio": count / total,
        "trainable_tensor_count": len(trainable),
        "invalid_names": [],
        "sample_names": [name for name, _ in trainable[:50]],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=LAB_ROOT / "configs" / "qwen3_1_7b_planner_lora_v1_local.yaml",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = _load_config(config_path)
    paths = _paths(config, args.output_dir)
    for label in ("model", "train", "validation", "manifest"):
        if not paths[label].exists():
            raise FileNotFoundError(f"{label} path does not exist: {paths[label]}")
    train_rows = _jsonl(paths["train"])
    validation_rows = _jsonl(paths["validation"])

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(paths["model"]),
        local_files_only=bool(config["model"].get("local_files_only", True)),
        trust_remote_code=bool(config["model"].get("trust_remote_code", True)),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    collator = PlannerCausalLMCollator(
        tokenizer,
        max_seq_length=int(config["data"]["max_seq_length"]),
    )
    diagnostics = [collator.diagnostics(row) for row in train_rows + validation_rows]
    if any(bool(row["truncated"]) for row in diagnostics):
        raise ValueError("Planner data would be truncated by max_seq_length")
    preflight = {
        "model": str(paths["model"]),
        "train_file": str(paths["train"]),
        "validation_file": str(paths["validation"]),
        "train_samples": len(train_rows),
        "validation_samples": len(validation_rows),
        "max_tokens": max(int(row["uncapped_total_tokens"]) for row in diagnostics),
        "max_seq_length": int(config["data"]["max_seq_length"]),
        "enable_thinking": False,
        "loss": "assistant_only_shifted_token_cross_entropy",
    }
    print(json.dumps({"preflight": preflight}, ensure_ascii=False), flush=True)
    if args.dry_run:
        return 0

    output = paths["output"]
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"training output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "preflight.json").write_text(
        json.dumps(preflight, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copy2(config_path, output / "training_config_source.yaml")

    import peft
    import torch
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, set_seed

    started = time.perf_counter()
    model = None
    trainer = None
    optimizer = None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Qwen3-1.7B Planner training")
        torch.cuda.reset_peak_memory_stats()
        set_seed(int(config["training"]["seed"]))
        dtype_name = str(config["model"].get("torch_dtype", "bfloat16"))
        dtype = getattr(torch, dtype_name)
        model = AutoModelForCausalLM.from_pretrained(
            str(paths["model"]),
            local_files_only=bool(config["model"].get("local_files_only", True)),
            trust_remote_code=bool(config["model"].get("trust_remote_code", True)),
            torch_dtype=dtype,
            attn_implementation=str(config["model"].get("attention_backend", "sdpa")),
            device_map="auto",
        )
        model.config.use_cache = False
        lora = config["lora"]
        model = peft.get_peft_model(
            model,
            peft.LoraConfig(
                r=int(lora["r"]),
                lora_alpha=int(lora["alpha"]),
                lora_dropout=float(lora["dropout"]),
                target_modules=list(lora["target_modules"]),
                task_type="CAUSAL_LM",
            ),
        )
        if bool(config["training"].get("gradient_checkpointing", True)):
            model.enable_input_require_grads()
        audit = _audit_trainables(model)
        print("Planner trainable audit: " + json.dumps(audit, sort_keys=True), flush=True)
        training = config["training"]
        arguments = TrainingArguments(
            output_dir=str(output),
            num_train_epochs=float(training["num_train_epochs"]),
            max_steps=int(args.max_steps) if args.max_steps is not None else -1,
            per_device_train_batch_size=int(training["per_device_train_batch_size"]),
            per_device_eval_batch_size=int(training["per_device_eval_batch_size"]),
            gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            warmup_ratio=float(training["warmup_ratio"]),
            lr_scheduler_type=str(training["lr_scheduler_type"]),
            logging_steps=int(training["logging_steps"]),
            eval_strategy="steps",
            eval_steps=int(training["eval_steps"]),
            save_strategy="steps",
            save_steps=int(training["save_steps"]),
            save_total_limit=int(training["save_total_limit"]),
            max_grad_norm=float(training["max_grad_norm"]),
            bf16=bool(training["bf16"]),
            fp16=False,
            gradient_checkpointing=bool(training["gradient_checkpointing"]),
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataloader_num_workers=0,
            dataloader_pin_memory=True,
            report_to="none",
            remove_unused_columns=False,
            seed=int(training["seed"]),
        )
        trainer = Trainer(
            model=model,
            args=arguments,
            train_dataset=MessageDataset(train_rows),
            eval_dataset=MessageDataset(validation_rows),
            data_collator=collator,
            processing_class=tokenizer,
        )
        result = trainer.train()
        trainer.save_model(str(output))
        tokenizer.save_pretrained(output / "tokenizer")
        trainer.state.save_to_json(str(output / "trainer_state.json"))
        report = {
            "success": True,
            **preflight,
            "model_config_sha256": _sha256(paths["model"] / "config.json"),
            "train_sha256": _sha256(paths["train"]),
            "validation_sha256": _sha256(paths["validation"]),
            "dataset_manifest_sha256": _sha256(paths["manifest"]),
            "config_sha256": _sha256(config_path),
            "adapter_sha256": _sha256(output / "adapter_model.safetensors"),
            "trainable_audit": audit,
            "global_step": int(trainer.state.global_step),
            "metrics": result.metrics,
            "duration_seconds": time.perf_counter() - started,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / (1024**2),
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "peft")
            },
        }
        (output / "training_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception as exc:
        (output / "failure.json").write_text(
            json.dumps(
                {"success": False, "error_type": type(exc).__name__, "error": str(exc)},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        raise
    finally:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        trainer = None
        model = None
        tokenizer = None
        gc.collect()
        if "torch" in locals() and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


if __name__ == "__main__":
    raise SystemExit(main())
