"""Qwen3-VL-4B LR + Visual Merger 诊断矩阵编排器。

本模块只编排已有 ``scripts/train_qwen3vl_lora.py`` 和统一评测入口，不复制
Dataset、Collator、Trainer 或 loss。每个实验都从同一个 Round1 adapter 生成
独立的解析配置和输出目录，支持状态文件、单实验失败继续、OOM 重试及 resume。
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from sat_rs_vlm.training.config import load_training_config
from sat_rs_vlm.training.vit_probe import sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SWEEP_CONFIG = (
    PROJECT_ROOT / "configs/experiments/qwen3vl_4b_lr_merger_sweep_4090.yaml"
)
STATUS_VALUES = {"PENDING", "RUNNING", "TRAINED", "EVALUATED", "ANALYZED", "FAILED"}
EXPERIMENT_PATTERN = re.compile(r"^A[123]$|^B[123]$")


class InfrastructureError(RuntimeError):
    """不可安全继续的全局输入错误。"""


class EvaluationOOMError(RuntimeError):
    """E1 评测显存不足，调用方可执行唯一的 4→2 fallback。"""


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    label: str
    phase: str
    lora_lr: float | None
    merger_lr: float
    vit_last_n: int
    main_merger: bool
    deepstack: bool = False
    patch_embed: bool = False


def load_sweep_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Sweep config does not exist: {config_path}")
    payload = dict(yaml.safe_load(config_path.read_text(encoding="utf-8")) or {})
    common = dict(payload.get("common", {}))
    experiments = dict(payload.get("experiments", {}))
    required = {
        "base_training_config",
        "probe_dataset",
        "probe_manifest",
        "evaluation_config",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"Sweep config is missing fields: {missing}")
    for experiment_id in ("A1", "A2", "A3", "B1", "B2", "B3"):
        if experiment_id not in experiments:
            raise ValueError(f"Sweep config must define {experiment_id}")
    payload["common"] = common
    payload["experiments"] = experiments
    return payload


def parse_experiment_specs(
    payload: Mapping[str, Any], selected_lr: float | None = None
) -> list[ExperimentSpec]:
    experiments = dict(payload["experiments"])
    specs: list[ExperimentSpec] = []
    for experiment_id in ("A1", "A2", "A3", "B1", "B2", "B3"):
        item = dict(experiments[experiment_id])
        raw_lr = item.get("lora_lr")
        if raw_lr in (None, "auto_from_phase_a"):
            raw_lr = selected_lr
        specs.append(
            ExperimentSpec(
                experiment_id=experiment_id,
                label=str(item.get("label", experiment_id)),
                phase="A" if experiment_id.startswith("A") else "B",
                lora_lr=float(raw_lr) if raw_lr is not None else None,
                merger_lr=float(item.get("merger_lr", 0.0)),
                vit_last_n=int(item.get("vit_last_n", 0)),
                main_merger=bool(item.get("main_merger", False)),
                deepstack=bool(item.get("deepstack", False)),
                patch_embed=bool(item.get("patch_embed", False)),
            )
        )
    return specs


def checkpoint_fingerprint(adapter_dir: str | Path) -> dict[str, Any]:
    root = Path(adapter_dir)
    if not root.is_dir():
        raise InfrastructureError(f"Initial adapter directory does not exist: {root}")
    config = root / "adapter_config.json"
    weights = root / "adapter_model.safetensors"
    if not config.is_file() or not weights.is_file():
        raise InfrastructureError(
            "Initial adapter must contain adapter_config.json and adapter_model.safetensors: "
            f"{root}"
        )
    manifest_path = root / "strategy_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {}
    )
    if manifest.get("strategy") not in (None, "lora"):
        raise InfrastructureError(f"Initial adapter is not a LoRA adapter: {root}")
    return {
        "path": str(root.resolve()),
        "adapter_config_sha256": sha256_file(config),
        "adapter_weights_sha256": sha256_file(weights),
        "base_model_fingerprint": manifest.get("base_model_fingerprint"),
        "manifest_sha256": (
            sha256_file(manifest_path) if manifest_path.is_file() else None
        ),
    }


def validate_probe_contract(
    train_file: str | Path,
    manifest_file: str | Path,
    protected_manifest: str | Path,
) -> dict[str, Any]:
    """验证 probe JSONL、manifest SHA、唯一 ID 和 E1/E2/E3 泄漏保护。"""

    train_path = Path(train_file)
    manifest_path = Path(manifest_file)
    protected_path = Path(protected_manifest)
    for path in (train_path, manifest_path, protected_path):
        if not path.is_file():
            raise InfrastructureError(f"Probe contract file does not exist: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_sha = sha256_file(train_path)
    expected_sha = str(manifest.get("output_sha256", ""))
    if not expected_sha or actual_sha != expected_sha:
        raise InfrastructureError(
            f"Probe dataset SHA mismatch: expected={expected_sha}, actual={actual_sha}"
        )
    rows: list[dict[str, Any]] = []
    with train_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not row.get("id"):
                raise InfrastructureError(
                    f"Invalid probe row at {train_path}:{line_number}"
                )
            rows.append(row)
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise InfrastructureError("Probe dataset contains duplicate sample IDs")
    protected_payload = json.loads(protected_path.read_text(encoding="utf-8"))
    protected: set[str] = set()
    for tier in dict(protected_payload.get("tiers", {})).values():
        if isinstance(tier, Mapping):
            protected.update(str(value) for value in tier.get("sample_ids", []))
    overlap = sorted(set(ids).intersection(protected))
    if overlap:
        raise InfrastructureError(
            f"Probe dataset overlaps protected evaluation IDs: {overlap[:5]}"
        )
    if int(manifest.get("total_samples", len(rows))) != len(rows):
        raise InfrastructureError("Probe manifest total_samples does not match JSONL")
    return {
        "path": str(train_path.resolve()),
        "sha256": actual_sha,
        "sample_count": len(rows),
        "unique_count": len(set(ids)),
        "protected_overlap_count": len(overlap),
        "manifest": str(manifest_path.resolve()),
    }


def evaluation_identity(evaluation_dir: str | Path) -> dict[str, Any]:
    root = Path(evaluation_dir)
    manifest = root / "evaluation_v1_5" / "evaluation_manifest.json"
    summary = root / "evaluation_v1_5" / "summary.json"
    if not manifest.is_file() or not summary.is_file():
        raise FileNotFoundError(f"Evaluation v1.5 artifacts are incomplete: {root}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("evaluation_tier") != "E1":
        raise InfrastructureError(
            f"Expected E1 evaluation, got {payload.get('evaluation_tier')}"
        )
    latency_context = dict(payload.get("latency_context", {}))
    metadata_path = root / "evaluation_metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else {}
    )
    return {
        "directory": str(root.resolve()),
        "tier": payload.get("evaluation_tier"),
        "tier_sha256": payload.get("evaluation_tier_sha256"),
        "sample_count": payload.get("evaluated_samples"),
        "eval_batch_size": latency_context.get("eval_batch_size"),
        "manifest_sha256": sha256_file(manifest),
        "summary": json.loads(summary.read_text(encoding="utf-8")),
        "metadata": metadata,
    }


def build_evaluation_command(
    *,
    evaluation_config: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    batch_size: int,
) -> list[str]:
    """构造统一 E1 命令，显式传递 batch override，便于审计和单测。"""

    return [
        "scripts/evaluate_rs_vlm.py",
        "--config",
        str(evaluation_config),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output_dir),
        "--batch-size",
        str(batch_size),
    ]


def clear_cuda_cache() -> None:
    """清理当前编排进程可见的 CUDA cache；无 torch/GPU 时安全跳过。"""

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        # 评测本身在子进程中执行，清理失败不应掩盖原始 OOM 原因。
        return


def write_status(path: str | Path, status: str, **fields: Any) -> None:
    if status not in STATUS_VALUES:
        raise ValueError(f"Unknown sweep status: {status}")
    payload = {
        "status": status,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **fields,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _metric(summary: Mapping[str, Any], task: str, *names: str) -> float | None:
    """从 v1.5 summary 的 task metrics 中按别名读取数值。"""

    def walk(value: Any) -> Iterable[Mapping[str, Any]]:
        if isinstance(value, Mapping):
            yield value
            for child in value.values():
                yield from walk(child)
        elif isinstance(value, list):
            for child in value:
                yield from walk(child)

    for node in walk(summary):
        candidate = node.get(task)
        if isinstance(candidate, Mapping):
            for child in walk(candidate):
                metrics = child.get("metrics", child)
                if not isinstance(metrics, Mapping):
                    continue
                for name in names:
                    value = metrics.get(name)
                    if isinstance(value, Mapping):
                        value = value.get("value")
                    if isinstance(value, (int, float)):
                        return float(value)
    return None


def extract_e1_metrics(summary: Mapping[str, Any]) -> dict[str, float | None]:
    return {
        "caption_bleu1": _metric(summary, "captioning", "bleu_1_approx", "bleu1"),
        "caption_bleu4": _metric(summary, "captioning", "bleu_4_approx", "bleu4"),
        "caption_rouge_l": _metric(
            summary, "captioning", "rouge_l_f1_approx", "rouge_l"
        ),
        "caption_meteor": _metric(
            summary, "captioning", "meteor_exact_approx", "meteor"
        ),
        "caption_chrf": _metric(summary, "captioning", "chrf_approx", "chrf"),
        "caption_cider": _metric(
            summary, "captioning", "cider_d_single_reference_approx", "cider"
        ),
        "count_exact": _metric(summary, "counting", "acc_exact", "exact_accuracy"),
        "count_pm1": _metric(summary, "counting", "acc_within_1", "within_1"),
        "count_mae": _metric(summary, "counting", "mae"),
        "count_rmse": _metric(summary, "counting", "rmse"),
        "scene_normalized": _metric(
            summary, "scene_classification", "normalized_exact_match", "normalized"
        ),
        "scene_exact": _metric(summary, "scene_classification", "exact_match", "exact"),
        "scene_keyword": _metric(
            summary, "scene_classification", "keyword_hit", "keyword"
        ),
        "vqa_normalized": _metric(
            summary, "vqa", "normalized_exact_match", "normalized"
        ),
        "vqa_exact": _metric(summary, "vqa", "exact_match", "exact"),
        "vqa_keyword": _metric(summary, "vqa", "keyword_hit", "keyword"),
        "detection_miou": _metric(summary, "detection", "mean_iou", "miou"),
        "detection_iou50": _metric(summary, "detection", "iou_at_0_5", "iou50"),
        "detection_iou70": _metric(summary, "detection", "iou_at_0_7", "iou70"),
        "detection_parse": _metric(
            summary, "detection", "valid_json_rate", "parse_success_rate"
        ),
        "levir_f1": _metric(summary, "change_detection", "change_f1", "f1"),
        "parse_success": _metric(summary, "overall", "parse_success_rate"),
    }


def weak_score(metrics: Mapping[str, Any]) -> float | None:
    values = [
        metrics.get("count_exact"),
        metrics.get("count_pm1"),
        metrics.get("scene_normalized"),
        metrics.get("vqa_normalized"),
        metrics.get("detection_miou"),
    ]
    if any(not isinstance(value, (int, float)) for value in values):
        return None
    return float(
        0.30 * float(metrics["count_exact"])
        + 0.15 * float(metrics["count_pm1"])
        + 0.25 * float(metrics["scene_normalized"])
        + 0.20 * float(metrics["vqa_normalized"])
        + 0.10 * float(metrics["detection_miou"])
    )


def select_phase_a_lr(
    candidates: Iterable[Mapping[str, Any]],
    baseline: Mapping[str, Any] | None,
    *,
    fallback_lr: float = 5.0e-5,
) -> dict[str, Any]:
    """按 weak score 和质量 guard 选择 Phase B LoRA LR。"""

    metric_names = {
        "parse_success",
        "detection_miou",
        "levir_f1",
        "caption_rouge_l",
        "count_exact",
        "count_pm1",
        "scene_normalized",
        "vqa_normalized",
    }

    def record_metrics(record: Mapping[str, Any] | None) -> dict[str, Any]:
        if record is None:
            return {}
        nested = record.get("metrics")
        if isinstance(nested, Mapping):
            return dict(nested)
        return {name: record.get(name) for name in metric_names if name in record}

    baseline_metrics = record_metrics(baseline)
    guard_limits = {
        "detection_miou": 0.04,
        "levir_f1": 0.04,
        "caption_rouge_l": 0.04,
    }
    eligible: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for candidate in candidates:
        metrics = record_metrics(candidate)
        score = weak_score(metrics)
        reasons: list[str] = []
        if score is None or candidate.get("status") not in {"EVALUATED", "ANALYZED"}:
            reasons.append("missing_metrics_or_incomplete")
        if (metrics.get("parse_success") or 0.0) < 0.99:
            reasons.append("parse_success_guard")
        for key, limit in guard_limits.items():
            candidate_value = metrics.get(key)
            baseline_value = baseline_metrics.get(key)
            if isinstance(candidate_value, (int, float)) and isinstance(
                baseline_value, (int, float)
            ):
                if float(candidate_value) < float(baseline_value) - limit:
                    reasons.append(f"{key}_guard")
        record = {
            "experiment_id": candidate.get("experiment_id"),
            "lora_lr": candidate.get("lora_lr"),
            "weak_score": score,
            "guard_reasons": reasons,
        }
        if reasons:
            rejected.append(record)
        else:
            eligible.append(record)
    if eligible:
        selected = max(eligible, key=lambda item: float(item["weak_score"] or -1.0))
        return {
            "selected_lr": float(selected["lora_lr"]),
            "selection_mode": "weak_score_with_guards",
            "selected_experiment": selected.get("experiment_id"),
            "eligible": eligible,
            "rejected": rejected,
        }
    return {
        "selected_lr": fallback_lr,
        "selection_mode": "fallback_due_to_guard_failure",
        "selected_experiment": None,
        "eligible": eligible,
        "rejected": rejected,
    }


def materialize_training_config(
    base_config_path: str | Path,
    *,
    output_path: str | Path,
    model_dir: str | Path,
    processor_dir: str | Path,
    train_file: str | Path,
    val_file: str | Path,
    image_root: str | Path,
    initial_adapter: str | Path,
    output_dir: str | Path,
    spec: ExperimentSpec,
    common: Mapping[str, Any],
    max_steps: int,
    batch_size: int,
    gradient_accumulation_steps: int,
) -> dict[str, Any]:
    """基于现有正式训练 YAML 生成单实验解析配置。"""

    base_path = Path(base_config_path)
    payload = dict(yaml.safe_load(base_path.read_text(encoding="utf-8")) or {})
    payload["model"].update(
        {
            "model_dir": str(model_dir),
            "processor_dir": str(processor_dir),
            "local_files_only": True,
        }
    )
    payload["data"].update(
        {
            "train_file": str(train_file),
            "val_file": str(val_file),
            "image_root": str(image_root),
            "max_train_samples": None,
            "max_eval_samples": None,
            "max_seq_length": int(
                common.get(
                    "max_seq_length", payload["data"].get("max_seq_length", 1024)
                )
            ),
        }
    )
    payload["training"].update(
        {
            "output_dir": str(output_dir),
            "method": "lora",
            "num_train_epochs": None,
            "max_steps": int(max_steps),
            "per_device_train_batch_size": int(batch_size),
            "gradient_accumulation_steps": int(gradient_accumulation_steps),
            "learning_rate": float(spec.lora_lr or 5.0e-5),
            "save_steps": int(max_steps),
            "eval_steps": int(max_steps),
            "freeze_vision_encoder": True,
            "freeze_projector": True,
            "bf16": bool(common.get("bf16", True)),
            "fp16": bool(common.get("fp16", False)),
            "gradient_checkpointing": bool(common.get("gradient_checkpointing", True)),
            "seed": int(common.get("seed", 42)),
        }
    )
    payload["lora"]["initial_adapter_dir"] = str(initial_adapter)
    payload["vision_tuning"] = {
        "enabled": spec.phase == "B",
        "unfreeze_last_n_blocks": spec.vit_last_n,
        "train_main_merger": spec.main_merger,
        "train_deepstack_mergers": spec.deepstack,
        "train_patch_embed": spec.patch_embed,
    }
    # A 阶段是 LoRA-only：merger 没有可训练参数，但统一配置 schema 仍要求
    # learning rate 为正值。这个占位值不会创建 visual_merger optimizer group。
    visual_merger_lr = (
        float(spec.merger_lr)
        if spec.main_merger and spec.merger_lr > 0.0
        else float(common.get("visual_merger_lr", 1.0e-6))
    )
    payload["optimization"] = {
        "lora_lr": float(spec.lora_lr or 5.0e-5),
        "visual_merger_lr": visual_merger_lr,
        "vision_lr": float(common.get("vision_lr", 1.0e-6)),
    }
    payload["evaluation"] = {
        **dict(payload.get("evaluation", {})),
        "do_eval": False,
        "predict_with_generate": False,
    }
    payload["logging"] = {
        **dict(payload.get("logging", {})),
        "experiment_name": f"qwen3vl_4b_lr_merger_sweep_{spec.experiment_id}",
    }
    payload["vit_probe"] = {**dict(payload.get("vit_probe", {})), "enabled": False}
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    load_training_config(destination)
    return payload


def run_logged_command(
    command: list[str], log_path: str | Path, *, cwd: str | Path = PROJECT_ROOT
) -> int:
    destination = Path(log_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        completed = subprocess.run(
            command, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT, text=True
        )
        log.write(f"\n[exit_code={completed.returncode}]\n")
    return int(completed.returncode)


def is_oom_log(path: str | Path) -> bool:
    if not Path(path).is_file():
        return False
    text = Path(path).read_text(encoding="utf-8", errors="replace").lower()
    return "out of memory" in text or "cuda oom" in text or "cuda out of memory" in text


def checkpoint_is_complete(path: str | Path, *, visual: bool) -> bool:
    root = Path(path)
    required = [
        root / "adapter_config.json",
        root / "adapter_model.safetensors",
        root / "processor",
        root / "strategy_manifest.json",
    ]
    if visual:
        required.append(root / "visual_trainable_weights.safetensors")
    return all(item.exists() for item in required)


def enrich_strategy_manifest(
    checkpoint_dir: str | Path,
    *,
    spec: ExperimentSpec,
    initial_adapter: str | Path,
    base_model: str | Path,
    train_contract: Mapping[str, Any],
    eval_contract: Mapping[str, Any] | None,
    max_steps: int,
    effective_batch: int,
) -> dict[str, Any]:
    root = Path(checkpoint_dir)
    path = root / "strategy_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    payload.update(
        {
            "experiment_id": spec.experiment_id,
            "experiment_label": spec.label,
            "initial_adapter": str(Path(initial_adapter).resolve()),
            "base_model_path": str(Path(base_model).resolve()),
            "train_data_sha256": train_contract.get("sha256"),
            "evaluation_tier": "E1",
            "evaluation_tier_sha256": (eval_contract or {}).get("tier_sha256"),
            "max_steps": max_steps,
            "effective_batch": effective_batch,
            "loss_mode": "task_weighted",
            "experiment_parameters": {
                "lora_lr": spec.lora_lr,
                "merger_lr": spec.merger_lr,
                "vit_last_n": spec.vit_last_n,
                "main_merger": spec.main_merger,
            },
        }
    )
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def write_audit_for_lora_only(
    checkpoint_dir: str | Path, manifest: Mapping[str, Any]
) -> Path:
    """从 adapter state keys 生成 LoRA-only audit，供 A1/A2/A3 审计统一。"""

    root = Path(checkpoint_dir)
    weight_path = root / "adapter_model.safetensors"
    names: list[str] = []
    if weight_path.is_file():
        from safetensors import safe_open

        with safe_open(str(weight_path), framework="pt", device="cpu") as handle:
            names = sorted(key for key in handle.keys() if "lora_" in key.lower())
    audit = {
        "schema_version": "1.0",
        "lora": {
            "parameter_count": int(manifest.get("trainable_parameters", 0)),
            "tensor_count": len(names),
            "names": names,
        },
        "vision_blocks": {
            "parameter_count": 0,
            "tensor_count": 0,
            "block_indices": [],
            "names": [],
        },
        "visual_merger": {"parameter_count": 0, "tensor_count": 0, "names": []},
        "optional_visual": {"parameter_count": 0, "tensor_count": 0, "names": []},
        "other_trainable": [],
        "total_trainable": int(manifest.get("trainable_parameters", 0)),
        "total_parameters": int(manifest.get("total_parameters", 0)),
        "trainable_ratio": manifest.get("trainable_ratio"),
    }
    destination = root / "trainable_parameters.json"
    destination.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return destination


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_sweep_report(
    report_dir: str | Path,
    records: list[Mapping[str, Any]],
    selection: Mapping[str, Any],
    evaluation_context: Mapping[str, Any] | None = None,
) -> None:
    root = Path(report_dir)
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "selection": dict(selection),
        "experiments": records,
        "evaluation": dict(evaluation_context or {}),
        "reference": {
            "experiment": "2B_reference",
            "lora_base_ratio": 0.0345,
            "note": "historical reference scale only; no fabricated E1 metrics",
        },
    }
    (root / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(root / "summary.csv", [dict(record) for record in records])
    lines = [
        "# Qwen3-VL-4B LR + Visual Merger Sweep",
        "",
        "本报告用于快速方向诊断，不是论文式严格 ablation。所有实验从同一 Round1 adapter、同一 probe 数据和同一 E1 开始。",
        "",
        f"Phase B selected LR: `{selection.get('selected_lr')}`",
        f"Selection mode: `{selection.get('selection_mode')}`",
        f"Effective E1 eval batch size: `{dict(evaluation_context or {}).get('effective_eval_batch_size')}`",
        "",
    ]
    if not records:
        lines.append("当前尚未产生实验记录。")
    else:
        lines.extend(
            [
                "| experiment | status | lora lr | merger lr | vit | LoRA/Base | "
                "Count exact | Count +/-1 | Scene norm | VQA norm | Det mIoU | "
                "LEVIR F1 | Caption ROUGE-L | weak score |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for record in records:
            table_row = (
                "| {experiment_id} | {status} | {lora_lr} | {merger_lr} | {vit_last_n} | "
                "{lora_base_ratio} | {count_exact} | {count_pm1} | {scene_normalized} | "
                "{vqa_normalized} | {detection_miou} | {levir_f1} | {caption_rouge_l} | "
                "{weak_score} |"
            )
            lines.append(
                table_row.format(
                    **{
                        key: record.get(key)
                        for key in (
                            "experiment_id",
                            "status",
                            "lora_lr",
                            "merger_lr",
                            "vit_last_n",
                            "lora_base_ratio",
                            "count_exact",
                            "count_pm1",
                            "scene_normalized",
                            "vqa_normalized",
                            "detection_miou",
                            "levir_f1",
                            "caption_rouge_l",
                            "weak_score",
                        )
                    }
                )
            )
    lines.extend(
        [
            "",
            "## 自动回答",
            "",
            "- Q1/Q2：查看 LR、LoRA/Base ratio 与 Scene/Counting/VQA 的联动；没有完成 E1 的实验不会进入 selector。",
            "- Q3：A3 是否不稳定由 `experiment_status.json`、训练日志中的 NaN/OOM 和 E1 指标共同判断。",
            "- Q4-Q7：比较 B1/B2/B3 的弱任务分数和 Detection/LEVIR/Caption guard。",
            "- Q8：优先选择满足 guard 且 weak score 最高的 Phase A LR；若全部违反 guard，明确使用 5e-5 fallback。",
            "",
            "2B 的 3.45% 只作为历史量级参考，不是本轮优化目标。",
        ]
    )
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def maybe_plot_sweep(report_dir: str | Path, records: list[Mapping[str, Any]]) -> None:
    """生成轻量 PNG；没有 matplotlib 时保留 CSV/JSON 并记录 warning。"""

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        (Path(report_dir) / "plotting_unavailable.txt").write_text(
            "matplotlib is not installed; CSV/JSON remain authoritative.\n",
            encoding="utf-8",
        )
        return
    root = Path(report_dir)
    plot_dir = root / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    values = [
        record
        for record in records
        if record.get("status") in {"EVALUATED", "ANALYZED"}
    ]
    if not values:
        return
    plots = [
        ("lr_vs_lora_base_ratio", "lora_lr", "lora_base_ratio"),
        ("lr_vs_counting_exact", "lora_lr", "count_exact"),
        ("lr_vs_scene_normalized", "lora_lr", "scene_normalized"),
        ("lr_vs_vqa_normalized", "lora_lr", "vqa_normalized"),
        ("lora_ratio_vs_weak_score", "lora_base_ratio", "weak_score"),
        ("experiment_vs_major_metrics", "experiment_id", "weak_score"),
    ]
    for name, x_key, y_key in plots:
        x = [
            record.get(x_key)
            for record in values
            if isinstance(record.get(y_key), (int, float))
        ]
        y = [
            record.get(y_key)
            for record in values
            if isinstance(record.get(y_key), (int, float))
        ]
        if not x or not y:
            continue
        plt.figure(figsize=(7, 4))
        plt.plot(range(len(y)), y, marker="o")
        plt.xticks(
            range(len(y)),
            [
                str(item.get(x_key))
                for item in values
                if isinstance(item.get(y_key), (int, float))
            ],
            rotation=30,
        )
        plt.ylabel(y_key)
        plt.xlabel(x_key)
        plt.tight_layout()
        plt.savefig(plot_dir / f"{name}.png", dpi=140)
        plt.close()

    for experiment_id in ("A1", "A2", "A3"):
        source = root / "lora_analysis" / experiment_id / "lora_analysis.json"
        if not source.is_file():
            continue
        payload = json.loads(source.read_text(encoding="utf-8"))
        rows = [row for row in payload.get("rows", []) if row.get("layer") is not None]
        for suffix, key in (
            ("layer_lora_base_ratio", "delta_base_ratio"),
            ("layer_relative_change_from_r1", "relative_change_from_r1"),
        ):
            points = [row for row in rows if isinstance(row.get(key), (int, float))]
            if not points:
                continue
            plt.figure(figsize=(7, 4))
            plt.plot(
                [row["layer"] for row in points],
                [row[key] for row in points],
                marker="o",
            )
            plt.xlabel("layer index")
            plt.ylabel(key)
            plt.title(experiment_id)
            plt.tight_layout()
            plt.savefig(plot_dir / f"{experiment_id}_{suffix}.png", dpi=140)
            plt.close()

    merger_points: list[tuple[str, float, float]] = []
    record_by_id = {str(record.get("experiment_id")): record for record in records}
    for experiment_id in ("B1", "B2", "B3"):
        source = root / "merger_analysis" / experiment_id / "merger_analysis.json"
        if not source.is_file():
            continue
        payload = json.loads(source.read_text(encoding="utf-8"))
        updates = [
            row.get("relative_frobenius_delta")
            for row in payload.get("rows", [])
            if isinstance(row.get("relative_frobenius_delta"), (int, float))
        ]
        weak = record_by_id.get(experiment_id, {}).get("weak_score")
        if updates and isinstance(weak, (int, float)):
            merger_points.append(
                (
                    experiment_id,
                    sum(float(value) for value in updates) / len(updates),
                    float(weak),
                )
            )
    if merger_points:
        plt.figure(figsize=(7, 4))
        plt.scatter(
            [point[1] for point in merger_points], [point[2] for point in merger_points]
        )
        for experiment_id, x, y in merger_points:
            plt.annotate(experiment_id, (x, y))
        plt.xlabel("merger relative update")
        plt.ylabel("weak score")
        plt.tight_layout()
        plt.savefig(plot_dir / "merger_update_vs_weak_score.png", dpi=140)
        plt.close()


def shutdown_if_requested(enabled: bool) -> None:
    """只在显式 ``--shutdown`` 且 AutoDL 环境文件存在时关机。"""

    if not enabled:
        return
    env_file = Path("/root/autodl_env.sh")
    if not env_file.is_file():
        raise RuntimeError("Refusing shutdown: /root/autodl_env.sh is missing")
    for command in ("/usr/sbin/shutdown", "/sbin/shutdown"):
        if Path(command).is_file():
            subprocess.run([command, "-h", "now"], check=False)
            return
    subprocess.run(["poweroff"], check=False)
