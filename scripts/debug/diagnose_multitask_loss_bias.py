"""只读诊断当前 Qwen3-VL LoRA 多任务 batch 的 Causal-LM 长度偏置。

脚本从本地基座模型和已训练 LoRA adapter 加载模型，按任务组成混合 batch，在
``torch.no_grad()`` 下执行 labels forward。它不会训练、不会 backward、不会生成、
不会下载模型，也不会修改正式 Trainer 或 loss 实现。
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sat_rs_vlm.data.qwen3vl_collator import Qwen3VLDataCollator
from sat_rs_vlm.data.qwen3vl_dataset import Qwen3VLDataset
from sat_rs_vlm.models.qwen3vl_loader import load_qwen3vl
from sat_rs_vlm.training.config import (
    TrainingPathOverrides,
    apply_training_overrides,
    load_training_config,
    resolve_path,
)
from sat_rs_vlm.training.loss_diagnostics import (
    analyze_causal_lm_batch_loss,
    multitask_loss_bias_markdown,
    summarize_multitask_loss_bias,
)
from sat_rs_vlm.training.utils import (
    MODEL_DEPS_ERROR,
    model_input_device,
    move_to_device,
    resolve_torch_dtype,
    safe_import_model_dependencies,
    set_seed,
)

REQUIRED_TASKS = (
    "captioning",
    "detection",
    "counting",
    "vqa",
    "scene_classification",
)


@dataclass(frozen=True)
class DiagnosticPaths:
    """只保存本诊断实际需要的已解析路径，避免要求验证集或训练输出目录。"""

    model_source: str
    processor_source: str
    train_file: Path
    image_root: Path
    initial_adapter_dir: Path


def parse_args() -> argparse.Namespace:
    """解析本地诊断参数，不提供任何训练或写 checkpoint 的选项。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--processor-dir", default=None)
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--train-file", default=None)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--output-dir", default="reports/debug/multitask_loss_bias")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-batches", type=int, default=20)
    parser.add_argument("--samples-per-task", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--material-share-gap", type=float, default=0.10)
    return parser.parse_args()


def choose_mixed_batches(
    samples: list[dict[str, Any]],
    *,
    batch_size: int,
    num_batches: int,
    samples_per_task: int,
    seed: int,
) -> list[list[dict[str, Any]]]:
    """按任务均衡地构造随机混合 batch，确保每个 batch 至少含两个任务。

    参数：
        samples: 已归一化的 VRSBench 训练样本。
        batch_size: 每个 forward 的样本数，默认 4，显存不足时可设为 2。
        num_batches: 要执行的混合 batch 数。
        samples_per_task: 从每个任务可重复使用的候选池上限。
        seed: 控制候选截取与 batch 任务组合的随机种子。

    返回：确定性的样本 batch 列表；不读取图像、不进行 Collator 编码。
    """

    if batch_size < 2:
        raise ValueError("batch-size must be at least 2 to preserve mixed-task batches")
    if num_batches < 1 or samples_per_task < 1:
        raise ValueError("num-batches and samples-per-task must be positive")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        task = str(sample.get("task_type", "")).strip().lower()
        if task in REQUIRED_TASKS:
            grouped[task].append(sample)
    missing = [task for task in REQUIRED_TASKS if not grouped[task]]
    if missing:
        raise ValueError(f"Training JSONL is missing required tasks: {missing}")

    rng = random.Random(seed)
    pools: dict[str, list[dict[str, Any]]] = {}
    for task in REQUIRED_TASKS:
        candidates = list(grouped[task])
        rng.shuffle(candidates)
        pools[task] = candidates[:samples_per_task]
    cursors = {task: 0 for task in REQUIRED_TASKS}
    batches: list[list[dict[str, Any]]] = []
    for batch_index in range(num_batches):
        offset = batch_index % len(REQUIRED_TASKS)
        task_order = list(REQUIRED_TASKS[offset:]) + list(REQUIRED_TASKS[:offset])
        selected_tasks = [task_order[index % len(task_order)] for index in range(batch_size)]
        batch: list[dict[str, Any]] = []
        for task in selected_tasks:
            pool = pools[task]
            batch.append(pool[cursors[task] % len(pool)])
            cursors[task] += 1
        rng.shuffle(batch)
        batches.append(batch)
    return batches


def cuda_memory_snapshot(torch: Any) -> dict[str, float | None]:
    """读取当前设备的 CUDA 峰值显存；CPU 环境显式返回空值。"""

    if not bool(torch.cuda.is_available()):
        return {"peak_memory_allocated_mb": None, "peak_memory_reserved_mb": None}
    return {
        "peak_memory_allocated_mb": float(torch.cuda.max_memory_allocated() / (1024 * 1024)),
        "peak_memory_reserved_mb": float(torch.cuda.max_memory_reserved() / (1024 * 1024)),
    }


def load_diagnostic_model(args: argparse.Namespace) -> tuple[Any, DiagnosticPaths, Any, Any]:
    """按正式配置加载本地 FP16/BF16 基座与最终 LoRA adapter，并保持 eval 模式。"""

    config = load_training_config(args.config, allow_unresolved_env=True)
    config = apply_training_overrides(
        config,
        TrainingPathOverrides(
            model_dir=args.model_dir,
            processor_dir=args.processor_dir,
            train_file=args.train_file,
            image_root=args.image_root,
            initial_adapter_dir=args.adapter_dir,
        ),
    )
    model_source = str(config.model.model_dir or config.model.model_id or "")
    processor_source = str(config.model.processor_dir or config.model.processor_id or "")
    if not model_source or not processor_source:
        raise ValueError("Set a local model/processor directory in config or through CLI overrides")
    if config.lora.initial_adapter_dir is None:
        raise ValueError("Set lora.initial_adapter_dir in config or provide --adapter-dir")
    paths = DiagnosticPaths(
        model_source=str(resolve_path(model_source)),
        processor_source=str(resolve_path(processor_source)),
        train_file=resolve_path(config.data.train_file),
        image_root=resolve_path(config.data.image_root),
        initial_adapter_dir=resolve_path(config.lora.initial_adapter_dir),
    )
    modules = safe_import_model_dependencies(require_bitsandbytes=False)
    torch = modules["torch"]
    if not bool(torch.cuda.is_available()):
        raise RuntimeError("This real-model diagnostic requires CUDA; no CPU fallback is provided.")
    dtype = resolve_torch_dtype(torch, config.model.torch_dtype)
    model_kwargs = {
        "trust_remote_code": config.model.trust_remote_code,
        "local_files_only": True,
        "device_map": config.model.device_map,
        "attn_implementation": config.model.attn_implementation,
        "torch_dtype": dtype,
    }
    model, processor = load_qwen3vl(
        modules=modules,
        base_model=paths.model_source,
        processor_source=paths.processor_source,
        model_kwargs=model_kwargs,
        processor_kwargs={
            "trust_remote_code": config.model.trust_remote_code,
            "local_files_only": True,
        },
        adapter_path=str(paths.initial_adapter_dir),
    )
    return config, paths, model, processor


def main() -> int:
    """执行只读多任务 forward，写入 JSON/Markdown，随后释放本进程内模型资源。"""

    args = parse_args()
    if args.batch_size < 2:
        raise ValueError("--batch-size must be at least 2")
    if args.num_batches < 1:
        raise ValueError("--num-batches must be positive")
    if not 0.0 < args.material_share_gap < 1.0:
        raise ValueError("--material-share-gap must be between 0 and 1")
    try:
        config, paths, model, processor = load_diagnostic_model(args)
    except ImportError as exc:
        raise SystemExit(MODEL_DEPS_ERROR) from exc
    modules = safe_import_model_dependencies(require_bitsandbytes=False)
    torch = modules["torch"]
    set_seed(args.seed)
    dataset = Qwen3VLDataset(paths.train_file, skip_bad_samples=config.data.skip_bad_samples)
    batches = choose_mixed_batches(
        list(dataset),
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        samples_per_task=args.samples_per_task,
        seed=args.seed,
    )
    collator = Qwen3VLDataCollator(processor, config.data.max_seq_length, paths.image_root)
    input_device = model_input_device(model, torch)
    if bool(torch.cuda.is_available()):
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    batch_reports: list[dict[str, Any]] = []
    with torch.no_grad():
        for index, samples in enumerate(batches, start=1):
            batch = move_to_device(collator(samples), input_device, torch)
            outputs = model(**batch)
            report = analyze_causal_lm_batch_loss(
                logits=outputs.logits,
                labels=batch["labels"],
                sample_ids=[str(sample["id"]) for sample in samples],
                task_types=[str(sample["task_type"]) for sample in samples],
                torch=torch,
                model_batch_loss=float(outputs.loss.item()),
            )
            report["batch_index"] = index
            report["memory"] = cuda_memory_snapshot(torch)
            batch_reports.append(report)
            print(
                f"Diagnosed mixed batch {index}/{len(batches)}: "
                f"loss={float(outputs.loss.item()):.6f}, "
                f"tokens={report['batch_statistics']['supervised_tokens']}",
                flush=True,
            )
            del outputs, batch
    summary = summarize_multitask_loss_bias(
        batch_reports,
        material_share_gap=args.material_share_gap,
    )
    summary.update(
        {
            "schema_version": "1.0",
            "diagnostic_mode": "read_only_no_grad_no_backward_no_generation",
            "model": {
                "base_model": paths.model_source,
                "adapter": str(paths.initial_adapter_dir),
                "dtype": str(next(iter(model.parameters())).dtype).removeprefix("torch."),
                "input_device": str(input_device),
            },
            "configuration": {
                "train_file": str(paths.train_file),
                "image_root": str(paths.image_root),
                "max_seq_length": config.data.max_seq_length,
                "batch_size": args.batch_size,
                "num_batches": args.num_batches,
                "samples_per_task": args.samples_per_task,
                "seed": args.seed,
            },
            "execution_seconds": time.perf_counter() - started,
            "batch_reports": batch_reports,
        }
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(
        multitask_loss_bias_markdown(summary),
        encoding="utf-8",
    )
    print(f"Saved summary JSON: {output_dir / 'summary.json'}")
    print(f"Saved summary Markdown: {output_dir / 'summary.md'}")
    print(f"Peak CUDA memory: {summary['memory']}")
    print(f"Executed samples: {summary['sample_count']}")
    print(f"Completed in {summary['execution_seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
