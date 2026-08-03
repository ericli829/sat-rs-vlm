"""Benchmark short AutoDL LoRA runs and recommend a production batch configuration."""

from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATES = "4:4:true,8:2:true,16:1:true,4:4:false,8:2:false"
OOM_MARKERS = ("out of memory", "cuda error: out of memory", "cuda out of memory")


@dataclass(frozen=True)
class Candidate:
    batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool

    @property
    def name(self) -> str:
        checkpointing = "gc" if self.gradient_checkpointing else "no-gc"
        return f"bs{self.batch_size}-ga{self.gradient_accumulation_steps}-{checkpointing}"

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps


def parse_candidate(value: str) -> Candidate:
    parts = [part.strip() for part in value.split(":")]
    if len(parts) != 3:
        raise ValueError(f"Invalid candidate '{value}'; expected BATCH:ACCUM:true|false")
    batch_size = int(parts[0])
    accumulation = int(parts[1])
    if batch_size <= 0 or accumulation <= 0:
        raise ValueError("Batch size and gradient accumulation must be positive")
    bool_values = {"true": True, "false": False}
    checkpointing_text = parts[2].lower()
    if checkpointing_text not in bool_values:
        raise ValueError(f"Invalid gradient checkpointing value: {parts[2]}")
    return Candidate(batch_size, accumulation, bool_values[checkpointing_text])


def parse_candidates(value: str) -> list[Candidate]:
    candidates = [parse_candidate(item) for item in value.split(",") if item.strip()]
    if not candidates:
        raise ValueError("At least one benchmark candidate is required")
    names = [candidate.name for candidate in candidates]
    if len(names) != len(set(names)):
        raise ValueError("Benchmark candidates must be unique")
    return candidates


def load_yaml(path: Path) -> dict[str, Any]:
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def build_benchmark_config(
    base: dict[str, Any],
    candidate: Candidate,
    *,
    max_steps: int,
    max_train_samples: int,
) -> dict[str, Any]:
    config = copy.deepcopy(base)
    config.setdefault("experiment", {})["name"] = f"autodl_benchmark_{candidate.name}"
    training = config.setdefault("training", {})
    training.update(
        {
            "num_train_epochs": 1,
            "max_steps": max_steps,
            "per_device_train_batch_size": candidate.batch_size,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": candidate.gradient_accumulation_steps,
            "gradient_checkpointing": candidate.gradient_checkpointing,
            "logging_steps": 1,
            "eval_steps": max_steps + 1,
            "save_steps": max_steps + 1,
            "save_total_limit": 1,
        }
    )
    data = config.setdefault("data", {})
    data["max_train_samples"] = max_train_samples
    data["max_validation_samples"] = 8
    config.setdefault("evaluation", {})["do_eval"] = False
    config.setdefault("runtime", {})["mock"] = False
    return config


def build_recommended_config(
    base: dict[str, Any],
    candidate: Candidate,
) -> dict[str, Any]:
    config = copy.deepcopy(base)
    config.setdefault("experiment", {})["name"] = "autodl_lora_4090"
    config.setdefault("training", {}).update(
        {
            "per_device_train_batch_size": candidate.batch_size,
            "gradient_accumulation_steps": candidate.gradient_accumulation_steps,
            "gradient_checkpointing": candidate.gradient_checkpointing,
        }
    )
    return config


def build_recommended_smoke_config(
    base: dict[str, Any],
    candidate: Candidate,
) -> dict[str, Any]:
    config = copy.deepcopy(base)
    config.setdefault("experiment", {})["name"] = "autodl_lora_4090_smoke"
    config.setdefault("training", {}).update(
        {
            "per_device_train_batch_size": candidate.batch_size,
            "gradient_accumulation_steps": 1,
            "gradient_checkpointing": candidate.gradient_checkpointing,
        }
    )
    minimum_samples = candidate.batch_size * int(config["training"].get("max_steps", 5))
    data = config.setdefault("data", {})
    data["max_train_samples"] = max(int(data.get("max_train_samples") or 0), minimum_samples)
    return config


def gpu_snapshot() -> dict[str, float] | None:
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    first_gpu = completed.stdout.strip().splitlines()[0]
    try:
        utilization, used, total = [float(item.strip()) for item in first_gpu.split(",")]
    except (TypeError, ValueError):
        return None
    return {"utilization": utilization, "memory_used_mb": used, "memory_total_mb": total}


def read_report(output_dir: Path) -> dict[str, Any]:
    for path in (
        output_dir / "checkpoints/train_report.json",
        output_dir / "train_report.json",
    ):
        if path.is_file():
            return dict(json.loads(path.read_text(encoding="utf-8")))
    return {}


def as_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def run_candidate(
    candidate: Candidate,
    *,
    config_path: Path,
    env_config: Path,
    output_dir: Path,
    log_path: Path,
    max_steps: int,
    sample_interval: float,
    memory_limit_fraction: float,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/training/run_train.py"),
        "--environment",
        "autodl",
        "--env-config",
        str(env_config),
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
        "--skip-eval",
    ]
    environment = dict(os.environ)
    environment.setdefault("OMP_NUM_THREADS", "8")
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    gpu_samples: list[dict[str, float]] = []
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while process.poll() is None:
            snapshot = gpu_snapshot()
            if snapshot is not None:
                gpu_samples.append(snapshot)
            time.sleep(sample_interval)
        returncode = process.wait()
    wall_seconds = time.perf_counter() - started
    report = read_report(output_dir)
    log_text = log_path.read_text(encoding="utf-8", errors="replace").lower()
    if returncode == 0 and report.get("success") is True:
        status = "success"
    elif any(marker in log_text for marker in OOM_MARKERS):
        status = "oom"
    else:
        status = "failed"

    train_runtime = as_float(report.get("train_runtime_seconds"))
    samples_per_second = as_float(report.get("train_samples_per_second"))
    if samples_per_second is None and train_runtime and train_runtime > 0:
        samples_per_second = candidate.effective_batch_size * max_steps / train_runtime
    reported_peak = max(
        as_float(report.get("peak_memory_mb")) or 0.0,
        as_float(report.get("peak_reserved_memory_mb")) or 0.0,
    )
    sampled_peak = max(
        (sample["memory_used_mb"] for sample in gpu_samples),
        default=0.0,
    )
    peak_memory = max(reported_peak, sampled_peak) or None
    total_memory = gpu_samples[0]["memory_total_mb"] if gpu_samples else None
    memory_fraction = (
        peak_memory / total_memory if peak_memory is not None and total_memory else None
    )
    memory_safe = memory_fraction is None or memory_fraction <= memory_limit_fraction
    average_utilization = (
        statistics.fmean(sample["utilization"] for sample in gpu_samples)
        if gpu_samples
        else None
    )
    return {
        **asdict(candidate),
        "name": candidate.name,
        "effective_batch_size": candidate.effective_batch_size,
        "status": status,
        "returncode": returncode,
        "wall_seconds": wall_seconds,
        "train_runtime_seconds": train_runtime,
        "train_samples_per_second": samples_per_second,
        "train_steps_per_second": as_float(report.get("train_steps_per_second")),
        "average_gpu_utilization": average_utilization,
        "peak_memory_mb": peak_memory,
        "total_memory_mb": total_memory,
        "memory_fraction": memory_fraction,
        "memory_safe": memory_safe,
        "output_dir": str(output_dir),
        "log_file": str(log_path),
    }


def select_best(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    successful = [
        result
        for result in results
        if result.get("status") == "success"
        and result.get("memory_safe") is True
        and result.get("train_samples_per_second") is not None
    ]
    if not successful:
        return None
    return max(successful, key=lambda result: float(result["train_samples_per_second"]))


def format_number(value: Any, digits: int = 2) -> str:
    numeric = as_float(value)
    return "-" if numeric is None else f"{numeric:.{digits}f}"


def write_markdown(path: Path, results: list[dict[str, Any]], best: dict[str, Any] | None) -> None:
    lines = [
        "# AutoDL LoRA 性能基准",
        "",
        "| 配置 | 状态 | 样本/秒 | 训练秒数 | 平均 GPU% | 峰值显存 MB | 显存占比 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        fraction = as_float(result.get("memory_fraction"))
        fraction_text = "-" if fraction is None else f"{fraction * 100:.1f}%"
        lines.append(
            "| {name} | {status} | {throughput} | {runtime} | {util} | {memory} | "
            "{fraction} |".format(
                name=result["name"],
                status=result["status"],
                throughput=format_number(result.get("train_samples_per_second")),
                runtime=format_number(result.get("train_runtime_seconds")),
                util=format_number(result.get("average_gpu_utilization"), 1),
                memory=format_number(result.get("peak_memory_mb"), 0),
                fraction=fraction_text,
            )
        )
    lines.extend(["", f"推荐配置：{best['name']}" if best else "没有可安全推荐的配置。", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config",
        type=Path,
        default=PROJECT_ROOT / "configs/cloud/train_lora_autodl.yaml",
    )
    parser.add_argument(
        "--env-config",
        type=Path,
        default=PROJECT_ROOT / "configs/cloud/autodl.yaml",
    )
    parser.add_argument(
        "--smoke-config",
        type=Path,
        default=PROJECT_ROOT / "configs/cloud/train_lora_autodl_smoke.yaml",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/root/autodl-tmp/outputs/performance"),
    )
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-train-samples", type=int, default=512)
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--memory-limit-fraction", type=float, default=0.80)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_steps <= 0 or args.max_train_samples <= 0:
        raise ValueError("max-steps and max-train-samples must be positive")
    if args.sample_interval <= 0:
        raise ValueError("sample-interval must be positive")
    if not 0 < args.memory_limit_fraction <= 1:
        raise ValueError("memory-limit-fraction must be in (0, 1]")

    base_config = load_yaml(args.base_config.resolve())
    smoke_config = load_yaml(args.smoke_config.resolve())
    candidates = parse_candidates(args.candidates)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    benchmark_root = args.output_root.resolve() / f"benchmark_{timestamp}"
    config_root = benchmark_root / "configs"
    log_root = benchmark_root / "logs"
    run_root = benchmark_root / "runs"
    for path in (config_root, log_root, run_root):
        path.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for candidate in candidates:
        print(f"[benchmark] starting {candidate.name}", flush=True)
        generated_config = build_benchmark_config(
            base_config,
            candidate,
            max_steps=args.max_steps,
            max_train_samples=args.max_train_samples,
        )
        config_path = config_root / f"{candidate.name}.yaml"
        config_path.write_text(
            yaml.safe_dump(generated_config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        result = run_candidate(
            candidate,
            config_path=config_path,
            env_config=args.env_config.resolve(),
            output_dir=run_root / candidate.name,
            log_path=log_root / f"{candidate.name}.log",
            max_steps=args.max_steps,
            sample_interval=args.sample_interval,
            memory_limit_fraction=args.memory_limit_fraction,
        )
        results.append(result)
        print(
            f"[benchmark] {candidate.name}: status={result['status']} "
            f"samples/s={format_number(result.get('train_samples_per_second'))} "
            f"peak_mb={format_number(result.get('peak_memory_mb'), 0)}",
            flush=True,
        )

    best = select_best(results)
    summary = {
        "benchmark_root": str(benchmark_root),
        "base_config": str(args.base_config.resolve()),
        "smoke_config": str(args.smoke_config.resolve()),
        "max_steps": args.max_steps,
        "max_train_samples": args.max_train_samples,
        "memory_limit_fraction": args.memory_limit_fraction,
        "results": results,
        "recommended": best,
    }
    (benchmark_root / "benchmark_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(benchmark_root / "benchmark_summary.md", results, best)
    if best is not None:
        best_candidate = Candidate(
            batch_size=int(best["batch_size"]),
            gradient_accumulation_steps=int(best["gradient_accumulation_steps"]),
            gradient_checkpointing=bool(best["gradient_checkpointing"]),
        )
        recommended = build_recommended_config(base_config, best_candidate)
        (benchmark_root / "recommended_training_config.yaml").write_text(
            yaml.safe_dump(recommended, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        recommended_smoke = build_recommended_smoke_config(smoke_config, best_candidate)
        (benchmark_root / "recommended_smoke_config.yaml").write_text(
            yaml.safe_dump(recommended_smoke, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    print(json.dumps({"benchmark_root": str(benchmark_root), "recommended": best}, indent=2))
    return 0 if best is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
