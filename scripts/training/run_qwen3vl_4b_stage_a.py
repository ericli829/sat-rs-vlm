"""AutoDL Qwen3-VL-4B Stage-A full-cycle orchestration backend."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from sat_rs_vlm.data.cyclic_training import sha256_file

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_CONFIG = ROOT / "configs/data/autodl_qwen3vl_4b_stage_a.yaml"
DEFAULT_TRAIN_CONFIG = ROOT / "configs/train/qwen3vl_4b_stage_a_multisource_4090.yaml"
DEFAULT_EVAL_CONFIG = ROOT / "configs/eval/qwen3vl_4b_stage_a_e2_v2.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--cycle-index", type=int, default=0)
    parser.add_argument("--start-round", type=int, default=0)
    parser.add_argument("--end-round", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--data-config", default=str(DEFAULT_DATA_CONFIG))
    parser.add_argument("--train-config", default=str(DEFAULT_TRAIN_CONFIG))
    parser.add_argument("--skip-e2-eval", action="store_true")
    return parser.parse_args()


def _run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _adapter_valid(path: Path) -> bool:
    return (path / "adapter_config.json").is_file() and any(
        (path / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")
    )


def _latest_checkpoint(adapter_dir: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in adapter_dir.glob("checkpoint-*"):
        try:
            candidates.append((int(path.name.split("-")[-1]), path))
        except ValueError:
            continue
    return max(candidates, default=(0, None))[1]


def _previous_round_adapter(run_root: Path, round_index: int) -> Path | None:
    """round 0 从 base 开始；后续 round 只允许链接紧邻的上一轮 adapter。"""

    if round_index == 0:
        return None
    return run_root / f"round_{round_index - 1:03d}" / "adapter"


def _learning_rate(config_path: str | Path, round_index: int, override: float | None) -> float:
    if override is not None:
        return override
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    rates = [float(value) for value in config["cycle_training"]["learning_rates"]]
    return rates[min(round_index, len(rates) - 1)]


def _training_command(
    args: argparse.Namespace,
    *,
    train_file: Path,
    validation_file: Path,
    output_dir: Path,
    initial_adapter: Path | None,
    learning_rate: float,
    mode: str | None,
    resume_checkpoint: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/train_qwen3vl_lora.py",
        "--config",
        args.train_config,
        "--train-file",
        str(train_file),
        "--val-file",
        str(validation_file),
        "--output-dir",
        str(output_dir),
        "--learning-rate",
        str(learning_rate),
    ]
    if initial_adapter is not None:
        command.extend(["--initial-adapter", str(initial_adapter)])
    if args.max_train_samples is not None:
        command.extend(["--max-train-samples", str(args.max_train_samples)])
    if args.max_eval_samples is not None:
        command.extend(["--max-eval-samples", str(args.max_eval_samples)])
    if resume_checkpoint is not None:
        command.extend(["--resume-from-checkpoint", str(resume_checkpoint)])
    if mode:
        command.append(mode)
    return command


def main() -> int:
    args = parse_args()
    if args.cycle_index < 0 or args.start_round < 0:
        raise ValueError("cycle-index and start-round must be non-negative")
    if args.max_train_samples is not None and not (args.dry_run or args.forward_only):
        raise ValueError("--max-train-samples is forbidden for a formal full-cycle run")
    for name in ("QWEN3VL_4B_MODEL_DIR", "DATA_ROOT", "OUTPUT_ROOT"):
        if not os.environ.get(name):
            raise ValueError(f"Required environment variable is missing: {name}")
    model_dir = Path(os.environ["QWEN3VL_4B_MODEL_DIR"])
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"Qwen3-VL-4B config.json is missing: {model_dir}")

    cycle_dir = ROOT / "data/processed/multisource/qwen3vl_4b_stage_a"
    _run(
        [
            sys.executable,
            "scripts/data/prepare_multisource_training_data.py",
            "--config",
            args.data_config,
            "--build-cycle",
            "--cycle-index",
            str(args.cycle_index),
            "--cycle-output-dir",
            str(cycle_dir),
        ]
    )
    manifest_path = cycle_dir / "cycle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("global", {}).get("valid"):
        raise ValueError("Cycle manifest does not prove full coverage")
    if manifest.get("protected_evaluation", {}).get("overlap_count") != 0:
        raise ValueError("Cycle manifest reports protected E3 leakage")
    if args.prepare_only:
        print(f"Prepared validated cycle: {manifest_path}")
        return 0
    train_config_payload = yaml.safe_load(Path(args.train_config).read_text(encoding="utf-8"))
    training_profile = dict(train_config_payload["training"])
    effective_batch = int(training_profile["per_device_train_batch_size"]) * int(
        training_profile["gradient_accumulation_steps"]
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = Path(
        args.run_root
        or Path(os.environ["OUTPUT_ROOT"]) / f"qwen3vl_4b_stage_a_{timestamp}"
    )
    run_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, run_root / "cycle_manifest.json")
    rounds = list(manifest["rounds"])
    end_round = len(rounds) - 1 if args.end_round is None else args.end_round
    if end_round >= len(rounds) or args.start_round > end_round:
        raise ValueError(f"Round range must be within 0..{len(rounds) - 1}")

    initial_adapter = _previous_round_adapter(run_root, args.start_round)
    if initial_adapter is not None:
        if not _adapter_valid(initial_adapter):
            raise FileNotFoundError(f"Previous round adapter is required: {initial_adapter}")
    results: list[dict[str, Any]] = []
    for previous_index in range(args.start_round):
        previous_result = run_root / f"round_{previous_index:03d}" / "round_result.json"
        if previous_result.is_file():
            results.append(json.loads(previous_result.read_text(encoding="utf-8")))
    validation_file = Path(manifest["validation_file"])

    for round_index in range(args.start_round, end_round + 1):
        round_entry = rounds[round_index]
        train_file = Path(round_entry["train_file"])
        round_root = run_root / f"round_{round_index:03d}"
        adapter_dir = round_root / "adapter"
        if args.resume and _adapter_valid(adapter_dir):
            initial_adapter = adapter_dir
            result_path = round_root / "round_result.json"
            results.append(
                json.loads(result_path.read_text(encoding="utf-8"))
                if result_path.is_file()
                else {"round_index": round_index, "status": "reused_complete"}
            )
            continue
        lr = _learning_rate(args.train_config, round_index, args.learning_rate)
        resume_checkpoint = _latest_checkpoint(adapter_dir) if args.resume else None
        def round_command(mode: str | None, resume_path: Path | None = None) -> list[str]:
            return _training_command(
                args,
                train_file=train_file,
                validation_file=validation_file,
                output_dir=adapter_dir,
                initial_adapter=initial_adapter,
                learning_rate=lr,
                mode=mode,
                resume_checkpoint=resume_path,
            )

        _run(round_command("--dry-run"))
        if args.dry_run:
            results.append({"round_index": round_index, "status": "dry_run_passed"})
            continue
        if round_index == args.start_round:
            _run(round_command("--forward-only"))
            if args.forward_only:
                results.append({"round_index": round_index, "status": "forward_only_passed"})
                break
        started = time.perf_counter()
        _run(round_command(None, resume_checkpoint))
        if not _adapter_valid(adapter_dir):
            raise FileNotFoundError(
                f"Round training did not produce a valid adapter: {adapter_dir}"
            )
        train_report_path = adapter_dir / "smoke_train_report.json"
        train_report = json.loads(train_report_path.read_text(encoding="utf-8"))
        result = {
            "round_index": round_index,
            "status": "completed",
            "train_file": str(train_file),
            "train_file_sha256": round_entry["sha256"],
            "initial_adapter": str(initial_adapter) if initial_adapter else None,
            "output_adapter": str(adapter_dir),
            "learning_rate": lr,
            "samples": round_entry["sample_count"],
            "optimizer_steps": train_report.get("global_step"),
            "effective_batch": effective_batch,
            "runtime": train_report.get("train_runtime_seconds", time.perf_counter() - started),
            "peak_vram_mb": train_report.get("peak_memory_mb"),
        }
        (round_root / "round_result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        results.append(result)
        initial_adapter = adapter_dir

    if args.dry_run or args.forward_only:
        print(json.dumps({"run_root": str(run_root), "rounds": results}, indent=2))
        return 0
    if initial_adapter is None:
        raise RuntimeError("No completed adapter is available")
    final_adapter = run_root / "final_adapter"
    if final_adapter.exists():
        shutil.rmtree(final_adapter)
    shutil.copytree(initial_adapter, final_adapter)
    if not (final_adapter / "strategy_manifest.json").is_file():
        raise FileNotFoundError("Final adapter is missing strategy_manifest.json")
    final_strategy_manifest = json.loads(
        (final_adapter / "strategy_manifest.json").read_text(encoding="utf-8")
    )
    stage_result = {
        "schema_version": "1.0",
        "base_model": str(model_dir),
        "base_model_config_sha256": sha256_file(model_dir / "config.json"),
        "base_model_fingerprint": final_strategy_manifest.get("base_model_fingerprint"),
        "processor": str(model_dir),
        "cycle_manifest_sha256": sha256_file(manifest_path),
        "round_count": len(results),
        "rounds": results,
        "lora_target_audit": final_strategy_manifest.get("lora_target_audit"),
        "recommended_adapter": str(final_adapter),
    }
    (run_root / "stage_a_result.json").write_text(
        json.dumps(stage_result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not args.skip_e2_eval:
        _run(
            [
                sys.executable,
                "scripts/evaluate_rs_vlm.py",
                "--config",
                str(DEFAULT_EVAL_CONFIG),
                "--checkpoint",
                str(final_adapter),
                "--output-dir",
                str(run_root / "evaluation_e2_v2"),
            ]
        )
    print(f"Stage-A completed: {final_adapter}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
