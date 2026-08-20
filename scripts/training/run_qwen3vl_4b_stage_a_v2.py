"""Run the formal two-stage Qwen3-VL-4B Stage-A v2 workflow on AutoDL."""

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

from sat_rs_vlm.configuration.environment import expand_environment
from sat_rs_vlm.data.cyclic_training import sha256_file
from sat_rs_vlm.data.stage_a_v2 import (
    build_stage2_vrs_levir_dataset,
    load_validated_population_manifest,
)
from sat_rs_vlm.training.lr_merger_sweep import (
    clear_cuda_cache,
    extract_e1_metrics,
)
from sat_rs_vlm.training.stage_a_v2 import (
    R0_STAGE,
    R1_STAGE,
    adapter_is_complete,
    latest_trainer_checkpoint,
    model_fingerprint_from_directory,
    resolve_stage_epoch_plan,
    training_command,
    validate_r0_adapter_contract,
    validate_stage2_sampler_coverage,
)
from sat_rs_vlm.training.vit_probe import make_checkpoint_evaluable
from sat_rs_vlm.utils.jsonl import read_jsonl

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_CONFIG = ROOT / "configs/data/autodl_qwen3vl_4b_stage_a_v2.yaml"
DEFAULT_R0_CONFIG = ROOT / "configs/train/qwen3vl_4b_stage_a_v2_r0_strong_lora_4090.yaml"
DEFAULT_R1_CONFIG = ROOT / "configs/train/qwen3vl_4b_stage_a_v2_r1_visual_reinforce_4090.yaml"
DEFAULT_E1_CONFIG = ROOT / "configs/eval/qwen3vl_4b_stage_a_v2_e1_4090.yaml"
DEFAULT_E2_CONFIG = ROOT / "configs/eval/qwen3vl_4b_baseline_e2_v2.yaml"
STATES = {
    "POPULATION_READY",
    "R0_RUNNING",
    "R0_TRAINED",
    "R0_EVALUATED",
    "R1_DATA_READY",
    "R1_RUNNING",
    "R1_TRAINED",
    "R1_EVALUATED",
    "COMPLETED",
    "FAILED",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--data-config", default=str(DEFAULT_DATA_CONFIG))
    parser.add_argument("--r0-config", default=str(DEFAULT_R0_CONFIG))
    parser.add_argument("--r1-config", default=str(DEFAULT_R1_CONFIG))
    parser.add_argument("--e1-config", default=str(DEFAULT_E1_CONFIG))
    parser.add_argument("--e2-config", default=str(DEFAULT_E2_CONFIG))
    parser.add_argument("--r0-adapter", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--run-e2", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    return parser.parse_args()


def _load_yaml(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    # Runner 会通过 CLI 注入本轮 R0 adapter；预读 R1 配置时允许保留该占位符。
    return dict(expand_environment(payload, environ=os.environ, allow_unresolved=True))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _update_state(path: Path, state: str, **details: Any) -> dict[str, Any]:
    if state not in STATES:
        raise ValueError(f"Unknown Stage-A v2 state: {state}")
    previous = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    history = list(previous.get("history", []))
    history.append({"state": state, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    payload = {**previous, **details, "state": state, "history": history}
    _write_json(path, payload)
    return payload


def _run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("+ " + " ".join(command), flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
        code = process.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, command)


def _run_data_command(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _is_cuda_oom(log_path: Path) -> bool:
    if not log_path.is_file():
        return False
    text = log_path.read_text(encoding="utf-8", errors="ignore").lower()
    return "cuda out of memory" in text or "torch.outofmemoryerror" in text


def _evaluation_command(
    config: Path,
    checkpoint: Path,
    output_dir: Path,
    batch_size: int,
) -> list[str]:
    return [
        sys.executable,
        "scripts/evaluate_rs_vlm.py",
        "--config",
        str(config),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output_dir),
        "--batch-size",
        str(batch_size),
    ]


def _evaluation_identity(output_dir: Path, *, expected_tier: str) -> dict[str, Any]:
    root = output_dir / "evaluation_v1_5"
    manifest_path = root / "evaluation_manifest.json"
    summary_path = root / "summary.json"
    if not manifest_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"Evaluation artifacts are incomplete: {output_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tier = str(manifest.get("evaluation_tier", ""))
    if tier.upper() != expected_tier.upper():
        raise ValueError(f"Expected {expected_tier} evaluation, got {tier or 'unknown'}")
    metadata_path = output_dir / "evaluation_metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    )
    latency = dict(manifest.get("latency_context", {}))
    return {
        "directory": str(output_dir.resolve()),
        "tier": tier,
        "tier_sha256": manifest.get("evaluation_tier_sha256"),
        "sample_count": manifest.get("evaluated_samples"),
        "eval_batch_size": latency.get("eval_batch_size"),
        "manifest_sha256": sha256_file(manifest_path),
        "summary": json.loads(summary_path.read_text(encoding="utf-8")),
        "metadata": metadata,
    }


def _evaluate_once(
    *,
    config: Path,
    checkpoint: Path,
    output_dir: Path,
    log_path: Path,
    batch_size: int,
    resume: bool,
    expected_tier: str = "E1",
) -> dict[str, Any]:
    manifest = output_dir / "evaluation_v1_5" / "evaluation_manifest.json"
    if resume and manifest.is_file():
        identity = _evaluation_identity(output_dir, expected_tier=expected_tier)
        if identity.get("eval_batch_size") != batch_size:
            raise ValueError(
                "Existing evaluation batch mismatch during resume: "
                f"requested={batch_size}, actual={identity.get('eval_batch_size')}"
            )
        return identity
    try:
        _run_logged(
            _evaluation_command(config, checkpoint, output_dir, batch_size),
            log_path,
        )
    except subprocess.CalledProcessError as exc:
        if batch_size == 4 and _is_cuda_oom(log_path):
            clear_cuda_cache()
            raise RuntimeError("evaluation_cuda_oom_batch4") from exc
        raise
    identity = _evaluation_identity(output_dir, expected_tier=expected_tier)
    if identity.get("eval_batch_size") != batch_size:
        raise ValueError(
            "Evaluation batch mismatch: "
            f"requested={batch_size}, actual={identity.get('eval_batch_size')}"
        )
    return identity


def _archive_evaluation(directory: Path, suffix: str) -> None:
    if not directory.exists():
        return
    destination = directory.with_name(f"{directory.name}_{suffix}")
    counter = 1
    while destination.exists():
        counter += 1
        destination = directory.with_name(f"{directory.name}_{suffix}_{counter}")
    directory.rename(destination)


def _promote_half_checkpoint(
    adapter_dir: Path,
    half_step: int,
    *,
    require_visual_sidecar: bool = True,
) -> Path:
    """Promote a Trainer checkpoint while enforcing its visual-artifact contract."""

    source = adapter_dir / f"checkpoint-{half_step}"
    if not source.is_dir():
        raise FileNotFoundError(f"Expected half-epoch Trainer checkpoint is missing: {source}")
    make_checkpoint_evaluable(
        adapter_dir,
        source,
        checkpoint_step=half_step,
        require_visual_sidecar=require_visual_sidecar,
    )
    destination = adapter_dir.parent / "checkpoint-half"
    if destination.exists():
        return destination
    shutil.copytree(source, destination)
    return destination


def _metric_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float]:
    first = extract_e1_metrics(before["summary"])
    second = extract_e1_metrics(after["summary"])
    return {
        key: float(second[key]) - float(first[key])
        for key in sorted(set(first).intersection(second))
        if isinstance(first[key], (int, float)) and isinstance(second[key], (int, float))
    }


def _ensure_population(data_config_path: Path, config: dict[str, Any]) -> Path:
    population_dir = Path(str(config["population"]["output_dir"]))
    manifest_path = population_dir / "population_manifest.json"
    if not manifest_path.is_file():
        _run_data_command(
            [
                sys.executable,
                "scripts/data/prepare_multisource_training_data.py",
                "--config",
                str(data_config_path),
                "--build-population",
                "--population-output-dir",
                str(population_dir),
            ]
        )
    load_validated_population_manifest(manifest_path)
    return manifest_path


def _ensure_stage2(
    population_manifest: Path, data_config: dict[str, Any]
) -> tuple[Path, Path, dict[str, Any]]:
    stage2 = dict(data_config["stage2"])
    train_file = Path(str(stage2["output_file"]))
    manifest_file = Path(str(stage2["manifest_file"]))
    if not train_file.is_file() or not manifest_file.is_file():
        build_stage2_vrs_levir_dataset(
            population_manifest,
            output_file=train_file,
            manifest_file=manifest_file,
            seed=int(data_config.get("seed", 42)),
            vrs_per_levir=int(stage2.get("vrs_per_levir", 3)),
        )
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if sha256_file(train_file) != manifest.get("sha256"):
        raise ValueError("Stage2 train file SHA does not match stage2_manifest.json")
    current_population_sha = sha256_file(population_manifest)
    recorded_population_sha = manifest.get("population_manifest_sha256")
    if recorded_population_sha != current_population_sha:
        raise ValueError(
            "Stage2 population manifest SHA does not match the current canonical population: "
            f"recorded={recorded_population_sha!r}, current={current_population_sha!r}. "
            "Regenerate Stage2 data before training."
        )
    return train_file, manifest_file, manifest


def _copy_population_reports(run_root: Path, *paths: Path) -> None:
    destination = run_root / "population"
    destination.mkdir(parents=True, exist_ok=True)
    for path in paths:
        shutil.copy2(path, destination / path.name)


def _run_training_stage(
    *,
    stage: str,
    config_path: Path,
    train_file: Path,
    validation_file: Path,
    adapter_dir: Path,
    plan: Any,
    state_path: Path,
    initial_adapter: Path | None,
    resume: bool,
    mode: str | None,
    max_train_samples: int | None,
) -> None:
    if resume and adapter_is_complete(adapter_dir):
        return
    resume_checkpoint = latest_trainer_checkpoint(adapter_dir) if resume else None
    _update_state(
        state_path,
        "R0_RUNNING" if stage == R0_STAGE else "R1_RUNNING",
        active_stage=stage,
        adapter_continuation=str(initial_adapter) if initial_adapter else None,
        trainer_resume=str(resume_checkpoint) if resume_checkpoint else None,
    )
    command = training_command(
        sys.executable,
        config=config_path,
        train_file=train_file,
        validation_file=validation_file,
        output_dir=adapter_dir,
        save_steps=plan.half_checkpoint_step,
        initial_adapter=initial_adapter,
        resume_checkpoint=resume_checkpoint,
        mode=mode,
        max_train_samples=max_train_samples,
    )
    _run_logged(command, adapter_dir.parent / "logs" / "train.log")
    if mode is None and not adapter_is_complete(adapter_dir):
        raise ValueError(f"Training completed without a valid adapter: {adapter_dir}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.dry_run and args.forward_only:
        raise ValueError("--dry-run and --forward-only are mutually exclusive")
    if args.max_train_samples is not None and not (args.dry_run or args.forward_only):
        raise ValueError("--max-train-samples is allowed only for dry-run/forward-only")
    for name in ("QWEN3VL_4B_MODEL_DIR", "DATA_ROOT", "OUTPUT_ROOT"):
        if not os.environ.get(name):
            raise ValueError(f"Required environment variable is missing: {name}")
    model_dir = Path(os.environ["QWEN3VL_4B_MODEL_DIR"])
    model_fingerprint = model_fingerprint_from_directory(model_dir)
    data_config_path = Path(args.data_config)
    data_config = _load_yaml(data_config_path)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = Path(
        args.run_root or Path(os.environ["OUTPUT_ROOT"]) / f"qwen3vl_4b_stage_a_v2_{timestamp}"
    )
    setattr(args, "resolved_run_root", str(run_root))
    run_root.mkdir(parents=True, exist_ok=True)
    state_path = run_root / "run_manifest.json"
    if state_path.exists() and not args.resume:
        raise FileExistsError(f"Run root already contains state; use --resume: {run_root}")

    population_manifest = _ensure_population(data_config_path, data_config)
    population_payload, populations = load_validated_population_manifest(population_manifest)
    validation_record = dict(population_payload["validation"])
    validation_file = Path(str(validation_record["path"]))
    _update_state(
        state_path,
        "POPULATION_READY",
        model_fingerprint=model_fingerprint,
        population_manifest=str(population_manifest),
        population_manifest_sha256=sha256_file(population_manifest),
    )

    stage2_file: Path | None = None
    stage2_manifest_file: Path | None = None
    stage2_manifest: dict[str, Any] | None = None
    if args.prepare_only:
        stage2_file, stage2_manifest_file, stage2_manifest = _ensure_stage2(
            population_manifest, data_config
        )
        _copy_population_reports(run_root, population_manifest, stage2_manifest_file)
        return _update_state(
            state_path,
            "R1_DATA_READY",
            stage2_train_file=str(stage2_file),
            stage2_manifest=str(stage2_manifest_file),
            stage2=stage2_manifest,
        )

    r0_config = _load_yaml(args.r0_config)
    r1_config = _load_yaml(args.r1_config)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    r0_plan = resolve_stage_epoch_plan(
        len(populations["VRSBench"]),
        per_device_batch_size=int(r0_config["training"]["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(r0_config["training"]["gradient_accumulation_steps"]),
        world_size=world_size,
    )
    r0_root = run_root / "r0"
    r0_adapter = Path(args.r0_adapter) if args.r0_adapter else r0_root / "adapter"
    _write_json(r0_root / "training_plan.json", r0_plan.to_dict())
    mode = "--dry-run" if args.dry_run else "--forward-only" if args.forward_only else None
    if not args.r0_adapter:
        _run_training_stage(
            stage=R0_STAGE,
            config_path=Path(args.r0_config),
            train_file=Path(population_payload["populations"]["VRSBench"]["path"]),
            validation_file=validation_file,
            adapter_dir=r0_adapter,
            plan=r0_plan,
            state_path=state_path,
            initial_adapter=None,
            resume=args.resume,
            mode=mode,
            max_train_samples=args.max_train_samples,
        )
    if mode is not None and not adapter_is_complete(r0_adapter):
        result = _update_state(
            state_path,
            "R0_RUNNING",
            diagnostic_mode=mode,
            r0_diagnostic="passed",
            r1_diagnostic="deferred_until_a_valid_r0_adapter_is_supplied",
        )
        _write_json(run_root / "stage_a_v2_result.json", result)
        return result
    if mode is None:
        if not args.r0_adapter:
            _promote_half_checkpoint(
                r0_adapter,
                r0_plan.half_checkpoint_step,
                require_visual_sidecar=False,
            )
        _update_state(state_path, "R0_TRAINED", r0_adapter=str(r0_adapter))

    # 正式顺序要求先完成 R0 E1，再构建并训练 R1。诊断模式不做生成评测。
    e1_config = Path(args.e1_config)
    current_state = json.loads(state_path.read_text(encoding="utf-8"))
    effective_batch = int(current_state.get("effective_eval_batch_size", 4))
    if effective_batch not in {2, 4}:
        raise ValueError(f"Invalid persisted E1 batch size: {effective_batch}")
    fallback = current_state.get("evaluation_batch_fallback")
    if not isinstance(fallback, dict):
        fallback = None
    r0_eval_dir = r0_root / "evaluation_e1"
    r0_identity: dict[str, Any] | None = None
    if mode is None:
        try:
            r0_identity = _evaluate_once(
                config=e1_config,
                checkpoint=r0_adapter,
                output_dir=r0_eval_dir,
                log_path=r0_root / "logs" / "evaluation_e1.log",
                batch_size=effective_batch,
                resume=args.resume,
            )
        except RuntimeError as exc:
            if str(exc) != "evaluation_cuda_oom_batch4":
                raise
            _archive_evaluation(r0_eval_dir, "batch4_oom")
            effective_batch = 2
            fallback = {"requested": 4, "effective": 2, "reason": "cuda_oom"}
            _update_state(
                state_path,
                "R0_TRAINED",
                effective_eval_batch_size=2,
                evaluation_batch_fallback=fallback,
            )
            r0_identity = _evaluate_once(
                config=e1_config,
                checkpoint=r0_adapter,
                output_dir=r0_eval_dir,
                log_path=r0_root / "logs" / "evaluation_e1_batch2.log",
                batch_size=effective_batch,
                resume=False,
            )
        _update_state(
            state_path,
            "R0_EVALUATED",
            effective_eval_batch_size=effective_batch,
            evaluation_batch_fallback=fallback,
        )

    parent_audit = validate_r0_adapter_contract(
        r0_adapter,
        model_dir,
        expected_target_modules=r1_config["lora"]["target_modules"],
    )
    stage2_file, stage2_manifest_file, stage2_manifest = _ensure_stage2(
        population_manifest, data_config
    )
    frozen_stage2_rows = read_jsonl(stage2_file)
    sampler_audit = validate_stage2_sampler_coverage(
        frozen_stage2_rows,
        source_batch_pattern=r1_config["data"]["source_batch_pattern"],
        batch_size=int(r1_config["training"]["per_device_train_batch_size"]),
        seed=int(r1_config["training"]["seed"]),
    )
    _copy_population_reports(run_root, population_manifest, stage2_manifest_file)
    _update_state(
        state_path,
        "R1_DATA_READY",
        r0_parent_audit=parent_audit,
        stage2=stage2_manifest,
        stage2_sampler_audit=sampler_audit,
    )
    r1_plan = resolve_stage_epoch_plan(
        len(frozen_stage2_rows),
        per_device_batch_size=int(r1_config["training"]["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(r1_config["training"]["gradient_accumulation_steps"]),
        world_size=world_size,
    )
    r1_root = run_root / "r1"
    r1_adapter = r1_root / "adapter"
    _write_json(r1_root / "training_plan.json", r1_plan.to_dict())
    _run_training_stage(
        stage=R1_STAGE,
        config_path=Path(args.r1_config),
        train_file=stage2_file,
        validation_file=validation_file,
        adapter_dir=r1_adapter,
        plan=r1_plan,
        state_path=state_path,
        initial_adapter=r0_adapter,
        resume=args.resume,
        mode=mode,
        max_train_samples=args.max_train_samples,
    )
    if mode is not None:
        result = _update_state(
            state_path,
            "R1_RUNNING",
            diagnostic_mode=mode,
            r0_diagnostic="reused_valid_adapter",
            r1_diagnostic="passed",
        )
        _write_json(run_root / "stage_a_v2_result.json", result)
        return result
    _promote_half_checkpoint(
        r1_adapter,
        r1_plan.half_checkpoint_step,
        require_visual_sidecar=True,
    )
    _update_state(state_path, "R1_TRAINED", r1_adapter=str(r1_adapter))

    r1_eval_dir = r1_root / "evaluation_e1"
    assert r0_identity is not None
    try:
        r1_identity = _evaluate_once(
            config=e1_config,
            checkpoint=r1_adapter,
            output_dir=r1_eval_dir,
            log_path=r1_root / "logs" / "evaluation_e1.log",
            batch_size=effective_batch,
            resume=args.resume,
        )
    except RuntimeError as exc:
        if str(exc) != "evaluation_cuda_oom_batch4" or effective_batch != 4:
            raise
        _archive_evaluation(r1_eval_dir, "batch4_oom")
        _archive_evaluation(r0_eval_dir, "batch4")
        effective_batch = 2
        fallback = {"requested": 4, "effective": 2, "reason": "cuda_oom"}
        _update_state(
            state_path,
            "R1_TRAINED",
            effective_eval_batch_size=2,
            evaluation_batch_fallback=fallback,
        )
        r0_identity = _evaluate_once(
            config=e1_config,
            checkpoint=r0_adapter,
            output_dir=r0_eval_dir,
            log_path=r0_root / "logs" / "evaluation_e1_batch2.log",
            batch_size=2,
            resume=False,
        )
        r1_identity = _evaluate_once(
            config=e1_config,
            checkpoint=r1_adapter,
            output_dir=r1_eval_dir,
            log_path=r1_root / "logs" / "evaluation_e1_batch2.log",
            batch_size=2,
            resume=False,
        )
    if r0_identity.get("tier_sha256") != r1_identity.get("tier_sha256"):
        raise ValueError("R0 and R1 E1 tier SHA values differ")
    _update_state(state_path, "R1_EVALUATED", effective_eval_batch_size=effective_batch)

    e2_identity = None
    if args.run_e2:
        e2_identity = _evaluate_once(
            config=Path(args.e2_config),
            checkpoint=r1_adapter,
            output_dir=r1_root / "evaluation_e2",
            log_path=r1_root / "logs" / "evaluation_e2.log",
            batch_size=effective_batch,
            resume=args.resume,
            expected_tier="E2",
        )
    result = {
        "success": True,
        "state": "COMPLETED",
        "run_root": str(run_root),
        "population_manifest": str(population_manifest),
        "r0": {
            "adapter": str(r0_adapter),
            "plan": r0_plan.to_dict(),
            "evaluation": r0_identity,
        },
        "r1": {
            "adapter": str(r1_adapter),
            "plan": r1_plan.to_dict(),
            "stage2_manifest": str(stage2_manifest_file),
            "evaluation": r1_identity,
        },
        "evaluation": {
            "tier": "E1",
            "tier_sha256": r0_identity.get("tier_sha256"),
            "requested_batch_size": 4,
            "effective_batch_size": effective_batch,
            "fallback": fallback,
            "r0_to_r1_delta": _metric_delta(r0_identity, r1_identity),
            "e2": e2_identity,
        },
    }
    _write_json(run_root / "stage_a_v2_result.json", result)
    _update_state(state_path, "COMPLETED", result_file=str(run_root / "stage_a_v2_result.json"))
    return result


def main() -> int:
    args = parse_args()
    try:
        result = run(args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        resolved = getattr(args, "resolved_run_root", args.run_root)
        run_root = Path(resolved) if resolved else None
        if run_root is not None:
            try:
                _update_state(run_root / "run_manifest.json", "FAILED", error=str(exc))
            except Exception:
                pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
