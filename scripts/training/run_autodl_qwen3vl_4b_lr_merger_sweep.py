"""AutoDL 一键执行 Qwen3-VL-4B LR + Visual Merger 快速诊断矩阵。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sat_rs_vlm.training.lr_merger_sweep import (  # noqa: E402
    DEFAULT_SWEEP_CONFIG,
    EvaluationOOMError,
    ExperimentSpec,
    InfrastructureError,
    checkpoint_fingerprint,
    checkpoint_is_complete,
    build_evaluation_command,
    clear_cuda_cache,
    enrich_strategy_manifest,
    evaluation_identity,
    extract_e1_metrics,
    is_oom_log,
    load_sweep_config,
    materialize_training_config,
    maybe_plot_sweep,
    parse_experiment_specs,
    select_phase_a_lr,
    shutdown_if_requested,
    validate_probe_contract,
    weak_score,
    write_audit_for_lora_only,
    write_sweep_report,
    write_status,
    run_logged_command,
)
from sat_rs_vlm.training.weight_analysis import (  # noqa: E402
    analyze_lora_adapters,
    analyze_merger_sidecar,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_SWEEP_CONFIG)
    parser.add_argument("--initial-adapter", type=Path, required=True)
    parser.add_argument("--existing-vit-checkpoint", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--processor-dir", type=Path, default=None)
    parser.add_argument("--train-file", type=Path, default=None)
    parser.add_argument("--val-file", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument(
        "--baseline-evaluation-dir",
        type=Path,
        default=None,
        help="Reuse an existing legal E1 directory for Round1 baseline.",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--skip-baseline-eval", action="store_true")
    parser.add_argument("--skip-existing-vit", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument(
        "--only", default=None, help="Comma-separated A1,A2,A3,B1,B2,B3"
    )
    parser.add_argument("--shutdown", action="store_true")
    return parser.parse_args()


def _resolve(value: str | Path, *, base: Path = PROJECT_ROOT) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path if path.is_absolute() else base / path


def _required_env(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise InfrastructureError(f"Required environment variable is missing: {name}")
    return Path(value).expanduser()


def _load_base_payload(path: Path) -> dict[str, Any]:
    payload = dict(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    model = dict(payload.get("model", {}))
    data = dict(payload.get("data", {}))
    model_dir = model.get("model_dir")
    if isinstance(model_dir, str) and "${" in model_dir:
        model["model_dir"] = str(_required_env("QWEN3VL_4B_MODEL_DIR"))
    processor_dir = model.get("processor_dir")
    if isinstance(processor_dir, str) and "${" in processor_dir:
        model["processor_dir"] = str(_required_env("QWEN3VL_4B_MODEL_DIR"))
    payload["model"] = model
    payload["data"] = data
    return payload


def _promote_checkpoint(final_dir: Path, checkpoint_dir: Path) -> None:
    """为 checkpoint-100 补齐 processor/manifest，保留旧文件不覆盖。"""

    if not checkpoint_dir.is_dir():
        return
    for name in (
        "adapter_config.json",
        "adapter_model.safetensors",
        "strategy_manifest.json",
    ):
        source = final_dir / name
        destination = checkpoint_dir / name
        if source.is_file() and not destination.is_file():
            shutil.copy2(source, destination)
    source_processor = final_dir / "processor"
    destination_processor = checkpoint_dir / "processor"
    if source_processor.is_dir() and not destination_processor.exists():
        shutil.copytree(source_processor, destination_processor)
    for name in (
        "visual_trainable_weights.safetensors",
        "visual_trainable_manifest.json",
    ):
        source = final_dir / name
        destination = checkpoint_dir / name
        if source.is_file() and not destination.is_file():
            shutil.copy2(source, destination)


def _evaluate(
    *,
    evaluation_config: Path,
    checkpoint: Path,
    output_dir: Path,
    log_path: Path,
    resume: bool,
    batch_size: int,
    fallback: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if (
        resume
        and (output_dir / "evaluation_v1_5" / "evaluation_manifest.json").is_file()
    ):
        return evaluation_identity(output_dir)
    started = time.perf_counter()
    command = [
        sys.executable,
        *build_evaluation_command(
            evaluation_config=evaluation_config,
            checkpoint=checkpoint,
            output_dir=output_dir,
            batch_size=batch_size,
        ),
    ]
    code = run_logged_command(
        command,
        log_path,
    )
    if code != 0:
        if batch_size == 4 and is_oom_log(log_path):
            clear_cuda_cache()
            raise EvaluationOOMError(
                f"E1 evaluation CUDA OOM at batch_size=4: {output_dir}"
            )
        raise RuntimeError(f"Evaluation failed with exit code {code}: {output_dir}")
    identity = evaluation_identity(output_dir)
    actual_batch = identity.get("eval_batch_size")
    if actual_batch != batch_size:
        raise InfrastructureError(
            f"Evaluator batch mismatch: requested={batch_size}, manifest={actual_batch}"
        )
    runtime = time.perf_counter() - started
    sample_count = identity.get("sample_count")
    samples_per_second = (
        float(sample_count) / runtime
        if isinstance(sample_count, (int, float)) and runtime > 0
        else None
    )
    summary = identity.get("summary", {})
    source_metadata_path = output_dir / "evaluation_metadata.json"
    source_metadata = (
        json.loads(source_metadata_path.read_text(encoding="utf-8"))
        if source_metadata_path.is_file()
        else {}
    )
    peak_vram_mb = source_metadata.get("peak_vram_mb")
    resource = dict(
        dict(summary).get("p0_data_availability", {}).get("resource_benchmark", {})
    )
    if not isinstance(peak_vram_mb, (int, float)):
        peak_vram_mb = None
        for key in ("peak_vram_mb", "peak_memory_mb"):
            value = resource.get(key)
            if isinstance(value, (int, float)):
                peak_vram_mb = float(value)
                break
    metadata = {
        "eval_batch_size": batch_size,
        "sample_count": sample_count,
        "evaluation_runtime_seconds": runtime,
        "samples_per_second": samples_per_second,
        "peak_vram_mb": peak_vram_mb,
        "requested_eval_batch_size": 4,
        "evaluation_batch_fallback": dict(fallback) if fallback else None,
    }
    (output_dir / "evaluation_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    identity["metadata"] = metadata
    return identity


def _evaluation_record_fields(identity: Mapping[str, Any]) -> dict[str, Any]:
    metadata = dict(identity.get("metadata", {}))
    return {
        "eval_batch_size": metadata.get(
            "eval_batch_size", identity.get("eval_batch_size")
        ),
        "sample_count": metadata.get("sample_count", identity.get("sample_count")),
        "evaluation_runtime_seconds": metadata.get("evaluation_runtime_seconds"),
        "samples_per_second": metadata.get("samples_per_second"),
        "peak_vram_mb": metadata.get("peak_vram_mb"),
    }


def _make_record(
    spec: ExperimentSpec,
    *,
    status: str,
    metrics: dict[str, Any] | None = None,
    analysis: Mapping[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    metrics = dict(metrics or {})
    lora_global = dict(dict(analysis or {}).get("lora", {}).get("global", {}))
    return {
        "experiment_id": spec.experiment_id,
        "label": spec.label,
        "status": status,
        "phase": spec.phase,
        "lora_lr": spec.lora_lr,
        "merger_lr": spec.merger_lr,
        "vit_last_n": spec.vit_last_n,
        "main_merger": spec.main_merger,
        "lora_base_ratio": lora_global.get("delta_base_ratio"),
        **metrics,
        "weak_score": weak_score(metrics),
        "analysis_error": dict(analysis or {}).get("error"),
        "error": error,
    }


def run(args: argparse.Namespace) -> int:
    payload = load_sweep_config(args.config)
    common = dict(payload.get("common", {}))
    base_config = _resolve(payload["base_training_config"])
    if not base_config.is_file():
        raise InfrastructureError(f"Base training config does not exist: {base_config}")
    base_payload = _load_base_payload(base_config)
    model_dir = args.model_dir or _resolve(base_payload["model"]["model_dir"])
    processor_dir = args.processor_dir or model_dir
    image_root = args.image_root or _resolve(
        base_payload["data"].get("image_root", _required_env("DATA_ROOT"))
    )
    val_file = args.val_file or _resolve(base_payload["data"]["val_file"])
    train_file = args.train_file or _resolve(payload["probe_dataset"])
    probe_manifest = _resolve(payload["probe_manifest"])
    protected_manifest = _resolve(
        payload.get(
            "protected_evaluation_manifest",
            base_payload.get("vit_probe", {}).get(
                "protected_evaluation_manifest",
                "data/evaluation/tiers_v2/evaluation_tiers_manifest.json",
            ),
        )
    )
    output_root = (
        args.output_root
        or _resolve(os.environ.get("OUTPUT_ROOT", "outputs"))
        / "experiments/qwen3vl_4b_lr_merger_sweep"
    )
    report_dir = args.report_dir or output_root / "reports"
    evaluation_config = _resolve(payload["evaluation_config"])
    for path in (model_dir, processor_dir, val_file, evaluation_config):
        if not Path(path).exists():
            raise InfrastructureError(f"Required sweep asset does not exist: {path}")
    initial_fingerprint = checkpoint_fingerprint(args.initial_adapter)

    if not train_file.is_file():
        source_files = [
            str(_resolve(value))
            for value in payload.get("probe_source_train_files", [])
        ]
        builder_config = base_config
        command = [
            sys.executable,
            "scripts/training/build_qwen3vl_4b_vit_probe_dataset.py",
            "--config",
            str(builder_config),
            "--output-dir",
            str(train_file.parent),
        ]
        for source in source_files:
            command.extend(["--source-train-file", source])
        if not args.prepare_only and not args.dry_run and not args.forward_only:
            code = run_logged_command(command, output_root / "prepare_probe.log")
            if code != 0:
                raise InfrastructureError("Existing deterministic probe builder failed")
    if not train_file.is_file():
        raise InfrastructureError(f"Probe train file does not exist: {train_file}")
    train_contract = validate_probe_contract(
        train_file, probe_manifest, protected_manifest
    )
    if (
        output_root.exists()
        and any(output_root.iterdir())
        and not args.resume
        and not args.prepare_only
        and not args.dry_run
        and not args.forward_only
    ):
        raise InfrastructureError(
            f"Output root is not empty; refusing to overwrite an existing sweep: {output_root}. "
            "Use a new --output-root or --resume."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    requested_eval_batch_size = int(common.get("eval_batch_size", 4))
    if requested_eval_batch_size != 4:
        raise InfrastructureError(
            "This 4090 sweep requires common.eval_batch_size=4; only the fixed 4→2 "
            "CUDA OOM fallback is supported."
        )
    effective_eval_batch_size = requested_eval_batch_size
    evaluation_batch_fallback: dict[str, Any] | None = None
    (output_root / "preflight.json").write_text(
        json.dumps(
            {
                "initial_adapter": initial_fingerprint,
                "train_contract": train_contract,
                "model_dir": str(model_dir),
                "evaluation_config": str(evaluation_config),
                "requested_eval_batch_size": requested_eval_batch_size,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if args.prepare_only:
        write_sweep_report(
            report_dir,
            [],
            {"selection_mode": "prepare_only", "selected_lr": None},
            {
                "requested_eval_batch_size": requested_eval_batch_size,
                "effective_eval_batch_size": effective_eval_batch_size,
                "evaluation_batch_fallback": evaluation_batch_fallback,
            },
        )
        (output_root / "sweep_status.json").write_text(
            json.dumps(
                {
                    "status": "PREPARED",
                    "train_contract": train_contract,
                    "requested_eval_batch_size": requested_eval_batch_size,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "status": "PREPARED",
                    "output_root": str(output_root),
                    "train_contract": train_contract,
                    "requested_eval_batch_size": requested_eval_batch_size,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    eval_contract: dict[str, Any] | None = None
    records: list[dict[str, Any]] = []
    baseline_record: dict[str, Any] | None = None

    def evaluate_with_fallback(
        *, checkpoint: Path, output_dir: Path, log_path: Path, resume: bool
    ) -> dict[str, Any]:
        nonlocal effective_eval_batch_size, evaluation_batch_fallback
        try:
            return _evaluate(
                evaluation_config=evaluation_config,
                checkpoint=checkpoint,
                output_dir=output_dir,
                log_path=log_path,
                resume=resume,
                batch_size=effective_eval_batch_size,
                fallback=evaluation_batch_fallback,
            )
        except EvaluationOOMError:
            if effective_eval_batch_size != 4:
                raise
            clear_cuda_cache()
            effective_eval_batch_size = 2
            evaluation_batch_fallback = {
                "requested": 4,
                "effective": 2,
                "reason": "cuda_oom",
            }
            print(
                "WARNING: E1 batch_size=4 CUDA OOM; switching the entire sweep to batch_size=2."
            )
            return _evaluate(
                evaluation_config=evaluation_config,
                checkpoint=checkpoint,
                output_dir=output_dir,
                log_path=log_path,
                resume=False,
                batch_size=effective_eval_batch_size,
                fallback=evaluation_batch_fallback,
            )

    if not args.skip_baseline_eval and not args.dry_run and not args.forward_only:
        baseline_dir = args.baseline_evaluation_dir or output_root / "baseline_e1"
        baseline_log = output_root / "logs" / "baseline_e1.log"
        baseline_identity = evaluate_with_fallback(
            checkpoint=args.initial_adapter,
            output_dir=baseline_dir,
            log_path=baseline_log,
            resume=args.resume,
        )
        baseline_batch = baseline_identity.get("eval_batch_size")
        if baseline_batch not in {2, 4}:
            raise InfrastructureError(
                f"Baseline E1 manifest has no supported eval batch size: {baseline_batch}"
            )
        effective_eval_batch_size = int(baseline_batch)
        eval_contract = baseline_identity
        baseline_metrics = extract_e1_metrics(baseline_identity["summary"])
        baseline_record = {
            "experiment_id": "baseline",
            "status": "EVALUATED",
            **baseline_metrics,
            "weak_score": weak_score(baseline_metrics),
            "checkpoint": str(args.initial_adapter),
            **_evaluation_record_fields(baseline_identity),
        }
        provenance_path = baseline_dir / "evaluation_provenance.json"
        if provenance_path.is_file():
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            recorded = dict(provenance.get("checkpoint", {}))
            if recorded.get("adapter_weights_sha256") not in (
                None,
                initial_fingerprint["adapter_weights_sha256"],
            ):
                raise InfrastructureError(
                    "Reused baseline E1 provenance does not match Round1 adapter"
                )
        else:
            if args.baseline_evaluation_dir is not None:
                print(
                    "WARNING: reused baseline E1 has no checkpoint provenance; "
                    "tier identity was validated, but the source directory was left unchanged."
                )
            else:
                provenance_path.write_text(
                    json.dumps(
                        {
                            "checkpoint": initial_fingerprint,
                            "evaluation": eval_contract,
                            "provenance_status": "recorded_by_sweep",
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
        records.append(baseline_record)

    max_steps = int(args.max_steps or common.get("max_steps", 100))
    batch_size = int(common.get("batch_size", 4))
    grad_accum = int(common.get("gradient_accumulation_steps", 4))
    selected_ids = (
        {value.strip() for value in str(args.only).split(",")}
        if args.only
        else {"A1", "A2", "A3", "B1", "B2", "B3"}
    )
    if any(
        not value or value not in {"A1", "A2", "A3", "B1", "B2", "B3"}
        for value in selected_ids
    ):
        raise ValueError("--only accepts only A1,A2,A3,B1,B2,B3")

    specs_a = [spec for spec in parse_experiment_specs(payload) if spec.phase == "A"]
    if args.dry_run or args.forward_only:
        for spec in parse_experiment_specs(payload, selected_lr=5.0e-5):
            if spec.experiment_id not in selected_ids:
                continue
            experiment_dir = output_root / spec.experiment_id
            config_path = experiment_dir / "resolved_training_config.yaml"
            materialize_training_config(
                base_config,
                output_path=config_path,
                model_dir=model_dir,
                processor_dir=processor_dir,
                train_file=train_file,
                val_file=val_file,
                image_root=image_root,
                initial_adapter=args.initial_adapter,
                output_dir=experiment_dir / "checkpoint",
                spec=spec,
                common=common,
                max_steps=max_steps,
                batch_size=batch_size,
                gradient_accumulation_steps=grad_accum,
            )
            mode = "--dry-run" if args.dry_run else "--forward-only"
            run_logged_command(
                [
                    sys.executable,
                    "scripts/train_qwen3vl_lora.py",
                    "--config",
                    str(config_path),
                    mode,
                ],
                experiment_dir / "orchestration.log",
            )
        write_sweep_report(
            report_dir,
            records,
            {"selection_mode": "dry_run_or_forward_only", "selected_lr": None},
            {
                "requested_eval_batch_size": requested_eval_batch_size,
                "effective_eval_batch_size": effective_eval_batch_size,
                "evaluation_batch_fallback": evaluation_batch_fallback,
            },
        )
        print(
            f"Sweep {('dry-run' if args.dry_run else 'forward-only')} completed: {output_root}"
        )
        return 0

    def run_one(spec: ExperimentSpec) -> dict[str, Any]:
        nonlocal eval_contract
        experiment_dir = output_root / spec.experiment_id
        checkpoint_dir = experiment_dir / "checkpoint"
        status_path = experiment_dir / "experiment_status.json"
        existing = (
            json.loads(status_path.read_text(encoding="utf-8"))
            if status_path.is_file()
            else {}
        )
        if existing.get("status") == "ANALYZED" and args.resume:
            return dict(
                existing.get(
                    "record",
                    {"experiment_id": spec.experiment_id, "status": "ANALYZED"},
                )
            )
        if existing.get("status") == "FAILED" and not args.retry_failed and args.resume:
            return {
                "experiment_id": spec.experiment_id,
                "status": "FAILED",
                "error": existing.get("error", "previous failure"),
            }
        experiment_dir.mkdir(parents=True, exist_ok=True)
        write_status(status_path, "RUNNING", experiment_id=spec.experiment_id)
        config_path = experiment_dir / "resolved_training_config.yaml"
        materialize_training_config(
            base_config,
            output_path=config_path,
            model_dir=model_dir,
            processor_dir=processor_dir,
            train_file=train_file,
            val_file=val_file,
            image_root=image_root,
            initial_adapter=args.initial_adapter,
            output_dir=checkpoint_dir,
            spec=spec,
            common=common,
            max_steps=max_steps,
            batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
        )
        log_path = experiment_dir / "train.log"
        command = [
            sys.executable,
            "scripts/train_qwen3vl_lora.py",
            "--config",
            str(config_path),
            "--skip-eval",
        ]
        code = run_logged_command(command, log_path)
        if code != 0 and is_oom_log(log_path):
            retry_config = experiment_dir / "resolved_training_config_oom_retry.yaml"
            materialize_training_config(
                base_config,
                output_path=retry_config,
                model_dir=model_dir,
                processor_dir=processor_dir,
                train_file=train_file,
                val_file=val_file,
                image_root=image_root,
                initial_adapter=args.initial_adapter,
                output_dir=checkpoint_dir,
                spec=spec,
                common=common,
                max_steps=max_steps,
                batch_size=2,
                gradient_accumulation_steps=8,
            )
            code = run_logged_command(
                [
                    sys.executable,
                    "scripts/train_qwen3vl_lora.py",
                    "--config",
                    str(retry_config),
                    "--skip-eval",
                ],
                log_path,
            )
        if code != 0 or not checkpoint_is_complete(
            checkpoint_dir, visual=spec.phase == "B"
        ):
            error = (
                f"training failed with exit_code={code}"
                if code != 0
                else "checkpoint is incomplete"
            )
            write_status(
                status_path, "FAILED", experiment_id=spec.experiment_id, error=error
            )
            return _make_record(spec, status="FAILED", error=error)
        manifest_path = checkpoint_dir / "strategy_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if spec.phase == "A":
            write_audit_for_lora_only(checkpoint_dir, manifest)
        enriched = enrich_strategy_manifest(
            checkpoint_dir,
            spec=spec,
            initial_adapter=args.initial_adapter,
            base_model=model_dir,
            train_contract=train_contract,
            eval_contract=eval_contract,
            max_steps=max_steps,
            effective_batch=batch_size * grad_accum,
        )
        _promote_checkpoint(checkpoint_dir, checkpoint_dir / f"checkpoint-{max_steps}")
        (experiment_dir / "trainable_parameters.json").write_text(
            (checkpoint_dir / "trainable_parameters.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        write_status(
            status_path,
            "TRAINED",
            experiment_id=spec.experiment_id,
            checkpoint=str(checkpoint_dir),
            record={"experiment_id": spec.experiment_id, "status": "TRAINED"},
        )
        metrics: dict[str, Any] = {}
        evaluation_dir = experiment_dir / "evaluation_e1"
        try:
            identity = evaluate_with_fallback(
                checkpoint=checkpoint_dir,
                output_dir=evaluation_dir,
                log_path=experiment_dir / "evaluation.log",
                resume=args.resume,
            )
        except InfrastructureError:
            raise
        except Exception as exc:
            error = f"evaluation failed: {exc}"
            write_status(
                status_path,
                "FAILED",
                experiment_id=spec.experiment_id,
                checkpoint=str(checkpoint_dir),
                error=error,
            )
            return _make_record(spec, status="FAILED", error=error)
        if eval_contract is None:
            eval_contract = identity
        elif identity.get("tier_sha256") != eval_contract.get("tier_sha256"):
            raise InfrastructureError("Candidate E1 tier SHA differs from baseline")
        if identity.get("eval_batch_size") != effective_eval_batch_size:
            raise InfrastructureError(
                "Candidate E1 batch size differs from the sweep effective batch size"
            )
        metrics = extract_e1_metrics(identity["summary"])
        metrics.update(_evaluation_record_fields(identity))
        (evaluation_dir / "evaluation_provenance.json").write_text(
            json.dumps(
                {"checkpoint_manifest": enriched, "evaluation": identity},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        write_status(
            status_path,
            "EVALUATED",
            experiment_id=spec.experiment_id,
            checkpoint=str(checkpoint_dir),
            evaluation=str(evaluation_dir),
        )
        analysis_payload: dict[str, Any] = {}
        try:
            analysis_payload["lora"] = analyze_lora_adapters(
                args.initial_adapter,
                checkpoint_dir,
                model_dir=model_dir,
                output_dir=report_dir / "lora_analysis" / spec.experiment_id,
            )
            if spec.phase == "B":
                analysis_payload["merger"] = analyze_merger_sidecar(
                    checkpoint_dir,
                    model_dir=model_dir,
                    output_dir=report_dir / "merger_analysis" / spec.experiment_id,
                )
        except Exception as exc:  # analysis must not invalidate a completed checkpoint
            analysis_payload["error"] = str(exc)
        record = _make_record(
            spec, status="ANALYZED", metrics=metrics, analysis=analysis_payload
        )
        write_status(
            status_path,
            "ANALYZED",
            experiment_id=spec.experiment_id,
            checkpoint=str(checkpoint_dir),
            record=record,
            analysis=analysis_payload,
        )
        return record

    for spec in specs_a:
        if spec.experiment_id in selected_ids:
            records.append(run_one(spec))
    phase_a_records = [
        record
        for record in records
        if str(record.get("experiment_id", "")).startswith("A")
    ]
    baseline_for_selector = baseline_record or {"metrics": {}}
    selection = select_phase_a_lr(phase_a_records, baseline_for_selector)
    (report_dir / "phase_a_selection.json").parent.mkdir(parents=True, exist_ok=True)
    (report_dir / "phase_a_selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for spec in parse_experiment_specs(
        payload, selected_lr=float(selection["selected_lr"])
    ):
        if spec.phase == "B" and spec.experiment_id in selected_ids:
            records.append(run_one(spec))

    if (
        args.existing_vit_checkpoint
        and not args.skip_existing_vit
        and args.existing_vit_checkpoint.exists()
    ):
        existing_dir = output_root / "existing_vit_checkpoint"
        identity = evaluate_with_fallback(
            checkpoint=args.existing_vit_checkpoint,
            output_dir=existing_dir,
            log_path=output_root / "logs" / "existing_vit_checkpoint.log",
            resume=args.resume,
        )
        metrics = extract_e1_metrics(identity["summary"])
        metrics.update(_evaluation_record_fields(identity))
        records.append(
            {
                "experiment_id": "existing_vit_checkpoint",
                "status": "EVALUATED",
                **metrics,
                "weak_score": weak_score(metrics),
                "checkpoint": str(args.existing_vit_checkpoint),
            }
        )
    evaluation_context = {
        "requested_eval_batch_size": requested_eval_batch_size,
        "effective_eval_batch_size": effective_eval_batch_size,
        "evaluation_batch_fallback": evaluation_batch_fallback,
    }
    write_sweep_report(report_dir, records, selection, evaluation_context)
    maybe_plot_sweep(report_dir, records)
    status = (
        "COMPLETED_WITH_EXPERIMENT_FAILURES"
        if any(record.get("status") == "FAILED" for record in records)
        else "COMPLETED"
    )
    (output_root / "sweep_status.json").write_text(
        json.dumps(
            {
                "status": status,
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "records": records,
                "selection": selection,
                "evaluation": evaluation_context,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Sweep finished: {status}; reports={report_dir}")
    return 0 if status == "COMPLETED" else 1


def main() -> int:
    args = parse_args()
    if args.dry_run and args.forward_only:
        raise ValueError("--dry-run and --forward-only are mutually exclusive")
    if args.shutdown and (args.dry_run or args.forward_only or args.prepare_only):
        raise ValueError(
            "--shutdown is disabled for prepare-only, dry-run, and forward-only"
        )
    exit_code = 1
    try:
        exit_code = run(args)
    except (InfrastructureError, OSError, ValueError, RuntimeError) as exc:
        print(f"Sweep failed: {exc}", file=sys.stderr)
        output_root = (
            args.output_root
            or Path(os.environ.get("OUTPUT_ROOT", "outputs"))
            / "experiments/qwen3vl_4b_lr_merger_sweep"
        )
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "sweep_status.json").write_text(
            json.dumps(
                {"status": "INFRASTRUCTURE_FAILED", "error": str(exc)},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    finally:
        if args.shutdown and not (
            args.dry_run or args.forward_only or args.prepare_only
        ):
            try:
                shutdown_if_requested(True)
            except Exception as exc:
                print(f"Shutdown was refused: {exc}", file=sys.stderr)
                exit_code = max(exit_code, 1)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
