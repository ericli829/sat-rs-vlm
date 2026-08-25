"""Run resumable SEU sensitivity scans using Evaluation v1.5 and paired comparisons."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import re
from pathlib import Path
from typing import Any

from sat_rs_vlm.configuration.layered import (
    LayeredConfigRequest,
    load_layered_config,
    write_resolved_config,
)
from sat_rs_vlm.configuration.paths import PathConfig
from sat_rs_vlm.models.reliability.checksum import file_sha256
from sat_rs_vlm.training.experiment import write_json
from sat_rs_vlm.models.reliability.sensitivity import aggregate_sensitivity_conditions

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAYERED_TARGETS = {"language_model", "attention", "mlp", "lora_adapter", "lora_a", "lora_b", "visual_blocks"}
ALLOWED_TARGETS = {
    "all_parameters",
    "lora_adapter",
    "lora_a",
    "lora_b",
    "vision_encoder",
    "visual_blocks",
    "visual_merger",
    "language_model",
    "attention",
    "mlp",
    "embeddings",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/reliability/experiments/v15_sensitivity.yaml",
    )
    parser.add_argument("--environment", choices=("local", "autodl"), default="autodl")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-id", default="v15-seu-sensitivity")
    parser.add_argument("--targets", nargs="+")
    parser.add_argument("--layers", nargs="+", type=int)
    parser.add_argument("--bit-counts", nargs="+", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--evaluation-tier", choices=("E1", "E2", "E3"))
    parser.add_argument("--bit-planes", nargs="+", choices=("all", "sign", "exponent", "mantissa"))
    parser.add_argument(
        "--dry-run", action="store_true", help="Create and print a reproducible condition plan."
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate configured paths and report GPU visibility.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a matching interrupted run and skip completed conditions.",
    )
    parser.add_argument(
        "--pilot-only",
        action="store_true",
        help="Run only coverage-first pilot conditions; resume later without this flag.",
    )
    parser.add_argument("--activation-guard", action="store_true")
    parser.add_argument("--activation-guard-mode", choices=("research", "deployment"), default="research")
    parser.add_argument("--activation-patterns", nargs="+", default=["self_attn", "mlp"])
    parser.add_argument("--activation-max-abs", type=float, default=10000.0)
    return parser.parse_args()


def _environment() -> dict[str, str]:
    value = dict(os.environ)
    value.setdefault("PROJECT_ROOT", str(PROJECT_ROOT))
    return value


def load_config(args: argparse.Namespace) -> tuple[dict[str, Any], PathConfig]:
    environment_config = PROJECT_ROOT / (
        "configs/cloud/autodl.yaml" if args.environment == "autodl" else "configs/local/paths.yaml"
    )
    config = load_layered_config(
        LayeredConfigRequest(
            base_configs=(
                PROJECT_ROOT / "configs/base/default.yaml",
                PROJECT_ROOT / "configs/reliability/base.yaml",
            ),
            environment_config=environment_config,
            experiment_config=args.config.resolve(),
            cli_overrides={
                "paths.output_root": str(args.output_root) if args.output_root else None
            },
            project_root=PROJECT_ROOT,
        ),
        environ=_environment(),
    )
    fault = dict(config.get("fault", {}))
    if str(fault.get("layer_indices", "")).lower() == "auto":
        model_path = Path(str(config["model"]["base_model"])).expanduser()
        discovered = {"language": set(), "visual": set()}
        try:
            from safetensors import safe_open

            files = sorted(model_path.glob("*.safetensors"))
            for file_path in files:
                with safe_open(str(file_path), framework="pt", device="cpu") as handle:
                    for name in handle.keys():
                        indices = {int(value) for value in re.findall(r"(?:layers?|blocks?)\.(\d+)", name)}
                        region = "visual" if "visual" in name.lower() or "vision" in name.lower() else "language"
                        discovered[region].update(indices)
        except (ImportError, OSError):
            discovered = {"language": set(), "visual": set()}
        if not discovered["language"] and not discovered["visual"]:
            raise ValueError("fault.layer_indices=auto could not discover layers from model safetensors")
        config["fault"] = {
            **fault,
            "layer_indices": sorted(discovered["language"]),
            "visual_layer_indices": sorted(discovered["visual"]),
        }
    if args.evaluation_tier:
        evaluation = dict(config.get("evaluation", {}))
        config["evaluation"] = {**evaluation, "tier": args.evaluation_tier}
        config["data"] = {
            **dict(config.get("data", {})),
            "eval_manifest": str(PROJECT_ROOT / f"data/evaluation/tiers/{args.evaluation_tier.lower()}_{'quick' if args.evaluation_tier == 'E1' else 'standard' if args.evaluation_tier == 'E2' else 'full'}.jsonl"),
        }
    paths = PathConfig.from_mapping(
        config.get("paths", {}),
        project_root=PROJECT_ROOT,
        environ=_environment(),
        apply_environment_overrides=False,
    )
    return config, paths


def build_conditions(
    config: dict[str, Any],
    *,
    targets_override: list[str] | None = None,
    layers_override: list[int] | None = None,
    counts_override: list[int] | None = None,
    repeats_override: int | None = None,
    bit_planes_override: list[str] | None = None,
) -> list[dict[str, Any]]:
    fault = dict(config.get("fault", {}))
    targets = targets_override or list(
        fault.get(
            "sensitivity_targets",
            ["attention", "mlp", "vision_encoder", "embeddings", "lora_adapter"],
        )
    )
    layers = (
        layers_override if layers_override is not None else list(fault.get("layer_indices", []))
    )
    visual_layers = layers_override if layers_override is not None else list(fault.get("visual_layer_indices", layers))
    counts = counts_override or list(fault.get("bit_flip_counts", [1]))
    repeats = repeats_override if repeats_override is not None else int(fault.get("repeats", 1))
    planes = bit_planes_override or list(fault.get("bit_planes", ["all"]))
    if not targets or not counts or repeats < 1 or any(value < 1 for value in counts):
        raise ValueError("targets, bit counts and repeats must be positive")
    unknown = sorted(set(targets).difference(ALLOWED_TARGETS))
    if unknown:
        raise ValueError(f"unsupported fault target(s): {unknown}")
    conditions: list[dict[str, Any]] = []
    base_seed = int(dict(config.get("experiment", {})).get("seed", 2026))
    ordinal = 0
    for target in targets:
        target_layers = visual_layers if target == "visual_blocks" else layers
        selected_layers: list[int | None] = target_layers if target in LAYERED_TARGETS and target_layers else [None]
        for layer in selected_layers:
            for bit_plane in planes:
                for count in counts:
                    for repeat in range(repeats):
                        conditions.append(
                            {
                                "id": (
                                    f"{target}_layer_{layer if layer is not None else 'all'}_"
                                    f"{bit_plane}_bits_{count}_repeat_{repeat}"
                                ),
                                "target": target,
                                "layers": [] if layer is None else [layer],
                                "bit_plane": bit_plane,
                                "num_bits": count,
                                "repeat": repeat,
                                "seed": base_seed + ordinal * 1009,
                            }
                        )
                        ordinal += 1
    return conditions


def validate_condition_plan(conditions: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate condition identity and reproducibility metadata."""
    required = {"id", "target", "layers", "bit_plane", "num_bits", "repeat", "seed"}
    missing = [row.get("id", "<unknown>") for row in conditions if not required.issubset(row)]
    ids = [str(row.get("id")) for row in conditions]
    seeds = [row.get("seed") for row in conditions]
    duplicate_ids = sorted({item for item in ids if ids.count(item) > 1})
    duplicate_seeds = sorted({item for item in seeds if seeds.count(item) > 1})
    invalid_bits = [row.get("id") for row in conditions if not isinstance(row.get("num_bits"), int) or row["num_bits"] < 1]
    invalid_planes = [row.get("id") for row in conditions if row.get("bit_plane") not in {"all", "sign", "exponent", "mantissa"}]
    return {
        "valid": not (missing or duplicate_ids or duplicate_seeds or invalid_bits or invalid_planes),
        "num_conditions": len(conditions),
        "missing_fields": missing,
        "duplicate_ids": duplicate_ids,
        "duplicate_seeds": duplicate_seeds,
        "invalid_bit_counts": invalid_bits,
        "invalid_bit_planes": invalid_planes,
    }


def prioritize_coverage_first(conditions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run a small cross-target pilot before the exhaustive condition matrix."""
    if not conditions:
        return []
    min_bits = min(int(row["num_bits"]) for row in conditions)
    target_order = list(dict.fromkeys(str(row["target"]) for row in conditions))
    plane_order = list(dict.fromkeys(str(row["bit_plane"]) for row in conditions))
    representative: dict[str, set[tuple[int, ...]]] = {}
    for target in target_order:
        layer_sets = list(dict.fromkeys(
            tuple(int(value) for value in row.get("layers", []))
            for row in conditions if row["target"] == target
        ))
        representative[target] = (
            set(layer_sets)
            if len(layer_sets) <= 3
            else {layer_sets[0], layer_sets[len(layer_sets) // 2], layer_sets[-1]}
        )
    pilot_ids: set[str] = set()
    pilot: list[dict[str, Any]] = []
    for target in target_order:
        for layers in sorted(representative[target]):
            for plane in plane_order:
                match = next((
                    row for row in conditions
                    if row["target"] == target
                    and tuple(row.get("layers", [])) == layers
                    and row["bit_plane"] == plane
                    and int(row["num_bits"]) == min_bits
                    and int(row["repeat"]) == 0
                ), None)
                if match is not None:
                    pilot_ids.add(str(match["id"]))
                    pilot.append({**match, "phase": "pilot"})
    return pilot + [
        {**row, "phase": "full"}
        for row in conditions if str(row["id"]) not in pilot_ids
    ]


def _configured_paths(config: dict[str, Any]) -> dict[str, Path]:
    model, data = dict(config["model"]), dict(config["data"])
    paths = {
        "base_model": Path(str(model["base_model"])).expanduser(),
        "processor": Path(str(model.get("processor_id", model["base_model"]))).expanduser(),
        "eval_manifest": Path(str(data["eval_manifest"])).expanduser(),
        "dataset_root": Path(str(data["dataset_root"])).expanduser(),
    }
    if model.get("adapter_path"):
        paths["adapter"] = Path(str(model["adapter_path"])).expanduser()
    for name in ("eval_manifest",):
        if not paths[name].is_absolute():
            paths[name] = PROJECT_ROOT / paths[name]
    tiers_manifest = config.get("evaluation", {}).get("tiers_manifest")
    if tiers_manifest:
        manifest_path = Path(str(tiers_manifest)).expanduser()
        paths["tiers_manifest"] = manifest_path if manifest_path.is_absolute() else PROJECT_ROOT / manifest_path
    return paths


def evaluation_identity(config: dict[str, Any]) -> dict[str, Any]:
    """Bind a run to the exact evaluation file and contract it used."""
    data = dict(config.get("data", {}))
    evaluation = dict(config.get("evaluation", {}))
    eval_path = Path(str(data.get("eval_manifest", ""))).expanduser()
    if not eval_path.is_absolute():
        eval_path = PROJECT_ROOT / eval_path
    return {
        "tier": evaluation.get("tier"),
        "eval_file": str(eval_path),
        "sha256": file_sha256(eval_path) if eval_path.is_file() else None,
        "tiers_manifest": evaluation.get("tiers_manifest"),
        "contract": evaluation.get("contract", "configs/eval/evaluation_contract_v1.5.yaml"),
    }


def preflight_report(config: dict[str, Any]) -> dict[str, Any]:
    paths = _configured_paths(config)
    report: dict[str, Any] = {
        "success": all(
            path.is_dir()
            if name in {"base_model", "processor", "adapter", "dataset_root"}
            else path.is_file()
            for name, path in paths.items()
        ),
        "paths": {
            name: {
                "path": str(path),
                "exists": path.exists(),
                "kind": "directory" if path.is_dir() else "file" if path.is_file() else "missing",
            }
            for name, path in paths.items()
        },
    }
    evaluation = dict(config.get("evaluation", {}))
    tier = evaluation.get("tier")
    if tier:
        from sat_rs_vlm.evaluation.tiers import validate_tier_asset

        eval_path = paths["eval_manifest"]
        manifest_value = evaluation.get("tiers_manifest")
        manifest_path = Path(str(manifest_value)).expanduser() if manifest_value else PROJECT_ROOT / "data/evaluation/tiers/evaluation_tiers_manifest.json"
        if not manifest_path.is_absolute():
            manifest_path = PROJECT_ROOT / manifest_path
        try:
            report["tier"] = validate_tier_asset(
                tier=str(tier), eval_file=eval_path, manifest_path=manifest_path
            )
            report["success"] = bool(report["success"])
        except (FileNotFoundError, ValueError, OSError) as exc:
            report["success"] = False
            report["tier"] = {
                "tier": str(tier),
                "eval_file": str(eval_path),
                "tiers_manifest": str(manifest_path),
                "valid": False,
                "error": str(exc),
            }
    try:
        import torch

        report["cuda"] = {
            "available": bool(torch.cuda.is_available()),
            "count": int(torch.cuda.device_count()),
            "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except ImportError:
        report["cuda"] = {
            "available": False,
            "count": 0,
            "name": None,
            "error": "torch is not installed",
        }
    return report


def _write_eval_config(config: dict[str, Any], destination: Path) -> Path:
    model, data = dict(config["model"]), dict(config["data"])
    payload = {
        "model": {
            "base_model": model["base_model"],
            "processor_id": model.get("processor_id", model["base_model"]),
            "adapter_path": model.get("adapter_path"),
            "local_files_only": bool(model.get("local_files_only", True)),
            "trust_remote_code": bool(model.get("trust_remote_code", True)),
            "device_map": model.get("device_map", "auto"),
            "torch_dtype": model.get("torch_dtype", "float16"),
            "attn_implementation": model.get("attn_implementation", "sdpa"),
        },
        "data": {
            "eval_file": data["eval_manifest"],
            "image_root": data["dataset_root"],
            "max_eval_samples": data.get("max_eval_samples"),
            "max_seq_length": int(data.get("max_seq_length", 1024)),
            "eval_batch_size": int(data.get("eval_batch_size", 1)),
            "group_by_task": bool(data.get("group_by_task", True)),
            "log_every_samples": int(data.get("log_every_samples", 20)),
        },
        "generation": dict(config.get("generation", {})),
        "evaluation": {
            "contract": str(
                config.get("evaluation", {}).get(
                    "contract", "configs/eval/evaluation_contract_v1.5.yaml"
                )
            ),
            "strict": True,
            "semantic": bool(config.get("evaluation", {}).get("semantic", False)),
            "tier": config.get("evaluation", {}).get("tier"),
            "tiers_manifest": config.get("evaluation", {}).get("tiers_manifest"),
            "latency_semantics": "batch_amortized_per_sample",
        },
        "output": {},
    }
    write_resolved_config(payload, destination)
    return destination


def _run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"subprocess failed ({result.returncode}); see {log}")


def _condition_complete(directory: Path, condition: dict[str, Any] | None = None) -> bool:
    injection = directory / "fault_injection_summary.json"
    comparison = directory / "comparison" / "comparison.json"
    if not injection.is_file() or not comparison.is_file():
        return False
    try:
        fault = json.loads(injection.read_text(encoding="utf-8"))
        compare = json.loads(comparison.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(fault, dict) or not isinstance(compare, dict):
        return False
    if not fault.get("schema_version") or "actual_bit_flips" not in fault or "overall" not in compare:
        return False
    if condition is not None:
        if fault.get("condition_id") not in {None, condition.get("id")}:
            return False
        if fault.get("planned_bit_flips") not in {None, condition.get("num_bits")}:
            return False
        if fault.get("actual_bit_flips") != len(fault.get("records", [])):
            return False
        if fault.get("execution_status") not in {None, "completed", "completed_guarded"}:
            return False
    return True


def _load_condition_result(directory: Path, condition: dict[str, Any]) -> dict[str, Any]:
    return {
        **condition,
        "injection": json.loads(
            (directory / "fault_injection_summary.json").read_text(encoding="utf-8")
        ),
        "comparison": json.loads(
            (directory / "comparison" / "comparison.json").read_text(encoding="utf-8")
        ),
    }


def _write_progress(
    root: Path,
    *,
    status: str,
    completed: int,
    total: int,
    current: str | None = None,
    error: Exception | None = None,
) -> None:
    payload: dict[str, Any] = {
        "schema_version": "2.0",
        "status": status,
        "completed_conditions": completed,
        "total_conditions": total,
        "current_condition": current,
    }
    if error is not None:
        payload["error_type"] = type(error).__name__
        payload["error"] = str(error)
    write_json(root / "sensitivity_progress.json", payload)


def main() -> int:
    args = parse_args()
    config, paths = load_config(args)
    conditions = build_conditions(
        config,
        targets_override=args.targets,
        layers_override=args.layers,
        counts_override=args.bit_counts,
        repeats_override=args.repeats,
        bit_planes_override=args.bit_planes,
    )
    if str(config.get("fault", {}).get("execution_order", "coverage_first")) == "coverage_first":
        conditions = prioritize_coverage_first(conditions)
    plan_validation = validate_condition_plan(conditions)
    if not plan_validation["valid"]:
        raise ValueError(f"Invalid condition plan: {plan_validation}")
    report = preflight_report(config)
    if args.preflight:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    paths.create_output_directories()
    root = paths.output_root / "reliability" / "v15_sensitivity" / args.run_id
    plan = {
        "schema_version": "2.1",
        "evaluation_identity": evaluation_identity(config),
        "conditions": conditions,
    }
    if root.exists():
        if not args.resume:
            raise FileExistsError(
                f"run directory already exists: {root}; use --resume to continue it"
            )
        previous = json.loads((root / "condition_plan.json").read_text(encoding="utf-8"))
        if previous != plan:
            raise ValueError("resume plan differs from existing plan; choose a new run-id")
    else:
        root.mkdir(parents=True)
        write_resolved_config(config, root / "config_resolved.yaml")
        write_json(root / "condition_plan.json", plan)
    if args.dry_run:
        pilot_count = sum(condition.get("phase") == "pilot" for condition in conditions)
        print(
            json.dumps(
                {"dry_run": True, "run_dir": str(root), "num_conditions": len(conditions), "pilot_conditions": pilot_count},
                ensure_ascii=False,
            )
        )
        return 0
    if not report["success"]:
        raise SystemExit(
            "Preflight failed: configured model, adapter, data manifest, "
            "or dataset path is missing. Run with --preflight for details."
        )
    if not report["cuda"]["available"]:
        raise SystemExit(
            "CUDA is unavailable; real_inference is intentionally blocked. "
            "Run --preflight for details."
        )

    eval_config = _write_eval_config(config, root / "evaluation_config.yaml")
    baseline_dir = root / "baseline"
    try:
        if not (baseline_dir / "evaluation_v1_5" / "metrics.json").is_file():
            _run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts/reliability/run_fault_evaluation.py"),
                    "--config",
                    str(eval_config),
                    "--output-dir",
                    str(baseline_dir),
                ],
                root / "logs" / "baseline.log",
            )
        summary: list[dict[str, Any]] = []
        pilot_stopped = False
        for condition in conditions:
            if args.pilot_only and condition.get("phase") == "full":
                pilot_stopped = True
                break
            directory = root / "conditions" / condition["id"]
            if _condition_complete(directory, condition):
                summary.append(_load_condition_result(directory, condition))
                continue
            _write_progress(
                root,
                status="running",
                completed=len(summary),
                total=len(conditions),
                current=condition["id"],
            )
            fault_command = [
                sys.executable,
                str(PROJECT_ROOT / "scripts/reliability/run_fault_evaluation.py"),
                "--config",
                str(eval_config),
                "--output-dir",
                str(directory),
                "--fault-target",
                condition["target"],
                "--fault-num-bits",
                str(condition["num_bits"]),
                "--fault-seed",
                str(condition["seed"]),
                "--fault-bit-plane",
                condition["bit_plane"],
            ]
            if args.activation_guard:
                fault_command.extend(
                    [
                        "--activation-guard",
                        "--activation-guard-mode",
                        args.activation_guard_mode,
                        "--activation-patterns",
                        *args.activation_patterns,
                        "--activation-max-abs",
                        str(args.activation_max_abs),
                    ]
                )
            if condition["layers"]:
                fault_command.extend(
                    ["--fault-layers", *(str(value) for value in condition["layers"])]
                )
            _run(fault_command, root / "logs" / f"{condition['id']}.log")
            comparison_dir = directory / "comparison"
            _run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts/evaluation/compare_evaluations.py"),
                    "--baseline-dir",
                    str(baseline_dir / "evaluation_v1_5"),
                    "--candidate-dir",
                    str(directory / "evaluation_v1_5"),
                    "--output-dir",
                    str(comparison_dir),
                    "--protect-repository",
                ],
                root / "logs" / f"{condition['id']}_comparison.log",
            )
            summary.append(_load_condition_result(directory, condition))
            _write_progress(root, status="running", completed=len(summary), total=len(conditions))
    except Exception as exc:
        _write_progress(
            root,
            status="failed",
            completed=len(summary) if "summary" in locals() else 0,
            total=len(conditions),
            error=exc,
        )
        raise
    write_json(
        root / "sensitivity_report.json",
        {
            "schema_version": "2.0",
            "method": "in_memory_seu_parameter_fault",
            "baseline_evaluation_dir": str(baseline_dir / "evaluation_v1_5"),
            "conditions": summary,
            "aggregated": aggregate_sensitivity_conditions(summary),
            "interpretation": (
                "Use each condition's paired comparison for task-metric deltas "
                "and Bootstrap confidence intervals."
            ),
        },
    )
    final_status = "pilot_completed" if pilot_stopped else "completed"
    _write_progress(root, status=final_status, completed=len(summary), total=len(conditions))
    print(
        json.dumps(
            {"success": True, "status": final_status, "run_dir": str(root), "num_conditions": len(summary)},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
