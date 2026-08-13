"""Run SEU sensitivity scans using Evaluation v1.5 and paired comparisons."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from sat_rs_vlm.configuration.layered import (
    LayeredConfigRequest,
    load_layered_config,
    write_resolved_config,
)
from sat_rs_vlm.configuration.paths import PathConfig
from sat_rs_vlm.training.experiment import write_json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAYERED_TARGETS = {"language_model", "attention", "mlp"}
ALLOWED_TARGETS = {
    "all_parameters",
    "lora_adapter",
    "lora_a",
    "lora_b",
    "vision_encoder",
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
    parser.add_argument("--bit-planes", nargs="+", choices=("all", "sign", "exponent", "mantissa"))
    parser.add_argument("--dry-run", action="store_true")
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
    paths = PathConfig.from_mapping(
        config.get("paths", {}),
        project_root=PROJECT_ROOT,
        environ=_environment(),
        apply_environment_overrides=False,
    )
    paths.create_output_directories()
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
    targets = targets_override or list(fault.get("sensitivity_targets", ["lora_adapter"]))
    layers = (
        layers_override if layers_override is not None else list(fault.get("layer_indices", []))
    )
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
        selected_layers: list[int | None] = (
            layers if target in LAYERED_TARGETS and layers else [None]
        )
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
    root = paths.output_root / "reliability" / "v15_sensitivity" / args.run_id
    if root.exists():
        raise FileExistsError(f"run directory already exists: {root}")
    root.mkdir(parents=True)
    write_resolved_config(config, root / "config_resolved.yaml")
    write_json(root / "condition_plan.json", {"schema_version": "2.0", "conditions": conditions})
    if args.dry_run:
        print(
            json.dumps(
                {"dry_run": True, "run_dir": str(root), "num_conditions": len(conditions)},
                ensure_ascii=False,
            )
        )
        return 0
    eval_config = _write_eval_config(config, root / "evaluation_config.yaml")
    baseline_dir = root / "baseline"
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
    for condition in conditions:
        directory = root / "conditions" / condition["id"]
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
        if condition["layers"]:
            fault_command.extend(["--fault-layers", *(str(value) for value in condition["layers"])])
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
        injection = json.loads(
            (directory / "fault_injection_summary.json").read_text(encoding="utf-8")
        )
        comparison = json.loads((comparison_dir / "comparison.json").read_text(encoding="utf-8"))
        summary.append({**condition, "injection": injection, "comparison": comparison})
        write_json(
            root / "sensitivity_progress.json",
            {
                "status": "running",
                "completed_conditions": len(summary),
                "total_conditions": len(conditions),
                "current_condition": condition["id"],
            },
        )
    write_json(
        root / "sensitivity_report.json",
        {
            "schema_version": "2.0",
            "method": "in_memory_seu_parameter_fault",
            "baseline_evaluation_dir": str(baseline_dir / "evaluation_v1_5"),
            "conditions": summary,
            "interpretation": (
                "Use each condition's paired comparison for task-metric deltas "
                "and Bootstrap confidence intervals."
            ),
        },
    )
    write_json(
        root / "sensitivity_progress.json",
        {
            "status": "completed",
            "completed_conditions": len(summary),
            "total_conditions": len(conditions),
        },
    )
    print(
        json.dumps(
            {"success": True, "run_dir": str(root), "num_conditions": len(summary)},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
