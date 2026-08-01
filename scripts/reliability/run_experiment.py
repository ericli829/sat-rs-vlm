"""运行真实 Qwen3-VL/LoRA bit flip 可靠性实验。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from sat_rs_vlm.application.reliability_service import ReliabilityExperimentService
from sat_rs_vlm.configuration.layered import LayeredConfigRequest, load_layered_config
from sat_rs_vlm.configuration.paths import PathConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/reliability/experiments/lora_bitflip.yaml",
    )
    parser.add_argument(
        "--mode",
        choices=("baseline", "inject", "compare", "protect", "recover", "full"),
        default="full",
    )
    parser.add_argument("--environment", choices=("local", "autodl"), default="autodl")
    parser.add_argument("--env-config", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--eval-manifest", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _environment_config(args: argparse.Namespace) -> Path:
    if args.env_config:
        return args.env_config.resolve()
    return PROJECT_ROOT / (
        "configs/cloud/autodl.yaml" if args.environment == "autodl" else "configs/local/paths.yaml"
    )


def _environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.setdefault("PROJECT_ROOT", str(PROJECT_ROOT))
    return environment


def _load_config(args: argparse.Namespace) -> tuple[dict[str, Any], PathConfig]:
    overrides = {
        "paths.output_root": str(args.output_root) if args.output_root else None,
        "model.adapter_path": str(args.adapter_path) if args.adapter_path else None,
        "data.eval_manifest": str(args.eval_manifest) if args.eval_manifest else None,
        "data.dataset_root": str(args.dataset_root) if args.dataset_root else None,
        "experiment.seed": args.seed,
    }
    config = load_layered_config(
        LayeredConfigRequest(
            base_configs=(
                PROJECT_ROOT / "configs/base/default.yaml",
                PROJECT_ROOT / "configs/reliability/base.yaml",
                PROJECT_ROOT / "configs/reliability/protection_suite.yaml",
            ),
            environment_config=_environment_config(args),
            experiment_config=args.config.resolve(),
            cli_overrides=overrides,
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


def main() -> int:
    args = parse_args()
    config, paths = _load_config(args)
    service = ReliabilityExperimentService(
        config,
        project_root=PROJECT_ROOT,
        output_root=paths.output_root,
        command=subprocess.list2cmdline(sys.argv),
    )
    layout = service.run(
        mode=args.mode,
        run_id=args.run_id,
        overwrite=args.overwrite,
        resume=args.resume,
    )
    print(
        json.dumps(
            {"success": True, "execution_mode": "real_inference", "run_dir": str(layout.root)},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
