"""在 CPU 上运行显式标记的可靠性 smoke_mock 流程。"""

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
        "--case",
        choices=(
            "tensor",
            "state-dict",
            "adapter-file",
            "output-guard",
            "recovery",
            "weight-clamp",
            "all",
        ),
        default="all",
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/reliability/local_smoke.yaml"
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.setdefault("PROJECT_ROOT", str(PROJECT_ROOT))
    environment.setdefault("DATA_ROOT", str(PROJECT_ROOT / "tests/fixtures/miniature_dataset"))
    environment.setdefault("MODEL_ROOT", str(PROJECT_ROOT / ".models"))
    environment.setdefault("OUTPUT_ROOT", str(PROJECT_ROOT / "outputs"))
    return environment


def _load_config(args: argparse.Namespace) -> tuple[dict[str, Any], PathConfig]:
    overrides = {
        "experiment.seed": args.seed,
        "paths.output_root": str(args.output_root) if args.output_root else None,
    }
    config = load_layered_config(
        LayeredConfigRequest(
            base_configs=(
                PROJECT_ROOT / "configs/base/default.yaml",
                PROJECT_ROOT / "configs/reliability/base.yaml",
                PROJECT_ROOT / "configs/reliability/protection_suite.yaml",
            ),
            environment_config=PROJECT_ROOT / "configs/local/paths.yaml",
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
        mode="full",
        smoke_case=args.case,
        run_id=args.run_id,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {"success": True, "execution_mode": "smoke_mock", "run_dir": str(layout.root)}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
