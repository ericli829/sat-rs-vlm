"""按 DatasetManifest 分片构建固定、均衡的可靠性评测 JSONL。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from sat_rs_vlm.configuration.layered import LayeredConfigRequest, load_layered_config
from sat_rs_vlm.configuration.paths import PathConfig
from sat_rs_vlm.data.reliability_manifest import (
    build_multisource_reliability_eval_manifest,
    build_reliability_eval_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--environment", choices=("local", "autodl"), default="local")
    parser.add_argument("--env-config", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-split")
    parser.add_argument("--samples-per-task", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _environment_config(args: argparse.Namespace) -> Path:
    if args.env_config:
        return args.env_config.resolve()
    return PROJECT_ROOT / (
        "configs/cloud/autodl.yaml" if args.environment == "autodl" else "configs/local/paths.yaml"
    )


def _load(args: argparse.Namespace) -> tuple[dict[str, Any], PathConfig]:
    environment = dict(os.environ)
    environment.setdefault("PROJECT_ROOT", str(PROJECT_ROOT))
    config = load_layered_config(
        LayeredConfigRequest(
            base_configs=(PROJECT_ROOT / "configs/base/default.yaml",),
            environment_config=_environment_config(args),
            experiment_config=args.config.resolve(),
            cli_overrides={
                "paths.dataset_root": str(args.dataset_root) if args.dataset_root else None,
                "data.dataset_manifest": str(args.manifest) if args.manifest else None,
                "data.reliability_eval_manifest": str(args.output) if args.output else None,
                "data.eval_split": args.source_split,
                "data.samples_per_task": args.samples_per_task,
                "experiment.seed": args.seed,
            },
            project_root=PROJECT_ROOT,
        ),
        environ=environment,
    )
    paths = PathConfig.from_mapping(
        config.get("paths", {}),
        project_root=PROJECT_ROOT,
        environ=environment,
        apply_environment_overrides=False,
    )
    return config, paths


def _output_path(data: dict[str, Any], dataset_root: Path, *, multisource: bool) -> str:
    configured = data.get("reliability_eval_manifest")
    if configured:
        return str(configured)
    if multisource:
        return str(
            dataset_root / "project_metadata" / "reliability" / "vrsbench_levircc_eval.jsonl"
        )
    legacy = data.get("eval_manifest")
    if legacy:
        return str(legacy)
    raise ValueError(
        "Reliability output path is not configured; set "
        "data.reliability_eval_manifest or data.eval_manifest"
    )


def main() -> int:
    args = parse_args()
    config, paths = _load(args)
    data = dict(config.get("data", {}))
    experiment = dict(config.get("experiment", {}))
    dataset_root = args.dataset_root or paths.dataset_root
    if dataset_root is None:
        raise ValueError("Dataset root is not configured")
    raw_sources = data.get("reliability_sources")
    output_path = _output_path(data, Path(dataset_root), multisource=bool(raw_sources))
    if raw_sources:
        sources = [dict(source) for source in list(raw_sources)]
        if args.manifest:
            sources[0]["dataset_manifest"] = str(args.manifest.resolve())
        if args.source_split:
            for source in sources:
                source["source_split"] = args.source_split
        if args.samples_per_task is not None:
            for source in sources:
                configured_tasks = source.get("task_samples")
                if isinstance(configured_tasks, dict):
                    source["task_samples"] = {
                        str(task): args.samples_per_task for task in configured_tasks
                    }
                else:
                    source["samples_per_task"] = args.samples_per_task
        statistics = build_multisource_reliability_eval_manifest(
            dataset_root,
            sources,
            output_path=output_path,
            samples_per_task=int(data.get("samples_per_task", 20)),
            seed=int(experiment.get("seed", 2026)),
            overwrite=args.overwrite,
        )
    else:
        statistics = build_reliability_eval_manifest(
            dataset_root,
            data["dataset_manifest"],
            source_split=str(data.get("eval_split", "validation")),
            output_path=output_path,
            samples_per_task=int(data.get("samples_per_task", 20)),
            seed=int(experiment.get("seed", 2026)),
            overwrite=args.overwrite,
        )
    print(json.dumps(statistics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
