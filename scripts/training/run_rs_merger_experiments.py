"""Run formal merger variants serially, one CUDA context per subprocess."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from sat_rs_vlm.models.reliability.checksum import file_sha256

DEFAULT_CONFIGS = (
    "configs/experiments/rs_count_merger_c2_cont_4090.yaml",
    "configs/experiments/rs_count_merger_c3_cont_4090.yaml",
    "configs/experiments/rs_count_merger_c4_wide_4090.yaml",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serial merger launcher; every variant runs in an independent subprocess."
    )
    parser.add_argument("--config", action="append", dest="configs", default=[])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-root")
    parser.add_argument("--launcher-log-dir", default="outputs/rs_merger_expert_launcher")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip only runs with a complete, provenance-validated final checkpoint.",
    )
    return parser.parse_args()


def build_experiment_command(
    *,
    python: str,
    config: str,
    output_root: str | None = None,
    max_train_samples: int | None = None,
    max_steps: int | None = None,
    dry_run: bool = False,
) -> list[str]:
    command = [
        python,
        "scripts/training/train_rs_merger_expert.py",
        "--config",
        config,
    ]
    if output_root:
        command.extend(["--output-root", output_root])
    if max_train_samples is not None:
        command.extend(["--max-train-samples", str(max_train_samples)])
    if max_steps is not None:
        command.extend(["--max-steps", str(max_steps)])
    if dry_run:
        command.append("--dry-run")
    return command


def run_serial_experiments(
    commands: list[list[str]],
    *,
    log_dir: str | Path,
    environ: dict[str, str] | None = None,
    skip_completed: bool = False,
) -> list[dict[str, Any]]:
    """Wait for each child to exit before creating the next CUDA context."""

    root = Path(log_dir)
    root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for index, command in enumerate(commands, 1):
        config_name = Path(command[command.index("--config") + 1]).stem
        config_path = Path(command[command.index("--config") + 1])
        output_root = None
        if "--output-root" in command:
            output_root = Path(command[command.index("--output-root") + 1])
        completed = _find_completed_run(
            config_path,
            output_root=output_root,
            environ=environ,
        )
        if skip_completed and completed is not None:
            results.append(
                {
                    "index": index,
                    "command": command,
                    "pid": None,
                    "return_code": 0,
                    "skipped": True,
                    "skip_reason": "complete checkpoint with provenance validation",
                    "completed_run": completed.as_posix(),
                    "cuda_context_boundary": "no child launched",
                }
            )
            (root / "launcher_results.json").write_text(
                json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            continue
        log_path = root / f"{index:02d}_{config_name}.log"
        started = time.time()
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=Path.cwd(),
                env=environ or os.environ.copy(),
            )
            return_code = process.wait()
        record = {
            "index": index,
            "command": command,
            "pid": process.pid,
            "return_code": return_code,
            "started_unix": started,
            "elapsed_seconds": time.time() - started,
            "log": log_path.as_posix(),
            "cuda_context_boundary": "child process exited before next launch",
        }
        results.append(record)
        (root / "launcher_results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)
    return results


def _find_completed_run(
    config_path: Path,
    *,
    output_root: Path | None,
    environ: dict[str, str] | None,
) -> Path | None:
    """Find a completed run by artifact/provenance, never by directory name alone."""

    environment = dict(os.environ, **(environ or {}))
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    experiment = str(payload.get("experiment", ""))
    configured_root = str(dict(payload.get("output", {})).get("root", ""))
    configured_root = os.path.expandvars(configured_root)
    if any(token in configured_root for token in ("${", "$")):
        for key, value in environment.items():
            configured_root = configured_root.replace(f"${{{key}}}", value)
            configured_root = configured_root.replace(f"${key}", value)
    base = output_root or Path(configured_root)
    if not base.is_absolute():
        base = Path.cwd() / base
    if not base.is_dir() or not experiment:
        return None
    model_cfg = dict(payload.get("model", {}))
    r1_checkpoint = Path(os.path.expandvars(str(model_cfg.get("r1_checkpoint", ""))))
    visual_sidecar = Path(os.path.expandvars(str(model_cfg.get("visual_sidecar", ""))))
    r1_manifest = r1_checkpoint / "strategy_manifest.json"
    for run in sorted(base.glob(f"{experiment}_*"), reverse=True):
        checkpoint = run / "checkpoint"
        manifest_path = checkpoint / "expert_manifest.json"
        weights = checkpoint / "expert_model.safetensors"
        resolved_config = checkpoint / "config_resolved.yaml"
        if not manifest_path.is_file() or not weights.is_file() or not resolved_config.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("expert_weights_sha256") != file_sha256(weights):
                continue
            if not r1_manifest.is_file() or not visual_sidecar.is_file():
                continue
            if manifest.get("source_r1_manifest_sha256") != file_sha256(r1_manifest):
                continue
            if manifest.get("source_visual_sidecar_sha256") != file_sha256(visual_sidecar):
                continue
            return checkpoint
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return None


def main() -> int:
    args = parse_args()
    configs = args.configs or list(DEFAULT_CONFIGS)
    commands = [
        build_experiment_command(
            python=args.python,
            config=config,
            output_root=args.output_root,
            max_train_samples=args.max_train_samples,
            max_steps=args.max_steps,
            dry_run=args.dry_run,
        )
        for config in configs
    ]
    results = run_serial_experiments(
        commands,
        log_dir=args.launcher_log_dir,
        skip_completed=args.skip_completed,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
