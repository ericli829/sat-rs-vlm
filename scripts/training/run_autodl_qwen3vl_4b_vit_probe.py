"""运行 Qwen3-VL-4B Round1 adapter + ViT last-2 的受控 E1 probe。

该入口是一个薄编排器：模型训练仍由 ``scripts/train_qwen3vl_lora.py`` 完成，
评测仍由 ``scripts/evaluate_rs_vlm.py`` 完成。本脚本只负责固定数据、阶段顺序、
中间 checkpoint 可评测化和最终 paired comparison。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sat_rs_vlm.training.config import load_training_config  # noqa: E402
from sat_rs_vlm.training.vit_probe import (  # noqa: E402
    build_probe_dataset,
    make_checkpoint_evaluable,
)


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--initial-adapter", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--source-train-file", action="append", default=None)
    parser.add_argument("--eval-config", type=Path, default=Path("configs/eval/qwen3vl_4b_baseline_e1_v2.yaml"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/experiments/qwen3vl_4b_vit_probe_last2"))
    parser.add_argument("--skip-baseline-eval", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def _run(command: list[str]) -> None:
    print("Running: " + " ".join(f'"{item}"' if " " in item else item for item in command))
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)


def _build_probe(config_path: Path, args: argparse.Namespace) -> Path:
    config = load_training_config(config_path)
    probe = config.vit_probe
    source_files = args.source_train_file or probe.source_train_files
    output_dir = _path(probe.output_dir)
    manifest = build_probe_dataset(
        source_files,
        output_dir=output_dir,
        protected_evaluation_manifest=_path(probe.protected_evaluation_manifest),
        target_samples=probe.target_samples,
        source_targets=probe.source_targets,
        task_targets=probe.task_targets,
        seed=probe.seed,
    )
    if manifest["protected_eval_overlap_count"] != 0:
        raise ValueError("Probe dataset has protected evaluation overlap")
    print(
        "Probe dataset ready: "
        f"samples={manifest['total_samples']}, sha256={manifest['output_sha256']}"
    )
    return output_dir / "train.jsonl"


def _training_command(
    config_path: Path,
    args: argparse.Namespace,
    train_file: Path,
    output_dir: Path,
    *,
    mode: str,
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/train_qwen3vl_lora.py"),
        "--config",
        str(config_path),
        "--initial-adapter",
        str(args.initial_adapter.resolve()),
        "--train-file",
        str(train_file.resolve()),
        "--output-dir",
        str(output_dir.resolve()),
    ]
    if mode == "dry-run":
        command.append("--dry-run")
    elif mode == "forward-only":
        command.append("--forward-only")
    else:
        steps = args.max_steps
        if steps is not None:
            command.extend(["--max-steps", str(steps)])
        command.extend(["--save-steps", str(min(100, steps or 200))])
    return command


def _promote_checkpoints(output_dir: Path) -> dict[int, Path]:
    promoted: dict[int, Path] = {}
    for step in (100, 200):
        checkpoint = output_dir / f"checkpoint-{step}"
        if not checkpoint.is_dir():
            continue
        make_checkpoint_evaluable(output_dir, checkpoint, checkpoint_step=step)
        promoted[step] = checkpoint
    if 100 not in promoted or 200 not in promoted:
        raise FileNotFoundError(
            f"Expected checkpoint-100 and checkpoint-200 under {output_dir}; "
            f"found={sorted(promoted)}"
        )
    return promoted


def _mark_probe_manifest(output_dir: Path) -> None:
    """把本次新产出的 manifest 标记为 probe，避免误认为 H1 正式实验。"""

    path = output_dir / "strategy_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Training did not produce strategy_manifest.json: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["training_stage"] = "qwen3vl_4b_vit_probe_last2"
    payload["experiment_type"] = "controlled_ablation"
    payload["probe_constraints"] = {
        "initial_adapter_only": True,
        "unfreeze_last_n_blocks": 2,
        "train_main_merger": False,
        "train_deepstack_mergers": False,
        "train_patch_embed": False,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _evaluate(checkpoint: Path, eval_config: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/evaluate_rs_vlm.py"),
            "--config",
            str(eval_config),
            "--checkpoint",
            str(checkpoint.resolve()),
            "--output-dir",
            str(output_dir.resolve()),
        ]
    )


def main() -> int:
    args = parse_args()
    if args.dry_run and args.forward_only:
        raise ValueError("--dry-run and --forward-only are mutually exclusive")
    config_path = _path(args.config)
    config = load_training_config(config_path)
    if not config.vit_probe.enabled:
        raise ValueError("vit_probe.enabled must be true for this runner")
    requested_steps = args.max_steps or config.training.max_steps or 200
    if requested_steps > config.vit_probe.max_steps_limit:
        raise ValueError(
            f"Probe max_steps={requested_steps} exceeds configured limit "
            f"{config.vit_probe.max_steps_limit}"
        )
    if config.vision_tuning.unfreeze_last_n_blocks != 2:
        raise ValueError("This controlled ablation requires unfreeze_last_n_blocks=2")
    if config.vision_tuning.train_main_merger:
        raise ValueError("This controlled ablation requires train_main_merger=false")
    if config.vision_tuning.train_deepstack_mergers or config.vision_tuning.train_patch_embed:
        raise ValueError("This controlled ablation keeps deepstack mergers and patch_embed frozen")
    if not args.initial_adapter.is_dir():
        raise FileNotFoundError(f"Initial Round1 adapter does not exist: {args.initial_adapter}")

    train_file = _build_probe(config_path, args)
    if args.prepare_only:
        return 0

    output_dir = _path(args.output_dir or config.training.output_dir)
    if args.dry_run:
        _run(_training_command(config_path, args, train_file, output_dir, mode="dry-run"))
        return 0
    if args.forward_only:
        _run(_training_command(config_path, args, train_file, output_dir, mode="forward-only"))
        return 0

    _run(_training_command(config_path, args, train_file, output_dir, mode="train"))
    _mark_probe_manifest(output_dir)
    if args.skip_eval or requested_steps < 200:
        return 0

    checkpoints = _promote_checkpoints(output_dir)
    report_dir = _path(args.report_dir)
    eval_config = _path(args.eval_config)
    baseline_dir = report_dir / "baseline_e1"
    checkpoint100_dir = report_dir / "checkpoint100_e1"
    checkpoint200_dir = report_dir / "checkpoint200_e1"
    if not args.skip_baseline_eval:
        _evaluate(args.initial_adapter, eval_config, baseline_dir)
    elif not baseline_dir.is_dir():
        print("Baseline E1 evaluation skipped; paired comparison will also be skipped.")
        _evaluate(checkpoints[100], eval_config, checkpoint100_dir)
        _evaluate(checkpoints[200], eval_config, checkpoint200_dir)
        return 0
    _evaluate(checkpoints[100], eval_config, checkpoint100_dir)
    _evaluate(checkpoints[200], eval_config, checkpoint200_dir)
    _run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/evaluation/compare_vit_probe.py"),
            "--baseline-dir",
            str(baseline_dir),
            "--checkpoint100-dir",
            str(checkpoint100_dir),
            "--checkpoint200-dir",
            str(checkpoint200_dir),
            "--output-dir",
            str(report_dir),
        ]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
