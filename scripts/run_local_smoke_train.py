"""本地 Qwen3-VL 最小训练测试封装脚本。"""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Run local Qwen3-VL smoke training checks.")
    parser.add_argument("--config", default="configs/train/qwen3vl_local_smoke.yaml")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--val-file", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", default="checkpoints/smoke/qwen3vl-local-smoke")
    parser.add_argument("--skip-forward-only", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--use-qlora", action="store_true")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--cpu-ok", action="store_true")
    return parser.parse_args()


def common_overrides(args: argparse.Namespace) -> list[str]:
    """构造通用 CLI 覆盖参数。"""

    return [
        "--config",
        args.config,
        "--model-dir",
        args.model_dir,
        "--train-file",
        args.train_file,
        "--val-file",
        args.val_file,
        "--image-root",
        args.image_root,
    ]


def run_step(name: str, command: list[str]) -> dict[str, object]:
    """执行一个 smoke step。"""

    print(f"\n== {name} ==")
    print("+ " + " ".join(command))
    started = time.perf_counter()
    result = subprocess.run(command, check=False)
    duration = time.perf_counter() - started
    return {"name": name, "returncode": result.returncode, "duration_seconds": duration}


def write_summary(summary: dict[str, object]) -> None:
    """写入 smoke summary。"""

    report_file = Path("reports/local_smoke_train_summary.json")
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Summary written to {report_file}")


def main() -> int:
    """脚本入口。"""

    args = parse_args()
    method = "qlora" if args.use_qlora else "lora"
    steps: list[tuple[str, list[str]]] = [
        (
            "validate assets",
            [
                sys.executable,
                "scripts/validate_training_assets.py",
                *common_overrides(args),
            ],
        ),
        (
            "dry run",
            [
                sys.executable,
                "scripts/train_qwen3vl_lora.py",
                *common_overrides(args),
                "--output-dir",
                args.output_dir,
                "--max-seq-length",
                str(args.max_seq_length),
                "--method",
                method,
                "--dry-run",
            ],
        ),
    ]
    if not args.skip_forward_only:
        steps.append(
            (
                "forward only",
                [
                    sys.executable,
                    "scripts/train_qwen3vl_lora.py",
                    *common_overrides(args),
                    "--output-dir",
                    args.output_dir,
                    "--max-seq-length",
                    str(args.max_seq_length),
                    "--method",
                    method,
                    "--forward-only",
                ],
            )
        )
    if not args.skip_train:
        steps.append(
            (
                "max_steps=2 train",
                [
                    sys.executable,
                    "scripts/train_qwen3vl_lora.py",
                    *common_overrides(args),
                    "--output-dir",
                    args.output_dir,
                    "--max-train-samples",
                    "4",
                    "--max-eval-samples",
                    "2",
                    "--max-steps",
                    "2",
                    "--max-seq-length",
                    str(args.max_seq_length),
                    "--method",
                    method,
                ],
            )
        )

    results: list[dict[str, object]] = []
    for name, command in steps:
        result = run_step(name, command)
        results.append(result)
        if result["returncode"] != 0:
            summary = {"success": False, "failed_step": name, "steps": results}
            write_summary(summary)
            print(f"Smoke train stopped at step: {name}")
            return int(result["returncode"])

    summary = {"success": True, "cpu_ok": args.cpu_ok, "steps": results}
    write_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
