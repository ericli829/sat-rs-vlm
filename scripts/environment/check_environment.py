"""检查基础依赖、可选模型栈、GPU、路径和磁盘，不加载模型。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

BASE_MODULES = ("pydantic", "yaml", "packaging", "typer", "fastapi", "uvicorn", "PIL")
MODEL_MODULES = (
    "torch",
    "torchvision",
    "transformers",
    "peft",
    "accelerate",
    "safetensors",
    "qwen_vl_utils",
    "scipy",
)
PATH_NAMES = (
    "PROJECT_ROOT",
    "DATA_ROOT",
    "MODEL_ROOT",
    "OUTPUT_ROOT",
    "CACHE_ROOT",
    "TMPDIR",
    "TENSORBOARD_ROOT",
    "BACKUP_ROOT",
    "HF_HOME",
    "HF_HUB_CACHE",
    "TORCH_HOME",
    "PIP_CACHE_DIR",
)


def parse_args() -> argparse.Namespace:
    """解析环境检查参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-model", action="store_true")
    parser.add_argument("--require-bitsandbytes", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def _available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _gpu_report() -> dict[str, Any]:
    if not _available("torch"):
        return {"available": False, "reason": "torch is not installed"}
    import torch

    available = bool(torch.cuda.is_available())
    report: dict[str, Any] = {"available": available}
    if available:
        report.update(
            {
                "name": torch.cuda.get_device_name(0),
                "count": torch.cuda.device_count(),
                "bf16_supported": bool(torch.cuda.is_bf16_supported()),
                "torch_cuda": torch.version.cuda,
            }
        )
    return report


def _path_report(
    dataset_root: Path | None,
    model_root: Path | None,
    output_root: Path | None,
) -> dict[str, Any]:
    dataset = dataset_root or (Path(os.environ["DATA_ROOT"]) if os.getenv("DATA_ROOT") else None)
    model = model_root or (Path(os.environ["MODEL_ROOT"]) if os.getenv("MODEL_ROOT") else None)
    output = output_root or (
        Path(os.environ["OUTPUT_ROOT"]) if os.getenv("OUTPUT_ROOT") else Path.cwd() / "outputs"
    )
    output.mkdir(parents=True, exist_ok=True)
    writable = False
    try:
        with tempfile.NamedTemporaryFile(dir=output, prefix=".write-check-", delete=True):
            writable = True
    except OSError:
        writable = False
    usage = shutil.disk_usage(output)
    return {
        "dataset_root": str(dataset) if dataset else None,
        "dataset_exists": bool(dataset and dataset.is_dir()),
        "model_root": str(model) if model else None,
        "model_exists": bool(model and model.is_dir()),
        "output_root": str(output.resolve()),
        "output_writable": writable,
        "output_free_gib": round(usage.free / 1024**3, 2),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    """构造完整检查报告。"""

    base = {name: _available(name) for name in BASE_MODULES}
    model = {name: _available(name) for name in MODEL_MODULES}
    optional = {"bitsandbytes": _available("bitsandbytes")}
    return {
        "ok": all(base.values()),
        "os": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "virtual_environment": os.environ.get("VIRTUAL_ENV"),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "base_dependencies": base,
        "model_dependencies": model,
        "optional_dependencies": optional,
        "gpu": _gpu_report(),
        "paths": _path_report(args.dataset_root, args.model_root, args.output_root),
        "path_environment": {name: os.environ.get(name) for name in PATH_NAMES},
    }


def main() -> int:
    """打印报告，并仅对明确要求的能力使用非零退出码。"""

    args = parse_args()
    report = build_report(args)
    failures: list[str] = []
    if not report["ok"]:
        failures.append("base dependencies are incomplete")
    if args.require_model and not all(report["model_dependencies"].values()):
        failures.append("model dependencies are incomplete")
    if args.require_bitsandbytes and not report["optional_dependencies"]["bitsandbytes"]:
        failures.append("bitsandbytes is unavailable")
    if args.require_gpu and not report["gpu"]["available"]:
        failures.append("CUDA GPU is unavailable")
    report["failures"] = failures
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
