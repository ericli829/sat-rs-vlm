"""训练资产检查脚本。

在真实训练前检查本地模型目录、processor 目录、JSONL 数据和图片路径是否可用，并
输出 reports/training_asset_check.json。默认会检查模型训练依赖是否已安装。
"""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from sat_rs_vlm.data.qwen3vl_dataset import sample_to_messages
    from sat_rs_vlm.training.config import (
        TrainingPathOverrides,
        apply_training_overrides,
        load_training_config,
        resolve_training_paths,
    )
    from sat_rs_vlm.utils.jsonl import read_jsonl
except ModuleNotFoundError as exc:
    missing = exc.name or "unknown"
    raise SystemExit(
        f"Missing required project dependency: {missing}\n"
        'Run: pip install -e ".[dev]"\n'
        'For model training dependencies, also run: pip install -e ".[model]"\n'
        "Make sure you run the command with the same Python interpreter used for this script."
    ) from exc

MODEL_DEPENDENCIES = ("torch", "transformers", "peft", "PIL", "qwen_vl_utils")
MODEL_DEPS_HINT = 'pip install -e ".[model]"'


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Validate local Qwen3-VL training assets.")
    parser.add_argument("--config", required=True, help="Training YAML config path.")
    parser.add_argument("--model-dir", default=None, help="Override local model directory.")
    parser.add_argument("--processor-dir", default=None, help="Override local processor directory.")
    parser.add_argument("--train-file", default=None, help="Override train JSONL path.")
    parser.add_argument("--val-file", default=None, help="Override val JSONL path.")
    parser.add_argument("--image-root", default=None, help="Override image root.")
    parser.add_argument(
        "--report-file",
        default="reports/training_asset_check.json",
        help="Output JSON report path.",
    )
    return parser.parse_args()


def build_overrides(args: argparse.Namespace) -> TrainingPathOverrides:
    """从 argparse Namespace 构造覆盖项。"""

    return TrainingPathOverrides(
        model_dir=args.model_dir,
        processor_dir=args.processor_dir,
        train_file=args.train_file,
        val_file=args.val_file,
        image_root=args.image_root,
    )


def require_path(path: Path, label: str, errors: list[str]) -> None:
    """检查路径存在。"""

    if not path.exists():
        errors.append(f"{label} does not exist: {path}")


def check_model_dir(path: Path, label: str, errors: list[str]) -> None:
    """检查模型或 processor 目录关键文件。"""

    require_path(path, label, errors)
    if not path.exists():
        return
    required_any = [
        path / "config.json",
        path / "tokenizer_config.json",
    ]
    for file_path in required_any:
        if not file_path.exists():
            errors.append(f"{label} missing required file: {file_path.name}")
    has_preprocessor = (path / "preprocessor_config.json").exists()
    has_processor = (path / "processor_config.json").exists()
    if not (has_preprocessor or has_processor):
        errors.append(f"{label} missing preprocessor_config.json or processor_config.json")


def resolve_image_path(image_path: str, image_root: Path) -> Path:
    """按绝对路径/相对 image_root 规则解析图片。"""

    path = Path(image_path).expanduser()
    return path if path.is_absolute() else image_root / path


def extract_image_paths(row: dict[str, Any]) -> list[str]:
    """从 messages 或内部格式中提取图片路径。"""

    images: list[str] = []
    for message in sample_to_messages(row):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if item.get("type") == "image":
                images.append(str(item.get("image", "")))
    return images


def check_jsonl(path: Path, image_root: Path, label: str, errors: list[str]) -> dict[str, Any]:
    """检查 JSONL 文件、样本结构和图片路径。"""

    info: dict[str, Any] = {"path": str(path), "num_checked": 0, "image_found": False}
    require_path(path, label, errors)
    if not path.exists():
        return info
    try:
        rows = list(read_jsonl(path))
    except Exception as exc:
        errors.append(f"{label} is not a valid JSONL file: {exc}")
        return info
    if not rows:
        errors.append(f"{label} is empty: {path}")
        return info
    for row in rows[:5]:
        info["num_checked"] += 1
        try:
            images = extract_image_paths(row)
        except Exception as exc:
            errors.append(f"{label} sample {row.get('id', '<unknown>')} is invalid: {exc}")
            continue
        if not images:
            errors.append(f"{label} sample {row.get('id', '<unknown>')} has no image")
        for image in images:
            resolved = resolve_image_path(image, image_root)
            if resolved.exists():
                info["image_found"] = True
            else:
                errors.append(
                    f"{label} sample {row.get('id', '<unknown>')} image missing: "
                    f"{image} (resolved: {resolved})"
                )
    if not info["image_found"]:
        errors.append(f"{label} did not contain any existing image in checked samples.")
    return info


def check_dependencies(errors: list[str]) -> dict[str, Any]:
    """检查训练依赖和 CUDA 状态。"""

    deps: dict[str, Any] = {}
    for name in MODEL_DEPENDENCIES:
        try:
            importlib.import_module(name)
            deps[name] = {"available": True}
        except (ImportError, OSError) as exc:
            deps[name] = {
                "available": False,
                "hint": MODEL_DEPS_HINT,
                "error": str(exc),
            }
            errors.append(
                f"Model dependency cannot be imported: {name}: {exc}. Run: {MODEL_DEPS_HINT}"
            )
    torch_module = deps.get("torch")
    if torch_module and torch_module.get("available"):
        torch = importlib.import_module("torch")
        cuda_available = bool(torch.cuda.is_available())
        deps["torch"]["version"] = getattr(torch, "__version__", "unknown")
        deps["torch"]["cuda_available"] = cuda_available
        if cuda_available:
            device_index = int(torch.cuda.current_device())
            props = torch.cuda.get_device_properties(device_index)
            deps["torch"]["device_name"] = torch.cuda.get_device_name(device_index)
            deps["torch"]["total_memory_mb"] = float(props.total_memory / (1024 * 1024))
        else:
            deps["torch"]["warning"] = (
                "CUDA is not available. Asset checks can run on CPU, but real training may be slow."
            )
    return deps


def write_report(report: dict[str, Any], report_file: Path) -> None:
    """写入 JSON 报告。"""

    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_training_assets(
    config_path: str | Path,
    overrides: TrainingPathOverrides | None = None,
    *,
    check_model_dependencies: bool = True,
    report_file: str | Path = "reports/training_asset_check.json",
) -> dict[str, Any]:
    """执行训练资产检查并返回报告。"""

    errors: list[str] = []
    config = load_training_config(config_path, allow_unresolved_env=True)
    if overrides is not None:
        config = apply_training_overrides(config, overrides)
    paths = resolve_training_paths(config)

    if paths.model_dir is None:
        errors.append("model_dir must point to a local directory for local training checks.")
    else:
        check_model_dir(paths.model_dir, "model_dir", errors)
    if paths.processor_dir is None:
        errors.append("processor_dir must point to a local directory for local training checks.")
    else:
        check_model_dir(paths.processor_dir, "processor_dir", errors)

    train_info = check_jsonl(paths.train_file, paths.image_root, "train_file", errors)
    val_info = check_jsonl(paths.val_file, paths.image_root, "val_file", errors)
    deps = check_dependencies(errors) if check_model_dependencies else {}

    report: dict[str, Any] = {
        "success": not errors,
        "errors": errors,
        "model_dir": str(paths.model_dir) if paths.model_dir else None,
        "processor_dir": str(paths.processor_dir) if paths.processor_dir else None,
        "train_file": train_info,
        "val_file": val_info,
        "image_root": str(paths.image_root),
        "dependencies": deps,
    }
    write_report(report, Path(report_file))
    return report


def main() -> int:
    """脚本入口。"""

    args = parse_args()
    try:
        report = validate_training_assets(
            args.config,
            build_overrides(args),
            report_file=args.report_file,
        )
    except Exception as exc:
        print(
            f"Training asset validation failed before report generation: {exc}\n"
            "Set LOCAL_MODEL_DIR/DATA_ROOT/TRAIN_JSONL/VAL_JSONL or pass "
            "--model-dir/--train-file/--val-file/--image-root."
        )
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
