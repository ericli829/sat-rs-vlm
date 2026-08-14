"""为保持原始目录不变的遥感数据集生成 project_metadata 清单。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sat_rs_vlm.utils.jsonl import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    """解析清单生成参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--test-file", type=Path)
    parser.add_argument("--dataset-name", default="VRSBench")
    parser.add_argument("--dataset-version", default="v1")
    parser.add_argument("--coordinate-max", type=float, default=1.0)
    parser.add_argument("--smoke-samples", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _copy_jsonl(
    source: Path | None,
    destination: Path,
    *,
    require_non_empty: bool = False,
) -> list[dict[str, Any]]:
    if source is not None and not source.is_file():
        raise FileNotFoundError(f"Source JSONL does not exist: {source.resolve()}")
    rows = list(read_jsonl(source)) if source is not None else []
    if require_non_empty and not rows:
        raise ValueError(f"Required source JSONL is empty: {source}")
    write_jsonl(destination, rows)
    return rows


def main() -> int:
    """创建独立元数据目录，不修改 VRSBench 原始图片和标注。"""

    args = parse_args()
    root = args.dataset_root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Dataset root does not exist: {root}")
    metadata = root / "project_metadata"
    if metadata.exists() and not args.overwrite:
        raise SystemExit(
            f"Metadata directory already exists: {metadata}. Pass --overwrite to replace files."
        )
    metadata.mkdir(parents=True, exist_ok=True)
    train_rows = _copy_jsonl(
        args.train_file,
        metadata / "train.jsonl",
        require_non_empty=True,
    )
    validation_rows = _copy_jsonl(
        args.validation_file,
        metadata / "validation.jsonl",
        require_non_empty=True,
    )
    test_rows = _copy_jsonl(args.test_file, metadata / "test.jsonl")
    smoke_rows = (train_rows + validation_rows)[: max(args.smoke_samples, 0)]
    write_jsonl(metadata / "smoke.jsonl", smoke_rows)

    manifest = {
        "schema_version": "1.0",
        "dataset_name": args.dataset_name,
        "dataset_version": args.dataset_version,
        "root_format": "external",
        "image_path_type": "relative",
        "coordinate_format": "xyxy",
        "coordinate_range": [0, args.coordinate_max],
        "splits": {
            "train": "project_metadata/train.jsonl",
            "validation": "project_metadata/validation.jsonl",
            "test": "project_metadata/test.jsonl",
            "smoke": "project_metadata/smoke.jsonl",
        },
        "statistics": "project_metadata/statistics.json",
        "classes": [],
    }
    (metadata / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    statistics = {
        "sample_counts": {
            "train": len(train_rows),
            "validation": len(validation_rows),
            "test": len(test_rows),
            "smoke": len(smoke_rows),
        }
    }
    (metadata / "statistics.json").write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    splits = metadata / "splits"
    splits.mkdir(exist_ok=True)
    for name, rows in (
        ("train", train_rows),
        ("validation", validation_rows),
        ("test", test_rows),
    ):
        (splits / f"{name}.txt").write_text(
            "\n".join(str(row.get("id", "")) for row in rows) + ("\n" if rows else ""),
            encoding="utf-8",
        )
    print(f"Created dataset manifest: {metadata / 'dataset_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
