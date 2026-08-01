"""从 manifest 数据集中抽取可独立搬运的微型 smoke 子集。"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path, PurePosixPath
from typing import Any

from sat_rs_vlm.data.manifest import load_dataset_manifest, load_manifest_split
from sat_rs_vlm.utils.jsonl import write_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    """解析 smoke 子集参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=32)
    parser.add_argument("--split", default="smoke")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _images(row: dict[str, Any]) -> list[str]:
    value = row.get("images", [])
    return [str(item) for item in value] if isinstance(value, list) else [str(value)]


def main() -> int:
    """复制选中样本和所引用图片，并生成 embedded manifest。"""

    args = parse_args()
    source = args.source_root.resolve()
    output = args.output_root.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output directory is not empty: {output}. Pass --overwrite.")
    manifest_path = source / "dataset_manifest.json"
    if not manifest_path.is_file():
        manifest_path = source / "project_metadata/dataset_manifest.json"
    manifest = load_dataset_manifest(manifest_path)
    rows = load_manifest_split(
        source,
        manifest,
        args.split,
        max_samples=max(args.sample_count, 0),
    )
    output.mkdir(parents=True, exist_ok=True)
    for row in rows:
        for relative in _images(row):
            source_image = source / PurePosixPath(relative)
            destination = output / PurePosixPath(relative)
            if not source_image.is_file():
                raise FileNotFoundError(f"Referenced image does not exist: {source_image}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_image, destination)
    for split in ("train", "validation", "test", "smoke"):
        write_jsonl(output / f"{split}.jsonl", rows if split in {"train", "smoke"} else [])
    payload = manifest.model_dump(mode="json")
    payload.update(
        {
            "dataset_name": f"{manifest.dataset_name}-smoke",
            "root_format": "embedded",
            "splits": {name: f"{name}.jsonl" for name in ("train", "validation", "test", "smoke")},
            "statistics": "statistics.json",
        }
    )
    (output / "dataset_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "statistics.json").write_text(
        json.dumps({"sample_counts": {"train": len(rows), "smoke": len(rows)}}, indent=2),
        encoding="utf-8",
    )
    print(f"Created smoke dataset with {len(rows)} samples at {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
