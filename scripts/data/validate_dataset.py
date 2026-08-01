"""校验 sat-rs-vlm 数据集 manifest、分片、图片和标注。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sat_rs_vlm.data.manifest import validate_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    """解析数据校验参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest-name", default="dataset_manifest.json")
    parser.add_argument("--sample-images", type=int, default=16)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--skip-image-decode", action="store_true")
    return parser.parse_args()


def main() -> int:
    """执行校验并以退出码表达成功或失败。"""

    args = parse_args()
    report = validate_dataset(
        args.dataset_root,
        manifest_name=args.manifest_name,
        sample_images=args.sample_images,
        verify_images=not args.skip_image_decode,
    )
    payload = report.model_dump(mode="json")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
