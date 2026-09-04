#!/usr/bin/env python3
"""Convert VRSBench referring annotations into Region Retriever JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def convert(
    annotation_dir: Path, image_dir: Path, output: Path, limit: int | None = None
) -> dict[str, int]:
    output.parent.mkdir(parents=True, exist_ok=True)
    stats = {"annotations": 0, "rows": 0, "missing_images": 0, "invalid_objects": 0}
    with output.open("w", encoding="utf-8") as handle:
        for annotation_path in sorted(annotation_dir.glob("*.json")):
            if limit is not None and stats["rows"] >= limit:
                break
            stats["annotations"] += 1
            try:
                payload = json.loads(annotation_path.read_text(encoding="utf-8"))
                image_name = str(payload["image"])
                image_path = image_dir / image_name
                if not image_path.is_file():
                    stats["missing_images"] += 1
                    continue
                with Image.open(image_path) as image:
                    width, height = image.size
                for obj in payload.get("objects", []):
                    query = str(obj.get("referring_sentence") or obj.get("obj_cls") or "").strip()
                    coords = obj.get("obj_coord")
                    if not query or not isinstance(coords, list) or len(coords) != 4:
                        stats["invalid_objects"] += 1
                        continue
                    # VRSBench stores normalized xyxy coordinates. Some boxes
                    # extend slightly beyond the image, so values above 1.0
                    # still need scaling before clipping to the image bounds.
                    x1, y1, x2, y2 = [float(value) for value in coords]
                    box = [
                        max(0.0, min(x1 * width, width)),
                        max(0.0, min(y1 * height, height)),
                        max(0.0, min(x2 * width, width)),
                        max(0.0, min(y2 * height, height)),
                    ]
                    if box[2] <= box[0] or box[3] <= box[1]:
                        stats["invalid_objects"] += 1
                        continue
                    handle.write(
                        json.dumps(
                            {
                                "id": f"{annotation_path.stem}_{obj.get('obj_id', stats['rows'])}",
                                "image": str(image_path.resolve()),
                                "query": query,
                                "gt_boxes": [box],
                                "category": obj.get("obj_cls"),
                                "source": "VRSBench",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    stats["rows"] += 1
                    if limit is not None and stats["rows"] >= limit:
                        break
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                stats["invalid_objects"] += 1
                continue
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    stats = convert(args.annotation_dir, args.image_dir, args.output, args.limit)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
