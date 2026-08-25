#!/usr/bin/env python3
"""Render one precomputed proposal row for coordinate/coverage inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_row(path: Path, sample_id: str | None) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        rows = payload
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    for row in rows:
        if isinstance(row, dict) and (sample_id is None or str(row.get("id")) == sample_id):
            return row
    raise ValueError(f"sample id not found in proposal file: {sample_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--proposal-json", type=Path, required=True)
    parser.add_argument("--sample-id")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-phrase")
    args = parser.parse_args()
    from PIL import Image, ImageDraw

    row = _load_row(args.proposal_json, args.sample_id)
    image = Image.open(args.image).convert("RGB")
    draw = ImageDraw.Draw(image)
    boxes = row.get("bbox_list", [])
    scores = row.get("bbox_scores", [])
    for index, box in enumerate(boxes):
        if not isinstance(box, list) or len(box) != 4:
            continue
        score = float(scores[index]) if index < len(scores) else 0.0
        coordinates = [float(value) for value in box]
        draw.rectangle(coordinates, outline=(255, 32, 32), width=2)
        draw.text((coordinates[0], coordinates[1]), f"{index}:{score:.3f}", fill=(255, 255, 0))
    phrase = args.target_phrase or row.get("target_phrase") or ""
    draw.text((8, 8), str(phrase), fill=(0, 255, 0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(json.dumps({"status": "ok", "output": str(args.output), "boxes": len(boxes)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
