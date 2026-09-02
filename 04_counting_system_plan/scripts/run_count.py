#!/usr/bin/env python3
"""对单张图或 Region 跑 COUNT。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from counting_system.executor import CountExecutor
from counting_system.image_ops import region_from_named
from counting_system.overlay import save_overlay
from counting_system.runtime import ImageRef
from counting_system.trace import TraceWriter, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--region", default="")
    parser.add_argument("--entire", action="store_true")
    parser.add_argument("--no-entire", action="store_true")
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--source-scale", type=int, default=1024)
    parser.add_argument("--score-threshold", type=float, default=0.2)
    parser.add_argument("--out", default=str(ROOT / "outputs" / "count"))
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    path = Path(args.image)
    with Image.open(path) as im:
        w, h = im.size
    image = ImageRef(path=str(path), image_id=path.stem, width=w, height=h)
    entire = True if args.entire else False if args.no_entire else not bool(args.region)
    visual = region_from_named(image, args.region) if args.region else image
    extra = {
        "detector": {"backend": args.backend},
        "gate": {"enabled": bool(args.gate)},
        "count": {"score_threshold": args.score_threshold},
        "scale": {"default_source_scale": args.source_scale},
    }
    executor = CountExecutor(extra)
    result = executor(
        visual,
        args.target,
        entire=entire,
        source_scale=args.source_scale,
        score_threshold=args.score_threshold,
        trace=TraceWriter(out),
    )
    overlay = save_overlay(image, result, out / "overlay.png", title=f"{args.target}")
    write_json(
        out / "result.json",
        {
            "count": result.count,
            "scalar": result.to_scalar().value,
            "num_detections": len(result.detections),
            "provenance": {k: v for k, v in result.provenance.items() if k != "raw_proposals"},
            "raw_count": result.provenance.get("raw_count"),
            "overlay": str(overlay),
        },
    )
    print(f"count={result.count} calls={result.provenance.get('detector_calls')} overlay={overlay}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
