#!/usr/bin/env python3
"""Fake E2E：合成 UHR 图 → tiling → FakeDetector → 去重融合 → overlay + JSON trace。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from counting_system.detector.fake import FakeDetector
from counting_system.executor import CountExecutor
from counting_system.overlay import save_overlay
from counting_system.synth import Blob, write_blob_image
from counting_system.trace import TraceWriter, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "outputs" / "fake_e2e"))
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--height", type=int, default=2048)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    blobs = [
        Blob((80, 80, 140, 140)),
        Blob((400, 120, 460, 180)),
        Blob((980, 90, 1040, 150)),  # 跨 tile 边界，测 ownership
        Blob((1600, 200, 1680, 280)),
        Blob((220, 1100, 280, 1160)),
        Blob((1500, 1500, 1580, 1580)),
        Blob((1900, 1900, 1960, 1960)),
    ]
    image = write_blob_image(str(out / "input.png"), width=args.width, height=args.height, blobs=blobs)
    detector = FakeDetector()
    executor = CountExecutor({"detector": {"backend": "fake"}}, detector=detector)
    trace = TraceWriter(out)
    result = executor(image, "ship", entire=True, trace=trace)
    overlay = save_overlay(image, result, out / "overlay.png", title="fake-e2e ship")
    write_json(
        out / "result.json",
        {
            "count": result.count,
            "expected": len(blobs),
            "ok": result.count == len(blobs),
            "fusion": result.provenance.get("fusion"),
            "detector_calls": result.provenance.get("detector_calls"),
            "overlay": str(overlay),
        },
    )
    print(f"count={result.count} expected={len(blobs)} overlay={overlay}")
    return 0 if result.count == len(blobs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
