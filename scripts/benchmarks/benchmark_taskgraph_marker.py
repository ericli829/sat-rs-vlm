"""Benchmark the vectorized TaskGraph marker finder on synthetic UHR images."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from PIL import Image, ImageDraw

from sat_rs_vlm.taskgraph.operators import GeometryExecutor
from sat_rs_vlm.taskgraph.runtime_types import ImageRef


def benchmark_size(size: int, directory: Path) -> dict[str, object]:
    path = directory / f"marker_{size}.png"
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    radius = max(8, size // 128)
    centers = ((size // 4, size // 4), (3 * size // 4, 3 * size // 4))
    for center_x, center_y in centers:
        draw.ellipse(
            (
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
            ),
            outline="red",
            width=max(3, radius // 4),
        )
    image.save(path)

    started = time.perf_counter()
    result = GeometryExecutor._marker(ImageRef(str(path)), "red", "circle")
    latency_ms = (time.perf_counter() - started) * 1000.0
    return {
        "size": [size, size],
        "pixels": size * size,
        "latency_ms": latency_ms,
        "region_count": len(result.regions),
        "component_count": result.provenance["component_count"],
        "implementation": result.provenance["implementation"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[512, 2048, 4096])
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if any(size < 32 for size in args.sizes):
        raise SystemExit("all benchmark sizes must be at least 32")
    with tempfile.TemporaryDirectory(prefix="taskgraph_marker_benchmark_") as directory:
        rows = [benchmark_size(size, Path(directory)) for size in args.sizes]
    report = {
        "schema_version": "taskgraph-marker-benchmark-v1",
        "results": rows,
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
