"""Compare committed/current CPU image paths on local UHR images, without model weights."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import subprocess
import sys
import tempfile
import time
import weakref
from pathlib import Path
from types import ModuleType, SimpleNamespace

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sat_rs_vlm.integrations.counting import detector_bridge  # noqa: E402
from sat_rs_vlm.integrations.detectors.protocol import ProposalResult  # noqa: E402
from sat_rs_vlm.integrations.detectors.tiled import (  # noqa: E402
    TiledProposalProvider,
    generate_tiles,
)
from sat_rs_vlm.integrations.retrievers import cache, openclip  # noqa: E402


def committed_module(ref: str, relative_path: str, package: str) -> ModuleType:
    code = subprocess.check_output(
        ["git", "show", f"{ref}:{relative_path}"], cwd=PROJECT_ROOT, encoding="utf-8"
    )
    module = ModuleType(f"{package}._benchmark_original_{Path(relative_path).stem}")
    module.__package__ = package
    module.__file__ = str(PROJECT_ROOT / relative_path)
    sys.modules[module.__name__] = module
    exec(compile(code, module.__file__, "exec"), module.__dict__)
    return module


class PixelDetector:
    """Read transported pixels and return deterministic proposals, without model compute."""

    model_id = "cpu-pixel-verifier"

    def __init__(self) -> None:
        self.records: dict[str, str] = {}
        self.transferred_bytes: dict[str, int] = {}

    def predict(self, path: Path, phrase: str) -> ProposalResult:
        with Image.open(path) as source, source.convert("RGB") as rgb:
            key = path.stem
            self.records[key] = hashlib.sha256(rgb.tobytes()).hexdigest()
            self.transferred_bytes[key] = path.stat().st_size
            width, height = rgb.size
        return ProposalResult([[0, 0, width, height]], [0.9], 0, "pixel", self.model_id)

    def close(self) -> None:
        pass


def cache_hit_run(provider_class, cache_module, image_path, checkpoint, cache_dir, boxes):
    provider = provider_class(
        {
            "checkpoint": str(checkpoint),
            "cache_dir": str(cache_dir),
            "batch_size": 48,
        }
    )
    hashes = 0
    crops = 0
    decodes = 0
    original_hash = cache_module._file_sha256
    original_crop = provider._crop
    original_decode = provider._decoded_image

    def hash_image(path):
        nonlocal hashes
        hashes += 1
        return original_hash(path)

    def crop_image(*args):
        nonlocal crops
        crops += 1
        return original_crop(*args)

    def decode_image(*args):
        nonlocal decodes
        decodes += 1
        return original_decode(*args)

    cache_module._file_sha256 = hash_image
    provider._crop = crop_image
    provider._decoded_image = decode_image
    started = time.perf_counter()
    try:
        result = provider.score_regions(image_path, "building", boxes)
        elapsed = time.perf_counter() - started
        assert result.metadata["score_cache_hits"] == len(boxes)
        return {
            "seconds": elapsed,
            "hash_calls": hashes,
            "hashed_bytes": hashes * image_path.stat().st_size,
            "pixel_decode_calls": decodes,
            "crop_calls": crops,
        }, result.scores
    finally:
        cache_module._file_sha256 = original_hash
        provider.close()
        gc.collect()


def tiled_run(provider_class, image_path, image_format=None):
    base = PixelDetector()
    config = {
        "tile_size": 1333,
        "overlap_ratio": 0.15,
        "parallel_workers": 5,
        "parallel_max_workers": 5,
        "proposal_cache_size": 0,
    }
    if image_format:
        config["tile_image_format"] = image_format
    provider = provider_class(base, config, base_provider_name="pixel")
    started = time.perf_counter()
    try:
        result = provider.predict(image_path, "building")
        return {
            "seconds": time.perf_counter() - started,
            "tile_count": result.metadata["tile_count"],
            "transferred_bytes": sum(base.transferred_bytes.values()),
        }, (result.boxes_xyxy, result.scores, base.records)
    finally:
        provider.close()
        gc.collect()


def counting_retention(bridge_class, image_path):
    bridge = bridge_class(PixelDetector())
    references = []
    try:
        with Image.open(image_path) as source, source.convert("RGB") as rgb:
            for index, box in enumerate(generate_tiles(rgb.width, rgb.height, 1333, 0.15)[:12]):
                crop = rgb.crop(box)
                references.append(weakref.ref(crop))
                request = detector_bridge.DetectionRequest(
                    image=crop,
                    target=SimpleNamespace(name="building"),
                    texts="building",
                    tile=SimpleNamespace(crop_xyxy=box, tile_id=str(index), scale_id="native"),
                )
                bridge.detect(request)
                del crop, request
        gc.collect()
        retained = [ref() for ref in references if ref() is not None]
        result = {
            "calls": len(references),
            "retained_images": len(retained),
            "retained_rgb_payload_bytes": sum(i.width * i.height * 3 for i in retained),
        }
        del retained
        return result
    finally:
        bridge.close()
        if hasattr(bridge, "calls"):
            bridge.calls.clear()
        gc.collect()


def comparison(original, updated):
    old = statistics.median(r["seconds"] for r in original)
    new = statistics.median(r["seconds"] for r in updated)
    return {
        "original_runs": original,
        "updated_runs": updated,
        "original_median_seconds": old,
        "updated_median_seconds": new,
        "speedup": old / new,
        "reduction_percent": (1 - new / old) * 100,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-ref", default="c77e098")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--images", nargs="+", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    root = args.image_root.resolve()
    images = args.images or [
        next((root / "xlrs_bench_samples_20260827").glob("01_anomaly_*.jpg")),
        next((root / "xlrs_bench_samples_20260827").glob("02_spatial_relation_*.jpg")),
        next((root / "mme_counting_20260829").glob("01_03553_*.png")),
    ]
    images = [p.resolve() for p in images]
    if any(not p.is_relative_to(root) for p in images):
        parser.error("all input images must be inside --image-root")
    # These are the user's known UHR fixtures, including trusted 100 MP images.
    Image.MAX_IMAGE_PIXELS = None
    package = "sat_rs_vlm.integrations"
    original_cache = committed_module(
        args.baseline_ref,
        "src/sat_rs_vlm/integrations/retrievers/cache.py",
        f"{package}.retrievers",
    )
    original_clip = committed_module(
        args.baseline_ref,
        "src/sat_rs_vlm/integrations/retrievers/openclip.py",
        f"{package}.retrievers",
    )
    original_clip.retrieval_cache_key = original_cache.retrieval_cache_key
    original_tiled = committed_module(
        args.baseline_ref,
        "src/sat_rs_vlm/integrations/detectors/tiled.py",
        f"{package}.detectors",
    )
    original_count = committed_module(
        args.baseline_ref,
        "src/sat_rs_vlm/integrations/counting/detector_bridge.py",
        f"{package}.counting",
    )
    report = {
        "baseline_commit": subprocess.check_output(
            ["git", "rev-parse", args.baseline_ref], cwd=PROJECT_ROOT, encoding="utf-8"
        ).strip(),
        "scope": "CPU preprocessing only; deterministic pixel verifier, no model inference",
        "cache_conditions": "all score hits; file cache warmed by cache setup; alternating order",
        "workers": 5,
        "repeats": args.repeats,
        "images": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="uhr_cpu_benchmark_") as temporary:
        checkpoint = Path(temporary) / "placeholder.pt"
        checkpoint.touch()
        for index, image_path in enumerate(images):
            with Image.open(image_path) as source:
                width, height = source.size
            boxes = [
                (x, y, min(x + 1333, width), min(y + 1333, height))
                for y in [round(i * (height - 1333) / 7) for i in range(8)]
                for x in [round(i * (width - 1333) / 7) for i in range(8)]
            ]
            cache_dir = Path(temporary) / f"scores_{index}"
            score_cache = cache.RetrievalCache(cache_dir)
            identity = cache.retrieval_image_identity(image_path)
            declared = openclip.OpenCLIPRetrieverProvider(
                {
                    "checkpoint": str(checkpoint),
                    "cache_dir": str(cache_dir),
                    "batch_size": 48,
                }
            )
            for box in boxes:
                key = cache.retrieval_cache_key(
                    image_path=image_path,
                    region_xyxy=box,
                    query="building",
                    provider=declared.provider_name,
                    model_identity={
                        "checkpoint": str(declared.checkpoint),
                        "model_id": declared.model_id,
                    },
                    parameters=declared.parameters,
                    image_identity=identity,
                )
                score_cache.put(key, 0.5)
            declared.close()
            samples = {name: [] for name in ("cache_old", "cache_new", "tile_old", "tile_new")}
            expected_scores = expected_tiles = None
            for repeat in range(args.repeats):
                order = ("old", "new") if repeat % 2 == 0 else ("new", "old")
                for version in order:
                    print(
                        f"[{index + 1}/{len(images)}] {image_path.name}: "
                        f"repeat {repeat + 1} {version} cache",
                        flush=True,
                    )
                    cls, mod = (
                        (original_clip.OpenCLIPRetrieverProvider, original_cache)
                        if version == "old"
                        else (openclip.OpenCLIPRetrieverProvider, cache)
                    )
                    measurement, scores = cache_hit_run(
                        cls, mod, image_path, checkpoint, cache_dir, boxes
                    )
                    expected_scores = scores if expected_scores is None else expected_scores
                    assert scores == expected_scores
                    samples[f"cache_{version}"].append(measurement)
                    print(f"  {measurement}", flush=True)
                for version in order:
                    print(
                        f"[{index + 1}/{len(images)}] {image_path.name}: "
                        f"repeat {repeat + 1} {version} tiled",
                        flush=True,
                    )
                    cls = (
                        original_tiled.TiledProposalProvider
                        if version == "old"
                        else TiledProposalProvider
                    )
                    measurement, pixels = tiled_run(cls, image_path)
                    expected_tiles = pixels if expected_tiles is None else expected_tiles
                    assert pixels == expected_tiles, (
                        "tile pixels, global boxes, scores/order changed"
                    )
                    samples[f"tile_{version}"].append(measurement)
                    print(f"  {measurement}", flush=True)
            entry = {
                "path": str(image_path),
                "size": [width, height],
                "file_bytes": image_path.stat().st_size,
                "cache_hits": comparison(samples["cache_old"], samples["cache_new"]),
                "tiled_transport": comparison(samples["tile_old"], samples["tile_new"]),
                "equal_scores_boxes_and_tile_pixels": True,
            }
            if index == 0:
                entry["counting_retention"] = {
                    "original": counting_retention(
                        original_count.CountingProposalDetectorBridge, image_path
                    ),
                    "updated": counting_retention(
                        detector_bridge.CountingProposalDetectorBridge, image_path
                    ),
                }
            report["images"].append(entry)
            args.output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    print(f"Report: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
