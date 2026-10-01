from __future__ import annotations

import gc
import hashlib
import json
import threading
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from sat_rs_vlm.integrations.counting.detector_bridge import (
    CountingProposalDetectorBridge,
    DetectionRequest,
)
from sat_rs_vlm.integrations.detectors.protocol import ProposalResult
from sat_rs_vlm.integrations.detectors.tiled import TiledProposalProvider
from sat_rs_vlm.integrations.retrievers import cache as cache_module
from sat_rs_vlm.integrations.retrievers.cache import retrieval_cache_key
from sat_rs_vlm.integrations.retrievers.openclip import OpenCLIPRetrieverProvider


def test_score_cache_hits_hash_once_without_decoding_or_cropping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "source.png"
    Image.new("RGB", (80, 80), "white").save(image_path)
    checkpoint = tmp_path / "clip.pt"
    checkpoint.touch()
    provider = OpenCLIPRetrieverProvider(
        {"checkpoint": str(checkpoint), "cache_dir": str(tmp_path / "cache")}
    )
    boxes = [(i, i, i + 8, i + 8) for i in range(64)]
    # Construct legacy keys independently, proving existing cache files remain usable.
    stat = image_path.stat()
    for box in boxes:
        payload = {
            "schema_version": "uhr-retrieval-cache-v1",
            "image_identity": {
                "path": str(image_path.resolve()),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            },
            "bbox": [float(v) for v in box],
            "query": "building",
            "provider": provider.provider_name,
            "model_identity": {"checkpoint": str(checkpoint), "model_id": provider.model_id},
            "parameters": provider.parameters,
        }
        key = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
        ).hexdigest()
        provider.cache.put(key, 0.75)
    hashes = []
    original_hash = cache_module._file_sha256

    def hash_source(path: Path) -> str:
        hashes.append(path)
        return original_hash(path)

    def unexpected_pixels(*_args, **_kwargs):
        pytest.fail("cache hit decoded or cropped source pixels")

    monkeypatch.setattr(cache_module, "_file_sha256", hash_source)
    monkeypatch.setattr(provider, "_decoded_image", unexpected_pixels)
    monkeypatch.setattr(provider, "_crop", unexpected_pixels)
    monkeypatch.setattr(provider, "_load", unexpected_pixels)
    try:
        result = provider.score_regions(image_path, "building", boxes)
        assert result.scores == [0.75] * 64
        assert result.metadata["score_cache_hits"] == 64
        assert hashes == [image_path.resolve()]
    finally:
        provider.close()


def test_batch_identity_does_not_reuse_hash_after_source_changes(tmp_path: Path) -> None:
    image_path = tmp_path / "source.png"
    Image.new("RGB", (20, 20), "white").save(image_path)
    params = dict(
        image_path=image_path,
        region_xyxy=(0, 0, 10, 10),
        query="building",
        provider="clip",
        model_identity="model",
        parameters={},
    )
    identity = cache_module.retrieval_image_identity(image_path)
    before = retrieval_cache_key(**params)
    assert before == retrieval_cache_key(**params, image_identity=identity)
    Image.new("RGB", (20, 20), "black").save(image_path)
    after_identity = cache_module.retrieval_image_identity(image_path)
    assert identity["sha256"] != after_identity["sha256"]
    assert before != retrieval_cache_key(**params, image_identity=after_identity)


def test_openclip_crops_only_missing_embeddings(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    image_path = tmp_path / "source.png"
    Image.new("RGB", (60, 20), "white").save(image_path)
    checkpoint = tmp_path / "clip.pt"
    checkpoint.touch()
    provider = OpenCLIPRetrieverProvider({"checkpoint": str(checkpoint), "batch_size": 1})

    class Model:
        def encode_text(self, _tokens):
            return torch.tensor([[1.0, 0.0]])

        def encode_image(self, images):
            return images

    provider._torch = torch
    provider._model = Model()
    provider._preprocess = lambda image: torch.tensor([float(image.width), float(image.height)])
    provider._tokenizer = lambda _values: torch.tensor([[1, 2]])
    provider._resolved_device = "cpu"
    try:
        first = provider.score_regions(image_path, "building", [(0, 0, 20, 20), (20, 0, 40, 20)])
        mixed = provider.score_regions(image_path, "ship", [(0, 0, 20, 20), (40, 0, 60, 20)])
        cached = provider.score_regions(image_path, "car", [(0, 0, 20, 20), (40, 0, 60, 20)])
        assert first.metadata["actual_crop_count"] == 2
        assert mixed.metadata["actual_crop_count"] == 1
        assert cached.metadata["actual_crop_count"] == 0
        assert cached.metadata["image_decode_skipped"] is True
        assert first.scores == mixed.scores == cached.scores
    finally:
        provider.close()


@pytest.mark.parametrize("image_format", ["bmp", "png"])
@pytest.mark.parametrize("prefetch", [False, True])
def test_tile_preparation_is_bounded_and_preserves_pixels_and_global_order(
    tmp_path: Path, image_format: str, prefetch: bool
) -> None:
    image_path = tmp_path / "source.png"
    image = Image.new("RGB", (60, 10))
    for x in range(60):
        for y in range(10):
            image.putpixel((x, y), (x, y, 127))
    image.save(image_path)
    lock = threading.Lock()
    paths = []

    class Provider:
        model_id = "pixel-fixture"

        def predict(self, path: Path, _phrase: str) -> ProposalResult:
            with lock:
                paths.append(path)
                # Six tiles must not have been materialized before the first inference.
                assert len(list(path.parent.iterdir())) <= 2 * (1 + prefetch)
            index = int(path.stem.split("_")[-1])
            with Image.open(path) as tile:
                expected = image.crop((index * 10, 0, index * 10 + 10, 10))
                assert tile.convert("RGB").tobytes() == expected.tobytes()
                expected.close()
            return ProposalResult([[0, 0, 10, 10]], [0.9], 0, "fixture", self.model_id)

        def close(self):
            pass

    provider = TiledProposalProvider(
        Provider(),
        {
            "tile_size": 10,
            "overlap_ratio": 0,
            "parallel_workers": 2,
            "tile_image_format": image_format,
            "tile_prefetch": prefetch,
        },
        base_provider_name="fixture",
    )
    try:
        result = provider.predict(image_path, "building")
        assert result.boxes_xyxy == [[i * 10, 0, i * 10 + 10, 10] for i in range(6)]
        assert [tile["tile_id"] for tile in result.metadata["tiles"]] == list(range(6))
        assert len(paths) == 6
        assert all(path.suffix == f".{image_format}" and not path.exists() for path in paths)
    finally:
        provider.close()
        image.close()


def test_tiled_failure_removes_transport_files(tmp_path: Path) -> None:
    image_path = tmp_path / "source.png"
    Image.new("RGB", (10, 10), "white").save(image_path)
    paths = []

    class Provider:
        def predict(self, path: Path, _phrase: str):
            paths.append(path)
            raise RuntimeError("detector failed")

        def close(self):
            pass

    provider = TiledProposalProvider(Provider(), {"tile_size": 10}, base_provider_name="fixture")
    try:
        with pytest.raises(RuntimeError, match="detector failed"):
            provider.predict(image_path, "building")
        assert paths and all(not path.exists() for path in paths)
    finally:
        provider.close()


def test_count_bridge_releases_images_and_preserves_coordinate_transform() -> None:
    class Provider:
        def predict(self, path: Path, _phrase: str) -> ProposalResult:
            with Image.open(path) as tile:
                assert tile.size == (10, 10)
                assert tile.getpixel((0, 0)) == (255, 0, 0)
            return ProposalResult([[1, 2, 5, 6]], [0.8], 0, "fixture", "fixture")

        def close(self):
            pass

    bridge = CountingProposalDetectorBridge(Provider())
    image_refs = []
    try:
        for _ in range(10):
            image = Image.new("RGB", (10, 10), "red")
            image_refs.append(weakref.ref(image))
            request = DetectionRequest(
                image=image,
                target=SimpleNamespace(name="car"),
                tile=SimpleNamespace(
                    crop_xyxy=(100, 200, 120, 220), tile_id="t1", scale_id="native"
                ),
                texts="car",
            )
            result = bridge.detect(request)
            assert result.detections[0].bbox_xyxy_global == (102, 204, 110, 212)
            del request, image
        gc.collect()
        assert bridge.call_count == 10
        assert all(ref() is None for ref in image_refs)
    finally:
        bridge.close()
