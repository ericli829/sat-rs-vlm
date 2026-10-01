from __future__ import annotations

import threading
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from sat_rs_vlm.integrations.counting import detector_bridge
from sat_rs_vlm.integrations.detectors import tiled
from sat_rs_vlm.integrations.detectors.protocol import ProposalResult
from sat_rs_vlm.integrations.detectors.tile_transport import cpu_prepared_items


def test_cpu_preparation_overlaps_serial_inference_and_keeps_one_lookahead(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (40, 10), "white").save(image_path)
    next_ready = threading.Event()
    prepare_threads = []
    infer_threads = []
    saved_paths = []
    original_save = tiled.save_tile_image

    def save(image, path, image_format):
        original_save(image, path, image_format)
        saved_paths.append(path)
        prepare_threads.append(threading.get_ident())
        if path.stem == "tile_00001":
            next_ready.set()

    class Provider:
        def predict(self, path, _phrase):
            infer_threads.append(threading.get_ident())
            if path.stem == "tile_00000":
                assert next_ready.wait(3), "next CPU tile did not progress during inference"
                assert len(saved_paths) == 2, "more than one tile was prefetched"
            return ProposalResult([[0, 0, 10, 10]], [0.9], 0, "fixture", "fixture")

        def close(self):
            pass

    monkeypatch.setattr(tiled, "save_tile_image", save)
    provider = tiled.TiledProposalProvider(
        Provider(),
        {"tile_size": 10, "overlap_ratio": 0, "parallel_workers": 1},
        base_provider_name="fixture",
    )
    try:
        result = provider.predict(image_path, "building")
        assert result.metadata["parallel_workers"] == 1
        assert len(set(infer_threads)) == 1
        assert set(prepare_threads).isdisjoint(infer_threads)
        assert len(result.boxes_xyxy) == 4
        assert all(not p.exists() for p in saved_paths)
    finally:
        provider.close()


@pytest.mark.parametrize("early_exit", ["break", "inference_error", "prepare_error"])
def test_cpu_prefetch_releases_active_and_unused_results(early_exit: str) -> None:
    prepared = []
    released = []
    lookahead_started = threading.Event()

    def prepare(item):
        if item == 1:
            lookahead_started.set()
            if early_exit == "prepare_error":
                raise ValueError("preparation failed")
        prepared.append(item)
        return item

    with pytest.raises(RuntimeError) if early_exit == "inference_error" else nullcontext():
        with cpu_prepared_items(range(3), prepare, released.append) as items:
            for _ in items:
                assert lookahead_started.wait(3)
                if early_exit == "inference_error":
                    raise RuntimeError("inference failed")
                break
    assert sorted(prepared) == sorted(released)
    assert len(prepared) <= 2


def test_cpu_prefetch_propagates_preparation_error() -> None:
    released = []

    def prepare(item):
        if item == 1:
            raise ValueError("preparation failed")
        return item

    with pytest.raises(ValueError, match="preparation failed"):
        with cpu_prepared_items(range(3), prepare, released.append) as items:
            list(items)
    assert released == [0]


def test_count_bridge_prefetch_keeps_inference_serial_and_preserves_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared_next = threading.Event()
    infer_threads = []
    prepare_threads = []
    paths = []
    original_save = detector_bridge.save_tile_image

    def save(image, path, image_format):
        original_save(image, path, image_format)
        paths.append(path)
        prepare_threads.append(threading.get_ident())
        if path.stem == "tile_00001":
            prepared_next.set()

    def requests():
        for i in range(4):
            image = Image.new("RGB", (10, 10), "red")
            try:
                yield detector_bridge.DetectionRequest(
                    image=image,
                    texts="car",
                    target=SimpleNamespace(name="car"),
                    tile=SimpleNamespace(
                        crop_xyxy=(i * 20, 0, i * 20 + 20, 20),
                        tile_id=str(i),
                        scale_id="native",
                    ),
                )
            finally:
                image.close()

    class Provider:
        def predict(self, path, _phrase):
            infer_threads.append(threading.get_ident())
            if path.stem == "tile_00000":
                assert prepared_next.wait(3)
                assert len(paths) == 2
            return ProposalResult([[1, 2, 5, 6]], [0.8], 0, "fixture", "fixture")

        def close(self):
            pass

    monkeypatch.setattr(detector_bridge, "save_tile_image", save)
    bridge = detector_bridge.CountingProposalDetectorBridge(Provider())
    try:
        results = list(bridge.detect_many(requests()))
        assert bridge.call_count == 4
        assert [r.detections[0].bbox_xyxy_global for r in results] == [
            (i * 20 + 2, 4, i * 20 + 10, 12) for i in range(4)
        ]
        assert len(set(infer_threads)) == 1
        assert set(prepare_threads).isdisjoint(infer_threads)
        assert all(not p.exists() for p in paths)
    finally:
        bridge.close()


@pytest.mark.parametrize("inference_error", [False, True])
def test_count_executor_drives_prefetch_and_closes_owned_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, inference_error: bool
) -> None:
    from counting_system import executor

    from sat_rs_vlm.integrations.counting import CountingSystemProvider
    from sat_rs_vlm.taskgraph.providers import CountingRequest
    from sat_rs_vlm.taskgraph.runtime_types import ImageRef
    from sat_rs_vlm.taskgraph.schema import TargetSpec

    path = tmp_path / "count.png"
    Image.new("RGB", (40, 10), "red").save(path)
    second_crop = threading.Event()
    crops = []
    transported_paths = []
    original_crop = executor.crop_tile

    def crop_tile(*args):
        crop = original_crop(*args)
        crops.append(crop)
        if len(crops) == 2:
            second_crop.set()
        return crop

    class Provider:
        def predict(self, path, _phrase):
            transported_paths.append(path)
            if len(transported_paths) == 1:
                assert second_crop.wait(3), "COUNT executor did not overlap CPU cropping"
                if inference_error:
                    raise ValueError("fixture failure")
            return ProposalResult([[1, 2, 5, 6]], [0.8], 0, "fixture", "fixture")

        def close(self):
            pass

    monkeypatch.setattr(executor, "crop_tile", crop_tile)
    bridge = detector_bridge.CountingProposalDetectorBridge(Provider())
    provider = CountingSystemProvider(
        config={
            "scale": {
                "global": {"enabled": False},
                "native": {"enabled": True, "tile_size": 10, "overlap": 0},
                "fine": {"enabled": False},
            },
            "gate": {"enabled": False},
        },
        detector=bridge,
    )
    try:
        with (
            pytest.raises(RuntimeError, match="fixture failure")
            if inference_error
            else nullcontext()
        ):
            result = provider.count(
                CountingRequest(
                    ImageRef(str(path), width=40, height=10), TargetSpec(category="car"), True
                )
            )
            assert result.count == 4
            assert result.metadata["detector_calls"] == 4
    finally:
        provider.close()
    assert all(not path.exists() for path in transported_paths)
    for crop in crops:
        with pytest.raises(ValueError, match="closed image"):
            crop.getpixel((0, 0))
