from pathlib import Path

from counting_system.detector.fake import FakeDetector
from counting_system.executor import CountExecutor
from counting_system.image_ops import region_from_named
from counting_system.overlay import save_overlay
from counting_system.synth import Blob, write_blob_image
from counting_system.trace import TraceWriter


def test_fake_e2e_whole_image(tmp_path: Path):
    blobs = [
        Blob((120, 130, 200, 210)),
        Blob((1500, 140, 1580, 220)),
        Blob((130, 1500, 210, 1580)),
        Blob((1480, 1480, 1560, 1560)),
    ]
    image = write_blob_image(str(tmp_path / "in.png"), width=2048, height=2048, blobs=blobs)
    executor = CountExecutor(
        {
            "detector": {"backend": "fake"},
            "scale": {
                "global": {"enabled": True},
                "native": {"enabled": True, "tile_size": 1024, "overlap": 256},
                "fine": {"enabled": True, "only_for_tiny": False, "tile_size": 512, "overlap": 128},
            },
            "count": {"score_threshold": 0.05, "nms_iou": 0.4, "cross_scale_iou": 0.3},
        },
        detector=FakeDetector(),
    )
    result = executor(image, "ship", entire=True, trace=TraceWriter(tmp_path))
    overlay = save_overlay(image, result, tmp_path / "overlay.png", title="e2e")
    assert overlay.exists()
    assert (tmp_path / "trace.jsonl").exists()
    assert result.count == len(blobs)
    assert result.provenance["detector_calls"] >= 1
    assert result.provenance["fusion"]["raw"] >= result.count


def test_region_count_is_exhaustive(tmp_path: Path):
    blobs = [
        Blob((80, 80, 140, 140)),
        Blob((200, 90, 260, 150)),
        Blob((1600, 1600, 1680, 1680)),
    ]
    image = write_blob_image(str(tmp_path / "in.png"), width=2048, height=2048, blobs=blobs)
    region = region_from_named(image, "TOP_LEFT")
    executor = CountExecutor({"detector": {"backend": "fake"}}, detector=FakeDetector())
    result = executor(region, "ship", entire=False)
    assert result.count == 2
    assert result.provenance["entire"] is False


def test_global_bbox_mapping_roundtrip(tmp_path: Path):
    blob = Blob((1100, 200, 1180, 280))
    image = write_blob_image(str(tmp_path / "in.png"), width=2048, height=1024, blobs=[blob])
    executor = CountExecutor(
        {
            "scale": {
                "global": {"enabled": False},
                "native": {"enabled": True, "tile_size": 1024, "overlap": 256},
                "fine": {"enabled": False},
            }
        },
        detector=FakeDetector(),
    )
    result = executor(image, "vehicle", entire=True)
    assert result.count == 1
    x0, y0, x1, y1 = result.detections.detections[0].bbox_xyxy_global
    assert abs(x0 - 1100) <= 3
    assert abs(y0 - 200) <= 3
    assert abs(x1 - 1180) <= 3
    assert abs(y1 - 280) <= 3
