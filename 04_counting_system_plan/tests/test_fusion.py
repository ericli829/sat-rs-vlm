from counting_system.fusion import cross_scale_fusion, same_scale_dedup
from counting_system.runtime import Detection, ImageRef
from counting_system.tiling import Tile


def _tile(tile_id: str, scale: str, crop, core) -> Tile:
    image = ImageRef(path="x", width=1000, height=1000)
    return Tile(
        tile_id=tile_id,
        scale_id=scale,
        image=image,
        crop_xyxy=crop,
        core_xyxy=core,
        detector_input=1024,
        scope_xyxy=(0, 0, 1000, 1000),
    )


def _det(box, score, scale, tile_id) -> Detection:
    return Detection(bbox_xyxy_global=box, score=score, label="ship", tile_id=tile_id, scale_id=scale)


def test_core_ownership_drops_overlap_duplicate():
    tiles = [
        _tile("native:0", "native", (0, 0, 600, 600), (0, 0, 500, 500)),
        _tile("native:1", "native", (400, 0, 1000, 600), (500, 0, 1000, 500)),
    ]
    dets = [
        _det((450, 100, 490, 140), 0.9, "native", "native:0"),  # center 470 in core0
        _det((450, 100, 490, 140), 0.8, "native", "native:1"),  # same center, not in core1
    ]
    kept = same_scale_dedup(dets, tiles, nms_iou=0.5)
    assert len(kept) == 1
    assert kept[0].tile_id == "native:0"


def test_cross_scale_one_to_one_keeps_fine():
    coarse = _det((10, 10, 50, 50), 0.4, "global", "global:0")
    fine = _det((12, 12, 48, 48), 0.9, "fine", "fine:0")
    fused = cross_scale_fusion([coarse, fine])
    assert len(fused) == 1
    assert fused[0].scale_id == "fine"


def test_cross_scale_one_to_many_drops_coarse():
    coarse = _det((0, 0, 100, 40), 0.5, "native", "native:0")
    f1 = _det((2, 2, 30, 30), 0.9, "fine", "fine:0")
    f2 = _det((60, 2, 90, 30), 0.8, "fine", "fine:1")
    fused = cross_scale_fusion([coarse, f1, f2], iou_thr=0.1)
    assert all(d.scale_id == "fine" for d in fused)
    assert len(fused) == 2


def test_cross_scale_keeps_unmatched_coarse():
    coarse = _det((200, 200, 240, 240), 0.7, "global", "global:0")
    fine = _det((10, 10, 30, 30), 0.9, "fine", "fine:0")
    fused = cross_scale_fusion([coarse, fine], iou_thr=0.5)
    scales = {d.scale_id for d in fused}
    assert scales == {"global", "fine"}
