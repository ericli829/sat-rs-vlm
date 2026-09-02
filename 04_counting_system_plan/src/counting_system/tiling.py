"""多尺度 tile：Global / Native / Fine，以及 tile ownership core。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from .runtime import BBox, ImageRef, Region
from .target import TargetSpec


@dataclass(slots=True)
class ScaleSpec:
    scale_id: str
    tile_size: int
    overlap: int
    detector_input: int
    upsample: bool = False


@dataclass(slots=True)
class Tile:
    tile_id: str
    scale_id: str
    image: ImageRef
    crop_xyxy: BBox
    core_xyxy: BBox
    detector_input: int
    scope_xyxy: BBox

    @property
    def crop_size(self) -> tuple[int, int]:
        x0, y0, x1, y1 = self.crop_xyxy
        return (max(1, int(round(x1 - x0))), max(1, int(round(y1 - y0))))

    def contains_center(self, point: tuple[float, float]) -> bool:
        x, y = point
        x0, y0, x1, y1 = self.core_xyxy
        return x0 <= x < x1 and y0 <= y < y1


def scope_from_input(image: ImageRef, region: Region | None = None) -> tuple[ImageRef, BBox]:
    if region is not None:
        return region.image, region.bbox_xyxy_global
    width = float(image.width or 0)
    height = float(image.height or 0)
    if width <= 0 or height <= 0:
        raise ValueError("ImageRef.width/height must be set before tiling")
    return image, (0.0, 0.0, width, height)


def _clamp_box(box: BBox, scope: BBox) -> BBox:
    x0, y0, x1, y1 = box
    sx0, sy0, sx1, sy1 = scope
    return (
        min(max(x0, sx0), sx1),
        min(max(y0, sy0), sy1),
        min(max(x1, sx0), sx1),
        min(max(y1, sy0), sy1),
    )


def _window_boxes(scope: BBox, tile_size: int, overlap: int) -> list[BBox]:
    sx0, sy0, sx1, sy1 = scope
    width = sx1 - sx0
    height = sy1 - sy0
    tile_size = max(int(tile_size), 1)
    overlap = min(max(int(overlap), 0), tile_size - 1)
    stride = max(tile_size - overlap, 1)
    boxes: list[BBox] = []
    y = sy0
    row = 0
    while True:
        x = sx0
        col = 0
        y1 = min(y + tile_size, sy1)
        y0 = y1 - tile_size if y1 - sy0 >= tile_size else sy0
        y0 = max(y0, sy0)
        while True:
            x1 = min(x + tile_size, sx1)
            x0 = x1 - tile_size if x1 - sx0 >= tile_size else sx0
            x0 = max(x0, sx0)
            boxes.append((x0, y0, x1, y1))
            if x1 >= sx1 - 1e-6:
                break
            x += stride
            col += 1
            if col > 10000:
                break
        if y1 >= sy1 - 1e-6:
            break
        y += stride
        row += 1
        if row > 10000:
            break
    # 去重（边界处可能重复）
    uniq: list[BBox] = []
    seen: set[tuple[int, int, int, int]] = set()
    for box in boxes:
        key = tuple(int(round(v)) for v in box)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(box)
    if not uniq:
        uniq = [scope]
    return uniq


def ownership_core(crop: BBox, scope: BBox, overlap: int) -> BBox:
    """tile 的 ownership core：检测中心落在 core 才归该 tile，避免重叠区双计。"""
    x0, y0, x1, y1 = crop
    sx0, sy0, sx1, sy1 = scope
    margin = max(int(overlap), 0) / 2.0
    cx0 = x0 if abs(x0 - sx0) < 1e-6 else x0 + margin
    cy0 = y0 if abs(y0 - sy0) < 1e-6 else y0 + margin
    cx1 = x1 if abs(x1 - sx1) < 1e-6 else x1 - margin
    cy1 = y1 if abs(y1 - sy1) < 1e-6 else y1 - margin
    if cx1 <= cx0:
        mid = (x0 + x1) / 2.0
        cx0, cx1 = mid, min(x1, mid + 1.0)
    if cy1 <= cy0:
        mid = (y0 + y1) / 2.0
        cy0, cy1 = mid, min(y1, mid + 1.0)
    return _clamp_box((cx0, cy0, cx1, cy1), scope)


def resolve_overlap(tile_size: int, cfg: dict) -> int:
    """从 pixel overlap 或 uhr-locator 风格的 overlap_ratio 解析 tile 重叠像素。"""
    if "overlap" in cfg and cfg["overlap"] is not None:
        return int(cfg["overlap"])
    ratio = cfg.get("overlap_ratio")
    if ratio is not None:
        return max(0, min(int(round(int(tile_size) * float(ratio))), int(tile_size) - 1))
    return 256


def build_scale_policy(
    *,
    source_scale: int = 1024,
    enable_global: bool = True,
    enable_native: bool = True,
    enable_fine: bool = True,
    native_tile: int = 1024,
    native_overlap: int = 256,
    fine_tile: int = 512,
    fine_overlap: int = 128,
    tiny_target: bool = False,
    only_fine_for_tiny: bool = True,
    entire: bool = True,
) -> list[ScaleSpec]:
    specs: list[ScaleSpec] = []
    if enable_global:
        specs.append(
            ScaleSpec(
                scale_id="global",
                tile_size=0,
                overlap=0,
                detector_input=int(source_scale),
                upsample=False,
            )
        )
    if enable_native:
        specs.append(
            ScaleSpec(
                scale_id="native",
                tile_size=int(native_tile),
                overlap=int(native_overlap),
                detector_input=int(source_scale),
                upsample=False,
            )
        )
    fine_ok = enable_fine and (tiny_target or not only_fine_for_tiny)
    if fine_ok:
        specs.append(
            ScaleSpec(
                scale_id="fine",
                tile_size=int(fine_tile),
                overlap=int(fine_overlap),
                detector_input=int(source_scale),
                upsample=True,
            )
        )
    if not entire:
        # Region 内必须 exhaustive：保留 native，按需 fine；global 仍可作大目标参考
        pass
    if not specs:
        specs.append(
            ScaleSpec(scale_id="native", tile_size=native_tile, overlap=native_overlap, detector_input=source_scale)
        )
    return specs


def iter_tiles(
    image: ImageRef,
    scope: BBox,
    spec: ScaleSpec,
) -> Iterator[Tile]:
    if spec.scale_id == "global" or spec.tile_size <= 0:
        crop = scope
        yield Tile(
            tile_id=f"{spec.scale_id}:0",
            scale_id=spec.scale_id,
            image=image,
            crop_xyxy=crop,
            core_xyxy=crop,
            detector_input=spec.detector_input,
            scope_xyxy=scope,
        )
        return
    for i, crop in enumerate(_window_boxes(scope, spec.tile_size, spec.overlap)):
        yield Tile(
            tile_id=f"{spec.scale_id}:{i}",
            scale_id=spec.scale_id,
            image=image,
            crop_xyxy=crop,
            core_xyxy=ownership_core(crop, scope, spec.overlap),
            detector_input=spec.detector_input,
            scope_xyxy=scope,
        )


def plan_tiles(
    image: ImageRef,
    scope: BBox,
    target: TargetSpec,
    config: dict,
    *,
    entire: bool,
    source_scale: int | None = None,
) -> list[Tile]:
    scale_cfg = config.get("scale") or {}
    native_cfg = scale_cfg.get("native") or {}
    fine_cfg = scale_cfg.get("fine") or {}
    global_cfg = scale_cfg.get("global") or {}
    specs = build_scale_policy(
        source_scale=int(source_scale or scale_cfg.get("default_source_scale") or 1333),
        enable_global=bool(global_cfg.get("enabled", True)),
        enable_native=bool(native_cfg.get("enabled", True)),
        enable_fine=bool(fine_cfg.get("enabled", True)),
        native_tile=int(native_cfg.get("tile_size", 1333)),
        native_overlap=resolve_overlap(int(native_cfg.get("tile_size", 1333)), native_cfg),
        fine_tile=int(fine_cfg.get("tile_size", 512)),
        fine_overlap=resolve_overlap(int(fine_cfg.get("tile_size", 512)), fine_cfg),
        tiny_target=target.tiny,
        only_fine_for_tiny=bool(fine_cfg.get("only_for_tiny", True)),
        entire=entire,
    )
    tiles: list[Tile] = []
    for spec in specs:
        tiles.extend(iter_tiles(image, scope, spec))
    return tiles
