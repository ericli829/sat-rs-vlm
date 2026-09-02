"""对齐 sat-rs-vlm feature/vlm-semantic-alignment 的 TaskGraph runtime_types。

COUNT 对外只交换这些 typed object；检测框等内部细节留在 Detection / CountResult。
"""

from __future__ import annotations

import math
from dataclasses import InitVar, asdict, dataclass, field, is_dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence, TypeAlias


BBox = tuple[float, float, float, float]


def _bbox(value: Sequence[float]) -> BBox:
    box = tuple(float(item) for item in value)
    if len(box) != 4 or not all(math.isfinite(item) for item in box):
        raise ValueError("bbox must contain four finite coordinates")
    x0, y0, x1, y1 = box
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    if x0 >= x1 or y0 >= y1:
        raise ValueError("bbox must have positive area")
    return (x0, y0, x1, y1)


@dataclass(frozen=True)
class ImageRef:
    uri_or_key: str = ""
    width: int | None = None
    height: int | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    path_init: InitVar[str | None] = None
    image_id_init: InitVar[str | None] = None

    def __post_init__(self, path_init: str | None, image_id_init: str | None) -> None:
        key = self.uri_or_key or (path_init or "")
        if not key:
            raise ValueError("ImageRef requires uri_or_key or path")
        object.__setattr__(self, "uri_or_key", key)
        prov = dict(self.provenance)
        if image_id_init:
            prov.setdefault("image_id", image_id_init)
        object.__setattr__(self, "provenance", prov)

    @property
    def path(self) -> Path:
        return Path(self.uri_or_key).expanduser()

    @property
    def image_id(self) -> str:
        return str(self.provenance.get("image_id") or Path(self.uri_or_key).stem)

    def with_size(self, width: int, height: int) -> ImageRef:
        if self.width == width and self.height == height:
            return self
        return replace(self, width=int(width), height=int(height))

    def resolved(self) -> Path:
        return self.path


_image_ref_init = ImageRef.__init__


def _compat_image_ref_init(
    self,
    uri_or_key: str = "",
    width: int | None = None,
    height: int | None = None,
    provenance: dict[str, Any] | None = None,
    path_init: str | None = None,
    image_id_init: str | None = None,
    path: str | None = None,
    image_id: str | None = None,
) -> None:
    _image_ref_init(
        self,
        uri_or_key=uri_or_key or path or "",
        width=width,
        height=height,
        provenance=dict(provenance or {}),
        path_init=path_init or path,
        image_id_init=image_id_init or image_id,
    )


ImageRef.__init__ = _compat_image_ref_init  # type: ignore[method-assign]


@dataclass(frozen=True)
class Region:
    image: ImageRef
    bbox_xyxy_global: BBox
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "bbox_xyxy_global", _bbox(self.bbox_xyxy_global))

    @property
    def bbox_xyxy(self) -> BBox:
        return self.bbox_xyxy_global

    @property
    def region_id(self) -> str:
        return str(self.provenance.get("region_id") or "")

    @property
    def label(self) -> str:
        return str(self.provenance.get("label") or "")

    @property
    def width(self) -> float:
        return self.bbox_xyxy_global[2] - self.bbox_xyxy_global[0]

    @property
    def height(self) -> float:
        return self.bbox_xyxy_global[3] - self.bbox_xyxy_global[1]


@dataclass(frozen=True)
class RegionSet:
    regions: tuple[Region, ...]
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "regions", tuple(self.regions))

    def __len__(self) -> int:
        return len(self.regions)


@dataclass(frozen=True)
class Entity:
    region: Region
    label: str
    score: float | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def bbox_xyxy(self) -> BBox:
        return self.region.bbox_xyxy_global

    @property
    def entity_id(self) -> str:
        return str(self.provenance.get("entity_id") or self.provenance.get("candidate_id") or "")


@dataclass(frozen=True)
class EntitySet:
    entities: tuple[Entity, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "entities", tuple(self.entities))

    def __len__(self) -> int:
        return len(self.entities)

    def __iter__(self) -> Iterator[Entity]:
        return iter(self.entities)


class SelectStatus(str, Enum):
    OK = "OK"
    EMPTY = "EMPTY"
    AMBIGUOUS = "AMBIGUOUS"
    UNRESOLVED = "UNRESOLVED"
    ERROR = "ERROR"


@dataclass(frozen=True)
class SelectResult:
    selected: EntitySet | Region | RegionSet
    status: SelectStatus
    method: str
    reason: str | None = None
    confidence: float | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScalarInt:
    value: int
    provenance: dict[str, Any] = field(default_factory=dict)


class SelectResultConsumptionError(ValueError):
    """Downstream consumer attempted to use an unsafe SELECT state."""


def unwrap_select_result(
    value: Any,
    *,
    allow_empty: bool,
    require_single: bool = False,
    consumer: str = "downstream",
) -> Any:
    """COUNT/GROUP 允许 EMPTY；单对象消费者拒绝 AMBIGUOUS/UNRESOLVED/ERROR。"""
    if not isinstance(value, SelectResult):
        return value
    if value.status is SelectStatus.EMPTY:
        if not allow_empty:
            raise SelectResultConsumptionError(
                f"{consumer} refuses SELECT status {value.status.value}"
            )
    elif value.status is not SelectStatus.OK:
        raise SelectResultConsumptionError(
            f"{consumer} refuses SELECT status {value.status.value}"
        )
    selected = value.selected
    if not require_single:
        return selected
    if isinstance(selected, EntitySet) and len(selected.entities) == 1:
        return selected.entities[0]
    if isinstance(selected, RegionSet) and len(selected.regions) == 1:
        return selected.regions[0]
    if isinstance(selected, Region):
        return selected
    cardinality = (
        len(selected.entities)
        if isinstance(selected, EntitySet)
        else len(selected.regions)
        if isinstance(selected, RegionSet)
        else 0
    )
    raise SelectResultConsumptionError(
        f"{consumer} requires one selected object, got {cardinality}"
    )


@dataclass(slots=True)
class Detection:
    """COUNT 内部检测框；不进入 TaskGraph 对外接口。"""

    bbox_xyxy_global: BBox
    score: float
    label: str
    tile_id: str = ""
    scale_id: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.bbox_xyxy_global = _bbox(self.bbox_xyxy_global)

    @property
    def center(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bbox_xyxy_global
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)

    def to_entity(self, image: ImageRef, index: int = 0) -> Entity:
        entity_id = self.provenance.get("entity_id") or f"{self.scale_id}:{self.tile_id}:{index}"
        return Entity(
            Region(image, self.bbox_xyxy_global, provenance={"source": "detection"}),
            self.label,
            float(self.score),
            provenance={**dict(self.provenance), "entity_id": entity_id, "tile_id": self.tile_id},
        )


@dataclass(slots=True)
class DetectionSet:
    detections: list[Detection] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.detections)

    def __iter__(self) -> Iterator[Detection]:
        return iter(self.detections)

    def filter_score(self, threshold: float) -> DetectionSet:
        return DetectionSet([d for d in self.detections if d.score >= threshold])

    def to_entity_set(self, image: ImageRef) -> EntitySet:
        return EntitySet(tuple(det.to_entity(image, i) for i, det in enumerate(self.detections)))


@dataclass(slots=True)
class CountResult:
    """内部完整结果；TaskGraph 节点只暴露 to_scalar()。"""

    count: int
    detections: DetectionSet = field(default_factory=DetectionSet)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_scalar(self) -> ScalarInt:
        return ScalarInt(int(self.count), provenance=dict(self.provenance))

    def to_entity_set(self, image: ImageRef) -> EntitySet:
        return self.detections.to_entity_set(image)


VisualInput = ImageRef | Region | EntitySet | SelectResult
RuntimeObject: TypeAlias = ImageRef | Region | RegionSet | Entity | EntitySet | SelectResult | ScalarInt


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        payload = asdict(value)
        payload["__type__"] = type(value).__name__
        return payload
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def detections_from_entities(entities: Iterable[Entity], *, scale_id: str = "given") -> DetectionSet:
    dets: list[Detection] = []
    for i, ent in enumerate(entities):
        dets.append(
            Detection(
                bbox_xyxy_global=ent.bbox_xyxy,
                score=float(ent.score if ent.score is not None else 1.0),
                label=ent.label,
                tile_id="upstream",
                scale_id=scale_id,
                provenance={"source": "entityset", "entity_id": ent.entity_id or str(i)},
            )
        )
    return DetectionSet(dets)
