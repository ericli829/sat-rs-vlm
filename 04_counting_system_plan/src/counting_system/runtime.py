"""TaskGraph 运行时对象：节点间传 typed object，不用字符串串联。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


BBox = tuple[float, float, float, float]


def _as_bbox(value: Sequence[float]) -> BBox:
    if len(value) != 4:
        raise ValueError(f"bbox must have 4 numbers, got {value!r}")
    x0, y0, x1, y1 = (float(v) for v in value)
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return (x0, y0, x1, y1)


@dataclass(slots=True)
class ImageRef:
    path: str
    image_id: str = ""
    width: int | None = None
    height: int | None = None

    def resolved(self) -> Path:
        return Path(self.path)


@dataclass(slots=True)
class Region:
    image: ImageRef
    bbox_xyxy: BBox
    region_id: str = ""
    label: str = ""

    def __post_init__(self) -> None:
        self.bbox_xyxy = _as_bbox(self.bbox_xyxy)

    @property
    def width(self) -> float:
        return self.bbox_xyxy[2] - self.bbox_xyxy[0]

    @property
    def height(self) -> float:
        return self.bbox_xyxy[3] - self.bbox_xyxy[1]


@dataclass(slots=True)
class Entity:
    bbox_xyxy: BBox
    score: float = 1.0
    label: str = ""
    entity_id: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.bbox_xyxy = _as_bbox(self.bbox_xyxy)


@dataclass(slots=True)
class EntitySet:
    entities: list[Entity] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.entities)

    def __iter__(self) -> Iterator[Entity]:
        return iter(self.entities)


@dataclass(slots=True)
class ScalarInt:
    value: int


@dataclass(slots=True)
class Detection:
    bbox_xyxy_global: BBox
    score: float
    label: str
    tile_id: str = ""
    scale_id: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.bbox_xyxy_global = _as_bbox(self.bbox_xyxy_global)

    @property
    def center(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bbox_xyxy_global
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)

    def to_entity(self, index: int = 0) -> Entity:
        return Entity(
            bbox_xyxy=self.bbox_xyxy_global,
            score=self.score,
            label=self.label,
            entity_id=self.provenance.get("entity_id") or f"{self.scale_id}:{self.tile_id}:{index}",
            provenance=dict(self.provenance),
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

    def to_entity_set(self) -> EntitySet:
        return EntitySet([det.to_entity(i) for i, det in enumerate(self.detections)])


@dataclass(slots=True)
class CountResult:
    count: int
    detections: DetectionSet = field(default_factory=DetectionSet)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_scalar(self) -> ScalarInt:
        return ScalarInt(int(self.count))


VisualInput = ImageRef | Region | EntitySet


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        payload = asdict(value)
        payload["__type__"] = type(value).__name__
        return payload
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
                score=float(ent.score),
                label=ent.label,
                tile_id="upstream",
                scale_id=scale_id,
                provenance={"source": "entityset", "entity_id": ent.entity_id or str(i)},
            )
        )
    return DetectionSet(dets)
