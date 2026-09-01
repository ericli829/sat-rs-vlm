"""Deterministic and capability-backed TaskGraph operator executors."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from math import hypot
from statistics import median
from typing import Protocol, cast

from PIL import Image

from .choice_config import ChoiceSystemConfig
from .input_composer import InputComposer
from .providers import (
    ChoiceScoringRequest,
    DetectionProvider,
    DetectionRequest,
    RegionRetrievalRequest,
    RegionRetrieverProvider,
    SemanticVLMProvider,
    VLMRequest,
    parse_selection_indices,
)
from .runtime_types import (
    Answer,
    Boolean,
    Entity,
    EntitySet,
    ImageRef,
    Label,
    LabelSet,
    Region,
    RegionSet,
    RouteContext,
    RuntimeObject,
    ScalarInt,
    SelectResult,
    SelectStatus,
)
from .schema import GraphNode, OperatorName, TargetSpec


@dataclass(frozen=True)
class OperatorOutcome:
    value: RuntimeObject
    provider: str


@dataclass(frozen=True)
class OperatorContext:
    question: str
    choices: tuple[str, ...]
    composer: InputComposer


class OperatorExecutor(Protocol):
    provider_name: str

    def execute(
        self,
        node: GraphNode,
        inputs: dict[str, RuntimeObject | list[RuntimeObject]],
        context: OperatorContext,
    ) -> OperatorOutcome: ...


def _image(value: RuntimeObject) -> ImageRef:
    if isinstance(value, SelectResult):
        return _image(value.selected)
    if isinstance(value, ImageRef):
        return value
    if isinstance(value, Region):
        return value.image
    if isinstance(value, Entity):
        return value.region.image
    if isinstance(value, EntitySet) and value.entities:
        return value.entities[0].region.image
    raise TypeError(f"cannot resolve image from {type(value).__name__}")


def _dimensions(value: ImageRef | Region) -> tuple[int, int]:
    image = value if isinstance(value, ImageRef) else value.image
    if image.width and image.height:
        return image.width, image.height
    with Image.open(image.path.resolve()) as source:
        width, height = source.size
        return int(width), int(height)


def _scope_box(value: ImageRef | Region) -> tuple[float, float, float, float]:
    if isinstance(value, Region):
        return value.bbox_xyxy_global
    width, height = _dimensions(value)
    return 0.0, 0.0, float(width), float(height)


class GeometryExecutor:
    provider_name = "geometry"

    _POSITION: dict[str, tuple[float, float, float, float]] = {
        "TOP": (0.0, 0.0, 1.0, 0.5),
        "BOTTOM": (0.0, 0.5, 1.0, 1.0),
        "LEFT": (0.0, 0.0, 0.5, 1.0),
        "RIGHT": (0.5, 0.0, 1.0, 1.0),
        "CENTER": (0.25, 0.25, 0.75, 0.75),
        "TOP_LEFT": (0.0, 0.0, 0.5, 0.5),
        "TOP_RIGHT": (0.5, 0.0, 1.0, 0.5),
        "BOTTOM_LEFT": (0.0, 0.5, 0.5, 1.0),
        "BOTTOM_RIGHT": (0.5, 0.5, 1.0, 1.0),
        "TOP_CENTER": (0.25, 0.0, 0.75, 0.5),
        "BOTTOM_CENTER": (0.25, 0.5, 0.75, 1.0),
        "CENTER_LEFT": (0.0, 0.25, 0.5, 0.75),
        "CENTER_RIGHT": (0.5, 0.25, 1.0, 0.75),
    }

    def _position_region(self, source: ImageRef | Region, position: str) -> Region:
        image = source if isinstance(source, ImageRef) else source.image
        x0, y0, x1, y1 = _scope_box(source)
        rx0, ry0, rx1, ry1 = self._POSITION[position]
        width, height = x1 - x0, y1 - y0
        return Region(
            image,
            (x0 + rx0 * width, y0 + ry0 * height, x0 + rx1 * width, y0 + ry1 * height),
            {"operator": "REGION", "position": position},
        )

    @staticmethod
    def _marker(source: ImageRef | Region, color: str | None) -> RegionSet:
        image = source if isinstance(source, ImageRef) else source.image
        scope = _scope_box(source)
        marker_color = (color or "red").casefold()

        def matches(pixel: tuple[int, int, int]) -> bool:
            red, green, blue = pixel
            if marker_color == "green":
                return green >= 140 and green >= red + 40 and green >= blue + 40
            if marker_color == "blue":
                return blue >= 140 and blue >= red + 40 and blue >= green + 40
            if marker_color == "yellow":
                return red >= 150 and green >= 150 and blue <= 130
            return red >= 160 and red >= green + 60 and red >= blue + 60

        with Image.open(image.path.resolve()) as raw:
            crop = raw.convert("RGB").crop(scope)
            pixels = crop.load()
            hits: set[tuple[int, int]] = set()
            for y in range(crop.height):
                for x in range(crop.width):
                    if matches(pixels[x, y]):
                        hits.add((x, y))
        if not hits:
            return RegionSet((), {"operator": "FIND_MARKER", "color": color})

        components: list[set[tuple[int, int]]] = []
        remaining = set(hits)
        while remaining:
            seed = min(remaining, key=lambda point: (point[1], point[0]))
            remaining.remove(seed)
            component = {seed}
            stack = [seed]
            while stack:
                x, y = stack.pop()
                for delta_y in (-1, 0, 1):
                    for delta_x in (-1, 0, 1):
                        neighbor = (x + delta_x, y + delta_y)
                        if neighbor in remaining:
                            remaining.remove(neighbor)
                            component.add(neighbor)
                            stack.append(neighbor)
            components.append(component)

        offset_x, offset_y = scope[0], scope[1]
        regions = tuple(
            Region(
                image,
                (
                    min(point[0] for point in component) + offset_x,
                    min(point[1] for point in component) + offset_y,
                    max(point[0] for point in component) + offset_x + 1,
                    max(point[1] for point in component) + offset_y + 1,
                ),
                {
                    "operator": "FIND_MARKER",
                    "color": marker_color,
                    "component_pixels": len(component),
                },
            )
            for component in sorted(
                components,
                key=lambda item: (
                    min(point[1] for point in item),
                    min(point[0] for point in item),
                ),
            )
        )
        return RegionSet(
            regions,
            {"operator": "FIND_MARKER", "color": marker_color, "component_count": len(regions)},
        )

    @staticmethod
    def _group(entities: EntitySet, mode: str) -> EntitySet:
        if not entities.entities:
            return EntitySet((), {**entities.provenance, "group": mode, "groups": []})
        indexed = list(enumerate(entities.entities))

        def center(item: tuple[int, Entity]) -> tuple[float, float]:
            box = item[1].region.bbox_xyxy_global
            return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0

        widths = [
            item.region.bbox_xyxy_global[2] - item.region.bbox_xyxy_global[0]
            for item in entities.entities
        ]
        heights = [
            item.region.bbox_xyxy_global[3] - item.region.bbox_xyxy_global[1]
            for item in entities.entities
        ]
        groups: list[list[tuple[int, Entity]]] = []
        if mode in {"ROW", "COLUMN"}:
            axis = 1 if mode == "ROW" else 0
            tolerance = 0.75 * median(heights if mode == "ROW" else widths)
            for item in sorted(
                indexed, key=lambda value: (center(value)[axis], center(value)[1 - axis])
            ):
                coordinate = center(item)[axis]
                if not groups:
                    groups.append([item])
                    continue
                group_coordinate = sum(center(member)[axis] for member in groups[-1]) / len(
                    groups[-1]
                )
                if abs(coordinate - group_coordinate) <= tolerance:
                    groups[-1].append(item)
                else:
                    groups.append([item])
            secondary_axis = 0 if mode == "ROW" else 1
            for group in groups:
                group.sort(key=lambda value: center(value)[secondary_axis])
        else:
            threshold = 2.5 * median(
                hypot(width, height) for width, height in zip(widths, heights, strict=True)
            )
            remaining = set(range(len(indexed)))
            while remaining:
                seed = min(remaining)
                remaining.remove(seed)
                member_indices = [seed]
                stack = [seed]
                while stack:
                    current = stack.pop()
                    current_center = center(indexed[current])
                    neighbors = [
                        other
                        for other in sorted(remaining)
                        if hypot(
                            current_center[0] - center(indexed[other])[0],
                            current_center[1] - center(indexed[other])[1],
                        )
                        <= threshold
                    ]
                    for other in neighbors:
                        remaining.remove(other)
                        member_indices.append(other)
                        stack.append(other)
                groups.append([indexed[item] for item in member_indices])
            groups.sort(
                key=lambda group: (
                    min(center(item)[1] for item in group),
                    min(center(item)[0] for item in group),
                )
            )
            for group in groups:
                group.sort(key=lambda item: (center(item)[1], center(item)[0]))

        ordered: list[Entity] = []
        group_metadata: list[dict[str, object]] = []
        for group_id, group in enumerate(groups):
            member_indices = [index for index, _ in group]
            boxes = [entity.region.bbox_xyxy_global for _, entity in group]
            group_metadata.append(
                {
                    "group_id": group_id,
                    "source_indices": member_indices,
                    "bbox_xyxy_global": [
                        min(box[0] for box in boxes),
                        min(box[1] for box in boxes),
                        max(box[2] for box in boxes),
                        max(box[3] for box in boxes),
                    ],
                }
            )
            ordered.extend(
                replace(
                    entity,
                    provenance={**entity.provenance, "group_id": group_id, "group_mode": mode},
                )
                for _, entity in group
            )
        return EntitySet(
            tuple(ordered),
            {
                **entities.provenance,
                "group": mode,
                "group_count": len(groups),
                "groups": group_metadata,
            },
        )

    @staticmethod
    def _resolve_route_endpoint(
        value: RuntimeObject, role: str
    ) -> tuple[Entity | Region, dict[str, object]]:
        if isinstance(value, (Entity, Region)):
            return value, {"policy": "single", "selected_index": 0}
        if not isinstance(value, EntitySet):
            raise TypeError(f"BUILD_ROUTE_CONTEXT.{role} must be Entity, EntitySet, or Region")
        if not value.entities:
            raise ValueError(f"BUILD_ROUTE_CONTEXT.{role} EntitySet is empty")
        if len(value.entities) == 1:
            return value.entities[0], {"policy": "single", "selected_index": 0}
        scored = [
            (index, entity)
            for index, entity in enumerate(value.entities)
            if entity.score is not None
        ]
        if not scored:
            raise ValueError(f"BUILD_ROUTE_CONTEXT.{role} is ambiguous: multiple unscored entities")
        selected_index, selected = max(scored, key=lambda item: (item[1].score, -item[0]))
        return selected, {
            "policy": "highest_score",
            "selected_index": selected_index,
            "selected_score": selected.score,
        }

    @staticmethod
    def _route_context(
        image_value: RuntimeObject, start: RuntimeObject, goal: RuntimeObject
    ) -> RouteContext:
        image = _image(image_value)
        selected_start, start_selection = GeometryExecutor._resolve_route_endpoint(start, "start")
        selected_goal, goal_selection = GeometryExecutor._resolve_route_endpoint(goal, "goal")
        start_region = (
            selected_start.region if isinstance(selected_start, Entity) else selected_start
        )
        goal_region = selected_goal.region if isinstance(selected_goal, Entity) else selected_goal
        if (
            start_region.image.uri_or_key != image.uri_or_key
            or goal_region.image.uri_or_key != image.uri_or_key
        ):
            raise ValueError("route endpoints must reference BUILD_ROUTE_CONTEXT.image")
        width, height = _dimensions(image)
        x0 = max(0.0, min(start_region.bbox_xyxy_global[0], goal_region.bbox_xyxy_global[0]))
        y0 = max(0.0, min(start_region.bbox_xyxy_global[1], goal_region.bbox_xyxy_global[1]))
        x1 = min(
            float(width), max(start_region.bbox_xyxy_global[2], goal_region.bbox_xyxy_global[2])
        )
        y1 = min(
            float(height), max(start_region.bbox_xyxy_global[3], goal_region.bbox_xyxy_global[3])
        )
        pad = max(8.0, 0.1 * max(x1 - x0, y1 - y0))
        context_box = (
            max(0.0, x0 - pad),
            max(0.0, y0 - pad),
            min(width, x1 + pad),
            min(height, y1 + pad),
        )
        context_region = Region(
            image,
            context_box,
            {"coordinate_transform": {"origin_global": list(context_box[:2])}},
        )
        return RouteContext(
            image=image,
            start=selected_start,
            goal=selected_goal,
            context_region=context_region,
            provenance={
                "provider": "geometry",
                "ambiguity_policy": "single_or_highest_score_else_error",
                "start_selection": start_selection,
                "goal_selection": goal_selection,
            },
        )

    def execute(
        self,
        node: GraphNode,
        inputs: dict[str, RuntimeObject | list[RuntimeObject]],
        context: OperatorContext,
    ) -> OperatorOutcome:
        if node.op is OperatorName.REGION:
            source = inputs["image"]
            if not isinstance(source, (ImageRef, Region)):
                raise TypeError("REGION.image must be ImageRef or Region")
            value: RuntimeObject = self._position_region(source, str(node.params["position"]))
        elif node.op is OperatorName.REGION_FROM_BBOX:
            image = inputs["image"]
            if not isinstance(image, ImageRef):
                raise TypeError("REGION_FROM_BBOX.image must be ImageRef")
            width, height = _dimensions(image)
            source_size = node.params.get("image_size")
            bbox = tuple(float(item) for item in node.params["bbox"])
            scale_x = scale_y = 1.0
            if source_size is not None:
                scale_x = width / float(source_size[0])
                scale_y = height / float(source_size[1])
                bbox = (
                    bbox[0] * scale_x,
                    bbox[1] * scale_y,
                    bbox[2] * scale_x,
                    bbox[3] * scale_y,
                )
            bbox = (
                max(0.0, min(float(width), bbox[0])),
                max(0.0, min(float(height), bbox[1])),
                max(0.0, min(float(width), bbox[2])),
                max(0.0, min(float(height), bbox[3])),
            )
            value = Region(
                image,
                bbox,
                {
                    "operator": node.op.value,
                    "source_image_size": list(source_size) if source_size is not None else None,
                    "actual_image_size": [width, height],
                    "scale_xy": [scale_x, scale_y],
                },
            )
        elif node.op is OperatorName.FIND_MARKER:
            source = inputs["image"]
            if not isinstance(source, (ImageRef, Region)):
                raise TypeError("FIND_MARKER.image must be ImageRef or Region")
            value = self._marker(source, node.params["marker"].get("color"))
        elif node.op is OperatorName.GROUP:
            entities = inputs["entities"]
            if not isinstance(entities, EntitySet):
                raise TypeError("GROUP.entities must be EntitySet")
            value = self._group(entities, str(node.params["mode"]))
        elif node.op is OperatorName.ABS_DIFF:
            left, right = inputs["a"], inputs["b"]
            if not isinstance(left, ScalarInt) or not isinstance(right, ScalarInt):
                raise TypeError("ABS_DIFF requires ScalarInt a and b")
            value = ScalarInt(abs(left.value - right.value), {"provider": self.provider_name})
        elif node.op is OperatorName.BUILD_ROUTE_CONTEXT:
            image_value, start, goal = inputs["image"], inputs["start"], inputs["goal"]
            if isinstance(image_value, list) or isinstance(start, list) or isinstance(goal, list):
                raise TypeError("BUILD_ROUTE_CONTEXT inputs must be named single refs")
            value = self._route_context(image_value, start, goal)
        else:
            raise ValueError(f"GeometryExecutor cannot execute {node.op.value}")
        return OperatorOutcome(value, self.provider_name)


class LocateExecutor:
    def __init__(
        self,
        detection: DetectionProvider,
        retriever: RegionRetrieverProvider,
        *,
        semantic_categories: set[str] | None = None,
    ) -> None:
        self.detection = detection
        self.retriever = retriever
        self.semantic_categories = {item.casefold() for item in (semantic_categories or set())}
        self.provider_name = "locate"

    def execute(
        self,
        node: GraphNode,
        inputs: dict[str, RuntimeObject | list[RuntimeObject]],
        context: OperatorContext,
    ) -> OperatorOutcome:
        scope = inputs["image"]
        if not isinstance(scope, (ImageRef, Region)):
            raise TypeError("LOCATE.image must be ImageRef or Region")
        target = TargetSpec.model_validate(node.params["target"])
        if target.category.casefold() in self.semantic_categories:
            result = self.retriever.retrieve(
                RegionRetrievalRequest(
                    scope,
                    target.phrase(),
                    search_scope=scope if isinstance(scope, Region) else None,
                    max_candidates=8,
                )
            )
            entities = EntitySet(
                tuple(
                    Entity(item.region, target.category, item.relevance_score, item.provenance)
                    for item in result.candidates
                ),
                {"provider": result.provider, "capability": "region_retrieval"},
            )
            return OperatorOutcome(entities, result.provider)
        detected = self.detection.detect(DetectionRequest(scope, target, "LOCATE"))
        return OperatorOutcome(detected.detections, detected.provider)


class CountExecutor:
    def __init__(self, detection: DetectionProvider) -> None:
        self.detection = detection
        self.provider_name = detection.provider_name

    def execute(
        self,
        node: GraphNode,
        inputs: dict[str, RuntimeObject | list[RuntimeObject]],
        context: OperatorContext,
    ) -> OperatorOutcome:
        if "entities" in inputs:
            entities = inputs["entities"]
            if isinstance(entities, SelectResult):
                if entities.status in {
                    SelectStatus.UNRESOLVED,
                    SelectStatus.ERROR,
                    SelectStatus.AMBIGUOUS,
                }:
                    raise ValueError(
                        f"COUNT refuses SELECT status {entities.status.value}; "
                        "resolve the selection in the graph before counting"
                    )
                entities = entities.selected
            if not isinstance(entities, EntitySet):
                raise TypeError("COUNT.entities must be EntitySet")
            return OperatorOutcome(
                ScalarInt(
                    len(entities.entities), {"provider": "cardinality", "source": "EntitySet"}
                ),
                "cardinality",
            )
        scope = inputs.get("image")
        if not isinstance(scope, (ImageRef, Region)):
            raise TypeError("COUNT requires image/Region or EntitySet")
        target = TargetSpec.model_validate(node.params["target"])
        detected = self.detection.detect(DetectionRequest(scope, target, "COUNT"))
        return OperatorOutcome(
            ScalarInt(
                len(detected.detections.entities),
                {"provider": detected.provider, "detection": detected.detections.provenance},
            ),
            detected.provider,
        )


class SelectExecutor:
    def __init__(
        self,
        semantic: SemanticVLMProvider,
        choice_config: ChoiceSystemConfig | None = None,
    ) -> None:
        self.semantic = semantic
        self.provider_name = semantic.provider_name
        self.choice_config = choice_config or ChoiceSystemConfig()

    @staticmethod
    def _sort_entities(entities: tuple[Entity, ...], order: str) -> list[Entity]:
        if order in {"TOP_TO_BOTTOM", "BOTTOM_TO_TOP"}:

            def key(item: Entity) -> tuple[float, float]:
                return item.region.bbox_xyxy_global[1], item.region.bbox_xyxy_global[0]

            reverse = order == "BOTTOM_TO_TOP"
        else:

            def key(item: Entity) -> tuple[float, float]:
                return item.region.bbox_xyxy_global[0], item.region.bbox_xyxy_global[1]

            reverse = order == "RIGHT_TO_LEFT"
        return sorted(entities, key=key, reverse=reverse)

    @staticmethod
    def _empty_like(value: EntitySet | Region | RegionSet) -> EntitySet | RegionSet:
        if isinstance(value, EntitySet):
            return EntitySet((), {"select_empty": True})
        return RegionSet((), {"select_empty": True})

    @staticmethod
    def _items(value: EntitySet | Region | RegionSet) -> tuple[Entity | Region, ...]:
        if isinstance(value, EntitySet):
            return value.entities
        if isinstance(value, RegionSet):
            return value.regions
        return (value,)

    @staticmethod
    def _item_region(item: Entity | Region) -> Region:
        return item.region if isinstance(item, Entity) else item

    @staticmethod
    def _candidate_ids(items: tuple[Entity | Region, ...], indices: tuple[int, ...]) -> list[str]:
        return [str(items[index].provenance.get("candidate_id")) for index in indices]

    @staticmethod
    def _ordinal_key(item: Entity, order: str) -> tuple[float, float]:
        box = item.region.bbox_xyxy_global
        if order in {"TOP_TO_BOTTOM", "BOTTOM_TO_TOP"}:
            return box[1], box[0]
        return box[0], box[1]

    @staticmethod
    def _rank_value(item: Entity, criterion: str) -> float:
        if "score" in criterion:
            return item.score or 0.0
        box = item.region.bbox_xyxy_global
        return (box[2] - box[0]) * (box[3] - box[1])

    @staticmethod
    def _selected_like(
        source: EntitySet | Region | RegionSet,
        indices: tuple[int, ...],
        *,
        provenance: dict[str, object],
    ) -> EntitySet | Region | RegionSet:
        items = SelectExecutor._items(source)
        selected = tuple(items[index] for index in indices)
        if isinstance(source, EntitySet):
            return EntitySet(
                tuple(item for item in selected if isinstance(item, Entity)), provenance
            )
        if isinstance(source, RegionSet):
            return RegionSet(
                tuple(item for item in selected if isinstance(item, Region)), provenance
            )
        if selected:
            return cast(Region, selected[0])  # A singleton Region is its own selected value.
        return RegionSet((), provenance)

    @staticmethod
    def _with_candidate_ids(
        value: EntitySet | Region | RegionSet,
    ) -> EntitySet | Region | RegionSet:
        """Guarantee a stable ID at the SELECT boundary without changing Entity."""

        def region_with_id(region: Region, index: int) -> Region:
            provenance = dict(region.provenance)
            provenance.setdefault("candidate_id", f"candidate_{index + 1:04d}")
            return replace(region, provenance=provenance)

        if isinstance(value, EntitySet):
            entities = []
            for index, entity in enumerate(value.entities):
                provenance = dict(entity.provenance)
                provenance.setdefault("candidate_id", f"candidate_{index + 1:04d}")
                entities.append(replace(entity, provenance=provenance))
            return EntitySet(tuple(entities), dict(value.provenance))
        if isinstance(value, RegionSet):
            return RegionSet(
                tuple(region_with_id(region, index) for index, region in enumerate(value.regions)),
                dict(value.provenance),
            )
        return region_with_id(value, 0)

    @staticmethod
    def _unwrap(value: RuntimeObject, *, role: str) -> EntitySet | Region | RegionSet:
        if isinstance(value, SelectResult):
            if value.status in {
                SelectStatus.UNRESOLVED,
                SelectStatus.ERROR,
                SelectStatus.AMBIGUOUS,
            }:
                raise ValueError(
                    f"SELECT.{role} has unresolved upstream status {value.status.value}"
                )
            return value.selected
        if isinstance(value, (EntitySet, Region, RegionSet)):
            return value
        raise TypeError(f"SELECT.{role} must be EntitySet, Region, or RegionSet")

    @staticmethod
    def _single_reference(value: RuntimeObject | None) -> Region | None:
        if value is None:
            return None
        if isinstance(value, SelectResult):
            if value.status is not SelectStatus.OK:
                return None
            value = value.selected
        if isinstance(value, Entity):
            return value.region
        if isinstance(value, Region):
            return value
        if isinstance(value, EntitySet) and len(value.entities) == 1:
            return value.entities[0].region
        if isinstance(value, RegionSet) and len(value.regions) == 1:
            return value.regions[0]
        return None

    @staticmethod
    def _box_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
        return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0

    @staticmethod
    def _iou(
        left: tuple[float, float, float, float], right: tuple[float, float, float, float]
    ) -> float:
        intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
        intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
        intersection = intersection_width * intersection_height
        if not intersection:
            return 0.0
        left_area = (left[2] - left[0]) * (left[3] - left[1])
        right_area = (right[2] - right[0]) * (right[3] - right[1])
        return intersection / (left_area + right_area - intersection)

    @staticmethod
    def _intersects(
        left: tuple[float, float, float, float], right: tuple[float, float, float, float]
    ) -> bool:
        return max(left[0], right[0]) < min(left[2], right[2]) and max(left[1], right[1]) < min(
            left[3], right[3]
        )

    @staticmethod
    def _contains(
        outer: tuple[float, float, float, float], inner: tuple[float, float, float, float]
    ) -> bool:
        return (
            outer[0] <= inner[0]
            and outer[1] <= inner[1]
            and outer[2] >= inner[2]
            and outer[3] >= inner[3]
        )

    @staticmethod
    def _default_margin(scope: ImageRef | Region, requested: object | None) -> float:
        if requested is not None:
            if not isinstance(requested, (int, float)):
                raise TypeError("SELECT margin must be numeric")
            return float(requested)
        width, height = _dimensions(scope)
        return max(4.0, min(width, height) * 0.02)

    @staticmethod
    def _scope_for(
        inputs: dict[str, RuntimeObject | list[RuntimeObject]],
        candidates: EntitySet | Region | RegionSet,
        reference: Region | None,
    ) -> ImageRef | Region:
        scope = inputs.get("scope")
        if isinstance(scope, (ImageRef, Region)):
            return scope
        if isinstance(candidates, Region):
            return candidates
        if reference is not None:
            return reference.image
        items = SelectExecutor._items(candidates)
        if items:
            return SelectExecutor._item_region(items[0]).image
        raise ValueError("SELECT cannot infer a scope from empty candidates without scope input")

    def _result(
        self,
        selected: EntitySet | Region | RegionSet,
        *,
        status: SelectStatus,
        method: str,
        reason: str | None = None,
        confidence: float | None = None,
        provenance: dict[str, object] | None = None,
    ) -> OperatorOutcome:
        return OperatorOutcome(
            SelectResult(selected, status, method, reason, confidence, dict(provenance or {})),
            method,
        )

    def _semantic_select(
        self,
        candidates: EntitySet | Region | RegionSet,
        reference: RuntimeObject | None,
        relation: str,
        node: GraphNode,
        context: OperatorContext,
        provenance: dict[str, object],
    ) -> OperatorOutcome:
        items = self._items(candidates)
        named: dict[str, RuntimeObject | list[RuntimeObject]] = {"candidates": candidates}
        if reference is not None:
            named["reference"] = reference
        model_input = context.composer.compose_named(
            named,
            question=(
                f"Determine which candidate objects satisfy relation {relation} "
                "relative to the marked reference."
            ),
        )
        candidate_mapping = model_input.metadata.get("candidate_mapping")
        if not isinstance(candidate_mapping, dict) or not candidate_mapping:
            return self._result(
                self._empty_like(candidates),
                status=SelectStatus.ERROR,
                method="qwen3_vl_kv_cached_choice",
                reason="semantic selection canvas has no stable candidate mapping",
                provenance=provenance,
            )
        choice_ids = tuple(str(choice_id) for choice_id in candidate_mapping)
        selection_type = getattr(node.params.get("selection_type"), "value", None) or str(
            node.params.get("selection_type") or "MULTI"
        )
        answer_type = "CHOICE_SINGLE" if selection_type == "SINGLE" else "CHOICE_MULTI"
        try:
            scored = self.semantic.reason_and_choose(
                ChoiceScoringRequest(
                    model_input=model_input,
                    answer_type=answer_type,
                    choice_ids=choice_ids,
                    option_texts=tuple(
                        f"Candidate {choice_id}: "
                        f"{item.label if isinstance(item, Entity) else 'region'}"
                        for choice_id, item in zip(choice_ids, items, strict=True)
                    ),
                    final_suffix=self.choice_config.final_suffix,
                    multi_select_threshold=self.choice_config.multi_select_threshold,
                    purpose="select_relation",
                )
            )
            indices: tuple[int, ...] = tuple(
                int(candidate_mapping[choice_id]["index"])
                for choice_id in scored.selected_ids
                if isinstance(candidate_mapping.get(choice_id), dict)
                and isinstance(candidate_mapping[choice_id].get("index"), int)
            )
            if len(indices) != len(scored.selected_ids):
                raise RuntimeError("choice score returned an invalid SELECT candidate id")
            selected = self._selected_like(
                candidates, indices, provenance={"provider": scored.provider}
            )
            status = SelectStatus.OK if indices else SelectStatus.EMPTY
            stable_candidate_ids = self._candidate_ids(items, indices)
            return self._result(
                selected,
                status=status,
                method="qwen3_vl_kv_cached_choice",
                provenance={
                    **provenance,
                    "provider": scored.provider,
                    "candidate_ids": stable_candidate_ids,
                    "choice_labels": list(scored.selected_ids),
                    "scores": dict(scored.scores),
                    "score_method": scored.method,
                    "cache_reused": scored.cache_reused,
                    "latency_ms": dict(scored.latency_ms),
                    "reasoning_text": (
                        scored.reasoning_text
                        if self.choice_config.preserve_reasoning_text
                        else None
                    ),
                    "choice_metadata": dict(scored.metadata),
                },
            )
        except Exception as exc:
            # The finite-output mask is retained strictly as a compatibility
            # fallback for a backend that cannot expose a valid KV cache.
            try:
                result = self.semantic.infer(VLMRequest(model_input, "selection"))
                if result.text.strip().casefold() in {"none", "no", "empty", "null"}:
                    return self._result(
                        self._empty_like(candidates),
                        status=SelectStatus.EMPTY,
                        method="qwen3_vl_token_mask_fallback",
                        confidence=result.confidence,
                        provenance={
                            **provenance,
                            "provider": result.provider,
                            "fallback_reason": str(exc),
                        },
                    )
                indices = parse_selection_indices(result.text, len(items))
            except Exception as fallback_exc:
                return self._result(
                    self._empty_like(candidates),
                    status=SelectStatus.UNRESOLVED,
                    method="qwen3_vl_kv_cached_choice",
                    reason=f"semantic_selection_unresolved: {fallback_exc}",
                    provenance={**provenance, "cached_choice_error": str(exc)},
                )
            selected = self._selected_like(
                candidates, indices, provenance={"provider": result.provider}
            )
            status = SelectStatus.OK if indices else SelectStatus.EMPTY
            candidate_ids = self._candidate_ids(items, indices)
            return self._result(
                selected,
                status=status,
                method="qwen3_vl_token_mask_fallback",
                confidence=result.confidence,
                provenance={
                    **provenance,
                    "provider": result.provider,
                    "raw_response": result.text,
                    "candidate_ids": candidate_ids,
                    "fallback_reason": str(exc),
                },
            )

    def _direct_relation(
        self,
        candidates: EntitySet | Region | RegionSet,
        reference: Region,
        relation: str,
        *,
        margin: float,
        overlap_iou_threshold: float,
    ) -> tuple[tuple[int, ...], bool]:
        """Return clear matches plus whether a boundary case needs semantic fallback."""
        ref_box = reference.bbox_xyxy_global
        ref_x, ref_y = self._box_center(ref_box)
        selected: list[int] = []
        grey = False
        for index, item in enumerate(self._items(candidates)):
            box = self._item_region(item).bbox_xyxy_global
            x, y = self._box_center(box)
            if relation == "LEFT_OF":
                selected.extend([index] if x < ref_x - margin else [])
                grey = grey or abs(x - ref_x) <= margin
            elif relation == "RIGHT_OF":
                selected.extend([index] if x > ref_x + margin else [])
                grey = grey or abs(x - ref_x) <= margin
            elif relation == "ABOVE":
                selected.extend([index] if y < ref_y - margin else [])
                grey = grey or abs(y - ref_y) <= margin
            elif relation == "BELOW":
                selected.extend([index] if y > ref_y + margin else [])
                grey = grey or abs(y - ref_y) <= margin
            elif relation == "INSIDE":
                if self._contains(ref_box, box):
                    selected.append(index)
                elif self._intersects(ref_box, box):
                    grey = True
            elif relation == "OVERLAP":
                iou = self._iou(box, ref_box)
                if iou >= overlap_iou_threshold:
                    selected.append(index)
                elif iou > 0.0:
                    grey = True
            else:
                return (), True
        return tuple(selected), grey

    @staticmethod
    def _clip_box(
        box: tuple[float, float, float, float], scope: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        clipped = (
            max(scope[0], box[0]),
            max(scope[1], box[1]),
            min(scope[2], box[2]),
            min(scope[3], box[3]),
        )
        if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
            raise ValueError("requested subregion has no positive area inside current scope")
        return clipped

    def _subregion(
        self,
        inputs: dict[str, RuntimeObject | list[RuntimeObject]],
        candidates: EntitySet | Region | RegionSet,
        node: GraphNode,
    ) -> OperatorOutcome:
        input_reference = inputs.get("reference")
        reference = self._single_reference(
            input_reference if not isinstance(input_reference, list) else None
        )
        if reference is None:
            return self._result(
                self._empty_like(candidates),
                status=SelectStatus.UNRESOLVED,
                method="geometry",
                reason="SUBREGION requires one reference",
            )
        scope = self._scope_for(inputs, candidates, reference)
        scope_box = _scope_box(scope)
        margin = self._default_margin(scope, node.params.get("margin"))
        bx0, by0, bx1, by1 = reference.bbox_xyxy_global
        subregion = str(node.params["subregion"])
        mapping = {
            "LEFT_SIDE": (scope_box[0], scope_box[1], bx0 + margin, scope_box[3]),
            "RIGHT_SIDE": (bx1 - margin, scope_box[1], scope_box[2], scope_box[3]),
            "ABOVE": (scope_box[0], scope_box[1], scope_box[2], by0 + margin),
            "BELOW": (scope_box[0], by1 - margin, scope_box[2], scope_box[3]),
            "INSIDE": reference.bbox_xyxy_global,
            "AROUND": (bx0 - margin, by0 - margin, bx1 + margin, by1 + margin),
        }
        if subregion not in mapping:
            return self._result(
                self._empty_like(candidates),
                status=SelectStatus.UNRESOLVED,
                method="geometry",
                reason=f"SUBREGION {subregion} is outside SELECT v1 scope",
            )
        try:
            region = Region(
                scope.image if isinstance(scope, Region) else scope,
                self._clip_box(mapping[subregion], scope_box),
                {
                    "coordinate_system": "bbox_xyxy_global",
                    "select_mode": "SUBREGION",
                    "subregion": subregion,
                    "scope_bbox_xyxy_global": list(scope_box),
                    "reference_bbox_xyxy_global": list(reference.bbox_xyxy_global),
                    "margin_px": margin,
                },
            )
        except ValueError as exc:
            return self._result(
                self._empty_like(candidates),
                status=SelectStatus.UNRESOLVED,
                method="geometry",
                reason=str(exc),
            )
        return self._result(
            region, status=SelectStatus.OK, method="geometry", provenance=dict(region.provenance)
        )

    def execute(
        self,
        node: GraphNode,
        inputs: dict[str, RuntimeObject | list[RuntimeObject]],
        context: OperatorContext,
    ) -> OperatorOutcome:
        raw_candidates = inputs["candidates"]
        if isinstance(raw_candidates, list):
            raise TypeError("SELECT.candidates does not allow reference lists")
        candidates = self._with_candidate_ids(self._unwrap(raw_candidates, role="candidates"))
        mode = str(node.params["mode"])
        items = self._items(candidates)
        if not items:
            return self._result(
                self._empty_like(candidates),
                status=SelectStatus.EMPTY,
                method="geometry",
                reason="empty_candidates",
            )
        if mode == "SUBREGION":
            return self._subregion(inputs, candidates, node)
        if isinstance(candidates, EntitySet):
            entities = candidates.entities
            if mode == "ORDINAL":
                order = str(node.params["order"])
                ordered = self._sort_entities(entities, order)
                index = int(node.params["index"]) - 1
                if index >= len(ordered):
                    return self._result(
                        self._empty_like(candidates),
                        status=SelectStatus.UNRESOLVED,
                        method="geometry",
                        reason="ordinal_out_of_range",
                    )
                selected_key = self._ordinal_key(ordered[index], order)
                selected = tuple(
                    entity for entity in ordered if self._ordinal_key(entity, order) == selected_key
                )
                status = SelectStatus.AMBIGUOUS if len(selected) > 1 else SelectStatus.OK
                return self._result(
                    EntitySet(tuple(selected), {"select": mode}),
                    status=status,
                    method="geometry",
                    reason="ordinal_tie" if status is SelectStatus.AMBIGUOUS else None,
                )
            if mode == "EXTREME":
                direction = str(node.params["direction"])
                order = {
                    "LEFTMOST": "LEFT_TO_RIGHT",
                    "RIGHTMOST": "RIGHT_TO_LEFT",
                    "TOPMOST": "TOP_TO_BOTTOM",
                    "BOTTOMMOST": "BOTTOM_TO_TOP",
                }[direction]
                ordered = self._sort_entities(entities, order)
                best_key = self._ordinal_key(ordered[0], order)[0]
                selected = tuple(
                    entity for entity in ordered if self._ordinal_key(entity, order)[0] == best_key
                )
                status = SelectStatus.AMBIGUOUS if len(selected) > 1 else SelectStatus.OK
                return self._result(
                    EntitySet(selected, {"select": mode}),
                    status=status,
                    method="geometry",
                    reason="extreme_tie" if status is SelectStatus.AMBIGUOUS else None,
                )
            if mode == "RANK":
                criterion = str(node.params["criterion"]).casefold()
                ranked = sorted(entities, key=lambda item: self._rank_value(item, criterion))
                if str(node.params["order"]) == "DESCENDING":
                    ranked.reverse()
                index = int(node.params["rank"]) - 1
                if index >= len(ranked):
                    return self._result(
                        self._empty_like(candidates),
                        status=SelectStatus.UNRESOLVED,
                        method="geometry",
                        reason="rank_out_of_range",
                    )
                selected_value = self._rank_value(ranked[index], criterion)
                selected = tuple(
                    entity
                    for entity in ranked
                    if self._rank_value(entity, criterion) == selected_value
                )
                status = SelectStatus.AMBIGUOUS if len(selected) > 1 else SelectStatus.OK
                return self._result(
                    EntitySet(selected, {"select": mode}),
                    status=status,
                    method="geometry",
                    reason="rank_tie" if status is SelectStatus.AMBIGUOUS else None,
                )
        if mode == "RELATION":
            raw_reference = inputs.get("reference")
            reference_value = raw_reference if not isinstance(raw_reference, list) else None
            reference = self._single_reference(reference_value)
            relation = str(node.params["relation"])
            if reference is None:
                # A plural upstream reference is still a valid visual context
                # for fuzzy relations.  It is not valid for a deterministic
                # one-to-one geometry calculation.
                if (
                    relation in {"NEAR", "NEXT_TO", "AROUND", "BETWEEN"}
                    and reference_value is not None
                ):
                    return self._semantic_select(
                        candidates,
                        reference_value,
                        relation,
                        node,
                        context,
                        {
                            "relation": relation,
                            "coordinate_system": "bbox_xyxy_global",
                            "reference_cardinality": "multiple",
                        },
                    )
                return self._result(
                    self._empty_like(candidates),
                    status=SelectStatus.UNRESOLVED,
                    method="geometry",
                    reason="RELATION requires exactly one reference",
                )
            scope = self._scope_for(inputs, candidates, reference)
            margin = self._default_margin(scope, node.params.get("margin"))
            threshold = float(node.params.get("overlap_iou_threshold") or 0.10)
            indices, needs_semantic = self._direct_relation(
                candidates, reference, relation, margin=margin, overlap_iou_threshold=threshold
            )
            provenance = {
                "relation": relation,
                "coordinate_system": "bbox_xyxy_global",
                "margin_px": margin,
                "overlap_iou_threshold": threshold,
            }
            if needs_semantic:
                return self._semantic_select(
                    candidates, reference_value, relation, node, context, provenance
                )
            relation_selected = self._selected_like(
                candidates, indices, provenance={"select": "RELATION", **provenance}
            )
            status = SelectStatus.OK if indices else SelectStatus.EMPTY
            return self._result(
                relation_selected,
                status=status,
                method="geometry",
                provenance={
                    **provenance,
                    "candidate_ids": self._candidate_ids(self._items(candidates), indices),
                },
            )
        return self._result(
            self._empty_like(candidates),
            status=SelectStatus.UNRESOLVED,
            method="geometry",
            reason=f"unsupported SELECT mode {mode}",
        )


class SemanticExecutor:
    def __init__(
        self,
        provider: SemanticVLMProvider,
        *,
        provider_name: str | None = None,
        choice_config: ChoiceSystemConfig | None = None,
    ) -> None:
        self.provider = provider
        self.provider_name = provider_name or provider.provider_name
        self.choice_config = choice_config or ChoiceSystemConfig()

    @staticmethod
    def _label_set(text: str) -> tuple[str, ...]:
        return tuple(item.strip() for item in re.split(r"[,;\n]", text) if item.strip())

    def execute(
        self,
        node: GraphNode,
        inputs: dict[str, RuntimeObject | list[RuntimeObject]],
        context: OperatorContext,
    ) -> OperatorOutcome:
        if node.op is OperatorName.ATTRIBUTE:
            question = f"Determine the {node.params['attribute']} of the selected object."
        elif node.op is OperatorName.CLASSIFY:
            question = "Classify the selected visual source."
        elif node.op is OperatorName.MULTILABEL_CLASSIFY:
            question = "Return all applicable labels."
        elif node.op is OperatorName.MOTION:
            question = "Is the selected object moving? Answer true or false."
        elif node.op is OperatorName.RELATION:
            question = "What is the subject's relation to the reference?"
        elif node.op in {OperatorName.VLM_REASON, OperatorName.ROUTE_REASON}:
            configured = str(node.params["question"])
            question = context.question if configured == "$question" else configured
        else:
            question = context.question
        options = (
            context.choices
            if node.op in {OperatorName.ROUTE_REASON, OperatorName.MATCH_CHOICE}
            else ()
        )
        model_input = context.composer.compose_named(inputs, question=question, options=options)
        if node.op is OperatorName.ROUTE_REASON:
            choice_ids = tuple(chr(ord("A") + index) for index in range(len(context.choices)))
            answer_type = getattr(node.params.get("answer_type"), "value", None) or str(
                node.params.get("answer_type") or "CHOICE_SINGLE"
            )
            scored = self.provider.reason_and_choose(
                ChoiceScoringRequest(
                    model_input=model_input,
                    answer_type=answer_type,
                    choice_ids=choice_ids,
                    option_texts=context.choices,
                    final_suffix=self.choice_config.final_suffix,
                    multi_select_threshold=self.choice_config.multi_select_threshold,
                    purpose="route_choice",
                )
            )
            if not self.choice_config.preserve_reasoning_text:
                scored = replace(scored, reasoning_text=None)
            return OperatorOutcome(scored, scored.provider)
        result = self.provider.infer(VLMRequest(model_input, node.op.value.casefold()))
        if node.op in {OperatorName.ATTRIBUTE, OperatorName.CLASSIFY, OperatorName.RELATION}:
            value: RuntimeObject = Label(result.text, {"provider": result.provider})
        elif node.op is OperatorName.MULTILABEL_CLASSIFY:
            value = LabelSet(self._label_set(result.text), {"provider": result.provider})
        elif node.op is OperatorName.MOTION:
            value = Boolean(
                result.text.strip().casefold() in {"true", "yes", "1"},
                {"provider": result.provider},
            )
        else:
            value = Answer(result.text, result.confidence, {"provider": result.provider})
        return OperatorOutcome(value, result.provider)
