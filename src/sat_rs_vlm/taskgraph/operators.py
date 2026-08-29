"""Deterministic and capability-backed TaskGraph operator executors."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from math import hypot
from statistics import median
from typing import Protocol

from PIL import Image

from .input_composer import InputComposer
from .providers import (
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
    def __init__(self, semantic: SemanticVLMProvider) -> None:
        self.semantic = semantic
        self.provider_name = semantic.provider_name

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

    def execute(
        self,
        node: GraphNode,
        inputs: dict[str, RuntimeObject | list[RuntimeObject]],
        context: OperatorContext,
    ) -> OperatorOutcome:
        candidates = inputs["candidates"]
        mode = str(node.params["mode"])
        if isinstance(candidates, EntitySet):
            entities = candidates.entities
            if not entities:
                return OperatorOutcome(
                    candidates, "geometry" if mode != "RELATION" else self.provider_name
                )
            if mode == "ORDINAL":
                ordered = self._sort_entities(entities, str(node.params["order"]))
                selected = ordered[int(node.params["index"]) - 1 : int(node.params["index"])]
                return OperatorOutcome(EntitySet(tuple(selected), {"select": mode}), "geometry")
            if mode == "EXTREME":
                direction = str(node.params["direction"])
                order = {
                    "LEFTMOST": "LEFT_TO_RIGHT",
                    "RIGHTMOST": "RIGHT_TO_LEFT",
                    "TOPMOST": "TOP_TO_BOTTOM",
                    "BOTTOMMOST": "BOTTOM_TO_TOP",
                }[direction]
                return OperatorOutcome(
                    EntitySet((self._sort_entities(entities, order)[0],), {"select": mode}),
                    "geometry",
                )
            if mode == "RANK":
                criterion = str(node.params["criterion"]).casefold()
                if "score" in criterion:
                    ranked = sorted(entities, key=lambda item: item.score or 0.0)
                else:
                    ranked = sorted(
                        entities,
                        key=lambda item: (
                            (item.region.bbox_xyxy_global[2] - item.region.bbox_xyxy_global[0])
                            * (item.region.bbox_xyxy_global[3] - item.region.bbox_xyxy_global[1])
                        ),
                    )
                if str(node.params["order"]) == "DESCENDING":
                    ranked.reverse()
                index = int(node.params["rank"]) - 1
                return OperatorOutcome(
                    EntitySet(tuple(ranked[index : index + 1]), {"select": mode}), "geometry"
                )
            if mode == "RELATION":
                reference = inputs.get("reference")
                named: dict[str, RuntimeObject | list[RuntimeObject]] = {"candidates": candidates}
                if reference is not None:
                    named["reference"] = reference
                model_input = context.composer.compose_named(
                    named,
                    question=(
                        f"Select candidate ids satisfying relation {node.params['relation']}. "
                        "Return only candidate ids such as A, B, or C."
                    ),
                )
                result = self.semantic.infer(VLMRequest(model_input, "selection"))
                indices = parse_selection_indices(result.text, len(entities))
                return OperatorOutcome(
                    EntitySet(
                        tuple(entities[index] for index in indices), {"provider": result.provider}
                    ),
                    result.provider,
                )
        if mode == "SUBREGION" and isinstance(candidates, Region):
            helper = GeometryExecutor()
            mapping = {
                "LEFT_SIDE": "LEFT",
                "RIGHT_SIDE": "RIGHT",
                "ABOVE": "TOP",
                "BELOW": "BOTTOM",
                "INSIDE": "CENTER",
                "AROUND": "CENTER",
            }
            position = mapping.get(str(node.params["subregion"]))
            if position:
                return OperatorOutcome(helper._position_region(candidates, position), "geometry")
        raise TypeError(f"SELECT {mode} does not support {type(candidates).__name__}")


class SemanticExecutor:
    def __init__(self, provider: SemanticVLMProvider, *, provider_name: str | None = None) -> None:
        self.provider = provider
        self.provider_name = provider_name or provider.provider_name

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
