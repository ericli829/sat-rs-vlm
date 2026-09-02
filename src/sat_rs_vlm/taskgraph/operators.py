"""Deterministic and capability-backed TaskGraph operator executors."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from math import hypot
from statistics import median
from typing import Any, Protocol, cast

import numpy as np
from PIL import Image

from .capabilities import TargetCapability, TargetCapabilityClassifier
from .choice_config import ChoiceSystemConfig
from .execution_plan import NodeExecutionHint
from .input_composer import InputComposer
from .providers import (
    CachedChoiceUnavailableError,
    ChoiceScoringRequest,
    DetectionProvider,
    DetectionRequest,
    RegionRetrievalRequest,
    RegionRetrieverProvider,
    SemanticVLMProvider,
    VLMRequest,
    VLMResult,
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
    unwrap_select_result,
)
from .schema import AnswerType, GraphNode, OperatorName, SpatialRelation, TargetSpec
from .semantic_decision import SemanticDecisionConfig, SemanticDecisionLayer
from .semantic_prompts import semantic_question, semantic_reasoning_instruction


@dataclass(frozen=True)
class OperatorOutcome:
    value: RuntimeObject
    provider: str
    trace_metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class OperatorContext:
    question: str
    choices: tuple[str, ...]
    composer: InputComposer
    execution_hint: NodeExecutionHint | None = None


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
        return _image(
            unwrap_select_result(
                value,
                allow_empty=False,
                require_single=True,
                consumer="geometry image resolution",
            )
        )
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


def _scope_dimensions(value: ImageRef | Region) -> tuple[float, float]:
    if isinstance(value, Region):
        x0, y0, x1, y1 = value.bbox_xyxy_global
        return x1 - x0, y1 - y0
    width, height = _dimensions(value)
    return float(width), float(height)


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
    def _marker_components(mask: Any) -> list[dict[str, int]]:
        """Label an 8-connected mask using row runs, never one Python object per pixel."""

        parents: list[int] = []
        runs: list[tuple[int, int, int, int]] = []

        def new_label() -> int:
            label = len(parents)
            parents.append(label)
            return label

        def find(label: int) -> int:
            root = label
            while parents[root] != root:
                root = parents[root]
            while parents[label] != label:
                parent = parents[label]
                parents[label] = root
                label = parent
            return root

        def union(left: int, right: int) -> int:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                root = min(left_root, right_root)
                parents[max(left_root, right_root)] = root
            return find(left_root)

        previous: list[tuple[int, int, int]] = []
        for y in range(int(mask.shape[0])):
            row = mask[y]
            padded = np.empty(int(row.shape[0]) + 2, dtype=np.bool_)
            padded[0] = False
            padded[-1] = False
            padded[1:-1] = row
            transitions = np.flatnonzero(padded[1:] != padded[:-1])
            current: list[tuple[int, int, int]] = []
            for x0_raw, x1_raw in zip(transitions[0::2], transitions[1::2], strict=True):
                x0, x1 = int(x0_raw), int(x1_raw)
                overlaps = [
                    label
                    for previous_x0, previous_x1, label in previous
                    if previous_x1 >= x0 and previous_x0 <= x1
                ]
                if overlaps:
                    label = find(overlaps[0])
                    for other in overlaps[1:]:
                        label = union(label, other)
                else:
                    label = new_label()
                current.append((x0, x1, label))
                runs.append((y, x0, x1, label))
            previous = current

        components: dict[int, dict[str, int]] = {}
        for y, x0, x1, label in runs:
            root = find(label)
            component = components.setdefault(
                root,
                {"x0": x0, "y0": y, "x1": x1, "y1": y + 1, "pixels": 0},
            )
            component["x0"] = min(component["x0"], x0)
            component["y0"] = min(component["y0"], y)
            component["x1"] = max(component["x1"], x1)
            component["y1"] = max(component["y1"], y + 1)
            component["pixels"] += x1 - x0
        return sorted(components.values(), key=lambda item: (item["y0"], item["x0"]))

    @staticmethod
    def _marker(
        source: ImageRef | Region,
        color: str | None,
        shape: str | None = None,
    ) -> RegionSet:
        image = source if isinstance(source, ImageRef) else source.image
        scope = _scope_box(source)
        marker_color = (color or "red").casefold()

        with Image.open(image.path.resolve()) as raw:
            crop = raw.convert("RGB").crop(scope)
            pixels = np.asarray(crop, dtype=np.int16)
        red, green, blue = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
        if marker_color == "green":
            mask = (green >= 140) & (green - red >= 40) & (green - blue >= 40)
        elif marker_color == "blue":
            mask = (blue >= 140) & (blue - red >= 40) & (blue - green >= 40)
        elif marker_color == "yellow":
            mask = (red >= 150) & (green >= 150) & (blue <= 130)
        else:
            mask = (red >= 160) & (red - green >= 60) & (red - blue >= 60)
        components = GeometryExecutor._marker_components(mask)
        marker_shape = (shape or "unspecified").casefold()
        if not components:
            return RegionSet(
                (),
                {
                    "operator": "FIND_MARKER",
                    "color": marker_color,
                    "shape": marker_shape,
                    "component_count": 0,
                    "rejected_component_count": 0,
                    "implementation": "numpy_vectorized_mask_rle_union_find",
                },
            )

        minimum_pixels = max(4, min(64, math.ceil(crop.width * crop.height * 0.00001)))
        maximum_bbox_area = crop.width * crop.height * 0.2
        accepted: list[dict[str, int]] = []
        rejected: list[dict[str, object]] = []
        for component in components:
            component_x0 = component["x0"]
            component_y0 = component["y0"]
            component_x1 = component["x1"]
            component_y1 = component["y1"]
            component_width = component_x1 - component_x0
            component_height = component_y1 - component_y0
            bbox_area = component_width * component_height
            reason = None
            if (
                component["pixels"] < minimum_pixels
                or component_width < 2
                or component_height < 2
            ):
                reason = "too_small"
            elif bbox_area > maximum_bbox_area:
                reason = "too_large_for_artificial_marker"
            elif any(token in marker_shape for token in ("circle", "round", "ring")):
                aspect = component_width / float(component_height)
                if not 0.55 <= aspect <= 1.8:
                    reason = "shape_aspect_mismatch"
            if reason is None:
                accepted.append(component)
            else:
                rejected.append(
                    {
                        "bbox_xyxy_scope": [
                            component_x0,
                            component_y0,
                            component_x1,
                            component_y1,
                        ],
                        "pixels": component["pixels"],
                        "reason": reason,
                    }
                )

        offset_x, offset_y = scope[0], scope[1]
        regions = tuple(
            Region(
                image,
                (
                    component["x0"] + offset_x,
                    component["y0"] + offset_y,
                    component["x1"] + offset_x,
                    component["y1"] + offset_y,
                ),
                {
                    "operator": "FIND_MARKER",
                    "color": marker_color,
                    "shape": marker_shape,
                    "component_pixels": component["pixels"],
                },
            )
            for component in accepted
        )
        return RegionSet(
            regions,
            {
                "operator": "FIND_MARKER",
                "color": marker_color,
                "shape": marker_shape,
                "component_count": len(regions),
                "rejected_component_count": len(rejected),
                "rejected_components": rejected,
                "minimum_component_pixels": minimum_pixels,
                "implementation": "numpy_vectorized_mask_rle_union_find",
            },
        )

    @staticmethod
    def _group(entities: EntitySet, mode: str) -> EntitySet:
        if not entities.entities:
            return EntitySet(
                (),
                {
                    **entities.provenance,
                    "group": mode,
                    "group_count": 0,
                    "groups": [],
                },
            )
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
        if any(entity.score is None for entity in value.entities):
            raise ValueError(
                f"BUILD_ROUTE_CONTEXT.{role} is ambiguous: every candidate needs a score"
            )
        scored = list(enumerate(value.entities))
        ordered = sorted(scored, key=lambda item: cast(float, item[1].score), reverse=True)
        if abs(cast(float, ordered[0][1].score) - cast(float, ordered[1][1].score)) <= 1e-9:
            raise ValueError(f"BUILD_ROUTE_CONTEXT.{role} is ambiguous: highest score is tied")
        selected_index, selected = ordered[0]
        return selected, {
            "policy": "unique_highest_complete_scores",
            "selected_index": selected_index,
            "selected_score": selected.score,
            "candidate_scores": [entity.score for entity in value.entities],
        }

    @staticmethod
    def _route_context(
        image_value: RuntimeObject, start: RuntimeObject, goal: RuntimeObject
    ) -> RouteContext:
        image = _image(image_value)
        selected_start, start_selection = GeometryExecutor._resolve_route_endpoint(start, "start")
        selected_goal, goal_selection = GeometryExecutor._resolve_route_endpoint(goal, "goal")
        width, height = _dimensions(image)

        def clipped_endpoint(value: Entity | Region, role: str) -> Entity | Region:
            region = value.region if isinstance(value, Entity) else value
            box = region.bbox_xyxy_global
            clipped_box = (
                max(0.0, min(float(width), box[0])),
                max(0.0, min(float(height), box[1])),
                max(0.0, min(float(width), box[2])),
                max(0.0, min(float(height), box[3])),
            )
            if clipped_box[0] >= clipped_box[2] or clipped_box[1] >= clipped_box[3]:
                raise ValueError(f"route {role} endpoint is outside the source image")
            if clipped_box == box:
                return value
            clipped_region = replace(
                region,
                bbox_xyxy_global=clipped_box,
                provenance={
                    **region.provenance,
                    "route_endpoint_clipped_from": list(box),
                },
            )
            return (
                replace(value, region=clipped_region)
                if isinstance(value, Entity)
                else clipped_region
            )

        selected_start = clipped_endpoint(selected_start, "start")
        selected_goal = clipped_endpoint(selected_goal, "goal")
        start_region = (
            selected_start.region if isinstance(selected_start, Entity) else selected_start
        )
        goal_region = selected_goal.region if isinstance(selected_goal, Entity) else selected_goal
        if (
            start_region.image.uri_or_key != image.uri_or_key
            or goal_region.image.uri_or_key != image.uri_or_key
        ):
            raise ValueError("route endpoints must reference BUILD_ROUTE_CONTEXT.image")
        x0 = max(0.0, min(start_region.bbox_xyxy_global[0], goal_region.bbox_xyxy_global[0]))
        y0 = max(0.0, min(start_region.bbox_xyxy_global[1], goal_region.bbox_xyxy_global[1]))
        x1 = min(
            float(width), max(start_region.bbox_xyxy_global[2], goal_region.bbox_xyxy_global[2])
        )
        y1 = min(
            float(height), max(start_region.bbox_xyxy_global[3], goal_region.bbox_xyxy_global[3])
        )
        pad = max(8.0, 0.1 * max(x1 - x0, y1 - y0))
        padded_box = (
            max(0.0, x0 - pad),
            max(0.0, y0 - pad),
            min(width, x1 + pad),
            min(height, y1 + pad),
        )
        minimum_side = 256.0

        def expand_axis(low: float, high: float, limit: float) -> tuple[float, float]:
            target = min(limit, max(minimum_side, high - low))
            center = (low + high) / 2.0
            expanded_low = max(0.0, center - target / 2.0)
            expanded_high = min(limit, expanded_low + target)
            expanded_low = max(0.0, expanded_high - target)
            return expanded_low, expanded_high

        context_x0, context_x1 = expand_axis(padded_box[0], padded_box[2], float(width))
        context_y0, context_y1 = expand_axis(padded_box[1], padded_box[3], float(height))
        context_box = (context_x0, context_y0, context_x1, context_y1)
        context_region = Region(
            image,
            context_box,
            {
                "coordinate_transform": {"origin_global": list(context_box[:2])},
                "endpoint_union_bbox_xyxy_global": [x0, y0, x1, y1],
                "padded_bbox_xyxy_global": list(padded_box),
                "minimum_context_side": minimum_side,
            },
        )
        return RouteContext(
            image=image,
            start=selected_start,
            goal=selected_goal,
            context_region=context_region,
            provenance={
                "provider": "geometry",
                "ambiguity_policy": "single_or_unique_highest_complete_scores_else_error",
                "start_selection": start_selection,
                "goal_selection": goal_selection,
                "context_bbox_xyxy_global": list(context_box),
                "context_size": [context_x1 - context_x0, context_y1 - context_y0],
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
            source = inputs["image"]
            if not isinstance(source, (ImageRef, Region)):
                raise TypeError("REGION_FROM_BBOX.image must be ImageRef or Region")
            image = source if isinstance(source, ImageRef) else source.image
            width, height = _dimensions(source)
            source_size = node.params.get("image_size")
            bbox = tuple(float(item) for item in node.params["bbox"])
            if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                raise ValueError("REGION_FROM_BBOX bbox must have positive area before clipping")
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
            scope = _scope_box(source)
            bbox = (
                max(scope[0], min(scope[2], bbox[0])),
                max(scope[1], min(scope[3], bbox[1])),
                max(scope[0], min(scope[2], bbox[2])),
                max(scope[1], min(scope[3], bbox[3])),
            )
            if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                raise ValueError("REGION_FROM_BBOX bbox does not intersect its input scope")
            value = Region(
                image,
                bbox,
                {
                    "operator": node.op.value,
                    "source_image_size": list(source_size) if source_size is not None else None,
                    "actual_image_size": [width, height],
                    "scale_xy": [scale_x, scale_y],
                    "input_scope_bbox_xyxy_global": list(scope),
                    "coordinates": "absolute_xyxy_global",
                },
            )
        elif node.op is OperatorName.FIND_MARKER:
            source = inputs["image"]
            if not isinstance(source, (ImageRef, Region)):
                raise TypeError("FIND_MARKER.image must be ImageRef or Region")
            value = self._marker(
                source,
                node.params["marker"].get("color"),
                node.params["marker"].get("shape"),
            )
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
        capability_classifier: TargetCapabilityClassifier | None = None,
    ) -> None:
        self.detection = detection
        self.retriever = retriever
        self.semantic_categories = {item.casefold() for item in (semantic_categories or set())}
        self.capability_classifier = capability_classifier or TargetCapabilityClassifier(
            legacy_region_overrides=self.semantic_categories
        )
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
        decision = self.capability_classifier.classify(target)
        decision_metadata = {"capability_decision": decision.to_dict()}
        if decision.effective_capability is TargetCapability.RETRIEVER:
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
                {
                    "provider": result.provider,
                    "capability": "region_retrieval",
                    **decision_metadata,
                },
            )
            return OperatorOutcome(
                entities,
                result.provider,
                {
                    **decision_metadata,
                    "capability": "RETRIEVER",
                    "latency_ms": result.latency_ms,
                    "activated_provider": result.provider,
                    "stage": "retriever",
                    "provider_metadata": dict(result.metadata),
                },
            )
        detected = self.detection.detect(DetectionRequest(scope, target, "LOCATE"))
        return OperatorOutcome(
            detected.detections,
            detected.provider,
            {
                **decision_metadata,
                "capability": "DETECTOR",
                "latency_ms": detected.latency_ms,
                "activated_provider": detected.provider,
                "stage": "detector",
                "provider_metadata": dict(detected.metadata),
            },
        )


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
            raw_entities = inputs["entities"]
            if isinstance(raw_entities, list):
                raise TypeError("COUNT.entities must be a single EntitySet")
            entities = unwrap_select_result(
                raw_entities,
                allow_empty=True,
                consumer="COUNT.entities",
            )
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
            {
                "latency_ms": detected.latency_ms,
                "activated_provider": detected.provider,
                "stage": "detector",
                "provider_metadata": dict(detected.metadata),
            },
        )


@dataclass(frozen=True)
class RelationPartition:
    positive: tuple[int, ...]
    grey: tuple[int, ...]
    negative: tuple[int, ...]


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
    def _ordinal_primary(item: Entity | Region, order: str) -> float:
        region = item.region if isinstance(item, Entity) else item
        box = region.bbox_xyxy_global
        if order in {"TOP_TO_BOTTOM", "BOTTOM_TO_TOP"}:
            return box[1]
        return box[0]

    @staticmethod
    def _rank_value(item: Entity, criterion: str) -> float:
        if criterion == "score":
            if item.score is None:
                raise ValueError("rank_score_missing")
            return item.score
        if criterion != "bbox_area":
            raise ValueError(f"unsupported SELECT RANK criterion {criterion}")
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
        materialized = unwrap_select_result(
            value,
            allow_empty=role == "candidates",
            consumer=f"SELECT.{role}",
        )
        if isinstance(materialized, (EntitySet, Region, RegionSet)):
            return materialized
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
        width, height = _scope_dimensions(scope)
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

    @staticmethod
    def _selection_type(node: GraphNode) -> str:
        return str(node.params.get("selection_type") or "MULTI")

    @staticmethod
    def _cardinality_status(count: int, selection_type: str) -> SelectStatus:
        if count == 0:
            return SelectStatus.EMPTY
        if selection_type == "SINGLE" and count > 1:
            return SelectStatus.AMBIGUOUS
        return SelectStatus.OK

    @staticmethod
    def _image_key(image: ImageRef) -> str:
        return str(image.path.resolve()).casefold()

    @classmethod
    def _image_keys(cls, value: RuntimeObject | None) -> set[str]:
        if value is None:
            return set()
        if isinstance(value, ImageRef):
            return {cls._image_key(value)}
        if isinstance(value, Region):
            return {cls._image_key(value.image)}
        if isinstance(value, Entity):
            return {cls._image_key(value.region.image)}
        if isinstance(value, RegionSet):
            return {cls._image_key(item.image) for item in value.regions}
        if isinstance(value, EntitySet):
            return {cls._image_key(item.region.image) for item in value.entities}
        return set()

    @classmethod
    def _same_image_inputs(
        cls,
        candidates: EntitySet | Region | RegionSet,
        reference: RuntimeObject | None,
        scope: RuntimeObject | None,
    ) -> bool:
        keys = cls._image_keys(candidates)
        keys.update(cls._image_keys(reference))
        keys.update(cls._image_keys(scope))
        return len(keys) <= 1

    @staticmethod
    def _stable_index(items: tuple[Entity | Region, ...]) -> dict[str, int]:
        mapping = {
            str(item.provenance.get("candidate_id")): index for index, item in enumerate(items)
        }
        if len(mapping) != len(items) or "None" in mapping:
            raise ValueError("SELECT candidate_id values must be present and unique")
        return mapping

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
        *,
        clear_positive_indices: tuple[int, ...] = (),
        grey_indices: tuple[int, ...] | None = None,
    ) -> OperatorOutcome:
        items = self._items(candidates)
        stable_index = self._stable_index(items)
        all_indices = tuple(range(len(items)))
        semantic_indices = all_indices if grey_indices is None else grey_indices
        selection_type = self._selection_type(node)
        clear_ids = self._candidate_ids(items, clear_positive_indices)
        grey_ids = self._candidate_ids(items, semantic_indices)
        common_provenance = {
            **provenance,
            "selection_type": selection_type,
            "all_candidate_ids": self._candidate_ids(items, all_indices),
            "clear_positive_candidate_ids": clear_ids,
            "grey_candidate_ids": grey_ids,
        }
        if selection_type == "SINGLE" and len(clear_positive_indices) >= 2:
            selected = self._selected_like(
                candidates,
                clear_positive_indices,
                provenance={"select": "RELATION", **provenance},
            )
            return self._result(
                selected,
                status=SelectStatus.AMBIGUOUS,
                method="geometry",
                reason="single_selection_multiple_matches",
                provenance={
                    **common_provenance,
                    "semantic_positive_candidate_ids": [],
                    "final_candidate_ids": clear_ids,
                    "candidate_ids": clear_ids,
                },
            )

        semantic_candidates = self._selected_like(
            candidates,
            semantic_indices,
            provenance={"select": "RELATION_GREY_SUBSET", **provenance},
        )
        named: dict[str, RuntimeObject | list[RuntimeObject]] = {"candidates": semantic_candidates}
        if reference is not None:
            named["reference"] = reference
        model_input = context.composer.compose_named(
            named,
            question=(
                f"Analyze which candidate objects satisfy relation {relation} relative to "
                "the marked reference. Use candidate labels only as visual references during "
                "reasoning. A separate constrained verification step will determine the final "
                "selection."
            ),
        )
        candidate_mapping = model_input.metadata.get("candidate_mapping")
        if not isinstance(candidate_mapping, dict) or not candidate_mapping:
            return self._result(
                self._empty_like(candidates),
                status=SelectStatus.ERROR,
                method="qwen3_vl_kv_cached_choice",
                reason="semantic selection canvas has no stable candidate mapping",
                provenance=common_provenance,
            )
        choice_ids = tuple(str(choice_id) for choice_id in candidate_mapping)
        subset_items = self._items(semantic_candidates)
        if len(choice_ids) != len(subset_items):
            raise RuntimeError("semantic SELECT canvas mapping is not aligned with its grey subset")

        def restore_indices(selected_labels: tuple[str, ...]) -> tuple[int, ...]:
            restored: list[int] = []
            for choice_id in selected_labels:
                entry = candidate_mapping.get(choice_id)
                if not isinstance(entry, dict):
                    raise RuntimeError("choice score returned an invalid SELECT candidate label")
                candidate_id = entry.get("candidate_id")
                if not isinstance(candidate_id, str) or candidate_id not in stable_index:
                    raise RuntimeError("SELECT candidate mapping lost a stable candidate_id")
                restored.append(stable_index[candidate_id])
            if len(restored) != len(set(restored)):
                raise RuntimeError("choice score returned duplicate SELECT candidates")
            return tuple(restored)

        fallback_used = False
        fallback_reason: str | None = None
        fallback_type: str | None = None
        confidence: float | None = None
        semantic_provider: str
        scores: dict[str, float] = {}
        score_method: str
        cache_reused = False
        latency_ms: dict[str, float | None] = {}
        reasoning_text: str | None = None
        choice_metadata: dict[str, object] = {}
        try:
            scorer = getattr(self.semantic, "reason_and_choose", None)
            if not callable(scorer):
                raise CachedChoiceUnavailableError(
                    "semantic provider does not implement cached choice scoring"
                )
            scored = scorer(
                ChoiceScoringRequest(
                    model_input=model_input,
                    answer_type="CHOICE_MULTI",
                    choice_ids=choice_ids,
                    option_texts=tuple(
                        f"Candidate {choice_id}: "
                        f"{item.label if isinstance(item, Entity) else 'region'}"
                        for choice_id, item in zip(choice_ids, subset_items, strict=True)
                    ),
                    single_choice_suffix=self.choice_config.single_choice_suffix,
                    multi_verify_template=self.choice_config.multi_verify_template,
                    multi_select_threshold=self.choice_config.multi_select_threshold,
                    purpose="select_relation",
                )
            )
            if scored.answer_type != "CHOICE_MULTI":
                raise RuntimeError("semantic SELECT must use independent binary verification")
            semantic_positive_indices = restore_indices(scored.selected_ids)
            semantic_provider = scored.provider
            scores = dict(scored.scores)
            score_method = scored.method
            cache_reused = scored.cache_reused
            latency_ms = dict(scored.latency_ms)
            reasoning_text = (
                scored.reasoning_text if self.choice_config.preserve_reasoning_text else None
            )
            choice_metadata = dict(scored.metadata)
        except CachedChoiceUnavailableError as exc:
            fallback_reason = str(exc)
            fallback_type = type(exc).__name__
            if len(choice_ids) > 8:
                return self._result(
                    self._selected_like(
                        candidates,
                        clear_positive_indices,
                        provenance={"select": "RELATION", **provenance},
                    ),
                    status=SelectStatus.UNRESOLVED,
                    method="qwen3_vl_token_mask_fallback",
                    reason="safe_fallback_unavailable",
                    provenance={
                        **common_provenance,
                        "semantic_positive_candidate_ids": [],
                        "final_candidate_ids": clear_ids,
                        "candidate_ids": clear_ids,
                        "fallback_used": False,
                        "fallback_reason": fallback_reason,
                        "fallback_type": fallback_type,
                    },
                )
            result = self.semantic.infer(VLMRequest(model_input, "selection"))
            if result.metadata.get("constrained_decoding") is not True:
                return self._result(
                    self._selected_like(
                        candidates,
                        clear_positive_indices,
                        provenance={"select": "RELATION", **provenance},
                    ),
                    status=SelectStatus.UNRESOLVED,
                    method="qwen3_vl_token_mask_fallback",
                    reason="safe_fallback_unavailable",
                    provenance={
                        **common_provenance,
                        "semantic_positive_candidate_ids": [],
                        "final_candidate_ids": clear_ids,
                        "candidate_ids": clear_ids,
                        "fallback_used": False,
                        "fallback_reason": fallback_reason,
                        "fallback_type": fallback_type,
                        "provider": result.provider,
                    },
                )
            try:
                local_indices = parse_selection_indices(result.text, len(choice_ids))
            except ValueError:
                return self._result(
                    self._selected_like(
                        candidates,
                        clear_positive_indices,
                        provenance={"select": "RELATION", **provenance},
                    ),
                    status=SelectStatus.UNRESOLVED,
                    method="qwen3_vl_token_mask_fallback",
                    reason="safe_fallback_invalid_output",
                    provenance={
                        **common_provenance,
                        "semantic_positive_candidate_ids": [],
                        "final_candidate_ids": clear_ids,
                        "candidate_ids": clear_ids,
                        "fallback_used": True,
                        "fallback_reason": fallback_reason,
                        "fallback_type": fallback_type,
                        "provider": result.provider,
                    },
                )
            selected_labels = tuple(choice_ids[index] for index in local_indices)
            semantic_positive_indices = restore_indices(selected_labels)
            fallback_used = True
            confidence = result.confidence
            semantic_provider = result.provider
            score_method = "finite_token_mask"
            choice_metadata = {"raw_response": result.text, **result.metadata}

        final_indices = tuple(
            index
            for index in all_indices
            if index in set(clear_positive_indices).union(semantic_positive_indices)
        )
        semantic_ids = self._candidate_ids(items, semantic_positive_indices)
        final_ids = self._candidate_ids(items, final_indices)
        status = self._cardinality_status(len(final_indices), selection_type)
        method = "qwen3_vl_token_mask_fallback" if fallback_used else "qwen3_vl_kv_cached_choice"
        return self._result(
            self._selected_like(
                candidates,
                final_indices,
                provenance={"provider": semantic_provider, "select": "RELATION"},
            ),
            status=status,
            method=method,
            reason=(
                "single_selection_multiple_matches" if status is SelectStatus.AMBIGUOUS else None
            ),
            confidence=confidence,
            provenance={
                **common_provenance,
                "provider": semantic_provider,
                "candidate_ids": final_ids,
                "semantic_positive_candidate_ids": semantic_ids,
                "final_candidate_ids": final_ids,
                "choice_labels": [
                    choice_id
                    for choice_id, entry in candidate_mapping.items()
                    if isinstance(entry, dict) and entry.get("candidate_id") in set(semantic_ids)
                ],
                "scores": scores,
                "score_method": score_method,
                "cache_reused": cache_reused,
                "latency_ms": latency_ms,
                "reasoning_text": reasoning_text,
                "choice_metadata": choice_metadata,
                "fallback_used": fallback_used,
                "fallback_reason": fallback_reason,
                "fallback_type": fallback_type,
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
    ) -> RelationPartition:
        """Partition candidates into deterministic yes, uncertain, and no sets."""
        ref_box = reference.bbox_xyxy_global
        ref_x, ref_y = self._box_center(ref_box)
        positive: list[int] = []
        grey: list[int] = []
        negative: list[int] = []
        for index, item in enumerate(self._items(candidates)):
            box = self._item_region(item).bbox_xyxy_global
            x, y = self._box_center(box)
            if relation == "LEFT_OF":
                bucket = (
                    positive
                    if x < ref_x - margin
                    else grey
                    if abs(x - ref_x) <= margin
                    else negative
                )
            elif relation == "RIGHT_OF":
                bucket = (
                    positive
                    if x > ref_x + margin
                    else grey
                    if abs(x - ref_x) <= margin
                    else negative
                )
            elif relation == "ABOVE":
                bucket = (
                    positive
                    if y < ref_y - margin
                    else grey
                    if abs(y - ref_y) <= margin
                    else negative
                )
            elif relation == "BELOW":
                bucket = (
                    positive
                    if y > ref_y + margin
                    else grey
                    if abs(y - ref_y) <= margin
                    else negative
                )
            elif relation == "INSIDE":
                if self._contains(ref_box, box):
                    bucket = positive
                elif self._intersects(ref_box, box):
                    bucket = grey
                else:
                    bucket = negative
            elif relation == "OVERLAP":
                iou = self._iou(box, ref_box)
                if iou >= overlap_iou_threshold:
                    bucket = positive
                elif iou > 0.0:
                    bucket = grey
                else:
                    bucket = negative
            else:
                bucket = grey
            bucket.append(index)
        return RelationPartition(tuple(positive), tuple(grey), tuple(negative))

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
        safe_inputs = dict(inputs)
        safe_inputs["candidates"] = candidates
        for role in ("reference", "scope"):
            raw_value = inputs.get(role)
            if isinstance(raw_value, list):
                raise TypeError(f"SELECT.{role} does not allow reference lists")
            if raw_value is not None:
                safe_inputs[role] = unwrap_select_result(
                    raw_value,
                    allow_empty=False,
                    require_single=role == "scope",
                    consumer=f"SELECT.{role}",
                )
        mode = str(node.params["mode"])
        items = self._items(candidates)
        reference_input = safe_inputs.get("reference")
        scope_input = safe_inputs.get("scope")
        reference_value = reference_input if not isinstance(reference_input, list) else None
        scope_value = scope_input if not isinstance(scope_input, list) else None
        if not self._same_image_inputs(candidates, reference_value, scope_value):
            return self._result(
                self._empty_like(candidates),
                status=SelectStatus.UNRESOLVED,
                method="geometry",
                reason="cross_image_select_inputs",
                provenance={
                    "mode": mode,
                    "coordinate_system": "bbox_xyxy_global",
                },
            )
        try:
            self._stable_index(items)
        except ValueError as exc:
            return self._result(
                self._empty_like(candidates),
                status=SelectStatus.ERROR,
                method="geometry",
                reason="invalid_candidate_ids",
                provenance={"error": str(exc)},
            )
        if not items:
            return self._result(
                self._empty_like(candidates),
                status=SelectStatus.EMPTY,
                method="geometry",
                reason="empty_candidates",
            )
        if mode == "SUBREGION":
            return self._subregion(safe_inputs, candidates, node)
        if isinstance(candidates, RegionSet) and mode in {"ORDINAL", "EXTREME"}:
            if mode == "ORDINAL":
                order = str(node.params["order"])
                requested_index = int(node.params["index"]) - 1
            else:
                order = {
                    "LEFTMOST": "LEFT_TO_RIGHT",
                    "RIGHTMOST": "RIGHT_TO_LEFT",
                    "TOPMOST": "TOP_TO_BOTTOM",
                    "BOTTOMMOST": "BOTTOM_TO_TOP",
                }[str(node.params["direction"])]
                requested_index = 0
            reverse = order in {"BOTTOM_TO_TOP", "RIGHT_TO_LEFT"}
            indexed_regions = sorted(
                enumerate(candidates.regions),
                key=lambda item: self._ordinal_primary(item[1], order),
                reverse=reverse,
            )
            if requested_index >= len(indexed_regions):
                return self._result(
                    self._empty_like(candidates),
                    status=SelectStatus.UNRESOLVED,
                    method="geometry",
                    reason="ordinal_out_of_range",
                )
            selected_key = self._ordinal_primary(
                indexed_regions[requested_index][1],
                order,
            )
            selected_indices = tuple(
                source_index
                for source_index, region in indexed_regions
                if self._ordinal_primary(region, order) == selected_key
            )
            status = SelectStatus.AMBIGUOUS if len(selected_indices) > 1 else SelectStatus.OK
            return self._result(
                self._selected_like(
                    candidates,
                    selected_indices,
                    provenance={"select": mode, "coordinate_system": "bbox_xyxy_global"},
                ),
                status=status,
                method="geometry",
                reason=(f"{mode.casefold()}_tie" if status is SelectStatus.AMBIGUOUS else None),
            )
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
                selected_key = self._ordinal_primary(ordered[index], order)
                selected = tuple(
                    entity
                    for entity in ordered
                    if self._ordinal_primary(entity, order) == selected_key
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
                best_key = self._ordinal_primary(ordered[0], order)
                selected = tuple(
                    entity for entity in ordered if self._ordinal_primary(entity, order) == best_key
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
                if criterion == "score" and any(entity.score is None for entity in entities):
                    return self._result(
                        self._empty_like(candidates),
                        status=SelectStatus.UNRESOLVED,
                        method="geometry",
                        reason="rank_score_missing",
                        provenance={"mode": mode, "criterion": criterion},
                    )
                ranked = sorted(
                    entities,
                    key=lambda item: self._rank_value(item, criterion),
                )
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
            scope = self._scope_for(safe_inputs, candidates, reference)
            margin = self._default_margin(scope, node.params.get("margin"))
            configured_threshold = node.params.get("overlap_iou_threshold")
            threshold = float(0.10 if configured_threshold is None else configured_threshold)
            partition = self._direct_relation(
                candidates, reference, relation, margin=margin, overlap_iou_threshold=threshold
            )
            all_indices = tuple(range(len(items)))
            provenance = {
                "mode": mode,
                "relation": relation,
                "selection_type": self._selection_type(node),
                "coordinate_system": "bbox_xyxy_global",
                "scope_bbox_xyxy_global": list(_scope_box(scope)),
                "margin_px": margin,
                "overlap_iou_threshold": threshold,
                "all_candidate_ids": self._candidate_ids(items, all_indices),
                "clear_positive_candidate_ids": self._candidate_ids(items, partition.positive),
                "grey_candidate_ids": self._candidate_ids(items, partition.grey),
                "clear_negative_candidate_ids": self._candidate_ids(items, partition.negative),
            }
            if partition.grey:
                return self._semantic_select(
                    candidates,
                    reference_value,
                    relation,
                    node,
                    context,
                    provenance,
                    clear_positive_indices=partition.positive,
                    grey_indices=partition.grey,
                )
            relation_selected = self._selected_like(
                candidates,
                partition.positive,
                provenance={"select": "RELATION", **provenance},
            )
            status = self._cardinality_status(len(partition.positive), self._selection_type(node))
            final_ids = self._candidate_ids(items, partition.positive)
            return self._result(
                relation_selected,
                status=status,
                method="geometry",
                reason=(
                    "single_selection_multiple_matches"
                    if status is SelectStatus.AMBIGUOUS
                    else None
                ),
                provenance={
                    **provenance,
                    "semantic_positive_candidate_ids": [],
                    "final_candidate_ids": final_ids,
                    "candidate_ids": final_ids,
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
        model_role: str | None = None,
        choice_config: ChoiceSystemConfig | None = None,
        semantic_config: SemanticDecisionConfig | None = None,
    ) -> None:
        self.provider = provider
        self.provider_name = provider_name or provider.provider_name
        self.model_role = model_role or getattr(provider, "role", None)
        self.choice_config = choice_config or ChoiceSystemConfig()
        self.semantic_config = semantic_config or SemanticDecisionConfig()
        self.decisions = SemanticDecisionLayer(provider, self.semantic_config)

    @staticmethod
    def _choice_ids(options: tuple[str, ...]) -> tuple[str, ...]:
        if not options or len(options) > 26:
            raise ValueError("fused benchmark choice requires between 1 and 26 options")
        return tuple(chr(ord("A") + index) for index in range(len(options)))

    @staticmethod
    def _trace_metadata(provenance: dict[str, object]) -> dict[str, object]:
        return {
            key: provenance[key]
            for key in (
                "execution_mode",
                "semantic_method",
                "cache_reused",
                "final_choice_fusion",
                "fusion_reason",
                "model_id",
                "model_role",
                "latency_ms",
                "semantic_decision_total_ms",
            )
            if key in provenance
        }

    @staticmethod
    def _free_provenance(
        result: VLMResult,
        *,
        fusion_reason: str,
    ) -> dict[str, object]:
        return {
            "provider": result.provider,
            "model_id": str(result.metadata.get("model_id", "unknown")),
            "method": "free_text_generation",
            "canonical": False,
            "execution_mode": "free_text",
            "semantic_method": "free_generation",
            "cache_reused": False,
            "final_choice_fusion": False,
            "fusion_reason": fusion_reason,
            "generation_metadata": dict(result.metadata),
        }

    def _final_fusion(
        self,
        node: GraphNode,
        inputs: dict[str, RuntimeObject | list[RuntimeObject]],
        context: OperatorContext,
        hint: NodeExecutionHint,
    ) -> OperatorOutcome:
        question = semantic_question(node, hint.question, final_choice_fusion=True)
        model_input = context.composer.compose_named(
            inputs,
            question=question,
            options=hint.options,
        )
        answer_type = hint.answer_type
        if answer_type is None:
            raise RuntimeError("eligible final choice fusion is missing answer_type")
        purpose = (
            "route_choice"
            if node.op is OperatorName.ROUTE_REASON
            else f"final_{node.op.value.casefold()}_choice_fusion"
        )
        scored = self.provider.reason_and_choose(
            ChoiceScoringRequest(
                model_input=model_input,
                answer_type=answer_type.value,
                choice_ids=self._choice_ids(hint.options),
                option_texts=hint.options,
                single_choice_suffix=self.choice_config.single_choice_suffix,
                multi_verify_template=self.choice_config.multi_verify_template,
                multi_select_threshold=self.choice_config.multi_select_threshold,
                purpose=purpose,
            )
        )
        metadata = {
            **scored.metadata,
            "input_metadata": dict(model_input.metadata),
            "execution_mode": "final_choice_fused",
            "semantic_method": "kv_cached_final_choice",
            "final_choice_fusion": True,
            "fusion_reason": "eligible",
        }
        scored = replace(
            scored,
            reasoning_text=(
                scored.reasoning_text if self.choice_config.preserve_reasoning_text else None
            ),
            metadata=metadata,
        )
        trace_metadata: dict[str, object] = {
            "execution_mode": "final_choice_fused",
            "semantic_method": "kv_cached_final_choice",
            "cache_reused": scored.cache_reused,
            "final_choice_fusion": True,
            "fusion_reason": "eligible",
            "model_id": scored.model_id,
            "latency_ms": dict(scored.latency_ms),
        }
        if self.model_role is not None:
            trace_metadata["model_role"] = self.model_role
        route_context = model_input.metadata.get("route_context")
        if route_context is not None:
            trace_metadata["route_context"] = route_context
        return OperatorOutcome(scored, scored.provider, trace_metadata)

    def _free_text(
        self,
        node: GraphNode,
        inputs: dict[str, RuntimeObject | list[RuntimeObject]],
        context: OperatorContext,
        *,
        options: tuple[str, ...] = (),
    ) -> OperatorOutcome:
        hint = context.execution_hint
        fusion_reason = hint.fusion_reason if hint is not None else "not_final_source"
        question = semantic_question(node, context.question, final_choice_fusion=False)
        model_input = context.composer.compose_named(inputs, question=question, options=options)
        result = self.provider.infer(VLMRequest(model_input, node.op.value.casefold()))
        provenance = self._free_provenance(result, fusion_reason=fusion_reason)
        if self.model_role is not None:
            provenance["model_role"] = self.model_role
        if node.op in {OperatorName.ATTRIBUTE, OperatorName.CLASSIFY}:
            value: RuntimeObject = Label(result.text.strip(), provenance)
        else:
            value = Answer(result.text, result.confidence, provenance)
        return OperatorOutcome(value, result.provider, self._trace_metadata(provenance))

    def execute(
        self,
        node: GraphNode,
        inputs: dict[str, RuntimeObject | list[RuntimeObject]],
        context: OperatorContext,
    ) -> OperatorOutcome:
        hint = context.execution_hint
        if hint is not None and hint.final_choice_fusion:
            return self._final_fusion(node, inputs, context, hint)

        # Preserve the frozen same-4B ROUTE_REASON path. Normal runtime graphs
        # mark it eligible; this compatibility branch only serves direct executor use.
        if node.op is OperatorName.ROUTE_REASON:
            compatibility_hint = NodeExecutionHint(
                node_id=node.id,
                final_choice_fusion=True,
                answer_type=AnswerType.CHOICE_SINGLE,
                options=context.choices,
                question=context.question,
                fusion_reason="eligible",
            )
            return self._final_fusion(node, inputs, context, compatibility_hint)

        fusion_reason = hint.fusion_reason if hint is not None else "not_final_source"
        question = semantic_question(node, context.question, final_choice_fusion=False)
        instruction = semantic_reasoning_instruction(node)

        candidates: tuple[str, ...] | None = None
        if node.op is OperatorName.RELATION:
            candidates = tuple(relation.value for relation in SpatialRelation)
        elif node.op is OperatorName.CLASSIFY:
            candidates = tuple(str(item) for item in (node.params.get("label_space") or ()))
            if not candidates:
                return self._free_text(node, inputs, context)
        elif node.op is OperatorName.MULTILABEL_CLASSIFY:
            candidates = tuple(str(item) for item in node.params["label_space"])
        elif node.op is OperatorName.ATTRIBUTE:
            candidates = self.semantic_config.attribute_values(str(node.params["attribute"]))
            if not candidates:
                return self._free_text(node, inputs, context)
        elif node.op is OperatorName.MOTION:
            candidates = ("YES", "NO")
        elif node.op is OperatorName.VLM_REASON:
            configured_choices = node.params.get("choices")
            options = (
                context.choices
                if configured_choices == "$choices"
                else tuple(configured_choices or ())
            )
            return self._free_text(node, inputs, context, options=options)
        elif node.op is OperatorName.MATCH_CHOICE:
            return self._free_text(node, inputs, context, options=context.choices)
        else:
            return self._free_text(node, inputs, context)

        model_input = context.composer.compose_named(
            inputs,
            question=question,
            options=candidates,
        )
        value: RuntimeObject
        if node.op is OperatorName.MOTION:
            moving, provenance = self.decisions.verify(
                model_input,
                purpose="semantic_motion",
                reasoning_instruction=instruction,
            )
            provenance = {**provenance, "fusion_reason": fusion_reason}
            if self.model_role is not None:
                provenance["model_role"] = self.model_role
            value = Boolean(moving, provenance)
        elif node.op is OperatorName.MULTILABEL_CLASSIFY:
            decision = self.decisions.choose_many(
                model_input,
                candidates,
                purpose="semantic_multilabel_classify",
                reasoning_instruction=instruction,
            )
            provenance = {**decision.provenance, "fusion_reason": fusion_reason}
            if self.model_role is not None:
                provenance["model_role"] = self.model_role
            value = LabelSet(decision.values, provenance)
        else:
            purpose = {
                OperatorName.RELATION: "semantic_relation",
                OperatorName.CLASSIFY: "semantic_classify",
                OperatorName.ATTRIBUTE: "semantic_attribute",
            }[node.op]
            decision = self.decisions.choose_one(
                model_input,
                candidates,
                purpose=purpose,
                reasoning_instruction=instruction,
            )
            provenance = {**decision.provenance, "fusion_reason": fusion_reason}
            if self.model_role is not None:
                provenance["model_role"] = self.model_role
            value = Label(decision.values[0], provenance)
        return OperatorOutcome(
            value,
            str(provenance["provider"]),
            self._trace_metadata(provenance),
        )
