"""Deterministic and capability-backed TaskGraph operator executors."""

from __future__ import annotations

import re
from dataclasses import dataclass
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


def _region(value: RuntimeObject) -> Region:
    if isinstance(value, Region):
        return value
    if isinstance(value, Entity):
        return value.region
    if isinstance(value, EntitySet) and value.entities:
        return value.entities[0].region
    raise TypeError(f"cannot resolve region from {type(value).__name__}")


def _dimensions(value: ImageRef | Region) -> tuple[int, int]:
    image = value if isinstance(value, ImageRef) else value.image
    if image.width and image.height:
        return image.width, image.height
    with Image.open(image.path.resolve()) as source:
        return source.size


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
        named = {
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "yellow": (255, 255, 0),
        }
        target = named.get((color or "red").casefold(), named["red"])
        with Image.open(image.path.resolve()) as raw:
            crop = raw.convert("RGB").crop(scope)
            pixels = crop.load()
            hits = []
            for y in range(crop.height):
                for x in range(crop.width):
                    pixel = pixels[x, y]
                    if all(pixel[index] >= target[index] - 60 for index in range(3)) and (
                        max(pixel) - min(pixel) >= 80
                    ):
                        hits.append((x, y))
        if not hits:
            return RegionSet((), {"operator": "FIND_MARKER", "color": color})
        offset_x, offset_y = scope[0], scope[1]
        box = (
            min(item[0] for item in hits) + offset_x,
            min(item[1] for item in hits) + offset_y,
            max(item[0] for item in hits) + offset_x + 1,
            max(item[1] for item in hits) + offset_y + 1,
        )
        return RegionSet((Region(image, box, {"operator": "FIND_MARKER"}),))

    @staticmethod
    def _route_context(
        image_value: RuntimeObject, start: RuntimeObject, goal: RuntimeObject
    ) -> RouteContext:
        image = _image(image_value)
        start_region, goal_region = _region(start), _region(goal)
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
        context_region = Region(
            image,
            (max(0.0, x0 - pad), max(0.0, y0 - pad), min(width, x1 + pad), min(height, y1 + pad)),
            {"coordinate_transform": {"origin_global": [x0, y0]}},
        )
        return RouteContext(
            image=image,
            start=start,  # type: ignore[arg-type]
            goal=goal,  # type: ignore[arg-type]
            context_region=context_region,
            provenance={"provider": "geometry", "ambiguity_policy": "highest_score_first"},
        )

    def execute(
        self,
        node: GraphNode,
        inputs: dict[str, RuntimeObject | list[RuntimeObject]],
        context: OperatorContext,
    ) -> OperatorOutcome:
        if node.op is OperatorName.REGION:
            source = inputs["image"]
            assert isinstance(source, (ImageRef, Region))
            value: RuntimeObject = self._position_region(source, str(node.params["position"]))
        elif node.op is OperatorName.REGION_FROM_BBOX:
            image = inputs["image"]
            assert isinstance(image, ImageRef)
            value = Region(image, tuple(node.params["bbox"]), {"operator": node.op.value})
        elif node.op is OperatorName.FIND_MARKER:
            source = inputs["image"]
            assert isinstance(source, (ImageRef, Region))
            value = self._marker(source, node.params["marker"].get("color"))
        elif node.op is OperatorName.GROUP:
            entities = inputs["entities"]
            if not isinstance(entities, EntitySet):
                raise TypeError("GROUP.entities must be EntitySet")
            value = EntitySet(
                entities.entities, {**entities.provenance, "group": node.params["mode"]}
            )
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
                RegionRetrievalRequest(scope, target.phrase(), max_candidates=8)
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
                sources: list[RuntimeObject] = [candidates]
                reference = inputs.get("reference")
                if isinstance(reference, list):
                    sources.extend(reference)
                elif reference is not None:
                    sources.append(reference)
                model_input = context.composer.compose(
                    sources,
                    question=(
                        f"Select candidate ids satisfying relation {node.params['relation']}. "
                        "Return only numeric candidate ids."
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
        sources = []
        for value in inputs.values():
            sources.extend(value if isinstance(value, list) else [value])
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
        model_input = context.composer.compose(sources, question=question, options=options)
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
