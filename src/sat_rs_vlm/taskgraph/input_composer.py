"""Typed multi-source materialization for semantic models and choice resolution."""

from __future__ import annotations

import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

from .providers import ModelInput
from .runtime_types import (
    Answer,
    Boolean,
    Entity,
    EntitySet,
    Evidence,
    EvidenceSet,
    ImageRef,
    Label,
    LabelSet,
    Region,
    RegionSet,
    RouteContext,
    RuntimeObject,
    ScalarFloat,
    ScalarInt,
)


class InputComposer:
    """Preserve visual/structured channels instead of coercing objects with str()."""

    def __init__(self, output_dir: str | Path | None = None) -> None:
        self._temporary = None
        if output_dir is None:
            self._temporary = tempfile.TemporaryDirectory(prefix="taskgraph_inputs_")
            output_dir = self._temporary.name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._counter = 0

    def close(self) -> None:
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None

    def _next_path(self, suffix: str = ".png") -> Path:
        self._counter += 1
        return self.output_dir / f"visual_{self._counter:05d}{suffix}"

    def _crop(self, region: Region) -> str:
        source_path = region.image.path.resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"visual source does not exist: {source_path}")
        output = self._next_path()
        with Image.open(source_path) as source:
            source.convert("RGB").crop(region.bbox_xyxy_global).save(output)
        return str(output)

    def _entity_set_visual(self, entities: EntitySet) -> str:
        if not entities.entities:
            raise ValueError("cannot materialize an empty EntitySet")
        image = entities.entities[0].region.image
        output = self._next_path()
        with Image.open(image.path.resolve()) as source:
            canvas = source.convert("RGB")
            draw = ImageDraw.Draw(canvas)
            for index, entity in enumerate(entities.entities):
                draw.rectangle(entity.region.bbox_xyxy_global, outline="red", width=3)
                draw.text(
                    (entity.region.bbox_xyxy_global[0] + 2, entity.region.bbox_xyxy_global[1] + 2),
                    str(index),
                    fill="red",
                )
            canvas.save(output)
        return str(output)

    def _route_visual(self, context: RouteContext) -> str:
        if context.marker_visual_path:
            return context.marker_visual_path
        output = self._next_path()
        with Image.open(context.image.path.resolve()) as source:
            canvas = source.convert("RGB").crop(context.context_region.bbox_xyxy_global)
            draw = ImageDraw.Draw(canvas)
            for label, value, color in (
                ("START", context.start, "green"),
                ("GOAL", context.goal, "red"),
            ):
                entity = value.entities[0] if isinstance(value, EntitySet) else value
                region = entity.region if isinstance(entity, Entity) else entity
                x0, y0, x1, y1 = context.context_region.bbox_xyxy_global
                box = region.bbox_xyxy_global
                local = (box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0)
                draw.rectangle(local, outline=color, width=4)
                draw.text((local[0], local[1]), label, fill=color)
            canvas.save(output)
        return str(output)

    def _visuals(self, value: RuntimeObject) -> list[str]:
        if isinstance(value, ImageRef):
            return [str(value.path.resolve())]
        if isinstance(value, Region):
            return [self._crop(value)]
        if isinstance(value, RegionSet):
            return [self._crop(region) for region in value.regions]
        if isinstance(value, Entity):
            return [self._crop(value.region)]
        if isinstance(value, EntitySet):
            if len(value.entities) == 1:
                return [self._crop(value.entities[0].region)]
            return [self._entity_set_visual(value)]
        if isinstance(value, RouteContext):
            return [self._route_visual(value)]
        if isinstance(value, Evidence):
            return self._visuals(value.value)
        if isinstance(value, EvidenceSet):
            return [path for item in value.evidence for path in self._visuals(item.value)]
        return []

    @staticmethod
    def _structured(value: RuntimeObject, result_id: int) -> str | None:
        header = f"[result_{result_id}]\ntype: {type(value).__name__}"
        if isinstance(value, (ScalarInt, ScalarFloat, Boolean, Label)):
            return f"{header}\nvalue: {value.value}"
        if isinstance(value, LabelSet):
            values = "\n".join(f"- {item}" for item in value.values)
            return f"{header}\nvalues:\n{values}"
        if isinstance(value, Answer):
            return f"{header}\nvalue: {value.text}"
        if isinstance(value, Evidence):
            return InputComposer._structured(value.value, result_id)
        return None

    def compose(
        self,
        sources: list[RuntimeObject],
        *,
        question: str,
        options: list[str] | tuple[str, ...] | None = None,
    ) -> ModelInput:
        visuals: list[str] = []
        structured: list[str] = []
        for index, source in enumerate(sources, start=1):
            visuals.extend(self._visuals(source))
            value = self._structured(source, index)
            if value is not None:
                structured.append(value)
        return ModelInput(
            visual_inputs=tuple(visuals),
            structured_context="\n\n".join(structured),
            question=question,
            options=tuple(options or ()),
            metadata={"source_types": [type(source).__name__ for source in sources]},
        )
