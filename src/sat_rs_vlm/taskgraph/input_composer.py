"""Typed, role-aware materialization for semantic model inputs."""

from __future__ import annotations

import math
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from PIL import Image, ImageDraw

from .providers import ModelInput, ModelSource
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
    """Preserve input roles and coordinate transforms across model boundaries."""

    def __init__(
        self,
        output_dir: str | Path | None = None,
        *,
        candidate_halo_ratio: float = 0.2,
    ) -> None:
        if not 0.0 <= candidate_halo_ratio <= 1.0:
            raise ValueError("candidate_halo_ratio must be between 0 and 1")
        self._temporary = None
        if output_dir is None:
            self._temporary = tempfile.TemporaryDirectory(prefix="taskgraph_inputs_")
            output_dir = self._temporary.name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.candidate_halo_ratio = candidate_halo_ratio
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

    @staticmethod
    def _candidate_id(index: int) -> str:
        return chr(ord("A") + index) if index < 26 else f"C{index + 1}"

    @staticmethod
    def _regions(value: RuntimeObject) -> list[tuple[Region, Entity | None]]:
        if isinstance(value, Region):
            return [(value, None)]
        if isinstance(value, RegionSet):
            return [(region, None) for region in value.regions]
        if isinstance(value, Entity):
            return [(value.region, value)]
        if isinstance(value, EntitySet):
            return [(entity.region, entity) for entity in value.entities]
        return []

    def _candidate_canvas(
        self, named: Mapping[str, RuntimeObject | list[RuntimeObject]]
    ) -> tuple[str, str, dict[str, object], set[str]] | None:
        roles = {role.casefold(): value for role, value in named.items()}
        if "candidates" in roles:
            canvas_role = "CANDIDATE_CANVAS"
            selected_roles = ("candidates", "reference")
        elif "subject" in roles and "reference" in roles:
            canvas_role = "RELATION_CANVAS"
            selected_roles = ("subject", "reference")
        else:
            return None

        items: list[tuple[str, int, Region, Entity | None]] = []
        for role in selected_roles:
            value = roles.get(role)
            values = value if isinstance(value, list) else [value]
            for source in values:
                if source is None:
                    continue
                for index, (region, entity) in enumerate(self._regions(source)):
                    items.append((role, index, region, entity))
        if not items:
            return None
        images = {item[2].image.path.resolve() for item in items}
        if len(images) != 1:
            raise ValueError(f"{canvas_role} sources must refer to the same image")

        source_path = next(iter(images))
        x0 = min(item[2].bbox_xyxy_global[0] for item in items)
        y0 = min(item[2].bbox_xyxy_global[1] for item in items)
        x1 = max(item[2].bbox_xyxy_global[2] for item in items)
        y1 = max(item[2].bbox_xyxy_global[3] for item in items)
        with Image.open(source_path) as source:
            width, height = source.size
            halo = max(4.0, self.candidate_halo_ratio * max(x1 - x0, y1 - y0))
            canvas_box = (
                max(0, math.floor(x0 - halo)),
                max(0, math.floor(y0 - halo)),
                min(width, math.ceil(x1 + halo)),
                min(height, math.ceil(y1 + halo)),
            )
            canvas = source.convert("RGB").crop(canvas_box)
        draw = ImageDraw.Draw(canvas)
        candidate_mapping: dict[str, object] = {}
        role_mapping: dict[str, list[dict[str, object]]] = {}
        for role, index, region, entity in items:
            global_box = region.bbox_xyxy_global
            local_box = (
                global_box[0] - canvas_box[0],
                global_box[1] - canvas_box[1],
                global_box[2] - canvas_box[0],
                global_box[3] - canvas_box[1],
            )
            if role == "candidates":
                marker = self._candidate_id(index)
                color = "red"
            elif role == "subject":
                marker = "SUBJECT" if index == 0 else f"SUBJECT_{index + 1}"
                color = "green"
            elif canvas_role == "CANDIDATE_CANVAS":
                marker = "REF" if index == 0 else f"REF_{index + 1}"
                color = "blue"
            else:
                marker = "REFERENCE" if index == 0 else f"REFERENCE_{index + 1}"
                color = "blue"
            draw.rectangle(local_box, outline=color, width=4)
            draw.text((local_box[0] + 2, local_box[1] + 2), marker, fill=color)
            entry: dict[str, object] = {
                "index": index,
                "bbox_xyxy_global": list(global_box),
                "bbox_xyxy_canvas": list(local_box),
            }
            if entity is not None:
                entry.update({"label": entity.label, "score": entity.score})
            role_mapping.setdefault(role, []).append({"id": marker, **entry})
            if role == "candidates":
                candidate_mapping[marker] = entry

        output = self._next_path()
        canvas.save(output)
        metadata: dict[str, object] = {
            "canvas_kind": canvas_role,
            "canvas_bbox_xyxy_global": list(canvas_box),
            "canvas_size": list(canvas.size),
            "original_image_size": [width, height],
            "coordinate_transform": {
                "origin_global": [canvas_box[0], canvas_box[1]],
                "canvas_to_global": "translate",
            },
            "candidate_mapping": candidate_mapping,
            "role_mapping": role_mapping,
        }
        return str(output), canvas_role, metadata, set(selected_roles)

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
                    self._candidate_id(index),
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
                origin_x, origin_y, _, _ = context.context_region.bbox_xyxy_global
                box = region.bbox_xyxy_global
                local = (
                    box[0] - origin_x,
                    box[1] - origin_y,
                    box[2] - origin_x,
                    box[3] - origin_y,
                )
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
    def _structured(value: RuntimeObject, role: str) -> str | None:
        header = f"[{role}]\ntype: {type(value).__name__}"
        if isinstance(value, (ScalarInt, ScalarFloat, Boolean, Label)):
            return f"{header}\nvalue: {value.value}"
        if isinstance(value, LabelSet):
            values = "\n".join(f"- {item}" for item in value.values)
            return f"{header}\nvalues:\n{values}"
        if isinstance(value, Answer):
            return f"{header}\nvalue: {value.text}"
        if isinstance(value, Evidence):
            return InputComposer._structured(value.value, role)
        return None

    @staticmethod
    def _model_sources(
        named: Mapping[str, RuntimeObject | list[RuntimeObject]],
    ) -> tuple[ModelSource, ...]:
        sources: list[ModelSource] = []
        for role, value in named.items():
            values = value if isinstance(value, list) else [value]
            for index, item in enumerate(values):
                item_role = role if len(values) == 1 else f"{role}_{index + 1}"
                sources.append(ModelSource(item_role.upper(), item))
        return tuple(sources)

    def compose_named(
        self,
        named: Mapping[str, RuntimeObject | list[RuntimeObject]],
        *,
        question: str,
        options: list[str] | tuple[str, ...] | None = None,
    ) -> ModelInput:
        visuals: list[str] = []
        visual_roles: list[str] = []
        structured: list[str] = []
        metadata: dict[str, object] = {}
        consumed: set[str] = set()
        candidate_canvas = self._candidate_canvas(named)
        if candidate_canvas is not None:
            path, role, canvas_metadata, consumed = candidate_canvas
            visuals.append(path)
            visual_roles.append(role)
            metadata.update(canvas_metadata)
            role_mapping = cast(dict[str, list[dict[str, object]]], canvas_metadata["role_mapping"])
            role_lines = [
                f"{name.upper()}: " + "/".join(str(item["id"]) for item in entries)
                for name, entries in role_mapping.items()
            ]
            structured.append("[CANVAS_ROLES]\n" + "\n".join(role_lines))

        sources = self._model_sources(named)
        for source in sources:
            root_role = source.role.casefold().split("_", 1)[0]
            if root_role not in consumed:
                source_visuals = self._visuals(source.value)
                visuals.extend(source_visuals)
                visual_roles.extend([source.role] * len(source_visuals))
            value = self._structured(source.value, source.role)
            if value is not None:
                structured.append(value)
        metadata.update(
            {
                "source_roles": [source.role for source in sources],
                "source_types": [type(source.value).__name__ for source in sources],
                "visual_roles": visual_roles,
            }
        )
        return ModelInput(
            visual_inputs=tuple(visuals),
            visual_roles=tuple(visual_roles),
            structured_context="\n\n".join(structured),
            question=question,
            options=tuple(options or ()),
            metadata=metadata,
        )

    def compose(
        self,
        sources: list[RuntimeObject],
        *,
        question: str,
        options: list[str] | tuple[str, ...] | None = None,
    ) -> ModelInput:
        return self.compose_named(
            {f"result_{index}": source for index, source in enumerate(sources, start=1)},
            question=question,
            options=options,
        )
