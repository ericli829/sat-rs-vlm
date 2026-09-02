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
    SelectResult,
    unwrap_select_result,
)


class InputComposer:
    """Preserve input roles and coordinate transforms across model boundaries."""

    def __init__(
        self,
        output_dir: str | Path | None = None,
        *,
        candidate_halo_ratio: float = 0.2,
        entity_set_union_area_threshold: float = 0.55,
        entity_set_max_side: int = 1536,
        entity_set_max_crops: int = 16,
        route_max_side: int = 1536,
    ) -> None:
        if not 0.0 <= candidate_halo_ratio <= 1.0:
            raise ValueError("candidate_halo_ratio must be between 0 and 1")
        if route_max_side < 256:
            raise ValueError("route_max_side must be at least 256")
        if not 0.0 < entity_set_union_area_threshold <= 1.0:
            raise ValueError("entity_set_union_area_threshold must be in (0, 1]")
        if entity_set_max_side < 256:
            raise ValueError("entity_set_max_side must be at least 256")
        if entity_set_max_crops < 1:
            raise ValueError("entity_set_max_crops must be positive")
        self._temporary = None
        if output_dir is None:
            self._temporary = tempfile.TemporaryDirectory(prefix="taskgraph_inputs_")
            output_dir = self._temporary.name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.candidate_halo_ratio = candidate_halo_ratio
        self.entity_set_union_area_threshold = entity_set_union_area_threshold
        self.entity_set_max_side = entity_set_max_side
        self.entity_set_max_crops = entity_set_max_crops
        self.route_max_side = route_max_side
        self._counter = 0
        self._artifact_paths: list[str] = []

    def close(self) -> None:
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None

    def _next_path(self, suffix: str = ".png") -> Path:
        while True:
            self._counter += 1
            output = self.output_dir / f"visual_{self._counter:05d}{suffix}"
            if not output.exists():
                self._artifact_paths.append(str(output))
                return output

    def artifact_checkpoint(self) -> int:
        return len(self._artifact_paths)

    def artifact_paths_since(self, checkpoint: int) -> tuple[str, ...]:
        if checkpoint < 0 or checkpoint > len(self._artifact_paths):
            raise ValueError("artifact checkpoint is outside the composer history")
        return tuple(path for path in self._artifact_paths[checkpoint:] if Path(path).is_file())

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
        if isinstance(value, SelectResult):
            materialized = unwrap_select_result(
                value,
                allow_empty=False,
                consumer="InputComposer visual regions",
            )
            return InputComposer._regions(materialized)
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
            source_provenance = entity.provenance if entity is not None else region.provenance
            candidate_id = source_provenance.get("candidate_id")
            if candidate_id is not None:
                entry["candidate_id"] = str(candidate_id)
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

    def _entity_set_visuals_with_metadata(
        self, entities: EntitySet
    ) -> tuple[list[str], dict[str, object]]:
        if not entities.entities:
            raise ValueError("cannot materialize an empty EntitySet")
        image = entities.entities[0].region.image
        image_key = image.path.resolve()
        if any(entity.region.image.path.resolve() != image_key for entity in entities.entities):
            raise ValueError("EntitySet visual sources must refer to the same image")
        boxes = [entity.region.bbox_xyxy_global for entity in entities.entities]
        union_box = (
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        )
        outputs: list[str] = []
        canvases_metadata: list[dict[str, object]] = []
        with Image.open(image_key) as source:
            rgb = source.convert("RGB")
            width, height = rgb.size
            image_area = float(width * height)
            union_area = (union_box[2] - union_box[0]) * (union_box[3] - union_box[1])
            union_area_ratio = union_area / image_area if image_area else 1.0
            strategy = (
                "union_crop"
                if union_area_ratio <= self.entity_set_union_area_threshold
                else "bounded_multi_crop"
            )
            if strategy == "union_crop":
                selected_candidate_indices = list(range(len(boxes)))
                omitted_candidate_indices: list[int] = []
                crop_sources = [(None, union_box)]
            else:
                ranked_indices = sorted(
                    range(len(boxes)),
                    key=lambda index: (
                        -(
                            float(entities.entities[index].score)
                            if entities.entities[index].score is not None
                            and math.isfinite(float(entities.entities[index].score))
                            else float("-inf")
                        ),
                        index,
                    ),
                )
                selected_candidate_indices = sorted(ranked_indices[: self.entity_set_max_crops])
                selected_set = set(selected_candidate_indices)
                omitted_candidate_indices = [
                    index for index in range(len(boxes)) if index not in selected_set
                ]
                crop_sources = [(index, boxes[index]) for index in selected_candidate_indices]
            for source_index, source_box in crop_sources:
                source_width = source_box[2] - source_box[0]
                source_height = source_box[3] - source_box[1]
                halo_px = max(
                    4.0,
                    self.candidate_halo_ratio * max(source_width, source_height),
                )
                crop_box = (
                    max(0, math.floor(source_box[0] - halo_px)),
                    max(0, math.floor(source_box[1] - halo_px)),
                    min(width, math.ceil(source_box[2] + halo_px)),
                    min(height, math.ceil(source_box[3] + halo_px)),
                )
                canvas = rgb.crop(crop_box)
                crop_size = canvas.size
                resize_scale = min(1.0, self.entity_set_max_side / float(max(crop_size)))
                if resize_scale < 1.0:
                    render_size = (
                        max(1, round(crop_size[0] * resize_scale)),
                        max(1, round(crop_size[1] * resize_scale)),
                    )
                    canvas = canvas.resize(render_size, Image.Resampling.LANCZOS)
                else:
                    render_size = crop_size
                candidate_metadata: list[dict[str, object]] = []
                draw = ImageDraw.Draw(canvas)
                for index, entity in enumerate(entities.entities):
                    if index not in selected_candidate_indices:
                        continue
                    box = entity.region.bbox_xyxy_global
                    clipped = (
                        max(float(crop_box[0]), box[0]),
                        max(float(crop_box[1]), box[1]),
                        min(float(crop_box[2]), box[2]),
                        min(float(crop_box[3]), box[3]),
                    )
                    if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
                        continue
                    local_box = tuple(
                        (clipped[position] - crop_box[position % 2]) * resize_scale
                        for position in range(4)
                    )
                    marker = self._candidate_id(index)
                    draw.rectangle(local_box, outline="red", width=max(2, round(3 * resize_scale)))
                    draw.text((local_box[0] + 2, local_box[1] + 2), marker, fill="red")
                    candidate_metadata.append(
                        {
                            "id": marker,
                            "index": index,
                            "label": entity.label,
                            "score": entity.score,
                            "bbox_xyxy_global": list(box),
                            "bbox_xyxy_local": list(local_box),
                        }
                    )
                output = self._next_path()
                canvas.save(output)
                outputs.append(str(output))
                canvases_metadata.append(
                    {
                        "source_candidate_index": source_index,
                        "crop_bbox_xyxy_global": list(crop_box),
                        "halo_px": halo_px,
                        "crop_size": list(crop_size),
                        "render_size": list(render_size),
                        "global_to_local": {
                            "operation": "translate_then_scale",
                            "origin_global": [crop_box[0], crop_box[1]],
                            "scale_xy": [resize_scale, resize_scale],
                        },
                        "candidate_boxes": candidate_metadata,
                    }
                )
        return outputs, {
            "strategy": strategy,
            "union_bbox_xyxy_global": list(union_box),
            "union_area_ratio": union_area_ratio,
            "union_area_threshold": self.entity_set_union_area_threshold,
            "halo_ratio": self.candidate_halo_ratio,
            "original_image_size": [width, height],
            "crop_count": len(outputs),
            "requested_candidate_count": len(boxes),
            "selected_candidate_indices": selected_candidate_indices,
            "omitted_candidate_indices": omitted_candidate_indices,
            "max_crops": self.entity_set_max_crops,
            "selection_policy": (
                "all_candidates_for_union_crop"
                if strategy == "union_crop"
                else "score_descending_then_source_index_top_k"
            ),
            "max_render_side": self.entity_set_max_side,
            "canvases": canvases_metadata,
            "whole_image_visual_used": False,
        }

    def _entity_set_visual(self, entities: EntitySet) -> str:
        visuals, _ = self._entity_set_visuals_with_metadata(entities)
        if len(visuals) != 1:
            raise ValueError("distributed EntitySet requires multi-image composition")
        return visuals[0]

    def _route_visual_with_metadata(self, context: RouteContext) -> tuple[str, dict[str, object]]:
        if context.marker_visual_path:
            return context.marker_visual_path, {
                "prompt_version": "route-v1",
                "marker_source": "precomputed",
                "context_bbox_xyxy_global": list(context.context_region.bbox_xyxy_global),
            }
        output = self._next_path()
        crop_box = tuple(
            int(value)
            for value in (
                math.floor(context.context_region.bbox_xyxy_global[0]),
                math.floor(context.context_region.bbox_xyxy_global[1]),
                math.ceil(context.context_region.bbox_xyxy_global[2]),
                math.ceil(context.context_region.bbox_xyxy_global[3]),
            )
        )
        with Image.open(context.image.path.resolve()) as source:
            canvas = source.convert("RGB").crop(crop_box)
            crop_size = canvas.size
            resize_scale = min(1.0, self.route_max_side / float(max(crop_size)))
            if resize_scale < 1.0:
                render_size = (
                    max(1, round(crop_size[0] * resize_scale)),
                    max(1, round(crop_size[1] * resize_scale)),
                )
                canvas = canvas.resize(render_size, Image.Resampling.LANCZOS)
            else:
                render_size = crop_size
            draw = ImageDraw.Draw(canvas)
            marker_width = max(3, round(4 * resize_scale))
            endpoint_mapping: dict[str, object] = {}
            for label, value, color in (
                ("START", context.start, "green"),
                ("GOAL", context.goal, "red"),
            ):
                entity = value.entities[0] if isinstance(value, EntitySet) else value
                region = entity.region if isinstance(entity, Entity) else entity
                box = region.bbox_xyxy_global
                local = (
                    (box[0] - crop_box[0]) * resize_scale,
                    (box[1] - crop_box[1]) * resize_scale,
                    (box[2] - crop_box[0]) * resize_scale,
                    (box[3] - crop_box[1]) * resize_scale,
                )
                draw.rectangle(local, outline=color, width=marker_width)
                center = ((local[0] + local[2]) / 2.0, (local[1] + local[3]) / 2.0)
                radius = max(4.0, 6.0 * resize_scale)
                draw.ellipse(
                    (
                        center[0] - radius,
                        center[1] - radius,
                        center[0] + radius,
                        center[1] + radius,
                    ),
                    outline=color,
                    width=marker_width,
                )
                text_y = max(0.0, local[1] - 12.0)
                draw.text((max(0.0, local[0]), text_y), label, fill=color)
                endpoint_mapping[label.casefold()] = {
                    "bbox_xyxy_global": list(box),
                    "bbox_xyxy_render": list(local),
                    "marker_color": color,
                }
            canvas.save(output)
        return str(output), {
            "prompt_version": "route-v1",
            "marker_source": "runtime",
            "marker_style": "bbox_plus_center_ring",
            "context_bbox_xyxy_global": list(crop_box),
            "context_size": list(crop_size),
            "render_size": list(render_size),
            "resize_scale": resize_scale,
            "max_side": self.route_max_side,
            "coordinate_transform": {
                "global_to_render": "translate_then_scale",
                "origin_global": [crop_box[0], crop_box[1]],
                "scale_xy": [resize_scale, resize_scale],
            },
            "endpoints": endpoint_mapping,
        }

    def _route_visual(self, context: RouteContext) -> str:
        return self._route_visual_with_metadata(context)[0]

    def _visuals(self, value: RuntimeObject) -> list[str]:
        if isinstance(value, SelectResult):
            materialized = unwrap_select_result(
                value,
                allow_empty=False,
                consumer="InputComposer visuals",
            )
            return self._visuals(materialized)
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
            return self._entity_set_visuals_with_metadata(value)[0]
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
        if isinstance(value, Region):
            return f"{header}\nbbox_xyxy_global: {list(value.bbox_xyxy_global)}"
        if isinstance(value, RegionSet):
            boxes = "\n".join(f"- {list(region.bbox_xyxy_global)}" for region in value.regions)
            return f"{header}\nboxes_xyxy_global:\n{boxes or '- []'}"
        if isinstance(value, Entity):
            return (
                f"{header}\nlabel: {value.label}\nscore: {value.score}\n"
                f"bbox_xyxy_global: {list(value.region.bbox_xyxy_global)}"
            )
        if isinstance(value, EntitySet):
            entities = "\n".join(
                f"- index: {index}; label: {entity.label}; score: {entity.score}; "
                f"bbox_xyxy_global: {list(entity.region.bbox_xyxy_global)}"
                for index, entity in enumerate(value.entities)
            )
            return f"{header}\nentities:\n{entities or '- []'}"
        if isinstance(value, RouteContext):
            start = value.start.entities[0] if isinstance(value.start, EntitySet) else value.start
            goal = value.goal.entities[0] if isinstance(value.goal, EntitySet) else value.goal
            start_region = start.region if isinstance(start, Entity) else start
            goal_region = goal.region if isinstance(goal, Entity) else goal
            return (
                f"{header}\nstart_bbox_xyxy_global: {list(start_region.bbox_xyxy_global)}\n"
                f"goal_bbox_xyxy_global: {list(goal_region.bbox_xyxy_global)}\n"
                f"context_bbox_xyxy_global: {list(value.context_region.bbox_xyxy_global)}"
            )
        if isinstance(value, Evidence):
            nested = InputComposer._structured(value.value, role)
            return (
                f"{nested}\nevidence_description: {value.description}"
                if nested is not None
                else None
            )
        if isinstance(value, EvidenceSet):
            entries = [
                InputComposer._structured(item, f"{role}_{index + 1}")
                for index, item in enumerate(value.evidence)
            ]
            return "\n\n".join(item for item in entries if item is not None) or None
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

    @staticmethod
    def _materialize_select_values(
        named: Mapping[str, RuntimeObject | list[RuntimeObject]],
    ) -> dict[str, RuntimeObject | list[RuntimeObject]]:
        materialized: dict[str, RuntimeObject | list[RuntimeObject]] = {}
        for role, value in named.items():
            values = value if isinstance(value, list) else [value]
            safe_values = [
                unwrap_select_result(
                    item,
                    allow_empty=False,
                    consumer=f"InputComposer.{role}",
                )
                for item in values
            ]
            materialized[role] = safe_values if isinstance(value, list) else safe_values[0]
        return materialized

    def compose_named(
        self,
        named: Mapping[str, RuntimeObject | list[RuntimeObject]],
        *,
        question: str,
        options: list[str] | tuple[str, ...] | None = None,
    ) -> ModelInput:
        named = self._materialize_select_values(named)
        if "before" in named and "after" in named:
            # Temporal ordering is semantic, not JSON/dict insertion order.
            named = {
                "before": named["before"],
                "after": named["after"],
                **{role: value for role, value in named.items() if role not in {"before", "after"}},
            }
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
                if isinstance(source.value, RouteContext):
                    route_path, route_metadata = self._route_visual_with_metadata(source.value)
                    source_visuals = [route_path]
                    metadata["route_context"] = route_metadata
                elif isinstance(source.value, EntitySet) and len(source.value.entities) > 1:
                    source_visuals, entity_set_metadata = self._entity_set_visuals_with_metadata(
                        source.value
                    )
                    existing = cast(list[dict[str, object]], metadata.setdefault("entity_sets", []))
                    existing.append({"role": source.role, **entity_set_metadata})
                else:
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
                "visual_paths": list(visuals),
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
