"""Deterministic position-choice resolution from global entity coordinates."""

from __future__ import annotations

import re
from pathlib import Path

from .runtime_types import ChoiceScoreResult, Entity, EntitySet, RuntimeObject


class SpatialPositionChoiceResolver:
    _OPTION_PREFIX = re.compile(r"^\s*[\(\[]?[A-Z][\)\].:]?\s*")

    @classmethod
    def _body(cls, option: str) -> str:
        return cls._OPTION_PREFIX.sub("", option).strip().casefold()

    @classmethod
    def _predicate(cls, option: str) -> str | None:
        body = cls._body(option)
        if "doesn't feature" in body or "does not feature" in body or "no position" in body:
            return None
        vertical = None
        horizontal = None
        if re.search(r"upper|top|north", body):
            vertical = "top"
        elif re.search(r"lower|bottom|low|south", body):
            vertical = "bottom"
        if re.search(r"left|west", body):
            horizontal = "left"
        elif re.search(r"right|east", body):
            horizontal = "right"
        middle = bool(re.search(r"middle|center|centre", body))
        edge = "edge" in body
        corner = "corner" in body
        if corner and vertical and horizontal:
            return f"{vertical}_{horizontal}"
        if edge:
            if horizontal:
                return f"middle_{horizontal}_edge" if middle else f"{horizontal}_edge"
            if vertical:
                return f"middle_{vertical}_edge" if middle else f"{vertical}_edge"
            return None
        if vertical and horizontal:
            return f"{vertical}_{horizontal}"
        if horizontal:
            return f"middle_{horizontal}" if middle else horizontal
        if vertical:
            return f"middle_{vertical}" if middle else vertical
        if middle:
            return "center"
        return None

    @classmethod
    def _is_absence(cls, option: str) -> bool:
        body = cls._body(option)
        return (
            "doesn't feature" in body
            or "does not feature" in body
            or "no position" in body
        )

    @staticmethod
    def _is_position_question(question: str | None) -> bool:
        if not question:
            return False
        return bool(
            re.search(
                r"\bwhere\b|\bposition\b|\blocated\b|\bside\b|\bcorner\b|\bedge\b|"
                r"\bupper\b|\blower\b|\bleft\b|\bright\b|\btop\b|\bbottom\b|"
                r"位置|方位|哪边|角落|边缘|左边|右边|上方|下方",
                question.casefold(),
            )
        )

    @staticmethod
    def _singleton(source: RuntimeObject) -> Entity | None:
        if isinstance(source, Entity):
            if source.provenance.get("fallback_required"):
                return None
            return source
        if isinstance(source, EntitySet) and len(source.entities) == 1:
            resolution_status = source.provenance.get("resolution_status")
            unresolved = resolution_status in {"UNRESOLVED", "SEMANTIC_FALLBACK_RESOLVED"}
            if unresolved or source.provenance.get("fallback_required"):
                return None
            entity = source.entities[0]
            return None if entity.provenance.get("fallback_required") else entity
        return None

    @staticmethod
    def _image_size(entity: Entity) -> tuple[float, float]:
        image = entity.region.image
        if image.width and image.height:
            return float(image.width), float(image.height)
        from PIL import Image

        with Image.open(Path(image.uri_or_key).expanduser().resolve()) as source:
            return float(source.width), float(source.height)

    @staticmethod
    def _score(predicate: str, x: float, y: float) -> float:
        targets = {
            "upper_left": (0.0, 0.0),
            "upper_right": (1.0, 0.0),
            "bottom_left": (0.0, 1.0),
            "bottom_right": (1.0, 1.0),
            "top": (0.5, 0.0),
            "bottom": (0.5, 1.0),
            "left": (0.0, 0.5),
            "right": (1.0, 0.5),
            "center": (0.5, 0.5),
            "middle_top": (0.5, 0.0),
            "middle_bottom": (0.5, 1.0),
            "middle_left": (0.0, 0.5),
            "middle_right": (1.0, 0.5),
            "top_edge": (0.5, 0.0),
            "bottom_edge": (0.5, 1.0),
            "left_edge": (0.0, 0.5),
            "right_edge": (1.0, 0.5),
            "middle_top_edge": (0.5, 0.0),
            "middle_bottom_edge": (0.5, 1.0),
            "middle_left_edge": (0.0, 0.5),
            "middle_right_edge": (1.0, 0.5),
        }
        if predicate not in targets:
            return -1.0
        target_x, target_y = targets[predicate]
        distance = ((x - target_x) ** 2 + (y - target_y) ** 2) ** 0.5 / (2.0**0.5)
        return max(0.0, 1.0 - distance)

    def resolve(
        self,
        sources: tuple[RuntimeObject, ...],
        options: tuple[str, ...],
        *,
        question: str | None,
    ) -> ChoiceScoreResult | None:
        if len(sources) != 1:
            return None
        if not self._is_position_question(question):
            return None
        entity = self._singleton(sources[0])
        predicates = [self._predicate(option) for option in options]
        absence_ids = tuple(
            index for index, option in enumerate(options) if self._is_absence(option)
        )
        if entity is None:
            if (
                isinstance(sources[0], EntitySet)
                and not sources[0].entities
                and sources[0].provenance.get("resolution_status", "EMPTY") == "EMPTY"
                and len(absence_ids) == 1
            ):
                selected = chr(ord("A") + absence_ids[0])
                return ChoiceScoreResult(
                    selected_ids=(selected,),
                    scores={
                        chr(ord("A") + index): (1.0 if index == absence_ids[0] else 0.0)
                        for index in range(len(options))
                    },
                    answer_type="CHOICE_SINGLE",
                    reasoning_text=None,
                    provider="spatial_position_geometry",
                    model_id="none",
                    method="spatial_position_absence",
                    cache_reused=False,
                    latency_ms={"total_ms": 0.0},
                    metadata={"model_called": False, "resolution": "EMPTY_ENTITY_SET"},
                )
            return None
        width, height = self._image_size(entity)
        x1, y1, x2, y2 = entity.region.bbox_xyxy_global
        x = ((x1 + x2) / 2.0) / width
        y = ((y1 + y2) / 2.0) / height
        scored: dict[str, float] = {}
        for index, predicate in enumerate(predicates):
            choice_id = chr(ord("A") + index)
            scored[choice_id] = self._score(predicate, x, y) if predicate is not None else -1.0
        valid = [(score, index) for index, score in enumerate(scored.values()) if score >= 0.0]
        if not valid:
            return None
        valid.sort(key=lambda item: (-item[0], item[1]))
        if len(valid) > 1 and valid[0][0] - valid[1][0] <= 1e-6:
            return None
        selected_index = valid[0][1]
        selected_id = chr(ord("A") + selected_index)
        return ChoiceScoreResult(
            selected_ids=(selected_id,),
            scores=scored,
            answer_type="CHOICE_SINGLE",
            reasoning_text=None,
            provider="spatial_position_geometry",
            model_id="none",
            method="spatial_position_geometry",
            cache_reused=False,
            latency_ms={"total_ms": 0.0},
            metadata={
                "model_called": False,
                "question": question,
                "normalized_center": [x, y],
                "predicates": {
                    chr(ord("A") + index): predicate
                    for index, predicate in enumerate(predicates)
                },
                "bbox_xyxy_global": list(entity.region.bbox_xyxy_global),
            },
        )
