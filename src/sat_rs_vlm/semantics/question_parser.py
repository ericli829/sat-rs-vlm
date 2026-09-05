"""Rules-first question parsing for query-aware locator routing."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Protocol

from .mentions import extract_semantic_facts
from .types import RelationSpec, TaskSpec


class QueryParser(Protocol):
    parser_name: str

    def parse(self, question: str) -> TaskSpec:
        """Parse a natural-language question without loading a large model."""


_OPERATION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("count", re.compile(r"\bhow many\b|\bnumber of\b|多少|几(?:个|艘|架|辆|座|条)")),
    (
        "existence",
        re.compile(r"\b(?:are|is) there\b|\bdo you see\b|\bcan you see\b|是否有|有没有"),
    ),
    (
        "grounding",
        re.compile(r"\b(?:locate|localize|ground)\b|bounding box|bbox|框出|定位"),
    ),
    ("position", re.compile(r"\bwhere\b|what position|which part|在哪里|什么位置|位于何处")),
    (
        "category",
        re.compile(r"\bwhat (?:type|category|kind)\b|\bwhich category\b|什么类型|什么类别"),
    ),
    (
        "global_scene",
        re.compile(
            r"\bwhat (?:scene|land use|land cover)\b|scene classification|场景类型|土地利用"
        ),
    ),
    (
        "open_reasoning",
        re.compile(r"\bwhy\b|\bexplain\b|\bdescribe\b|\bwhat can be inferred\b|为什么|描述|推断"),
    ),
)

_ATTRIBUTE_PATTERNS: dict[str, re.Pattern[str]] = {
    "color": re.compile(r"\b(?:what|which) colou?r\b|颜色"),
    "size": re.compile(r"\b(?:what|which) size\b|how (?:large|small)|尺寸|大小"),
    "shape": re.compile(r"\b(?:what|which) shape\b|形状"),
    "orientation": re.compile(r"\borientation\b|which direction|朝向|方向"),
}

_SPATIAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("upper_left", re.compile(r"upper[- ]left|top[- ]left|north[- ]west|左上|西北")),
    ("upper_right", re.compile(r"upper[- ]right|top[- ]right|north[- ]east|右上|东北")),
    ("lower_left", re.compile(r"lower[- ]left|bottom[- ]left|south[- ]west|左下|西南")),
    ("lower_right", re.compile(r"lower[- ]right|bottom[- ]right|south[- ]east|右下|东南")),
    ("center", re.compile(r"\bcent(?:er|re)\b|middle|中央|中心")),
    ("left", re.compile(r"\bleft(?: side| part)?\b|左侧|左边")),
    ("right", re.compile(r"\bright(?: side| part)?\b|右侧|右边")),
    ("upper", re.compile(r"\bupper(?: side| part)?\b|\btop\b|上方|上部")),
    ("lower", re.compile(r"\blower(?: side| part)?\b|\bbottom\b|下方|下部")),
    ("north", re.compile(r"\bnorthern?\b|北部|北侧")),
    ("south", re.compile(r"\bsouthern?\b|南部|南侧")),
    ("east", re.compile(r"\beastern?\b|东部|东侧")),
    ("west", re.compile(r"\bwestern?\b|西部|西侧")),
)

_BBOX_PATTERN = re.compile(
    r"(?:bbox|box|region|区域|框)\s*[:=]?\s*[\[(]"
    r"\s*(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)\s*[, ]"
    r"\s*(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)\s*[\])]",
    flags=re.IGNORECASE,
)


def _given_bbox(question: str) -> tuple[float, float, float, float] | None:
    match = _BBOX_PATTERN.search(question)
    if match is None:
        return None
    values = tuple(float(value) for value in match.groups())
    if values[2] <= values[0] or values[3] <= values[1]:
        return None
    return values  # type: ignore[return-value]


class RuleBasedQueryParser:
    parser_name = "rules_v1"

    def __init__(self, ontology: dict[str, Any]) -> None:
        self.ontology = ontology

    def parse(self, question: str) -> TaskSpec:
        normalized = question.strip()
        if not normalized:
            raise ValueError("question must not be empty")
        lowered = normalized.lower()
        facts = extract_semantic_facts(lowered, self.ontology)
        attributes = tuple(
            name for name, pattern in _ATTRIBUTE_PATTERNS.items() if pattern.search(lowered)
        )
        relations = tuple(RelationSpec(*relation) for relation in facts.relations)
        operation = "attribute" if attributes else "unknown"
        if operation == "unknown":
            for candidate, pattern in _OPERATION_PATTERNS:
                if pattern.search(lowered):
                    operation = candidate
                    break
        if operation == "unknown" and relations:
            operation = "relation"

        spatial_scope = "global"
        for scope_name, pattern in _SPATIAL_PATTERNS:
            if pattern.search(lowered):
                spatial_scope = scope_name
                break
        bbox = _given_bbox(normalized)
        warnings = list(facts.warnings)
        if operation == "unknown":
            warnings.append("operation_unresolved")
        if operation in {"count", "existence", "attribute", "grounding"} and not facts.objects:
            warnings.append("detector_target_unresolved")
        if "bbox" in lowered and bbox is None:
            warnings.append("given_bbox_unresolved")

        if bbox is not None:
            scope = "given_bbox"
        elif spatial_scope != "global":
            scope = "regional"
        else:
            scope = "image"
        confidence = 0.2 if operation == "unknown" else 0.85
        if operation != "unknown" and facts.objects:
            confidence = 1.0
        elif relations:
            confidence = max(confidence, 0.75)

        return TaskSpec(
            raw_question=normalized,
            operation=operation,
            targets=tuple(facts.objects),
            attributes=attributes,
            relations=relations,
            spatial_scope=spatial_scope,
            scope=scope,
            multi_instance=operation in {"count", "existence"},
            given_bbox=bbox,
            confidence=confidence,
            parser_source=self.parser_name,
            warnings=tuple(dict.fromkeys(warnings)),
        )


class OptionalFallbackQueryParser:
    """Composition point for a future small-LM parser without requiring one."""

    parser_name = "rules_with_optional_fallback"

    def __init__(self, primary: QueryParser, fallback: QueryParser | None = None) -> None:
        self.primary = primary
        self.fallback = fallback

    def parse(self, question: str) -> TaskSpec:
        primary_result = self.primary.parse(question)
        if primary_result.operation != "unknown" or self.fallback is None:
            return primary_result
        fallback_result = self.fallback.parse(question)
        return replace(
            fallback_result,
            warnings=tuple(
                dict.fromkeys((*primary_result.warnings, *fallback_result.warnings))
            ),
        )
