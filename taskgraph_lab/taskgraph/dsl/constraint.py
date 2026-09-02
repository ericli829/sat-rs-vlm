"""Canonical DSL prefix grammar used by constrained Planner decoding.

The compiler output is the authoritative language.  This module recognizes
canonical compiler-style text only (no optional whitespace) and deliberately
does not replace the existing parser or semantic validator.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from itertools import chain

import regex

from taskgraph_lab.taskgraph.enums import (
    AnswerType,
    ExtremeDirection,
    GroupMode,
    IntentLabel,
    OperatorName,
    RegionPosition,
    SortOrder,
    SpatialRelation,
    SubregionType,
)
from taskgraph_lab.taskgraph.schema import INTRINSIC_ATTRIBUTES
from taskgraph_lab.taskgraph.type_checker import OUTPUT_TYPES, SIGNATURES


def _alternation(values: Iterable[str]) -> str:
    ordered = sorted({str(value) for value in values}, key=lambda item: (-len(item), item))
    if not ordered:
        return r"(?!)"
    return "(?:" + "|".join(re.escape(value) for value in ordered) + ")"


def _enum_pattern(enum_type: type[Enum]) -> str:
    return _alternation(str(item.value) for item in enum_type)


JSON_STRING = r'"(?:[^"\\\x00-\x1f]|\\["\\/bfnrt]|\\u[0-9A-Fa-f]{4})*"'
JSON_NUMBER = r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
POSITIVE_INTEGER = r"[1-9][0-9]*"
SCALAR = rf"(?:{JSON_STRING}|{JSON_NUMBER}|true|false)"
STRING_LIST = rf"(?:\[\]|\[{JSON_STRING}(?:,{JSON_STRING})*\])"
NONEMPTY_STRING_LIST = rf"\[{JSON_STRING}(?:,{JSON_STRING})*\]"
CHOICES_REFERENCE = re.escape(json.dumps("$choices"))

INTENT_PATTERN = regex.compile(rf"INTENT\({_enum_pattern(IntentLabel)}\)")
ANSWER_TYPE_PATTERN = _enum_pattern(AnswerType)


def _target_pattern() -> str:
    # The parser/compiler owns canonical attribute ordering.  The generation
    # grammar accepts any order and rejects duplicates when a node closes.
    name = _alternation(sorted(INTRINSIC_ATTRIBUTES))
    return rf"T\({JSON_STRING}(?:,{name}={SCALAR})*\)"


TARGET = _target_pattern()


@dataclass(frozen=True)
class PrefixAnalysis:
    valid_prefix: bool
    complete: bool = False
    reason: str | None = None
    node_count: int = 0
    current_node_complete: bool = False
    force_final: bool = False


@dataclass(frozen=True)
class _ProgramState:
    intent_seen: bool
    node_lines: tuple[str, ...]
    outputs: tuple[tuple[str, frozenset], ...]

    @property
    def output_map(self) -> dict[str, set]:
        return {name: set(values) for name, values in self.outputs}


def _full_or_partial(pattern: regex.Pattern, text: str) -> tuple[bool, bool]:
    match = pattern.fullmatch(text, partial=True)
    return match is not None, bool(match is not None and not match.partial)


def _reference_pattern(references: Iterable[str]) -> str:
    return _alternation(references)


def _reference_list(reference: str) -> str:
    return rf"\[{reference}(?:,{reference})*\]"


def _node_surface(line: str) -> str:
    match = re.match(r"^n[1-9][0-9]*=([A-Z_]+)\(", line)
    if match is None:
        raise ValueError(f"canonical node line has no surface operator: {line!r}")
    return match.group(1)


def _canonical_operator(surface: str) -> OperatorName:
    if surface == "REGION_BBOX":
        return OperatorName.REGION_FROM_BBOX
    if surface.startswith("SELECT_"):
        return OperatorName.SELECT
    if surface in {"COUNT", "COUNT_IMAGE", "COUNT_ENTITIES"}:
        return OperatorName.COUNT
    return OperatorName(surface)


def _normalized_node_signature(line: str) -> str:
    expression = line.split("=", 1)[1]
    return re.sub(r"\$n[1-9][0-9]*", "$n", expression)


def _has_repeated_cycle(lines: tuple[str, ...], repetitions: int) -> bool:
    if repetitions < 2:
        return False
    signatures = tuple(_normalized_node_signature(line) for line in lines)
    for period in (1, 2, 3):
        width = period * repetitions
        if len(signatures) < width:
            continue
        tail = signatures[-width:]
        if all(tail[index : index + period] == tail[:period] for index in range(0, width, period)):
            return True
    return False


def _has_duplicate_target_attribute(line: str) -> bool:
    """Detect duplicate T(...) keys after a node reaches a complete surface."""

    from .errors import DSLParseError
    from .grammar import parse_syntax

    node_id = line.split("=", 1)[0]
    try:
        parse_syntax(f"{line}\nFINAL(${node_id},TEXT)")
    except DSLParseError as exc:
        return "duplicate named argument" in str(exc)
    return False


class CanonicalDSLPrefixGrammar:
    """Recognize prefixes of the exact language emitted by the DSL compiler.

    Dynamic references are request-scoped: image references come from the
    caller and node references can point only to preceding contiguous nodes.
    ``max_nodes`` and ``repeat_guard_repetitions`` are operational decoding
    guards.  Leave both disabled when checking the compiler-language invariant.
    """

    def __init__(
        self,
        image_refs: Iterable[str],
        *,
        max_nodes: int | None = None,
        repeat_guard_repetitions: int | None = None,
    ) -> None:
        normalized: set[str] = set()
        for raw in image_refs:
            value = str(raw)
            value = value if value.startswith("$") else f"${value}"
            if re.fullmatch(r"\$image(?:0|[1-9][0-9]*)", value) is None:
                raise ValueError(f"invalid image reference for DSL constraint: {raw!r}")
            normalized.add(value)
        if not normalized:
            raise ValueError("DSL constraint requires at least one caller-provided image ref")
        if max_nodes is not None and max_nodes < 1:
            raise ValueError("max_nodes must be positive when configured")
        if repeat_guard_repetitions is not None and repeat_guard_repetitions < 2:
            raise ValueError("repeat_guard_repetitions must be >= 2 when configured")
        self.image_refs = tuple(sorted(normalized))
        self.max_nodes = max_nodes
        self.repeat_guard_repetitions = repeat_guard_repetitions

    def _allowed_references(self, node_count: int) -> tuple[str, ...]:
        nodes = tuple(f"$n{index}" for index in range(1, node_count + 1))
        return tuple(chain(self.image_refs, nodes))

    def _count_compact_references(
        self, references: tuple[str, ...], outputs: dict[str, set]
    ) -> tuple[str, ...]:
        allowed: list[str] = []
        for reference in references:
            if reference.startswith("$image"):
                # Importing RuntimeType solely for this branch would obscure the
                # contract; image refs can match COUNT.image and never entities.
                candidates = ["image"]
            else:
                actual = outputs.get(reference, set())
                candidates = [
                    role
                    for role in ("image", "entities")
                    if actual.intersection(SIGNATURES[OperatorName.COUNT][role])
                ]
            if len(candidates) == 1:
                allowed.append(reference)
        return tuple(allowed)

    def _node_pattern(self, state: _ProgramState) -> regex.Pattern:
        node_index = len(state.node_lines) + 1
        references = self._allowed_references(len(state.node_lines))
        ref = _reference_pattern(references)
        ref_or_null = rf"(?:{ref}|null)"
        ref_or_list = rf"(?:{ref}|{_reference_list(ref)})"
        count_ref = _reference_pattern(
            self._count_compact_references(references, state.output_map)
        )
        number4 = rf"\[{JSON_NUMBER},{JSON_NUMBER},{JSON_NUMBER},{JSON_NUMBER}\]"
        image_size = rf"\[{POSITIVE_INTEGER},{POSITIVE_INTEGER}\]"
        position = _enum_pattern(RegionPosition)
        relation = _enum_pattern(SpatialRelation)
        rank_order = _alternation(
            (SortOrder.ASCENDING.value, SortOrder.DESCENDING.value)
        )
        ordinal_order = _alternation(
            (
                SortOrder.TOP_TO_BOTTOM.value,
                SortOrder.BOTTOM_TO_TOP.value,
                SortOrder.LEFT_TO_RIGHT.value,
                SortOrder.RIGHT_TO_LEFT.value,
            )
        )
        extreme = _enum_pattern(ExtremeDirection)
        subregion = _enum_pattern(SubregionType)
        group = _enum_pattern(GroupMode)

        choices = CHOICES_REFERENCE
        vlm_variants = []
        for include_image in (False, True):
            for include_evidence in (False, True):
                values: list[str] = []
                if include_image:
                    values.append(rf"image={ref}")
                if include_evidence:
                    values.append(rf"evidence={ref_or_list}")
                values.append(rf"question={JSON_STRING}")
                values.append(rf"(?:,choices={choices})?")
                rendered = ",".join(values[:-1]) + values[-1]
                vlm_variants.append(rf"VLM_REASON\({rendered}\)")

        calls = [
            rf"REGION\({ref},{position}\)",
            rf"REGION_BBOX\({ref},{number4}(?:,{image_size})?\)",
            rf"FIND_MARKER\({ref},{JSON_STRING}(?:,{JSON_STRING})?\)",
            rf"LOCATE\({ref},{TARGET}\)",
            rf"SELECT_REL\({ref},{ref_or_null},{relation}\)",
            rf"SELECT_RANK\({ref},{ref_or_null},{JSON_STRING},{POSITIVE_INTEGER},{rank_order}\)",
            rf"SELECT_ORD\({ref},{ref_or_null},{POSITIVE_INTEGER},{ordinal_order}\)",
            rf"SELECT_EXTREME\({ref},{ref_or_null},{extreme}\)",
            rf"SELECT_SUBREGION\({ref},{ref_or_null},{subregion}\)",
            rf"GROUP\({ref},{group}\)",
            rf"COUNT\({count_ref},{TARGET},(?:true|false)\)",
            rf"COUNT_IMAGE\({ref},{TARGET},(?:true|false)\)",
            rf"COUNT_ENTITIES\({ref},{TARGET},(?:true|false)\)",
            rf"ATTRIBUTE\({ref},{JSON_STRING}(?:,{JSON_STRING})?\)",
            rf"CLASSIFY\({ref}(?:,{STRING_LIST})?\)",
            rf"MULTILABEL_CLASSIFY\({ref},{NONEMPTY_STRING_LIST}\)",
            rf"MOTION\({ref}\)",
            rf"RELATION\({ref},{ref}\)",
            rf"ABS_DIFF\({ref},{ref}\)",
            *vlm_variants,
            rf"BUILD_ROUTE_CONTEXT\({ref},{ref},{ref}\)",
            rf"ROUTE_REASON\({ref},{JSON_STRING},{choices}\)",
            rf"MATCH_CHOICE\({ref},{choices}\)",
        ]
        return regex.compile(rf"n{node_index}=(?:{'|'.join(calls)})")

    def _final_pattern(self, node_count: int) -> regex.Pattern:
        node_ref = _reference_pattern(f"$n{index}" for index in range(1, node_count + 1))
        source = rf"(?:{node_ref}|{_reference_list(node_ref)})"
        return regex.compile(
            rf"(?:FINAL\({source},{ANSWER_TYPE_PATTERN}\)|"
            rf"FINAL_QUESTION\({source},{ANSWER_TYPE_PATTERN},{JSON_STRING}\))"
        )

    def _append_node(self, state: _ProgramState, line: str) -> _ProgramState:
        surface = _node_surface(line)
        operator = _canonical_operator(surface)
        node_index = len(state.node_lines) + 1
        outputs = dict(state.outputs)
        outputs[f"$n{node_index}"] = frozenset(OUTPUT_TYPES[operator])
        return _ProgramState(
            intent_seen=state.intent_seen,
            node_lines=(*state.node_lines, line),
            outputs=tuple(outputs.items()),
        )

    def _repeat_blocked(self, node_lines: tuple[str, ...]) -> bool:
        repetitions = self.repeat_guard_repetitions
        return bool(repetitions and _has_repeated_cycle(node_lines, repetitions))

    def analyze(self, text: str) -> PrefixAnalysis:
        if "\r" in text:
            return PrefixAnalysis(False, reason="noncanonical_carriage_return")
        lines = text.split("\n")
        completed_lines = lines[:-1]
        current = lines[-1]
        state = _ProgramState(intent_seen=False, node_lines=(), outputs=())
        force_final = False

        for line in completed_lines:
            if not state.intent_seen and not state.node_lines:
                intent_valid, intent_full = _full_or_partial(INTENT_PATTERN, line)
                if intent_valid and intent_full:
                    state = _ProgramState(True, (), ())
                    continue
            if force_final:
                return PrefixAnalysis(
                    False,
                    reason="node_after_repeat_guard",
                    node_count=len(state.node_lines),
                    force_final=True,
                )
            if self.max_nodes is not None and len(state.node_lines) >= self.max_nodes:
                return PrefixAnalysis(False, reason="max_nodes", node_count=len(state.node_lines))
            node_pattern = self._node_pattern(state)
            node_valid, node_full = _full_or_partial(node_pattern, line)
            if not node_valid or not node_full:
                return PrefixAnalysis(
                    False,
                    reason="invalid_completed_line",
                    node_count=len(state.node_lines),
                )
            if _has_duplicate_target_attribute(line):
                return PrefixAnalysis(
                    False,
                    reason="duplicate_target_attribute",
                    node_count=len(state.node_lines),
                )
            candidate = self._append_node(state, line)
            if self._repeat_blocked(candidate.node_lines):
                force_final = True
            state = candidate

        if not state.intent_seen and not state.node_lines:
            intent_valid, intent_full = _full_or_partial(INTENT_PATTERN, current)
            node_valid, node_full = _full_or_partial(self._node_pattern(state), current)
            if intent_valid or node_valid:
                if node_full:
                    if _has_duplicate_target_attribute(current):
                        return PrefixAnalysis(
                            False,
                            reason="duplicate_target_attribute",
                            node_count=0,
                        )
                    candidate = self._append_node(state, current)
                    if self._repeat_blocked(candidate.node_lines):
                        return PrefixAnalysis(
                            True,
                            reason="repeat_guard",
                            node_count=len(candidate.node_lines),
                            current_node_complete=True,
                            force_final=True,
                        )
                return PrefixAnalysis(
                    True,
                    node_count=1 if node_full else 0,
                    current_node_complete=node_full,
                )
            return PrefixAnalysis(False, reason="invalid_program_start")

        final_valid = False
        final_full = False
        if state.node_lines:
            final_valid, final_full = _full_or_partial(
                self._final_pattern(len(state.node_lines)), current
            )
        if force_final:
            if final_full:
                return PrefixAnalysis(
                    True,
                    complete=True,
                    reason="repeat_guard",
                    node_count=len(state.node_lines),
                    force_final=True,
                )
            if final_valid:
                return PrefixAnalysis(
                    True,
                    reason="repeat_guard",
                    node_count=len(state.node_lines),
                    force_final=True,
                )
            return PrefixAnalysis(
                False,
                reason="forced_final_required",
                node_count=len(state.node_lines),
                force_final=True,
            )
        node_valid = False
        node_full = False
        if self.max_nodes is None or len(state.node_lines) < self.max_nodes:
            node_valid, node_full = _full_or_partial(self._node_pattern(state), current)
        if final_full:
            return PrefixAnalysis(
                True,
                complete=True,
                reason="final",
                node_count=len(state.node_lines),
            )
        if node_full:
            if _has_duplicate_target_attribute(current):
                return PrefixAnalysis(
                    False,
                    reason="duplicate_target_attribute",
                    node_count=len(state.node_lines),
                )
            candidate = self._append_node(state, current)
            if self._repeat_blocked(candidate.node_lines):
                return PrefixAnalysis(
                    True,
                    reason="repeat_guard",
                    node_count=len(candidate.node_lines),
                    current_node_complete=True,
                    force_final=True,
                )
        if final_valid or node_valid:
            return PrefixAnalysis(
                True,
                node_count=len(state.node_lines) + (1 if node_full else 0),
                current_node_complete=node_full,
            )
        reason = (
            "max_nodes"
            if self.max_nodes is not None and len(state.node_lines) >= self.max_nodes
            else "grammar_mismatch"
        )
        return PrefixAnalysis(False, reason=reason, node_count=len(state.node_lines))

    def accepts(self, text: str) -> bool:
        result = self.analyze(text)
        return result.valid_prefix and result.complete
