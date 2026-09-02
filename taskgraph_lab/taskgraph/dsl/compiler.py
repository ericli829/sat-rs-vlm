from __future__ import annotations

import json
import math
from collections.abc import Mapping
from enum import Enum
from typing import Any

from taskgraph_lab.taskgraph.canonicalize import canonicalize_target
from taskgraph_lab.taskgraph.enums import OperatorName, RuntimeType
from taskgraph_lab.taskgraph.schema import PlannerTarget
from taskgraph_lab.taskgraph.type_checker import OUTPUT_TYPES, SIGNATURES
from taskgraph_lab.taskgraph.validator import validate_candidate

from .errors import DSLCompileError


def _string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _literal(value: Any) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return _string(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DSLCompileError("non-finite numbers are not valid DSL literals")
        return json.dumps(value, allow_nan=False)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_literal(item) for item in value) + "]"
    raise DSLCompileError(f"unsupported DSL literal type: {type(value).__name__}")


def _ref(value: Any) -> str:
    if not isinstance(value, str) or not value.startswith("$"):
        raise DSLCompileError(f"expected canonical reference, got {value!r}")
    return value


def _reference_list(value: Any) -> str:
    if not isinstance(value, (list, tuple)):
        raise DSLCompileError("expected a canonical reference list")
    return "[" + ",".join(_ref(item) for item in value) + "]"


def _target(value: Any) -> str:
    if not isinstance(value, Mapping):
        raise DSLCompileError("TargetSpec must be a mapping")
    category = value.get("category")
    attributes = value.get("attributes", {})
    if not isinstance(category, str) or not isinstance(attributes, Mapping):
        raise DSLCompileError("invalid TargetSpec")
    parts = [_string(category)]
    for key in sorted(attributes):
        parts.append(f"{key}={_literal(attributes[key])}")
    return "T(" + ",".join(parts) + ")"


def _call(name: str, *args: str) -> str:
    return f"{name}({','.join(args)})"


def _named_call(name: str, values: list[tuple[str, str]]) -> str:
    return f"{name}({','.join(f'{key}={value}' for key, value in values)})"


def _static_types(reference: str, outputs: dict[str, set[RuntimeType]]) -> set[RuntimeType]:
    if reference.startswith("$image"):
        return {RuntimeType.IMAGE_REF}
    return outputs.get(reference, set())


def _count_surface(
    role: str,
    source: str,
    outputs: dict[str, set[RuntimeType]],
) -> str:
    actual = _static_types(source, outputs)
    candidates = [
        candidate
        for candidate in ("image", "entities")
        if actual.intersection(SIGNATURES[OperatorName.COUNT][candidate])
    ]
    if candidates == [role]:
        return "COUNT"
    return "COUNT_IMAGE" if role == "image" else "COUNT_ENTITIES"


def _compile_select(node: Mapping[str, Any]) -> str:
    inputs = node["inputs"]
    params = node["params"]
    reference = _ref(inputs["reference"]) if "reference" in inputs else "null"
    prefix = [_ref(inputs["candidates"]), reference]
    mode = params["mode"]
    if mode == "RELATION":
        return _call("SELECT_REL", *prefix, str(params["relation"]))
    if mode == "RANK":
        return _call(
            "SELECT_RANK",
            *prefix,
            _string(str(params["criterion"])),
            str(params["rank"]),
            str(params["order"]),
        )
    if mode == "ORDINAL":
        return _call(
            "SELECT_ORD", *prefix, str(params["index"]), str(params["order"])
        )
    if mode == "EXTREME":
        return _call("SELECT_EXTREME", *prefix, str(params["direction"]))
    if mode == "SUBREGION":
        return _call("SELECT_SUBREGION", *prefix, str(params["subregion"]))
    raise DSLCompileError(f"unsupported SELECT mode {mode!r}")


def _compile_node(node: Mapping[str, Any], outputs: dict[str, set[RuntimeType]]) -> str:
    op = OperatorName(node["op"])
    inputs = node["inputs"]
    params = node["params"]
    if op is OperatorName.REGION:
        return _call("REGION", _ref(inputs["image"]), str(params["position"]))
    if op is OperatorName.REGION_FROM_BBOX:
        args = [_ref(inputs["image"]), _literal(params["bbox"])]
        if "image_size" in params:
            args.append(_literal(params["image_size"]))
        return _call("REGION_BBOX", *args)
    if op is OperatorName.FIND_MARKER:
        marker = params["marker"]
        args = [_ref(inputs["image"]), _string(str(marker["shape"]))]
        if "color" in marker:
            args.append(_string(str(marker["color"])))
        return _call("FIND_MARKER", *args)
    if op is OperatorName.LOCATE:
        return _call("LOCATE", _ref(inputs["image"]), _target(params["target"]))
    if op is OperatorName.SELECT:
        return _compile_select(node)
    if op is OperatorName.GROUP:
        return _call("GROUP", _ref(inputs["entities"]), str(params["mode"]))
    if op is OperatorName.COUNT:
        roles = [name for name in ("image", "entities") if name in inputs]
        if len(roles) != 1:
            raise DSLCompileError("COUNT requires exactly one canonical source role")
        role = roles[0]
        source = _ref(inputs[role])
        surface = _count_surface(role, source, outputs)
        return _call(surface, source, _target(params["target"]), _literal(params["entire"]))
    if op is OperatorName.ATTRIBUTE:
        args = [_ref(inputs["entity"]), _string(str(params["attribute"]))]
        if "part" in params:
            args.append(_string(str(params["part"])))
        return _call("ATTRIBUTE", *args)
    if op is OperatorName.CLASSIFY:
        args = [_ref(inputs["input"])]
        if "label_space" in params:
            args.append(_literal(params["label_space"]))
        return _call("CLASSIFY", *args)
    if op is OperatorName.MULTILABEL_CLASSIFY:
        return _call(
            "MULTILABEL_CLASSIFY",
            _ref(inputs["input"]),
            _literal(params["label_space"]),
        )
    if op is OperatorName.MOTION:
        return _call("MOTION", _ref(inputs["input"]))
    if op is OperatorName.RELATION:
        return _call("RELATION", _ref(inputs["subject"]), _ref(inputs["reference"]))
    if op is OperatorName.ABS_DIFF:
        return _call("ABS_DIFF", _ref(inputs["a"]), _ref(inputs["b"]))
    if op is OperatorName.VLM_REASON:
        values: list[tuple[str, str]] = []
        if "image" in inputs:
            values.append(("image", _ref(inputs["image"])))
        if "evidence" in inputs:
            evidence = inputs["evidence"]
            rendered_evidence = (
                _reference_list(evidence) if isinstance(evidence, list) else _ref(evidence)
            )
            values.append(("evidence", rendered_evidence))
        values.append(("question", _string(str(params["question"]))))
        if "choices" in params:
            values.append(("choices", _literal(params["choices"])))
        return _named_call("VLM_REASON", values)
    if op is OperatorName.BUILD_ROUTE_CONTEXT:
        return _call(
            "BUILD_ROUTE_CONTEXT",
            _ref(inputs["image"]),
            _ref(inputs["start"]),
            _ref(inputs["goal"]),
        )
    if op is OperatorName.ROUTE_REASON:
        return _call(
            "ROUTE_REASON",
            _ref(inputs["context"]),
            _string(str(params["question"])),
            _literal(params["choices"]),
        )
    if op is OperatorName.MATCH_CHOICE:
        return _call(
            "MATCH_CHOICE", _ref(inputs["value"]), _literal(params["choices"])
        )
    raise DSLCompileError(f"unsupported canonical operator {op.value}")


def _image_inputs(payload: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    names: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, str) and value.startswith("$image"):
            names.add(value[1:])
        elif isinstance(value, Mapping):
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)
    return {name: {} for name in sorted(names)}


def compile_taskgraph_to_dsl(graph: PlannerTarget | Mapping[str, Any]) -> str:
    """Validate and deterministically serialize a canonical TaskGraph target."""
    try:
        canonical = canonicalize_target(graph)
        target, report = validate_candidate(canonical, inputs=_image_inputs(canonical))
        if target is None or not report.valid:
            details = "; ".join(f"{issue.code}: {issue.message}" for issue in report.errors)
            raise DSLCompileError(f"cannot compile invalid TaskGraph: {details}")
        canonical = canonicalize_target(target)
    except DSLCompileError:
        raise
    except Exception as exc:
        raise DSLCompileError(f"cannot compile invalid TaskGraph: {exc}") from exc

    lines: list[str] = []
    if "intent" in canonical:
        lines.append(_call("INTENT", str(canonical["intent"])))
    outputs: dict[str, set[RuntimeType]] = {}
    for node in canonical["nodes"]:
        expression = _compile_node(node, outputs)
        lines.append(f"{node['id']}={expression}")
        op = OperatorName(node["op"])
        outputs[f"${node['id']}"] = set(OUTPUT_TYPES[op])
    final = canonical["final"]
    source = (
        final["sources"][0]
        if len(final["sources"]) == 1
        else _reference_list(final["sources"])
    )
    if "question" in final:
        lines.append(
            _call(
                "FINAL_QUESTION",
                source,
                str(final["answer_type"]),
                _string(str(final["question"])),
            )
        )
    else:
        lines.append(_call("FINAL", source, str(final["answer_type"])))
    return "\n".join(lines)
