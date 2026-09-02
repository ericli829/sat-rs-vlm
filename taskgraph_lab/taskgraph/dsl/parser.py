from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from taskgraph_lab.taskgraph.enums import OperatorName, RuntimeType
from taskgraph_lab.taskgraph.schema import PlannerTarget
from taskgraph_lab.taskgraph.type_checker import OUTPUT_TYPES, SIGNATURES
from taskgraph_lab.taskgraph.validator import validate_candidate

from .errors import DSLParseError
from .grammar import Call, Program, Reference, Symbol, parse_syntax

_SURFACE_OPERATORS = {
    "REGION",
    "REGION_BBOX",
    "FIND_MARKER",
    "LOCATE",
    "SELECT_REL",
    "SELECT_RANK",
    "SELECT_ORD",
    "SELECT_EXTREME",
    "SELECT_SUBREGION",
    "GROUP",
    "COUNT",
    "COUNT_IMAGE",
    "COUNT_ENTITIES",
    "ATTRIBUTE",
    "CLASSIFY",
    "MULTILABEL_CLASSIFY",
    "MOTION",
    "RELATION",
    "ABS_DIFF",
    "VLM_REASON",
    "BUILD_ROUTE_CONTEXT",
    "ROUTE_REASON",
    "MATCH_CHOICE",
}


def _error(message: str) -> DSLParseError:
    return DSLParseError(message)


def _positional(call: Call, minimum: int, maximum: int | None = None) -> tuple[Any, ...]:
    maximum = minimum if maximum is None else maximum
    if call.kwargs:
        raise _error(f"{call.name} does not accept named arguments")
    if not minimum <= len(call.args) <= maximum:
        expected = str(minimum) if minimum == maximum else f"{minimum}..{maximum}"
        raise _error(f"{call.name} expects {expected} positional arguments")
    return call.args


def _reference(value: Any, *, node_only: bool = False) -> str:
    if not isinstance(value, Reference):
        raise _error("expected a reference")
    if node_only and not value.value.startswith("$n"):
        raise _error("expected a $nX node reference")
    return value.value


def _reference_list(value: Any, *, node_only: bool = False) -> list[str]:
    if not isinstance(value, list) or not value:
        raise _error("expected a non-empty reference list")
    return [_reference(item, node_only=node_only) for item in value]


def _reference_or_list(value: Any) -> str | list[str]:
    if isinstance(value, Reference):
        return _reference(value)
    return _reference_list(value)


def _source_list(value: Any) -> list[str]:
    if isinstance(value, Reference):
        return [_reference(value, node_only=True)]
    return _reference_list(value, node_only=True)


def _symbol(value: Any) -> str:
    if not isinstance(value, Symbol):
        raise _error("expected an unquoted canonical enum symbol")
    return value.value


def _string(value: Any) -> str:
    if not isinstance(value, str):
        raise _error("expected a JSON-compatible quoted string")
    return value


def _boolean(value: Any) -> bool:
    if not isinstance(value, bool):
        raise _error("expected true or false")
    return value


def _integer(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise _error("expected an integer")
    return value


def _number_list(value: Any, length: int) -> list[int | float]:
    if not isinstance(value, list) or len(value) != length:
        raise _error(f"expected a numeric list of length {length}")
    if any(not isinstance(item, (int, float)) or isinstance(item, bool) for item in value):
        raise _error(f"expected a numeric list of length {length}")
    return value


def _string_list(value: Any, *, non_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (non_empty and not value):
        raise _error("expected a list of quoted strings")
    if any(not isinstance(item, str) for item in value):
        raise _error("expected a list of quoted strings")
    return value


def _target(value: Any) -> dict[str, Any]:
    if not isinstance(value, Call) or value.name != "T":
        raise _error("expected TargetSpec constructor T(...)")
    if len(value.args) != 1 or not isinstance(value.args[0], str):
        raise _error("T requires exactly one quoted category string")
    attributes: dict[str, Any] = {}
    for key, attribute_value in value.kwargs:
        if key in attributes:
            raise _error(f"T has duplicate attribute {key!r}")
        if isinstance(attribute_value, (Call, Reference, Symbol, list)) or attribute_value is None:
            raise _error(f"T attribute {key!r} must be a scalar literal")
        attributes[key] = attribute_value
    return {"category": value.args[0], "attributes": attributes}


def _static_types(reference: str, outputs: dict[str, set[RuntimeType]]) -> set[RuntimeType]:
    if reference.startswith("$image"):
        return {RuntimeType.IMAGE_REF}
    return outputs.get(reference, set())


def _infer_count_role(reference: str, outputs: dict[str, set[RuntimeType]]) -> str:
    actual = _static_types(reference, outputs)
    candidates = [
        role
        for role in ("image", "entities")
        if actual.intersection(SIGNATURES[OperatorName.COUNT][role])
    ]
    if len(candidates) != 1:
        raise _error(
            "COUNT source role is not statically unique; use COUNT_IMAGE or COUNT_ENTITIES"
        )
    return candidates[0]


def _select_node(call: Call) -> tuple[dict[str, Any], dict[str, Any]]:
    signatures = {
        "SELECT_REL": ("RELATION", ("relation",)),
        "SELECT_RANK": ("RANK", ("criterion", "rank", "order")),
        "SELECT_ORD": ("ORDINAL", ("index", "order")),
        "SELECT_EXTREME": ("EXTREME", ("direction",)),
        "SELECT_SUBREGION": ("SUBREGION", ("subregion",)),
    }
    mode, param_names = signatures[call.name]
    args = _positional(call, 2 + len(param_names))
    inputs = {"candidates": _reference(args[0])}
    if args[1] is not None:
        inputs["reference"] = _reference(args[1])
    raw_params = args[2:]
    params: dict[str, Any] = {"mode": mode}
    for name, raw in zip(param_names, raw_params, strict=True):
        if name in {"rank", "index"}:
            params[name] = _integer(raw)
        elif name == "criterion":
            params[name] = _string(raw)
        else:
            params[name] = _symbol(raw)
    return inputs, params


def _lower_node(
    call: Call,
    outputs: dict[str, set[RuntimeType]],
) -> tuple[OperatorName, dict[str, str | list[str]], dict[str, Any]]:
    if call.name not in _SURFACE_OPERATORS:
        raise _error(f"unknown DSL operator {call.name!r}")
    if call.name == "REGION":
        source, position = _positional(call, 2)
        return OperatorName.REGION, {"image": _reference(source)}, {"position": _symbol(position)}
    if call.name == "REGION_BBOX":
        args = _positional(call, 2, 3)
        params: dict[str, Any] = {"bbox": _number_list(args[1], 4)}
        if len(args) == 3:
            params["image_size"] = _number_list(args[2], 2)
        return OperatorName.REGION_FROM_BBOX, {"image": _reference(args[0])}, params
    if call.name == "FIND_MARKER":
        args = _positional(call, 2, 3)
        marker = {"shape": _string(args[1])}
        if len(args) == 3:
            marker["color"] = _string(args[2])
        return OperatorName.FIND_MARKER, {"image": _reference(args[0])}, {"marker": marker}
    if call.name == "LOCATE":
        source, target = _positional(call, 2)
        return OperatorName.LOCATE, {"image": _reference(source)}, {"target": _target(target)}
    if call.name.startswith("SELECT_"):
        inputs, params = _select_node(call)
        return OperatorName.SELECT, inputs, params
    if call.name == "GROUP":
        source, mode = _positional(call, 2)
        return OperatorName.GROUP, {"entities": _reference(source)}, {"mode": _symbol(mode)}
    if call.name in {"COUNT", "COUNT_IMAGE", "COUNT_ENTITIES"}:
        source, target, entire = _positional(call, 3)
        source_ref = _reference(source)
        if call.name == "COUNT":
            role = _infer_count_role(source_ref, outputs)
        else:
            role = "image" if call.name == "COUNT_IMAGE" else "entities"
        return (
            OperatorName.COUNT,
            {role: source_ref},
            {"target": _target(target), "entire": _boolean(entire)},
        )
    if call.name == "ATTRIBUTE":
        args = _positional(call, 2, 3)
        params = {"attribute": _string(args[1])}
        if len(args) == 3:
            params["part"] = _string(args[2])
        return OperatorName.ATTRIBUTE, {"entity": _reference(args[0])}, params
    if call.name == "CLASSIFY":
        args = _positional(call, 1, 2)
        params = {}
        if len(args) == 2:
            params["label_space"] = _string_list(args[1])
        return OperatorName.CLASSIFY, {"input": _reference(args[0])}, params
    if call.name == "MULTILABEL_CLASSIFY":
        source, labels = _positional(call, 2)
        return (
            OperatorName.MULTILABEL_CLASSIFY,
            {"input": _reference(source)},
            {"label_space": _string_list(labels, non_empty=True)},
        )
    if call.name == "MOTION":
        (source,) = _positional(call, 1)
        return OperatorName.MOTION, {"input": _reference(source)}, {}
    if call.name == "RELATION":
        subject, reference = _positional(call, 2)
        return (
            OperatorName.RELATION,
            {"subject": _reference(subject), "reference": _reference(reference)},
            {},
        )
    if call.name == "ABS_DIFF":
        a, b = _positional(call, 2)
        return OperatorName.ABS_DIFF, {"a": _reference(a), "b": _reference(b)}, {}
    if call.name == "VLM_REASON":
        if call.args:
            raise _error("VLM_REASON accepts named arguments only")
        values = call.keyword_map()
        unknown = sorted(set(values) - {"image", "evidence", "question", "choices"})
        if unknown:
            raise _error(f"VLM_REASON has unknown arguments: {unknown}")
        if "question" not in values:
            raise _error("VLM_REASON requires question")
        inputs: dict[str, str | list[str]] = {}
        if "image" in values:
            inputs["image"] = _reference(values["image"])
        if "evidence" in values:
            inputs["evidence"] = _reference_or_list(values["evidence"])
        params = {"question": _string(values["question"])}
        if "choices" in values:
            choices = values["choices"]
            params["choices"] = (
                _string_list(choices) if isinstance(choices, list) else _string(choices)
            )
        return OperatorName.VLM_REASON, inputs, params
    if call.name == "BUILD_ROUTE_CONTEXT":
        image, start, goal = _positional(call, 3)
        return (
            OperatorName.BUILD_ROUTE_CONTEXT,
            {
                "image": _reference(image),
                "start": _reference(start),
                "goal": _reference(goal),
            },
            {},
        )
    if call.name == "ROUTE_REASON":
        context, question, choices = _positional(call, 3)
        raw_choices = _string_list(choices) if isinstance(choices, list) else _string(choices)
        return (
            OperatorName.ROUTE_REASON,
            {"context": _reference(context)},
            {"question": _string(question), "choices": raw_choices},
        )
    if call.name == "MATCH_CHOICE":
        value, choices = _positional(call, 2)
        raw_choices = _string_list(choices) if isinstance(choices, list) else _string(choices)
        return OperatorName.MATCH_CHOICE, {"value": _reference(value)}, {"choices": raw_choices}
    raise AssertionError(f"unhandled operator {call.name}")


def _image_inputs(values: Iterable[Any]) -> dict[str, dict[str, str]]:
    refs: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, str) and value.startswith("$image"):
            refs.add(value[1:])
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for value in values:
        visit(value)
    return {name: {} for name in sorted(refs)}


def _lower(program: Program) -> dict[str, Any]:
    intent: str | None = None
    if program.intent is not None:
        args = _positional(program.intent, 1)
        intent = _symbol(args[0])
    nodes: list[dict[str, Any]] = []
    outputs: dict[str, set[RuntimeType]] = {}
    seen: set[str] = set()
    for index, statement in enumerate(program.nodes, 1):
        expected = f"n{index}"
        if statement.node_id in seen:
            raise _error(f"duplicate node id {statement.node_id}")
        if statement.node_id != expected:
            raise _error(f"node ids must be contiguous dependency order; expected {expected}")
        seen.add(statement.node_id)
        op, inputs, params = _lower_node(statement.call, outputs)
        node = {
            "id": statement.node_id,
            "op": op.value,
            "inputs": inputs,
            "params": params,
        }
        nodes.append(node)
        outputs[f"${statement.node_id}"] = set(OUTPUT_TYPES[op])
    if program.final.name == "FINAL":
        sources, answer_type = _positional(program.final, 2)
        final = {"sources": _source_list(sources), "answer_type": _symbol(answer_type)}
    elif program.final.name == "FINAL_QUESTION":
        sources, answer_type, question = _positional(program.final, 3)
        final = {
            "sources": _source_list(sources),
            "question": _string(question),
            "answer_type": _symbol(answer_type),
        }
    else:
        raise _error(f"unknown final statement {program.final.name!r}")
    payload: dict[str, Any] = {"nodes": nodes, "final": final}
    if intent is not None:
        payload["intent"] = intent
    return payload


def parse_taskgraph_dsl_payload(text: str) -> dict[str, Any]:
    """Parse and lower DSL without conflating graph/runtime validation."""

    try:
        return _lower(parse_syntax(text))
    except DSLParseError:
        raise
    except Exception as exc:
        raise DSLParseError(str(exc)) from exc


def parse_taskgraph_dsl(text: str) -> PlannerTarget:
    """Parse constrained DSL and validate the resulting canonical PlannerTarget."""

    payload = parse_taskgraph_dsl_payload(text)
    inputs = _image_inputs(payload["nodes"])
    target, report = validate_candidate(payload, inputs=inputs)
    if target is None or not report.valid:
        details = "; ".join(f"{issue.code}: {issue.message}" for issue in report.errors)
        raise DSLParseError(f"DSL lowers to an invalid TaskGraph: {details}")
    return target
