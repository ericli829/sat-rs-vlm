from __future__ import annotations

from dataclasses import dataclass, field

from .enums import AnswerType, OperatorName, RuntimeType
from .schema import PlannerTarget

TypeSet = set[RuntimeType]


@dataclass
class TypeIssue:
    code: str
    message: str
    node_id: str | None = None


@dataclass
class TypeCheckResult:
    errors: list[TypeIssue] = field(default_factory=list)
    inferred: dict[str, TypeSet] = field(default_factory=dict)
    resolved_inputs: dict[str, dict[str, list[tuple[str, TypeSet]]]] = field(
        default_factory=dict
    )


ANY_EVIDENCE: TypeSet = set(RuntimeType)


SIGNATURES: dict[OperatorName, dict[str, TypeSet]] = {
    OperatorName.REGION: {"image": {RuntimeType.IMAGE_REF, RuntimeType.REGION}},
    OperatorName.REGION_FROM_BBOX: {"image": {RuntimeType.IMAGE_REF}},
    OperatorName.FIND_MARKER: {"image": {RuntimeType.IMAGE_REF, RuntimeType.REGION}},
    OperatorName.LOCATE: {"image": {RuntimeType.IMAGE_REF, RuntimeType.REGION}},
    OperatorName.SELECT: {
        "candidates": {RuntimeType.ENTITY_SET, RuntimeType.REGION, RuntimeType.REGION_SET},
        "reference": {RuntimeType.ENTITY, RuntimeType.ENTITY_SET, RuntimeType.REGION},
    },
    OperatorName.GROUP: {"entities": {RuntimeType.ENTITY_SET}},
    OperatorName.COUNT: {
        "image": {RuntimeType.IMAGE_REF, RuntimeType.REGION},
        "entities": {RuntimeType.ENTITY_SET},
    },
    OperatorName.ATTRIBUTE: {
        "entity": {RuntimeType.ENTITY, RuntimeType.ENTITY_SET, RuntimeType.REGION}
    },
    OperatorName.CLASSIFY: {
        "input": {RuntimeType.REGION, RuntimeType.ENTITY, RuntimeType.IMAGE_REF}
    },
    OperatorName.MULTILABEL_CLASSIFY: {"input": {RuntimeType.IMAGE_REF, RuntimeType.REGION}},
    OperatorName.MOTION: {
        "input": {RuntimeType.REGION, RuntimeType.ENTITY, RuntimeType.ENTITY_SET}
    },
    OperatorName.RELATION: {
        "subject": {RuntimeType.ENTITY, RuntimeType.ENTITY_SET, RuntimeType.REGION},
        "reference": {RuntimeType.ENTITY, RuntimeType.ENTITY_SET, RuntimeType.REGION},
    },
    OperatorName.ABS_DIFF: {
        "a": {RuntimeType.SCALAR_INT},
        "b": {RuntimeType.SCALAR_INT},
    },
    OperatorName.VLM_REASON: {
        "image": {RuntimeType.IMAGE_REF, RuntimeType.REGION},
        "evidence": ANY_EVIDENCE,
    },
    OperatorName.BUILD_ROUTE_CONTEXT: {
        "image": {RuntimeType.IMAGE_REF, RuntimeType.REGION},
        "start": {RuntimeType.ENTITY, RuntimeType.ENTITY_SET, RuntimeType.REGION},
        "goal": {RuntimeType.ENTITY, RuntimeType.ENTITY_SET, RuntimeType.REGION},
    },
    OperatorName.ROUTE_REASON: {"context": {RuntimeType.ROUTE_CONTEXT}},
    OperatorName.MATCH_CHOICE: {
        "value": {
            RuntimeType.SCALAR_INT,
            RuntimeType.SCALAR_FLOAT,
            RuntimeType.BOOLEAN,
            RuntimeType.LABEL,
            RuntimeType.LABEL_SET,
            RuntimeType.ANSWER,
        }
    },
}


OUTPUT_TYPES: dict[OperatorName, TypeSet] = {
    OperatorName.REGION: {RuntimeType.REGION},
    OperatorName.REGION_FROM_BBOX: {RuntimeType.REGION},
    OperatorName.FIND_MARKER: {RuntimeType.REGION, RuntimeType.REGION_SET},
    OperatorName.LOCATE: {RuntimeType.ENTITY_SET},
    OperatorName.SELECT: {
        RuntimeType.ENTITY,
        RuntimeType.ENTITY_SET,
        RuntimeType.REGION,
        RuntimeType.REGION_SET,
    },
    OperatorName.GROUP: {RuntimeType.ENTITY_SET, RuntimeType.REGION_SET},
    OperatorName.COUNT: {RuntimeType.SCALAR_INT},
    OperatorName.ATTRIBUTE: {
        RuntimeType.LABEL,
        RuntimeType.BOOLEAN,
        RuntimeType.SCALAR_FLOAT,
    },
    OperatorName.CLASSIFY: {RuntimeType.LABEL},
    OperatorName.MULTILABEL_CLASSIFY: {RuntimeType.LABEL_SET},
    OperatorName.MOTION: {RuntimeType.BOOLEAN},
    OperatorName.RELATION: {RuntimeType.LABEL},
    OperatorName.ABS_DIFF: {RuntimeType.SCALAR_INT},
    OperatorName.VLM_REASON: {
        RuntimeType.ANSWER,
        RuntimeType.LABEL,
        RuntimeType.BOOLEAN,
        RuntimeType.TEXT,
    },
    OperatorName.BUILD_ROUTE_CONTEXT: {RuntimeType.ROUTE_CONTEXT},
    OperatorName.ROUTE_REASON: {RuntimeType.ANSWER, RuntimeType.LABEL},
    OperatorName.MATCH_CHOICE: {RuntimeType.ANSWER},
}


FINAL_TYPES: dict[AnswerType, TypeSet] = {
    AnswerType.CHOICE_SINGLE: {RuntimeType.ANSWER, RuntimeType.LABEL},
    AnswerType.CHOICE_MULTI: {RuntimeType.ANSWER, RuntimeType.LABEL_SET},
    AnswerType.INTEGER: {RuntimeType.SCALAR_INT},
    AnswerType.BOOLEAN: {RuntimeType.BOOLEAN},
    AnswerType.LABEL: {RuntimeType.LABEL},
    AnswerType.LABEL_SET: {RuntimeType.LABEL_SET},
    AnswerType.TEXT: {RuntimeType.TEXT, RuntimeType.ANSWER},
}


OPTIONAL_INPUTS = {
    (OperatorName.COUNT, "image"),
    (OperatorName.COUNT, "entities"),
    (OperatorName.SELECT, "reference"),
    (OperatorName.VLM_REASON, "image"),
    (OperatorName.VLM_REASON, "evidence"),
}


VISUAL_ONLY_INPUTS = {
    (OperatorName.REGION, "image"),
    (OperatorName.REGION_FROM_BBOX, "image"),
    (OperatorName.FIND_MARKER, "image"),
    (OperatorName.LOCATE, "image"),
    (OperatorName.COUNT, "image"),
    (OperatorName.VLM_REASON, "image"),
    (OperatorName.BUILD_ROUTE_CONTEXT, "image"),
}

SINGLETON_SELECT_MODES = {"RANK", "ORDINAL", "EXTREME"}


def _refs(value: str | list[str]) -> list[str]:
    return value if isinstance(value, list) else [value]


def check_types(target: PlannerTarget, input_names: set[str]) -> TypeCheckResult:
    result = TypeCheckResult(
        inferred={f"${name}": {RuntimeType.IMAGE_REF} for name in sorted(input_names)}
    )
    nodes_by_ref = {f"${node.id}": node for node in target.nodes}
    for node in target.nodes:
        signature = SIGNATURES[node.op]
        required = {name for name in signature if (node.op, name) not in OPTIONAL_INPUTS}
        missing = required - set(node.inputs)
        unexpected = set(node.inputs) - set(signature)
        if missing:
            result.errors.append(
                TypeIssue("missing_input", f"missing required inputs: {sorted(missing)}", node.id)
            )
        if unexpected:
            result.errors.append(
                TypeIssue("unexpected_input", f"unexpected inputs: {sorted(unexpected)}", node.id)
            )
        if node.op is OperatorName.COUNT:
            count_inputs = set(node.inputs).intersection({"image", "entities"})
            if len(count_inputs) != 1:
                result.errors.append(
                    TypeIssue(
                        "count_input_xor",
                        "COUNT requires exactly one of image or entities",
                        node.id,
                    )
                )
        for name, raw in node.inputs.items():
            if name not in signature:
                continue
            for ref in _refs(raw):
                actual = result.inferred.get(ref)
                result.resolved_inputs.setdefault(node.id, {}).setdefault(name, []).append(
                    (ref, set(actual or set()))
                )
                if actual is None:
                    continue
                allowed = signature[name]
                if not actual.intersection(allowed):
                    result.errors.append(
                        TypeIssue(
                            "input_type_mismatch",
                            f"{name}={ref} has {sorted(t.value for t in actual)}, "
                            f"expected one of {sorted(t.value for t in allowed)}",
                            node.id,
                        )
                    )
                producer = nodes_by_ref.get(ref)
                if producer is not None and producer.op is OperatorName.SELECT:
                    if (node.op, name) in VISUAL_ONLY_INPUTS:
                        result.errors.append(
                            TypeIssue(
                                "select_result_not_visual_scope",
                                f"{node.op.value}.{name} cannot consume SELECT result {ref}; "
                                "use the selected entity/region as semantic evidence or a source "
                                "image scope",
                                node.id,
                            )
                        )
                    if node.op is OperatorName.ATTRIBUTE and name == "entity":
                        mode = producer.params.get("mode")
                        mode = getattr(mode, "value", mode)
                        if str(mode).upper() not in SINGLETON_SELECT_MODES:
                            result.errors.append(
                                TypeIssue(
                                    "attribute_requires_singleton",
                                    f"ATTRIBUTE.entity={ref} requires a singleton SELECT; "
                                    "add RANK, ORDINAL, or EXTREME before ATTRIBUTE",
                                    node.id,
                                )
                            )
        result.inferred[f"${node.id}"] = set(OUTPUT_TYPES[node.op])

    final_types = set().union(
        *(result.inferred.get(ref, set()) for ref in target.final.sources), set()
    )
    is_choice = target.final.answer_type in {
        AnswerType.CHOICE_SINGLE,
        AnswerType.CHOICE_MULTI,
    }
    if final_types and not is_choice and not final_types.intersection(
        FINAL_TYPES[target.final.answer_type]
    ):
        result.errors.append(
            TypeIssue(
                "final_type_mismatch",
                f"final sources {target.final.sources} have "
                f"{sorted(t.value for t in final_types)}, "
                f"incompatible with {target.final.answer_type.value}",
            )
        )
    return result
