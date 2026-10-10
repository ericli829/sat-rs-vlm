"""Build and compare procedural DSL references for synthetic localization questions."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from taskgraph_lab.taskgraph.canonicalize import canonicalize_target
from taskgraph_lab.taskgraph.dsl import parse_taskgraph_dsl
from taskgraph_lab.taskgraph.schema import PlannerTarget

TARGETS = {
    "building": ("buildings", {"color": "white", "shape": "square"}),
    "warehouse": ("warehouses", {"color": "gray", "shape": "rectangular"}),
    "storage tank": ("storage tanks", {"color": "white", "shape": "circular"}),
    "swimming pool": ("swimming pools", {"color": "blue", "shape": "rectangular"}),
    "tennis court": ("tennis courts", {"color": "blue", "shape": "rectangular"}),
    "parking area": ("parking areas", {"color": "gray", "shape": "rectangular"}),
    "sports field": ("sports fields", {"color": "green", "shape": "rectangular"}),
    "greenhouse": ("greenhouses", {"color": "white", "shape": "elongated"}),
    "ship": ("ships", {"color": "white", "shape": "elongated"}),
    "vehicle": ("vehicles", {"color": "white", "shape": "rectangular"}),
}
PLURAL_TO_SINGULAR = {plural: singular for singular, (plural, _) in TARGETS.items()}
SCOPES = {
    "lower-right": "BOTTOM_RIGHT",
    "upper-left": "TOP_LEFT",
    "upper-right": "TOP_RIGHT",
    "lower-left": "BOTTOM_LEFT",
    "central": "CENTER",
}
EXTREMES = {"topmost": "TOPMOST", "bottommost": "BOTTOMMOST", "leftmost": "LEFTMOST", "rightmost": "RIGHTMOST"}
ORDERS = {
    "left to right": "LEFT_TO_RIGHT",
    "right to left": "RIGHT_TO_LEFT",
    "top to bottom": "TOP_TO_BOTTOM",
    "bottom to top": "BOTTOM_TO_TOP",
}
ORDINALS = {"second": 2, "third": 3, "fourth": 4}


def quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def target_in_question(family: str, question: str) -> tuple[str, dict[str, str]]:
    text = question.casefold()
    if family == "anchor_context":
        text = re.split(r",\s*find the\s+", text, maxsplit=1)[1]
    elif family == "semantic_multi":
        text = re.sub(r"^find all\s+", "", text)
    elif family == "ordinal_control":
        text = re.sub(r"^order all\s+", "", text)
    elif family == "extreme_control":
        text = re.sub(r"^find the\s+", "", text)

    # The multi-object and ordinal templates use plural category names, while
    # TaskGraph TargetSpec stores the canonical singular category.
    for plural, singular in sorted(PLURAL_TO_SINGULAR.items(), key=lambda item: len(item[0]), reverse=True):
        text = re.sub(rf"\b{re.escape(plural)}\b", singular, text)

    alternatives = sorted(TARGETS, key=len, reverse=True)
    pattern = re.compile(
        r"(?<![a-z])(" + "|".join(re.escape(value) for value in alternatives) + r")(?![a-z])"
    )
    match = pattern.search(text)
    if match is None:
        raise ValueError(f"cannot resolve target class for {family}: {question}")
    category = match.group(1)
    attributes: dict[str, str] = {}
    if family in {"intrinsic_single", "intrinsic_context"}:
        prefix = text[: match.start()]
        attributes = dict(TARGETS[category][1])
        color = re.search(r"\b(white|gray|green|blue)\b", prefix)
        shape = re.search(r"\b(square|rectangular|circular|elongated)\b", prefix)
        if color is None or shape is None:
            raise ValueError(f"cannot resolve intrinsic attributes for {question}")
        attributes.update(color=color.group(1), shape=shape.group(1))
    return category, attributes


def target_expression(category: str, attributes: dict[str, str]) -> str:
    args = [quote(category)] + [f"{key}={quote(value)}" for key, value in attributes.items()]
    return f"T({','.join(args)})"


def create_gold(row: dict[str, Any]) -> str:
    family = row["family"]
    question = row["question"]
    category, attributes = target_in_question(family, question)
    target = target_expression(category, attributes)
    lines: list[str] = []

    if family == "intrinsic_single":
        lines.append(f"n1=LOCATE($image0,{target})")
        final_ref = "$n1"
        selection = "SINGLE"
    elif family in {"context_single", "intrinsic_context"}:
        lines.extend(
            [
                f"n1=LOCATE($image0,{target})",
                f"n2=SELECT_SEM($n1,null,{quote(question)},SINGLE)",
            ]
        )
        final_ref = "$n2"
        selection = "SINGLE"
    elif family == "scoped_context":
        scope_match = re.search(r"\b(lower-right|upper-left|upper-right|lower-left|central)\b", question)
        if scope_match is None:
            raise ValueError(f"cannot resolve explicit scope for {question}")
        position = SCOPES[scope_match.group(1)]
        lines.extend(
            [
                f"n1=REGION($image0,{position})",
                f"n2=LOCATE($n1,{target})",
                f"n3=SELECT_SEM($n2,null,{quote(question)},SINGLE,$n1)",
            ]
        )
        final_ref = "$n3"
        selection = "SINGLE"
    elif family == "anchor_context":
        anchor_match = re.search(
            r"\btopmost\s+(" + "|".join(re.escape(value) for value in sorted(TARGETS, key=len, reverse=True)) + r")\b",
            question,
            re.IGNORECASE,
        )
        if anchor_match is None:
            raise ValueError(f"cannot resolve reference category for {question}")
        anchor = anchor_match.group(1).casefold()
        lines.extend(
            [
                f"n1=LOCATE($image0,T({quote(anchor)}))",
                "n2=SELECT_EXTREME($n1,null,TOPMOST)",
                f"n3=LOCATE($image0,{target})",
                f"n4=SELECT_SEM($n3,$n2,{quote(question)},SINGLE)",
            ]
        )
        final_ref = "$n4"
        selection = "SINGLE"
    elif family == "semantic_multi":
        lines.extend(
            [
                f"n1=LOCATE($image0,{target})",
                f"n2=SELECT_SEM($n1,null,{quote(question)},MULTI)",
            ]
        )
        final_ref = "$n2"
        selection = "MULTI"
    elif family == "extreme_control":
        direction = next((value for key, value in EXTREMES.items() if re.search(rf"\b{key}\b", question, re.IGNORECASE)), None)
        if direction is None:
            raise ValueError(f"cannot resolve extreme direction for {question}")
        lines.extend(
            [
                f"n1=LOCATE($image0,{target})",
                f"n2=SELECT_EXTREME($n1,null,{direction})",
            ]
        )
        final_ref = "$n2"
        selection = "SINGLE"
    elif family == "ordinal_control":
        order = next((value for key, value in ORDERS.items() if key in question.casefold()), None)
        ordinal = next((value for key, value in ORDINALS.items() if re.search(rf"\b{key}\b", question, re.IGNORECASE)), None)
        if order is None or ordinal is None:
            raise ValueError(f"cannot resolve ordinal or sorting direction for {question}")
        lines.extend(
            [
                f"n1=LOCATE($image0,{target})",
                f"n2=SELECT_ORD($n1,null,{ordinal},{order})",
            ]
        )
        final_ref = "$n2"
        selection = "SINGLE"
    else:
        raise ValueError(f"unknown question family: {family}")

    answer_type = "CHOICE_MULTI" if selection == "MULTI" else "CHOICE_SINGLE"
    final_question = "Select all matching targets." if selection == "MULTI" else "Identify the selected target."
    lines.append(f"FINAL_QUESTION({final_ref},{answer_type},{quote(final_question)})")
    return "\n".join(lines)


def signature(target: PlannerTarget) -> dict[str, Any]:
    canonical = canonicalize_target(target)
    return {
        "nodes": canonical["nodes"],
        "sources": canonical["final"]["sources"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    questions = [
        json.loads(line)
        for line in args.questions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    predictions = {
        row["id"]: row
        for row in (
            json.loads(line)
            for line in args.predictions.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    if len(predictions) != len(questions) or {row["id"] for row in questions} != set(predictions):
        raise ValueError("question IDs and prediction IDs must match exactly")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    gold_path = output / "gold_dsl.jsonl"
    comparison_path = output / "gold_comparison.json"
    if gold_path.exists() or comparison_path.exists():
        raise FileExistsError("refusing to overwrite existing gold artifacts")

    comparisons: list[dict[str, Any]] = []
    family_stats: dict[str, Counter] = defaultdict(Counter)
    with gold_path.open("w", encoding="utf-8", newline="\n") as gold_handle:
        for row in questions:
            expected_dsl = create_gold(row)
            expected_target = parse_taskgraph_dsl(expected_dsl)
            gold_handle.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "family": row["family"],
                        "expected_dsl": expected_dsl,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            prediction = predictions[row["id"]]
            actual_dsl = prediction.get("generated_output")
            semantic_select_proposed = bool(actual_dsl and "SELECT_SEM(" in actual_dsl)
            semantic_select_in_valid_plan = (
                prediction.get("status") == "valid_plan" and semantic_select_proposed
            )
            actual_matches = False
            if prediction.get("status") == "valid_plan" and actual_dsl:
                # Parse the actual emitted DSL instead of the evaluator's
                # enriched TaskGraph record, which includes runtime-only
                # nullable fields that are outside the planner schema.
                actual_target = parse_taskgraph_dsl(actual_dsl)
                actual_matches = signature(expected_target) == signature(actual_target)
            family_stats[row["family"]]["total"] += 1
            family_stats[row["family"]]["exact_structure_matches"] += actual_matches
            comparisons.append(
                {
                    "id": row["id"],
                    "family": row["family"],
                    "prediction_status": prediction["status"],
                    "exact_node_and_dependency_match": actual_matches,
                    "select_sem_expected": "SELECT_SEM" in expected_dsl,
                    "select_sem_proposed": semantic_select_proposed,
                    "select_sem_in_valid_plan": semantic_select_in_valid_plan,
                    "expected_dsl": expected_dsl,
                    "predicted_dsl": actual_dsl,
                }
            )

    stats = {
        family: {**counts, "exact_structure_rate": counts["exact_structure_matches"] / counts["total"]}
        for family, counts in family_stats.items()
    }
    exact_count = sum(row["exact_node_and_dependency_match"] for row in comparisons)
    summary = {
        "question_count": len(questions),
        "gold_dsl_validated_count": len(questions),
        "exact_node_and_dependency_match_count": exact_count,
        "exact_node_and_dependency_match_rate": exact_count / len(questions),
        "semantic_select_expected_count": sum(row["select_sem_expected"] for row in comparisons),
        "semantic_select_generated_output_count": sum(row["select_sem_proposed"] for row in comparisons),
        "semantic_select_valid_plan_count": sum(row["select_sem_in_valid_plan"] for row in comparisons),
        "family_stats": stats,
        "comparison_scope": "exact nodes, input roles, params and final source; ignores intent, final question and answer type",
        "limitation": "Programmatically generated gold plans encode intended operators, not image-dependent answer correctness.",
    }
    comparison_path.write_text(
        json.dumps({"summary": summary, "samples": comparisons}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
