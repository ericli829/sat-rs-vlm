"""Text-reviewed compatibility and paired-image revisions for old choice labels.

Writes new-release plans only. The original SFT files are never edited, source
answers are never loaded, and this does not claim image-level correctness.
The finite input queue was reviewed against its English questions and old DSL.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path

from taskgraph_lab.taskgraph.canonicalize import canonicalize_target
from taskgraph_lab.taskgraph.dsl import (
    compile_taskgraph_to_dsl,
    parse_taskgraph_dsl,
    parse_taskgraph_dsl_payload,
)
from taskgraph_lab.taskgraph.validator import validate_candidate
from taskgraph_lab.tools.choice_source_inputs import normalize_hrscd_inputs

REVIEWED_SOURCE_SHA256 = "93ae4ddb7bd40f5d3afcc3e509a8d61362618e65fe42284cd6f525b272ba7155"

PLURAL_COLOR_IDS = {"2019", "2775", "0419", "0802", "0884", "2340", "3433", "3454"}
COLOR_PREFIX = re.compile(
    r"^(?:What colou?r (?:is|are) |What is the colou?r of |What's the color of )",
    re.IGNORECASE,
)


def _refs(node: dict) -> list[str]:
    return [
        ref
        for value in node["inputs"].values()
        for ref in (value if isinstance(value, list) else [value])
    ]


def _prune(graph: dict) -> dict:
    by_ref = {"$" + node["id"]: node for node in graph["nodes"]}
    needed = set(graph["final"]["sources"])
    stack = list(needed)
    while stack:
        ref = stack.pop()
        if ref in by_ref:
            for dependency in _refs(by_ref[ref]):
                if dependency not in needed:
                    needed.add(dependency)
                    stack.append(dependency)
    graph["nodes"] = [node for node in graph["nodes"] if "$" + node["id"] in needed]
    return graph


def _color(row: dict, graph: dict) -> dict:
    attribute = next(node for node in graph["nodes"] if node["op"] == "ATTRIBUTE")
    by_ref = {"$" + node["id"]: node for node in graph["nodes"]}
    source = by_ref[attribute["inputs"]["entity"]]
    candidate = source
    while candidate["op"] == "SELECT":
        candidate = by_ref[candidate["inputs"]["candidates"]]
    if candidate["op"] != "LOCATE":
        raise ValueError("reviewed color source must lead to LOCATE")
    target = deepcopy(candidate["params"]["target"])
    suffix = row["sample_id"].rsplit("_", 1)[1]
    # The roofs belong to one building complex; they are not separate buildings.
    if suffix == "0923":
        target = {"category": "building complex"}
    question = row["question"].rstrip("?")
    query = COLOR_PREFIX.sub("", question)
    if suffix == "2949":
        query = (
            "the car next to the red car on the road to the right below the center of the picture"
        )
    if query == question:
        raise ValueError(f"unreviewed color question prefix: {question}")
    selection = "MULTI" if suffix in PLURAL_COLOR_IDS else "SINGLE"
    inputs = {"candidates": "$revised_candidates", "scope": "$image0"}
    # Keep explicit geometric/ordinal reference calculations rather than ask
    # semantic selection to silently replace ordering or grouping.
    reference = source["inputs"].get("reference")
    if reference in by_ref and by_ref[reference]["op"] == "SELECT":
        params = by_ref[reference]["params"]
        if (
            params.get("mode") in {"ORDINAL", "EXTREME", "RANK"}
            and params.get("criterion") != "distance"
        ):
            inputs["reference"] = reference
    graph["nodes"].extend(
        [
            {
                "id": "revised_candidates",
                "op": "LOCATE",
                "inputs": {"image": "$image0"},
                "params": {"target": target},
            },
            {
                "id": "revised_selected",
                "op": "SELECT",
                "inputs": inputs,
                "params": {"mode": "SEMANTIC", "query": query, "selection_type": selection},
            },
        ]
    )
    if selection == "SINGLE":
        graph["nodes"].append(
            {
                "id": "revised_attribute",
                "op": "ATTRIBUTE",
                "inputs": {"entity": "$revised_selected"},
                "params": attribute["params"],
            }
        )
        graph["final"]["sources"] = ["$revised_attribute"]
    else:
        graph["final"]["sources"] = ["$revised_selected"]
        graph["final"]["question"] = row["question"]
    return _prune(graph)


def _scope(row: dict, graph: dict) -> dict:
    old_nodes = {"$" + node["id"]: node for node in graph["nodes"]}
    result = []
    is_count = any(node["op"] == "COUNT" for node in graph["nodes"])
    for node in graph["nodes"]:
        role = "image" if node["op"] in {"LOCATE", "COUNT"} else None
        producer = old_nodes.get(node["inputs"].get(role)) if role else None
        if (
            producer is None
            or producer["op"] != "SELECT"
            or producer["params"]["mode"] == "SUBREGION"
        ):
            result.append(node)
            continue
        reference = node["inputs"][role]
        candidates_id = node["id"] + "_candidates"
        target = node["params"]["target"]
        result.append(
            {
                "id": candidates_id,
                "op": "LOCATE",
                "inputs": {"image": "$image0"},
                "params": {"target": target},
            }
        )
        select_id = node["id"] if node["op"] == "LOCATE" else node["id"] + "_selected"
        category = target["category"]
        query = f"the {category} candidates inside the referenced region"
        if row["sample_id"].endswith("position_3708"):
            query = (
                "the yellow car on the first horizontal road counted from top to bottom "
                "in the picture"
            )
        elif row["sample_id"] == "xlrs_000037":
            query = (
                "the roundabout in the center of the referenced largest cluster of houses "
                "towards the bottom of the picture"
            )
        result.append(
            {
                "id": select_id,
                "op": "SELECT",
                "inputs": {
                    "candidates": "$" + candidates_id,
                    "reference": reference,
                    "scope": "$image0",
                },
                "params": {
                    "mode": "SEMANTIC",
                    "query": query,
                    "selection_type": "MULTI" if is_count else "SINGLE",
                },
            }
        )
        if node["op"] == "COUNT":
            node["inputs"] = {"entities": "$" + select_id}
            result.append(node)
    graph["nodes"] = result
    if row["sample_id"].endswith("position_3708"):
        graph["nodes"][0]["params"]["target"]["attributes"] = {"shape": "horizontal"}
    return graph


def compile_reviewed(row: dict) -> dict:
    graph = parse_taskgraph_dsl_payload(row["old_dsl"])
    normalized_inputs = normalize_hrscd_inputs(row["inputs"])
    if normalized_inputs != row["inputs"]:
        lower = row["question"].lower()
        category = next(
            (
                noun
                for text, noun in [
                    ("abandoned fields", "abandoned field"),
                    ("soccer fields", "soccer field"),
                    ("farms", "farm"),
                    ("houses", "house"),
                    ("lakes", "lake"),
                    ("trees", "tree"),
                    ("buildings", "building"),
                ]
                if text in lower
            ),
            None,
        )
        if category is None:
            raise ValueError("unreviewed paired-image target")
        nodes = []
        for index in range(2):
            start = 4 * index + 1
            image = f"$image{index}"
            nodes.extend(
                [
                    {
                        "id": f"n{start}",
                        "op": "FIND_MARKER",
                        "inputs": {"image": image},
                        "params": {"marker": {"shape": "circle", "color": "red"}},
                    },
                    {
                        "id": f"n{start + 1}",
                        "op": "LOCATE",
                        "inputs": {"image": image},
                        "params": {"target": {"category": category}},
                    },
                    {
                        "id": f"n{start + 2}",
                        "op": "SELECT",
                        "inputs": {
                            "candidates": f"$n{start + 1}",
                            "reference": f"$n{start}",
                            "scope": image,
                        },
                        "params": {
                            "mode": "SEMANTIC",
                            "selection_type": "MULTI",
                            "query": (
                                f"the {category} candidates within the areas marked by "
                                "the referenced red circles in this image"
                            ),
                        },
                    },
                    {
                        "id": f"n{start + 3}",
                        "op": "COUNT",
                        "inputs": {"entities": f"$n{start + 2}"},
                        "params": {"target": {"category": category}, "entire": False},
                    },
                ]
            )
        nodes.append(
            {"id": "n9", "op": "ABS_DIFF", "inputs": {"a": "$n4", "b": "$n8"}, "params": {}}
        )
        graph = {
            "intent": "CHANGE_COUNT",
            "nodes": nodes,
            "final": {"sources": ["$n9"], "answer_type": "CHOICE_SINGLE"},
        }
        rule = "legacy_explicit_hrscd_pair_marked_count_difference_v1"
    elif "_color_" in row["sample_id"]:
        graph = _color(row, graph)
        rule = "legacy_color_semantic_singleton_or_set_v1"
    elif row["sample_id"].endswith("count_0193"):
        graph = {
            "intent": "ATTRIBUTE_QUERY",
            "nodes": [
                {
                    "id": "n1",
                    "op": "LOCATE",
                    "inputs": {"image": "$image0"},
                    "params": {
                        "target": {"category": "building", "attributes": {"color": "white"}}
                    },
                },
                {
                    "id": "n2",
                    "op": "SELECT",
                    "inputs": {"candidates": "$n1", "scope": "$image0"},
                    "params": {
                        "mode": "SEMANTIC",
                        "selection_type": "SINGLE",
                        "query": (
                            "the white building on the upper left side of the intersection "
                            "next to the area above the picture"
                        ),
                    },
                },
                {
                    "id": "n3",
                    "op": "ATTRIBUTE",
                    "inputs": {"entity": "$n2"},
                    "params": {"attribute": "floors"},
                },
            ],
            "final": {"sources": ["$n3"], "answer_type": "CHOICE_SINGLE"},
        }
        rule = "legacy_floor_attribute_singleton_v1"
    else:
        graph = _scope(row, graph)
        rule = "legacy_visual_scope_to_candidate_filter_v1"
    mapping = {"$" + node["id"]: f"$n{index}" for index, node in enumerate(graph["nodes"], 1)}
    for index, node in enumerate(graph["nodes"], 1):
        node["id"] = f"n{index}"
        node["inputs"] = {
            name: [mapping.get(ref, ref) for ref in value]
            if isinstance(value, list)
            else mapping.get(value, value)
            for name, value in node["inputs"].items()
        }
    graph["final"]["sources"] = [mapping[ref] for ref in graph["final"]["sources"]]
    target, report = validate_candidate(
        graph,
        inputs=normalized_inputs,
        question=row["question"],
        question_type=row["question_type"],
    )
    if target is None or not report.valid:
        raise ValueError(f"{row['sample_id']}: {report.errors}")
    dsl = compile_taskgraph_to_dsl(target)
    if canonicalize_target(parse_taskgraph_dsl(dsl)) != canonicalize_target(target):
        raise ValueError("legacy revision changed during DSL roundtrip")
    return {
        "sample_id": row["sample_id"],
        "dsl": dsl,
        "rule_id": rule,
        "review_notes": (
            "English question and old DSL reviewed by Codex. Full referring conditions "
            "are retained; ordinal/group references and COUNT operations are preserved. "
            "Revision belongs only to the new release; no source answer or image inspected."
        ),
    }


def main() -> None:
    prepared = Path("taskgraph_lab/data/choice_expansion_codex_v1")
    reviewed_source = prepared / "legacy_revisions_needed.jsonl"
    if hashlib.sha256(reviewed_source.read_bytes()).hexdigest() != REVIEWED_SOURCE_SHA256:
        raise ValueError("Reviewed question content/order changed; fresh Codex review is required")
    rows = [
        json.loads(line)
        for line in (prepared / "legacy_revisions_needed.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    converted = [compile_reviewed(row) for row in rows]
    (prepared / "plans/legacy_revisions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in converted), encoding="utf-8"
    )
    print(json.dumps({"legacy_revisions": len(converted), "original_files_modified": False}))


if __name__ == "__main__":
    main()
