# ruff: noqa: E501
# Keep finite reviewed English descriptions and annotation rows intact for audit.
"""Human-reviewed MME legacy RANK compatibility corrections, without answers."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

from sat_rs_vlm.taskgraph.providers import PlannerRequest, Qwen3VLPlannerProvider
from sat_rs_vlm.taskgraph.runtime_types import ImageRef
from taskgraph_lab.taskgraph.canonicalize import canonicalize_target
from taskgraph_lab.taskgraph.dsl import compile_taskgraph_to_dsl, parse_taskgraph_dsl
from taskgraph_lab.taskgraph.validator import validate_candidate

LAB = Path(__file__).resolve().parents[1]
BASELINE = LAB / "data" / "planner_sft_hard_augmented_v1_20260901_173936" / "train.jsonl"
SOURCE_SHA256 = "fec2f0c4f7dbe5b996973bd22c2b2c9831598d222792961823b4b87f08ced9ad"
OUTPUT = LAB / "data" / "choice_expansion_codex_v1" / "plans" / "legacy_rank_mme.jsonl"

# Explicit per-question review: source suffix -> (node, complete target query,
# optional existing reference, visual scope). No arbitrary criterion guessing.
REVIEWED = {
    "color_0297": ("n3", "the wider road in the lower left corner of the picture", None, "$n1"),
    "color_1356": (
        "n3",
        "the tallest building on the left side of the center of the picture",
        None,
        "$n1",
    ),
    "color_1629": (
        "n3",
        "the tallest building in the middle at the top of the picture",
        None,
        "$n1",
    ),
    "color_0315": (
        "n3",
        "the higher building in the upper right corner of the picture",
        None,
        "$n1",
    ),
    "color_1671": (
        "n3",
        "the taller cylindrical building in the middle of the left side of the picture",
        None,
        "$n1",
    ),
    "color_1740": (
        "n7",
        "the vehicle nearest to the lower left corner of the distant traffic roundabout on the gray road in the bottom right corner of the picture",
        "$n4",
        "$image0",
    ),
    "color_2590": (
        "n3",
        "the tallest building in the upper left corner of the picture",
        None,
        "$n1",
    ),
    "count_0175": (
        "n4",
        "the building directly below the railway in the picture",
        "$n1",
        "$image0",
    ),
    "count_0181": ("n3", "the tallest building in the upper left area of the picture", None, "$n1"),
    "count_0235": (
        "n3",
        "the wider longitudinal road in the center area of the picture",
        None,
        "$n1",
    ),
    "count_0241": ("n2", "the widest horizontal road in the picture", None, "$image0"),
    "count_0334": ("n3", "the taller building in the left edge area of the picture", None, "$n1"),
    "count_1063": ("n3", "the widest waterway at the bottom of the picture", None, "$n1"),
    "position_0218": ("n2", "the highest building in the picture", None, "$image0"),
    "count_2848": (
        "n4",
        "the wider road at the bottom left of the river in the upper right corner of the picture",
        "$n1",
        "$image0",
    ),
}

SPECIAL = {
    "color_0306": """INTENT(ATTRIBUTE_QUERY)
n1=REGION($image0,TOP_LEFT)
n2=LOCATE($n1,T("ship",state="docked"))
n3=SELECT_SEM($n2,null,"docked ships in the river in the upper left area of the picture",MULTI,$n1)
n4=SELECT_ORD($n3,null,5,BOTTOM_TO_TOP)
n5=ATTRIBUTE($n4,"color")
FINAL($n5,CHOICE_SINGLE)""",
    "count_0253": """INTENT(RELATIONAL_COUNT)
n1=REGION($image0,TOP_RIGHT)
n2=LOCATE($n1,T("road section"))
n3=SELECT_SEM($n2,null,"the widest section of the road in the upper right corner of the picture",SINGLE,$n1)
n4=LOCATE($image0,T("lane"))
n5=SELECT_REL($n4,$n3,INSIDE)
n6=COUNT_ENTITIES($n5,T("lane"),false)
FINAL($n6,CHOICE_SINGLE)""",
    "count_0256": """INTENT(RELATIONAL_COUNT)
n1=LOCATE($image0,T("road"))
n2=SELECT_SEM($n1,null,"the widest road in the picture",SINGLE,$image0)
n3=SELECT_SUBREGION($n2,null,INSIDE)
n4=LOCATE($n3,T("road section"))
n5=SELECT_SEM($n4,null,"the widest section within the selected widest road",SINGLE,$n3)
n6=SELECT_SUBREGION($n5,null,INSIDE)
n7=COUNT_IMAGE($n6,T("lane"),false)
FINAL($n7,CHOICE_SINGLE)""",
}


def key(sample_id: str) -> str:
    return sample_id.removeprefix("mme_rs_perception_remote_sensing_")


def repair(source: dict) -> tuple[str, dict]:
    sample_key = key(source["id"])
    old_dsl = next(m["content"] for m in source["messages"] if m["role"] == "assistant")
    old_graph = canonicalize_target(parse_taskgraph_dsl(old_dsl))
    if sample_key in SPECIAL:
        graph = canonicalize_target(parse_taskgraph_dsl(SPECIAL[sample_key]))
        return compile_taskgraph_to_dsl(graph), {
            "special_review": sample_key,
            "old_node_count": len(old_graph["nodes"]),
            "new_node_count": len(graph["nodes"]),
        }
    node_id, query, reference, scope = REVIEWED[sample_key]
    graph = deepcopy(old_graph)
    found = False
    for node in graph["nodes"]:
        if node["id"] == node_id:
            if node["op"] != "SELECT" or node["params"].get("mode") != "RANK":
                raise ValueError(f"reviewed node {node_id} is no longer the expected RANK")
            old_params = deepcopy(node["params"])
            node["inputs"] = {"candidates": node["inputs"]["candidates"], "scope": scope}
            if reference is not None:
                node["inputs"]["reference"] = reference
            node["params"] = {"mode": "SEMANTIC", "query": query, "selection_type": "SINGLE"}
            found = True
        elif (
            node["op"] == "SELECT"
            and node["params"].get("mode") == "RANK"
            and node["params"].get("criterion") in {"size", "area", "bboxarea"}
        ):
            # Explicit existing area-rank alias: preserve the deterministic RANK.
            node["params"]["criterion"] = "bbox_area"
    if not found:
        raise ValueError(f"missing reviewed node {node_id}")
    return compile_taskgraph_to_dsl(graph), {
        "node_id": node_id,
        "previous_params": old_params,
        "new_query": query,
        "scope": scope,
        "reference": reference,
    }


def main() -> int:
    source_bytes = BASELINE.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != SOURCE_SHA256:
        raise ValueError("legacy baseline differs from the source read for this review")
    rows = [json.loads(line) for line in source_bytes.decode("utf-8-sig").splitlines() if line]
    selected = [row for row in rows if key(row["id"]) in set(REVIEWED) | set(SPECIAL)]
    if len(selected) != 18:
        raise ValueError(f"expected 18 reviewed MME cases, found {len(selected)}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    checks = []
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
        for source in selected:
            user = json.loads(next(m["content"] for m in source["messages"] if m["role"] == "user"))
            dsl, change = repair(source)
            graph = parse_taskgraph_dsl(dsl)
            _, validation = validate_candidate(
                graph,
                inputs=user["inputs"],
                question=user["question"],
                question_type=user["question_type"],
            )
            if not validation.valid:
                raise ValueError(f"{source['id']}: {validation.errors}")
            canonical = canonicalize_target(graph)
            if (
                canonicalize_target(parse_taskgraph_dsl(compile_taskgraph_to_dsl(canonical)))
                != canonical
            ):
                raise ValueError("roundtrip mismatch")
            request = PlannerRequest(
                question=user["question"],
                question_type=user["question_type"],
                choices=tuple(user["choices"] or ()),
                inputs={
                    "$" + name: ImageRef(value["uri_or_key"])
                    for name, value in user["inputs"].items()
                },
                sample_id=source["id"],
            )
            production = Qwen3VLPlannerProvider._to_production_graph(dsl, request)
            if any(
                n["op"] == "SELECT"
                and n["params"].get("mode") == "RANK"
                and n["params"].get("criterion") not in {"bbox_area", "score"}
                for n in canonical["nodes"]
            ):
                raise ValueError("unsupported rank remains")
            original_dsl = next(
                m["content"] for m in source["messages"] if m["role"] == "assistant"
            )
            original_graph = canonicalize_target(parse_taskgraph_dsl(original_dsl))
            if sum(n["op"] == "COUNT" for n in canonical["nodes"]) != sum(
                n["op"] == "COUNT" for n in original_graph["nodes"]
            ):
                raise ValueError("COUNT count changed")
            row = {
                "sample_id": source["id"],
                "dsl": dsl,
                "rule_id": "legacy_mme_reviewed_rank_contract_v1",
                "review_notes": "Read original English question and baseline DSL, without GT or images. Unsupported width/height/distance ranks use complete semantic conditions; deterministic area/ordinal/count semantics retained. Passed lab validation, canonical roundtrip and production bridge.",
                "revision_trace": change,
            }
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            checks.append(
                {
                    "sample_id": source["id"],
                    "question": user["question"],
                    "old_dsl": original_dsl,
                    "dsl": dsl,
                    "trace": change,
                    "lab_valid": True,
                    "canonical_roundtrip": True,
                    "production_bridge_valid": True,
                    "production_node_count": len(production.nodes),
                }
            )
    report = {
        "count": len(checks),
        "source_sha256": SOURCE_SHA256,
        "checks": checks,
        "visual_validation": False,
        "ground_truth_read": False,
        "api_calls": 0,
    }
    OUTPUT.with_suffix(".review.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "written": len(checks),
                "output": str(OUTPUT),
                "production_bridge_valid": len(checks),
                "count_preserved": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
