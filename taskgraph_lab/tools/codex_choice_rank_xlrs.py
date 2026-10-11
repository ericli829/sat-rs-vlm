# ruff: noqa: E501, E402
# Keep reviewed descriptions intact; direct-script imports follow path bootstrap.
"""Text-reviewed production compatibility repairs for 27 legacy XLRS plans.

Only unsupported RANK selectors are replaced. No source answers, images, model
weights, production rules, or old SFT files are used or modified.
"""

from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from sat_rs_vlm.taskgraph.providers import (
    PlannerRequest,
    Qwen3VLPlannerProvider,
)
from sat_rs_vlm.taskgraph.runtime_types import (
    ImageRef,
)
from taskgraph_lab.taskgraph.canonicalize import (
    canonicalize_target,
)
from taskgraph_lab.taskgraph.dsl import (
    CanonicalDSLPrefixGrammar,
    compile_taskgraph_to_dsl,
    parse_taskgraph_dsl,
    parse_taskgraph_dsl_payload,
)
from taskgraph_lab.taskgraph.validator import (
    validate_candidate,
)

BASELINE = ROOT / "taskgraph_lab/data/planner_sft_hard_augmented_v1_20260901_173936/train.jsonl"
OUTPUT = ROOT / "taskgraph_lab/data/choice_expansion_codex_v1/plans/legacy_rank_xlrs.jsonl"

# Each entry was individually authored after reading the English question and
# old DSL. Values are (old node ID, old criterion, complete target condition,
# optional existing reference override). These are target-selection queries,
# not copies of the whole multiple-choice or routing question.
REVIEWED = {
    "xlrs_000270": ("n5", "cluster_size", "the group of three closely situated lakes", None),
    "xlrs_000407": (
        "n3",
        "length",
        "the longest gray-roofed building in the top left corner of the image",
        None,
    ),
    "xlrs_000511": ("n4", "color darkness", "the lake with the darkest color in the image", None),
    "xlrs_000516": ("n2", "length", "the longer silver greenhouse in the image", None),
    "xlrs_000523": (
        "n4",
        "length",
        "the longest gray rectangular building in the cluster of houses in the lower left corner of the picture",
        None,
    ),
    "xlrs_001988": (
        "n8",
        "distance",
        "the nearest gray house to the right of the referenced blue building on the right side of the cluster of houses in the center",
        "$n5",
    ),
    "xlrs_002017": (
        "n7",
        "distance",
        "the nearest dark gray house to the upper right of the referenced orange house above the cluster of houses in the bottom right center of the image",
        "$n4",
    ),
    "xlrs_002020": (
        "n11",
        "distance",
        "the roundabout in the bottom right corner nearest to the referenced start road on the left side of the gray sports field above the largest cluster of houses in the top left corner",
        "$n8",
    ),
    "xlrs_002056": (
        "n6",
        "distance",
        "the nearest circular island to the upper left of the referenced largest rectangular pond at the top center of the image",
        "$n3",
    ),
    "xlrs_000391": (
        "n3",
        "length",
        "the longest black rectangular building in the gray area on the lower left side of the image",
        None,
    ),
    "xlrs_000395": (
        "n3",
        "length",
        "the longest gray rectangular building in the cluster of buildings on the right edge at the top right of the picture",
        None,
    ),
    "xlrs_000457": (
        "n5",
        "length",
        "the longest gray rectangular building in the cluster of buildings near the left edge at the bottom left of the picture",
        None,
    ),
    "xlrs_000468": ("n4", "length", "the longer orange circular track in the image", None),
    "xlrs_000491": (
        "n4",
        "roof area",
        "the building next to the highway with the largest white rectangular roof featuring numerous vents",
        None,
    ),
    "xlrs_001989": (
        "n7",
        "distance",
        "the nearest black circular pit to the left of the referenced start road on the left side of the cluster of houses in the top right corner of the picture",
        "$n4",
    ),
    "xlrs_001994": (
        "n7",
        "distance",
        "the nearest pond to the right of the referenced start road in the middle of the slightly larger cluster of houses at the top right corner of the picture",
        "$n4",
    ),
    "xlrs_002010": (
        "n2",
        "length",
        "the longest blue rectangular building located slightly above the center of the image",
        None,
    ),
    "xlrs_002046": (
        "n5",
        "distance",
        "the nearest green lake to the bottom right of the referenced blue building at the top right corner of the image",
        "$n2",
    ),
    "xlrs_002053": (
        "n10",
        "number of tennis courts",
        "the nearest row of four tennis courts in the bottom left corner relative to the referenced start road on the left side of the playground with a red track at the top of the picture",
        "$n6",
    ),
    "xlrs_002058": (
        "n9",
        "distance",
        "the nearest crossroad above the referenced start road on the left side of the light green rectangular building northeast of the cluster of houses at the bottom right corner",
        "$n6",
    ),
    "xlrs_000350": (
        "n3",
        "length",
        "the longest gray rectangular building in the cluster of buildings near the bottom edge on the right side of the image",
        None,
    ),
    "xlrs_000385": (
        "n5",
        "length",
        "the longest gray rectangular building in the cluster of buildings near the right edge at the top right of the image",
        None,
    ),
    "xlrs_000428": (
        "n3",
        "length",
        "the longest orange rectangular building in the upper right corner",
        None,
    ),
    "xlrs_000448": (
        "n3",
        "length",
        "the longest black rectangular building in the small town at the upper left edge of the picture",
        None,
    ),
    "xlrs_000683": (
        "n3",
        "length",
        "the longest green park in the central upper area of the image",
        None,
    ),
    "xlrs_002015": (
        "n5",
        "distance",
        "the nearest blue building to the left of the referenced large playground at the upper center",
        "$n2",
    ),
    "xlrs_002050": (
        "n5",
        "length",
        "the longer white building to the bottom left of the referenced deep blue swimming pool in the cluster of small houses in the bottom right corner",
        "$n2",
    ),
}


def candidate_scope(graph, candidate):
    by_ref = {"$" + node["id"]: node for node in graph["nodes"]}
    seen = set()
    while candidate in by_ref and candidate not in seen:
        seen.add(candidate)
        node = by_ref[candidate]
        if node["op"] == "LOCATE":
            return node["inputs"]["image"]
        if node["op"] == "SELECT":
            candidate = node["inputs"]["candidates"]
        elif node["op"] == "GROUP":
            candidate = node["inputs"]["entities"]
        else:
            raise ValueError(f"Unexpected candidate producer: {node['op']}")
    raise ValueError("No original LOCATE image/Region scope for reviewed RANK")


def convert(sample_id, user, original_dsl):
    node_id, criterion, query, reference_override = REVIEWED[sample_id]
    original = parse_taskgraph_dsl_payload(original_dsl)
    graph = deepcopy(original)
    changed = []
    for node in graph["nodes"]:
        if node["op"] != "SELECT" or node["params"].get("mode") != "RANK":
            continue
        old_criterion = node["params"]["criterion"]
        if old_criterion in {"size", "area", "bboxarea"}:
            node["params"]["criterion"] = "bbox_area"
            changed.append(
                {
                    "node_id": node["id"],
                    "from": old_criterion,
                    "to": "bbox_area",
                    "semantic_change": False,
                }
            )
            continue
        if old_criterion in {"bbox_area", "score"}:
            continue
        if node["id"] != node_id or old_criterion != criterion or node["params"]["rank"] != 1:
            raise ValueError(f"Unreviewed incompatible RANK: {sample_id} {node}")
        candidate = node["inputs"]["candidates"]
        new_inputs = dict(node["inputs"])
        if reference_override:
            new_inputs["reference"] = reference_override
        new_inputs["scope"] = candidate_scope(original, candidate)
        node["inputs"] = new_inputs
        node["params"] = {"mode": "SEMANTIC", "query": query, "selection_type": "SINGLE"}
        changed.append(
            {
                "node_id": node_id,
                "from": criterion,
                "to": "SEMANTIC",
                "reference": new_inputs.get("reference"),
                "scope": new_inputs["scope"],
            }
        )
    if not any(item["to"] == "SEMANTIC" for item in changed):
        raise ValueError(f"Missing reviewed RANK node in {sample_id}")
    target, report = validate_candidate(
        graph, inputs=user["inputs"], question=user["question"], question_type=user["question_type"]
    )
    if not report.valid or target is None:
        raise ValueError(f"{sample_id}: {report.errors}")
    dsl = compile_taskgraph_to_dsl(target)
    assert canonicalize_target(parse_taskgraph_dsl(dsl)) == canonicalize_target(target)
    grammar = CanonicalDSLPrefixGrammar(image_refs=["$" + key for key in user["inputs"]]).analyze(
        dsl
    )
    assert grammar.valid_prefix and grammar.complete
    request = PlannerRequest(
        question=user["question"],
        question_type=user["question_type"],
        choices=tuple(user["choices"]),
        inputs={"$" + key: ImageRef(spec["uri_or_key"]) for key, spec in user["inputs"].items()},
        sample_id=sample_id,
    )
    runtime_graph = Qwen3VLPlannerProvider._to_production_graph(dsl, request)
    assert len(runtime_graph.nodes) == len(original["nodes"])
    assert graph["final"] == original["final"]
    # None of the existing dedicated nodes may disappear or be silently edited.
    original_by_id = {node["id"]: node for node in original["nodes"]}
    for node in graph["nodes"]:
        if not any(item["node_id"] == node["id"] for item in changed):
            assert node == original_by_id[node["id"]]
    return {
        "sample_id": sample_id,
        "dsl": dsl,
        "rule_id": "legacy_xlrs_unsupported_rank_to_candidate_semantic_v1",
        "review_notes": {
            "changes": changed,
            "scope": "English question and old DSL individually reviewed; no GT or image read; unsupported length/distance/roof-area/cardinality predicates are explicit natural-language candidate selection; dedicated route/geometry/group nodes and final contract remain unchanged.",
            "validation": {
                "lab_original_input": True,
                "canonical_roundtrip": True,
                "constraint_complete": True,
                "production_bridge": True,
            },
            "original_question_dsl_sha256": hashlib.sha256(
                (user["question"] + "\n" + original_dsl).encode("utf-8")
            ).hexdigest(),
        },
    }


def main():
    found = {}
    # Only message content is accessed. GT and answer-audit metadata are not
    # teacher inputs and never enter the reviewed query or program.
    for line in BASELINE.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["id"] in REVIEWED:
            found[row["id"]] = (
                json.loads(row["messages"][1]["content"]),
                row["messages"][2]["content"],
            )
    if set(found) != set(REVIEWED):
        raise ValueError(f"Missing reviewed baseline IDs: {set(REVIEWED) - set(found)}")
    rows = [convert(sample_id, *found[sample_id]) for sample_id in sorted(REVIEWED)]
    OUTPUT.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "revisions": len(rows),
                "pending": 0,
                "production_bridge_pass": len(rows),
                "sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
                "output": str(OUTPUT),
            }
        )
    )


if __name__ == "__main__":
    main()
