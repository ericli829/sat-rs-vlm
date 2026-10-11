"""Codex-reviewed compiler for three explicit XLRS reference-box templates.

These are the exact three source templates inspected for the 386-row pack.
Only supplied question coordinates are extracted; no answer or category is
inferred from ground truth, and unmatched templates fail closed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from taskgraph_lab.taskgraph.canonicalize import canonicalize_target
from taskgraph_lab.taskgraph.dsl import compile_taskgraph_to_dsl, parse_taskgraph_dsl
from taskgraph_lab.taskgraph.validator import validate_candidate

TEMPLATES = {
    "Determine the color of the object within the given reference bounding box "
    "in the satellite image.": ("ATTRIBUTE_QUERY", "ATTRIBUTE", {"attribute": "color"}),
    "Identify the object category within the given bounding box in the satellite "
    "or aerial image.": ("OBJECT_CLASSIFICATION", "CLASSIFY", {}),
    "Determine whether the object within the given reference bounding box is in motion.": (
        "MOTION_QUERY",
        "MOTION",
        {},
    ),
}
COORDINATES = re.compile(
    r"Image resolution: (\d+) x (\d+)\. Bounding box: "
    r"\[(\d+), (\d+), (\d+), (\d+)\]\."
)


def compile_reviewed(sample: dict) -> dict:
    question = sample["question"]
    prefix, suffix = question.rsplit(" Image resolution: ", 1)
    if prefix not in TEMPLATES:
        raise ValueError(f"unreviewed source template: {question}")
    match = COORDINATES.fullmatch("Image resolution: " + suffix)
    if match is None:
        raise ValueError("reference coordinates do not match the reviewed format")
    width, height, x1, y1, x2, y2 = map(int, match.groups())
    if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
        raise ValueError("invalid supplied reference box")
    intent, op, params = TEMPLATES[prefix]
    graph = {
        "intent": intent,
        "nodes": [
            {
                "id": "n1",
                "op": "REGION_FROM_BBOX",
                "inputs": {"image": "$image0"},
                "params": {"bbox": [x1, y1, x2, y2], "image_size": [width, height]},
            },
            {
                "id": "n2",
                "op": op,
                "inputs": {"entity" if op == "ATTRIBUTE" else "input": "$n1"},
                "params": params,
            },
        ],
        "final": {"sources": ["$n2"], "answer_type": "CHOICE_SINGLE"},
    }
    target, report = validate_candidate(
        graph, inputs=sample["inputs"], question=question, question_type=sample["question_type"]
    )
    if target is None or not report.valid:
        raise ValueError(f"{sample['sample_id']}: {report.errors}")
    dsl = compile_taskgraph_to_dsl(target)
    if canonicalize_target(parse_taskgraph_dsl(dsl)) != canonicalize_target(target):
        raise ValueError("DSL roundtrip changed reviewed graph")
    return {
        "sample_id": sample["sample_id"],
        "dsl": dsl,
        "rule_id": "codex_reviewed_explicit_reference_bbox_" + op.lower() + "_v1",
        "review_notes": "Exact source template reviewed by Codex; coordinates copied only "
        "from the question; classification/color/motion answers are unknown. "
        "No source answer or image was inspected.",
    }


def main() -> None:
    prepared = Path("taskgraph_lab/data/choice_expansion_codex_v1")
    rows = [
        json.loads(line)
        for line in (prepared / "xlrs_bbox.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    results = [compile_reviewed(row) for row in rows]
    with (prepared / "plans/xlrs_bbox.jsonl").open("w", encoding="utf-8", newline="\n") as out:
        for row in results:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "converted": len(results),
                "templates": len(TEMPLATES),
                "visual_answers_verified": False,
            }
        )
    )


if __name__ == "__main__":
    main()
