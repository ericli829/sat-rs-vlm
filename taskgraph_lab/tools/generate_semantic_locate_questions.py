"""Generate English-only localization questions, never visual or graph labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

VERSION = "semantic-locate-questions-v2-en"
OBJECTS = (
    ("building", "buildings", "white square"),
    ("warehouse", "warehouses", "gray rectangular"),
    ("storage tank", "storage tanks", "white circular"),
    ("swimming pool", "swimming pools", "blue rectangular"),
    ("tennis court", "tennis courts", "blue rectangular"),
    ("parking area", "parking areas", "gray rectangular"),
    ("sports field", "sports fields", "green rectangular"),
    ("greenhouse", "greenhouses", "white elongated"),
    ("ship", "ships", "white elongated"),
    ("vehicle", "vehicles", "white rectangular"),
)
HORIZONTAL_CONTEXT = (
    "a row of trees",
    "a road",
    "a river",
    "a green space",
)
VERTICAL_CONTEXT = (
    "grass",
    "a path",
    "a wooded area",
    "a paved open area",
)
SCOPES = (
    "the lower-right part",
    "the upper-left part",
    "the upper-right part",
    "the lower-left part",
    "the central part",
)
EXTREMES = ("topmost", "bottommost", "leftmost", "rightmost")
ORDERS = ("left to right", "right to left", "top to bottom", "bottom to top")
ORDINALS = ("second", "third", "fourth")
FAMILIES = {
    "intrinsic_single": "Intrinsic target attributes",
    "context_single": "Two contextual spatial conditions",
    "intrinsic_context": "Target attributes and context",
    "scoped_context": "Semantic localization within an explicit scope",
    "anchor_context": "Explicit reference object and target context",
    "semantic_multi": "Multiple targets satisfying contextual conditions",
    "extreme_control": "Deterministic extreme control",
    "ordinal_control": "Ordinal control with an explicit sorting direction",
}


def render(family: str, obj: tuple, slots: dict) -> str:
    singular, plural, attribute = obj
    horizontal = slots["horizontal_context"]
    vertical = slots["vertical_context"]
    direction = slots["horizontal_direction"]
    vertical_direction = slots["vertical_direction"]
    context = f"{horizontal} to its {direction} and {vertical} {vertical_direction} it"
    if family == "intrinsic_single":
        return f"Find the {attribute} {singular} in the image."
    if family == "context_single":
        return f"Find the {singular} with {context} in the image."
    if family == "intrinsic_context":
        return f"Find the {attribute} {singular} with {context} in the image."
    if family == "scoped_context":
        return f"Within {slots['scope']} of the image, find the {singular} with {context}."
    if family == "anchor_context":
        # Always use a different anchor category; avoid an unclear self-reference.
        anchor = "warehouse" if singular == "building" else "building"
        return (
            f"Using the topmost {anchor} in the image as the reference, "
            f"find the {singular} below it with {horizontal} to the target's {direction}."
        )
    if family == "semantic_multi":
        return (
            f"Find all {plural} with {horizontal} to their {direction} "
            f"and {vertical} {vertical_direction} them in the image."
        )
    if family == "extreme_control":
        return f"Find the {slots['extreme']} {singular} in the image."
    if family == "ordinal_control":
        return (
            f"Order all {plural} in the image from {slots['order']} "
            f"and find the {slots['ordinal']} one."
        )
    raise ValueError(f"unknown family: {family}")


def generate(*, per_family: int = 10, seed: int = 20261010) -> list[dict]:
    if not 1 <= per_family <= 10:
        raise ValueError("per_family must be between 1 and 10 for this pilot vocabulary")
    rng = random.Random(seed)
    rows: list[dict] = []
    seen: set[str] = set()
    for family in FAMILIES:
        objects = list(OBJECTS)
        rng.shuffle(objects)
        for index in range(per_family):
            obj = objects[index % len(objects)]
            for _ in range(200):
                slots = {
                    "horizontal_context": rng.choice(HORIZONTAL_CONTEXT),
                    "vertical_context": rng.choice(VERTICAL_CONTEXT),
                    "horizontal_direction": rng.choice(("left", "right")),
                    "vertical_direction": rng.choice(("above", "below")),
                    "scope": rng.choice(SCOPES),
                    "extreme": rng.choice(EXTREMES),
                    "order": rng.choice(ORDERS),
                    "ordinal": rng.choice(ORDINALS),
                }
                if obj[0] == "building":
                    # A common anchor case makes cross-family manual review easy.
                    slots.update(
                        horizontal_context=HORIZONTAL_CONTEXT[0],
                        vertical_context=VERTICAL_CONTEXT[0],
                        horizontal_direction="right",
                        vertical_direction="below",
                        scope=SCOPES[0],
                        extreme=EXTREMES[0],
                        order=ORDERS[0],
                        ordinal=ORDINALS[1],
                    )
                question = render(family, obj, slots)
                normalized = re.sub(r"\s+", " ", question).strip().casefold()
                if normalized not in seen:
                    break
            else:
                raise ValueError(f"could not generate a unique question for {family}")
            seen.add(normalized)
            question_id = hashlib.sha256((family + "\n" + question).encode()).hexdigest()[:16]
            rows.append(
                {
                    "id": f"synthetic_locate_{question_id}_en",
                    "question": question,
                    "language": "en",
                    "family": family,
                }
            )
    return rows


def validate(rows: list[dict], *, per_family: int) -> dict:
    expected = len(FAMILIES) * per_family
    if len(rows) != expected or len({r["id"] for r in rows}) != expected:
        raise ValueError("incorrect row count or duplicate IDs")
    normalized = [re.sub(r"\s+", " ", r["question"]).strip().casefold() for r in rows]
    if len(set(normalized)) != expected:
        raise ValueError("duplicate question text")
    for row in rows:
        if set(row) != {"id", "question", "language", "family"}:
            raise ValueError("unexpected fields; questions must remain unlabeled and text-only")
        if not row["question"].strip() or any(c in row["question"] for c in "\r\n"):
            raise ValueError("empty or multiline question")
        if row["language"] != "en" or not row["question"].isascii():
            raise ValueError("this question pool must contain English questions only")
    counts = dict(Counter(r["family"] for r in rows))
    if counts != dict.fromkeys(FAMILIES, per_family):
        raise ValueError("unbalanced families")
    return {"question_count": expected, "family_counts": counts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--per-family", type=int, default=10, choices=range(1, 11))
    args = parser.parse_args()
    rows = generate(per_family=args.per_family, seed=args.seed)
    checked = validate(rows, per_family=args.per_family)
    output = args.output_dir.resolve()
    filenames = ["questions.jsonl", "questions_en.txt", "manifest.json"]
    if any((output / name).exists() for name in filenames):
        raise FileExistsError(f"refusing to replace an existing question set: {output}")
    output.mkdir(parents=True, exist_ok=True)
    contents = {"questions.jsonl": "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)}
    contents["questions_en.txt"] = "".join(r["question"] + "\n" for r in rows)
    hashes = {}
    for name, content in contents.items():
        data = content.encode("utf-8")
        (output / name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    manifest = {
        "version": VERSION,
        "seed": args.seed,
        "per_family": args.per_family,
        **checked,
        "language_counts": dict(Counter(r["language"] for r in rows)),
        "families": FAMILIES,
        "review_status": "synthetic_unlabeled_questions",
        "images_used": False,
        "answers_or_taskgraphs_generated": False,
        "source": "local English templates; not extracted from real dataset annotations",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "files_sha256": hashes,
        "limitations": [
            "Text checks do not establish target existence or visual ground truth.",
            "Template families are construction tags, not gold operator labels.",
            "Group related scenario and template variants in any future split.",
            "This raw question pool is not NormalizedSample or ready-to-train Planner SFT.",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), **checked}, ensure_ascii=False))


if __name__ == "__main__":
    main()
