"""Apply Codex-reviewed, answer-blind MME referent plans to one fixed shard.

This is a finite source-template conversion, not a generic gold-label oracle.
All original referring conditions remain in SELECT_SEM.query unless a dedicated
region, extreme or size-rank operator consumes that condition. No image or source
answer is read. The original question is never changed in the prepared seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from taskgraph_lab.datasets.base import NormalizedSample
from taskgraph_lab.tools.collect_choice_expansion import _audit_dsl

REVIEWED_SOURCE_SHA256 = "2a9619c4526a8f5cce9b8f77c94a6986bf52d3a8070616c0ca0f1c4a87cbc7f2"

DEFAULT = Path(__file__).resolve().parents[1] / "data/choice_expansion_codex_v1"

# These noun aliases were reviewed against all 425 source questions. Earliest
# noun wins, so a target roof/flag/car never becomes its background building.
NOUNS = {
    "basketball court": "basketball court",
    "basketball courts": "basketball court",
    "basketball ground": "basketball court",
    "football field": "football field",
    "sports field": "sports field",
    "sports ground": "sports field",
    "tennis court": "tennis court",
    "tennis courts": "tennis court",
    "table-tennis table": "table tennis table",
    "ping-pong table": "table tennis table",
    "swimming pool": "swimming pool",
    "cargo ship": "cargo ship",
    "cargo ships": "cargo ship",
    "tower crane": "tower crane",
    "hoisting equipment": "crane",
    "hoisting equipments": "crane",
    "traffic roundabout": "roundabout",
    "traffic roundabouts": "roundabout",
    "traffic intersection": "intersection",
    "traffic wheel": "roundabout",
    "traffic sign": "traffic sign",
    "parking lot": "parking area",
    "parking lots": "parking area",
    "open space": "open ground",
    "ground area": "ground area",
    "common entrance": "entrance",
    "building complex": "building complex",
    "building complexes": "building complex",
    "chinese architecture": "building",
    "river crossing bridge": "bridge",
    "double box truck": "truck",
    "observation tower": "observation tower",
    "sightseeing tower": "observation tower",
    "zebra crossing": "zebra crossing",
    "railway tracks": "railway track",
    "robotic arm": "robotic arm",
    "window sunshade": "window sunshade",
    "lamp post": "lamp post",
    "jungle gym": "jungle gym",
    "non-motorized vehicle lane": "cycle lane",
    "playground road": "running track",
    "building": "building",
    "buildings": "building",
    "buiding": "building",
    "house": "house",
    "houses": "house",
    "plane": "airplane",
    "planes": "airplane",
    "airplane": "airplane",
    "aircraft": "airplane",
    "ship": "ship",
    "ships": "ship",
    "boat": "boat",
    "boats": "boat",
    "yacht": "yacht",
    "car": "car",
    "cars": "car",
    "truck": "truck",
    "trucks": "truck",
    "van": "van",
    "vans": "van",
    "bus": "bus",
    "cruise": "cruise ship",
    "submarine": "submarine",
    "train": "train",
    "trains": "train",
    "vehicle": "vehicle",
    "vehicles": "vehicle",
    "roof": "roof",
    "roofs": "roof",
    "wall": "wall",
    "walls": "wall",
    "parasol": "parasol",
    "umbrella": "umbrella",
    "crane": "crane",
    "cranes": "crane",
    "excavator": "excavator",
    "forklift": "forklift",
    "bridge": "bridge",
    "bridges": "bridge",
    "stadium": "stadium",
    "stadiums": "stadium",
    "playground": "playground",
    "field": "field",
    "ground": "ground area",
    "area": "area",
    "pattern": "pattern",
    "rectangle": "rectangular area",
    "rectangles": "rectangular object",
    "square": "square area",
    "squares": "square structure",
    "object": "object",
    "objects": "object",
    "structure": "structure",
    "structures": "structure",
    "machine": "machine",
    "equipment": "crane",
    "facilities": "facility",
    "flag": "flag",
    "flags": "flag",
    "sign": "sign",
    "pipes": "pipe",
    "tower": "tower",
    "box": "box",
    "boxes": "box",
    "goods": "cargo box",
    "grass": "grassland",
    "grassland": "grassland",
    "river": "river",
    "bay": "bay",
    "pool": "pool",
    "fountain": "fountain",
    "roundabout": "roundabout",
    "crossroads": "intersection",
    "intersection": "intersection",
    "road": "road",
    "track": "track",
    "lanes": "lane",
    "lane": "lane",
    "reef": "reef",
    "turntable": "roundabout",
    "statue": "statue",
    "tent": "tent",
    "chimney": "chimney",
    "pier": "pier",
    "tree": "tree",
    "trees": "tree",
    "airport": "airport",
    "step": "step",
    "deck": "deck",
    "people": "person",
    "complexes": "building complex",
    "skyscraper": "skyscraper",
    "courses": "sports court",
    "sunshade": "sunshade",
    "pit": "pit",
    "entrance": "entrance",
}

# Source-specific corrections resolve malformed grammar or nonliteral nouns.
CATEGORY_OVERRIDES = {
    40: "structure",
    75: "structure",
    116: "square opening",
    157: "bay",
    128: "building",
    182: "building",
    239: "cargo box",
    290: "ground area",
    300: "roof",
    343: "sports court",
    361: "square ground area",
    369: "bridge",
    385: "tree",
    387: "tree",
    410: "square ground area",
    424: "ship",
}
PARTS = {
    "front": "front",
    "roof": "roof",
    "roofs": "roofs",
    "wall": "wall",
    "walls": "walls",
    "exterior wall": "exterior wall",
    "tops": "tops",
    "spire": "spire",
    "wings": "wings",
    "head": "front",
    "bow": "bow",
}
COLORS = (
    "blue and white",
    "white and green",
    "black-and-white",
    "blue-and-white",
    "blue-gray",
    "dark blue",
    "light blue",
    "dark brown",
    "bright red",
    "fluorescent blue",
    "yellow-green",
    "blue and grey",
    "blue and brown",
    "white and pink",
    "red and white",
    "gray",
    "grey",
    "white",
    "black",
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "silver",
    "pink",
    "purple",
)
SHAPES = (
    "h-shaped",
    "u-shaped",
    "y-shaped",
    "fan-shaped",
    "semi-circular",
    "disc-shaped",
    "half-ring shaped",
    "football shaped",
    "heart shaped",
    "oval shaped",
    "cone-shaped",
    "droplet-shaped",
    "cylindrical",
    "rectangular",
    "triangular",
    "circular",
    "square",
    "oval",
    "round",
    "trapezoidal",
)
REGIONS = {
    "upper left": "TOP_LEFT",
    "top left": "TOP_LEFT",
    "lower left": "BOTTOM_LEFT",
    "bottom left": "BOTTOM_LEFT",
    "upper right": "TOP_RIGHT",
    "top right": "TOP_RIGHT",
    "lower right": "BOTTOM_RIGHT",
    "bottom right": "BOTTOM_RIGHT",
    "middle left": "CENTER_LEFT",
    "middle right": "CENTER_RIGHT",
    "middle bottom": "BOTTOM_CENTER",
    "middle of the bottom": "BOTTOM_CENTER",
    "top middle": "TOP_CENTER",
    "at the top": "TOP",
    "above the picture": "TOP",
    "below the picture": "BOTTOM",
    "at the bottom": "BOTTOM",
    "in the middle": "CENTER",
    "center area": "CENTER",
    "on the left side": "LEFT",
    "on the right side": "RIGHT",
    "left edge": "LEFT",
    "right edge": "RIGHT",
    "left area": "LEFT",
    "right area": "RIGHT",
    "top area": "TOP",
    "bottom area": "BOTTOM",
    "upper area": "TOP",
    "lower area": "BOTTOM",
}


def quoted(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def referent(row: dict[str, Any]) -> tuple[str, bool, str | None]:
    question = row["question"].strip().rstrip("?\uff1f2").strip()
    kind = row["metadata"]["source_category"]
    plural = bool(re.search(r"\b(?:are|colors|roofs|walls|tops)\b", question, re.I))
    if kind == "count":
        text = re.sub(r"^How many\s+", "", question, flags=re.I)
        text = re.sub(r"\bare there\b|\bare\b", "", text, flags=re.I)
        text = " ".join(text.split())
        return text, True, None
    patterns = (
        r"^What (?:is|are) the colors? of\s+",
        r"^What'?s the color of\s+",
        r"^What colors? (?:is|are)\s+",
        r"^What color appears of\s+",
        r"^In the picture, where (?:is|are)\s+",
        r"^Where (?:is|are)\s+",
        r"^In which area(?: of the picture)? is\s+",
        r"^What area is\s+",
        r"^What is the location of\s+",
        r"^Which area is\s+",
    )
    text = question
    for pattern in patterns:
        updated, count = re.subn(pattern, "", text, count=1, flags=re.I)
        if count:
            text = updated
            break
    else:
        match = re.match(r"^What color (boat|car) is\s+(.+)$", question, re.I)
        if match:
            text = f"the {match.group(1)} {match.group(2)}"
        elif question.startswith("Which position in the picture is full of ships parked"):
            text = "the bay full of parked ships"
        else:
            raise ValueError(f"unreviewed question shell: {question}")
    text = re.sub(r"\blocated\b", "", text, flags=re.I)
    text = re.sub(r"\bin\s*$", "", text, flags=re.I)
    text = re.sub(r"\b(?:in|of) (?:this|the) (?:picture|image)$", "", text, flags=re.I)
    text = " ".join(text.split()).rstrip(" ,")
    text = re.sub(r"^the position of ", "", text, flags=re.I)
    part = None
    for phrase in sorted(PARTS, key=len, reverse=True):
        match = re.match(rf"^(?:the )?{re.escape(phrase)} (?:of|on)(?: the)? (.+)$", text, re.I)
        if match and not re.match(
            r"(?:upper|lower|top|bottom|middle|left|right)\b", match.group(1), re.I
        ):
            part, text = PARTS[phrase], "the " + match.group(1)
            break
    # Possessive wording in these two reviewed questions refers to a roof part.
    match = re.match(r"^the building's roof (.+)$", text, re.I)
    if match:
        part, text = "roof", "the building " + match.group(1)
    return text, plural, part


def noun_and_attributes(text: str, index: int) -> tuple[str, dict[str, Any], str, str]:
    matches = []
    for alias, category in NOUNS.items():
        for match in re.finditer(rf"\b{re.escape(alias)}\b", text, re.I):
            if alias == "square" and re.match(
                r"\s+(?:buildings?|houses?|roofs?|fields?|attics?)\b", text[match.end() :], re.I
            ):
                continue
            matches.append((match.start(), -len(alias), category, match))
    if not matches:
        if index == 40:
            return "structure", {"pattern": "U-shape"}, text, ""
        raise ValueError(f"unreviewed target noun: {text}")
    _, _, category, match = min(matches, key=lambda item: item[:2])
    category = CATEGORY_OVERRIDES.get(index, category)
    prefix = text[: match.start()].casefold()
    suffix = text[match.end() :].strip()
    attrs: dict[str, Any] = {}
    for color in COLORS:
        if re.search(rf"\b{re.escape(color)}\b", prefix):
            if re.search(rf"\b{re.escape(color)}-roofed\b", prefix):
                attrs["has_part"] = color + " roof"
            else:
                attrs["color"] = color
            break
    for shape in SHAPES:
        if shape in prefix and category not in {
            "square area",
            "rectangular area",
            "square structure",
        }:
            attrs["shape"] = shape
            break
    if re.search(r"\bsmall\b", prefix):
        attrs["size"] = "small"
    elif re.search(r"\b(?:big|large)\b", prefix):
        attrs["size"] = "large"
    return category, attrs, prefix, suffix


def target(category: str, attrs: dict[str, Any]) -> str:
    return (
        "T("
        + quoted(category)
        + "".join(f",{key}={quoted(value)}" for key, value in sorted(attrs.items()))
        + ")"
    )


def build_plan(row: dict[str, Any], index: int) -> dict[str, Any]:
    kind = row["metadata"]["source_category"]
    question = row["question"]
    # Absence and dominant-color requests require an enumerated visible set.
    special = {
        132: (
            "ship",
            "docked ships in the bottom right corner of the picture",
            "Which color occurs most frequently among the selected docked ships?",
        ),
        342: (
            "truck",
            "trucks in the truck parking area in the upper left corner of the picture",
            "Which color occurs most frequently among the selected trucks?",
        ),
        326: (
            "car",
            "all visible cars",
            "Which option describes a car color that is absent from the image?",
        ),
        397: (
            "boat",
            "all visible boats",
            "Which option describes a boat color that is absent from the image?",
        ),
    }
    notes = [
        "All 425 question shells and target nouns were reviewed; "
        "no source answer or image was read."
    ]
    if index in special:
        category, query, residual = special[index]
        lines = [
            "INTENT(OTHER)",
            f"n1=LOCATE($image0,T({quoted(category)}))",
            f"n2=SELECT_SEM($n1,null,{quoted(query)},MULTI,$image0)",
            f"FINAL_QUESTION($n2,CHOICE_SINGLE,{quoted(residual)})",
        ]
        rule = "mme-enumerated-color-distribution-or-absence-v1"
    elif index == 40:
        # The U is a configuration of circles, not an individual circle.
        lines = [
            "INTENT(OTHER)",
            'n1=LOCATE($image0,T("structure",color="red",shape="circular"))',
            "n2=GROUP($n1,CLUSTER)",
            "n3=SELECT_SEM($n2,null,"
            '"the U-shaped group formed by connected red circular structures",SINGLE,$image0)',
            "FINAL_QUESTION($n3,CHOICE_SINGLE,"
            '"Where in the original image is the selected target group?")',
        ]
        rule = "mme-explicit-group-configuration-semantic-v1"
    elif index in {37, 50, 54}:
        # These source questions omit the adjacent target's category. Keep that
        # unresolved judgment in FINAL rather than inventing an object class.
        anchor = re.split(r"\bnext to\b", question, maxsplit=1, flags=re.I)[1].rstrip("?")
        category, attrs, _, _ = noun_and_attributes(anchor, index)
        lines = [
            "INTENT(OTHER)",
            f"n1=LOCATE($image0,{target(category, attrs)})",
            f"n2=SELECT_SEM($n1,null,{quoted(anchor)},SINGLE,$image0)",
            "FINAL_QUESTION($n2,CHOICE_SINGLE,"
            '"What color is visible next to the selected reference object?")',
        ]
        notes.append(
            "Source omits the adjacent target category; "
            "retain that judgment as a final residual question."
        )
        rule = "mme-underspecified-adjacent-color-reference-v1"
    else:
        description, plural, part = referent(row)
        category, attrs, prefix, suffix = noun_and_attributes(description, index)
        if part:
            # Plural roofs/walls on a single house are attributes of one
            # referent, not evidence that several houses must be selected.
            head = description[: len(description) - len(suffix)] if suffix else description
            plural = bool(
                re.search(r"\b(?:buildings|houses|vans|planes|ships|boats)\b", head, re.I)
            )
        # A singular part on several objects is still a plural visual source.
        if kind == "count":
            plural = True
        if plural and " and " in str(attrs.get("color", "")):
            # "white and pink roofs" may denote a collection containing either
            # color. Do not require every roof to be two-colored before SELECT.
            attrs.pop("color")
        extreme = next(
            (
                direction
                for word, direction in (("leftmost", "LEFTMOST"), ("rightmost", "RIGHTMOST"))
                if word in prefix
            ),
            None,
        )
        rank = (
            "ASCENDING"
            if re.search(r"\bsmallest\b", prefix)
            else "DESCENDING"
            if re.search(r"\b(?:largest|biggest)\b", prefix)
            else None
        )
        cleaned = (
            re.sub(
                r"\b(?:leftmost|rightmost|largest|biggest|smallest)\s+", "", description, flags=re.I
            )
            if extreme or rank
            else description
        )
        if extreme or rank:
            cleaned = re.sub(r"^the one of (?:the )?", "the ", cleaned, flags=re.I)
        # Absolute image regions are deterministic only when the entire suffix
        # is a location; nested landmark locations remain in the semantic query.
        region = None
        simple_suffix = suffix.casefold()
        region_suffix = bool(re.match(r"^(?:in|at|on|along|above|below)\b", simple_suffix))
        if region_suffix and not re.search(
            r"\b(?:building|house|river|water|road|car|ship|roof|shore|bay|lot|pool|track|field|"
            r"corner along)\b",
            simple_suffix,
        ):
            for phrase, position in REGIONS.items():
                if phrase in simple_suffix:
                    region = position
                    break
        # All nontrivial relations/patterns/groups remain complete free text.
        unexplained_prefix = re.sub(
            r"\b(?:the|a|an|small|big|large|leftmost|rightmost|largest|biggest|smallest)\b",
            "",
            prefix,
        )
        for value in (*COLORS, *SHAPES):
            unexplained_prefix = unexplained_prefix.replace(value, "")
        complex_prefix = bool(unexplained_prefix.strip(" -"))
        semantic = bool(
            (suffix and region is None) or complex_prefix or (plural and kind != "count")
        )
        scope = "$image0"
        lines = [
            "INTENT(RELATIONAL_COUNT)"
            if kind == "count" and semantic
            else "INTENT(SIMPLE_COUNT)"
            if kind == "count"
            else "INTENT(OTHER)"
        ]
        node = 0
        if region:
            node += 1
            lines.append(f"n{node}=REGION($image0,{region})")
            scope = f"$n{node}"
        if kind == "count" and not semantic and not extreme and not rank:
            node += 1
            entire = "true" if scope == "$image0" else "false"
            lines.append(f"n{node}=COUNT({scope},{target(category, attrs)},{entire})")
            lines.append(f"FINAL($n{node},CHOICE_SINGLE)")
            rule = "mme-dedicated-region-or-whole-count-v1"
        else:
            node += 1
            lines.append(f"n{node}=LOCATE({scope},{target(category, attrs)})")
            source = f"$n{node}"
            if semantic:
                node += 1
                selection = "MULTI" if plural or extreme or rank else "SINGLE"
                lines.append(
                    f"n{node}=SELECT_SEM({source},null,{quoted(cleaned)},{selection},{scope})"
                )
                source = f"$n{node}"
            if extreme:
                node += 1
                lines.append(f"n{node}=SELECT_EXTREME({source},null,{extreme})")
                source = f"$n{node}"
            if rank:
                node += 1
                lines.append(f'n{node}=SELECT_RANK({source},null,"bbox_area",1,{rank})')
                source = f"$n{node}"
            if kind == "count":
                node += 1
                lines.append(f"n{node}=COUNT_ENTITIES({source},{target(category, attrs)},false)")
                lines.append(f"FINAL($n{node},CHOICE_SINGLE)")
                rule = "mme-full-query-semantic-set-then-count-v1"
            else:
                residual = (
                    (
                        "What color or colors are visible on the "
                        + part
                        + " of the selected target or targets?"
                        if part
                        else "What color or colors are visible on the selected object or objects?"
                    )
                    if kind == "color"
                    else "Where in the original image is the selected target or target group?"
                )
                lines.append(f"FINAL_QUESTION({source},CHOICE_SINGLE,{quoted(residual)})")
                rule = (
                    "mme-intrinsic-or-region-visual-residual-v1"
                    if not semantic
                    else "mme-full-referent-semantic-visual-residual-v1"
                )
            if index in {61, 98, 388}:
                notes.append(
                    "Nested source ordering is retained in the complete semantic query; "
                    "no unsupported diagonal ordering is invented."
                )
            if index == 312:
                notes.append(
                    "The source's 'sets of cranes' is treated as equipment units, "
                    "not an inferred spatial clustering of otherwise separate cranes."
                )
    sample = NormalizedSample.model_validate(row)
    dsl = "\n".join(lines)
    _, audit = _audit_dsl(dsl, sample)
    if not audit["valid"]:
        raise ValueError(f"{row['sample_id']}: {audit}")
    return {
        "sample_id": row["sample_id"],
        "dsl": audit["canonical_dsl"],
        "rule_id": rule,
        "review_notes": " ".join(notes),
        "source_agent": "choice_teacher_audit",
        "source_shard_index": index,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=DEFAULT)
    args = parser.parse_args()
    reviewed_source = args.prepared_dir / "mme_b.jsonl"
    if hashlib.sha256(reviewed_source.read_bytes()).hexdigest() != REVIEWED_SOURCE_SHA256:
        raise ValueError("Reviewed question content/order changed; fresh Codex review is required")
    rows = [
        json.loads(line)
        for line in (args.prepared_dir / "mme_b.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    results, errors = [], []
    for index, row in enumerate(rows):
        try:
            results.append(build_plan(row, index))
        except Exception as exc:
            errors.append(
                {
                    "index": index,
                    "sample_id": row["sample_id"],
                    "question": row["question"],
                    "error": str(exc),
                }
            )
    output = args.prepared_dir / "plans/mme_b.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in results
        ),
        encoding="utf-8",
    )
    (output.parent / "mme_b_errors.json").write_text(
        json.dumps(errors, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "total": len(rows),
                "valid": len(results),
                "errors": len(errors),
                "rules": Counter(row["rule_id"] for row in results),
            },
            ensure_ascii=False,
        )
    )
    for error in errors:
        print(json.dumps(error, ensure_ascii=True))
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
