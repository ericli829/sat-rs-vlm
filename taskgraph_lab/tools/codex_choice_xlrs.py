# ruff: noqa: E501, E402
# Keep reviewed descriptions intact; direct-script imports follow path bootstrap.
"""Codex-authored conversion rules for the 449 reviewed XLRS expansion questions.

The immutable input contains original questions/options but deliberately no GT.
These rules are limited to those reviewed questions, not a general teacher or
gold extractor. Every output is parsed, validated, and canonicalized.
"""

from __future__ import annotations

import collections
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from taskgraph_lab.taskgraph.canonicalize import (
    canonicalize_target,
)
from taskgraph_lab.taskgraph.dsl import (
    compile_taskgraph_to_dsl,
    parse_taskgraph_dsl,
)
from taskgraph_lab.taskgraph.validator import (
    validate_candidate,
)

BASE = ROOT / "taskgraph_lab/data/choice_expansion_codex_v1"
REVIEWED_QUESTION_FINGERPRINT = "7cc1a5dcf65d63b7f8c35c7eb30deab144c6b6bc6865171bbfc439fbf64cda70"


def normalize_reviewed_temporal_inputs(row):
    """Split the two reviewed compound HRSCD asset keys, never invent layout."""
    if row["sample_id"] not in {"xlrs_001929", "xlrs_001954"} or "image1" in row["inputs"]:
        return row
    original = row["inputs"].get("image0", {}).get("uri_or_key", "")
    match = re.fullmatch(r"(HRSCD[^:]*\.tif):(HRSCD[^:]*\.tif)", original, re.I)
    if not match or not re.search(r"\btwo (?:pictures|images)\b", row["question"], re.I):
        raise ValueError("Reviewed temporal sample no longer matches its paired-asset contract")
    result = dict(row)
    result["inputs"] = {
        f"image{i}": {"type": "image", "uri_or_key": uri} for i, uri in enumerate(match.groups())
    }
    result["metadata"] = {
        **row.get("metadata", {}),
        "original_compound_uri_or_key": original,
        "temporal_inputs_normalized": True,
    }
    return result


def j(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class Plan:
    def __init__(self):
        self.lines = []

    def add(self, expression):
        ref = f"$n{len(self.lines) + 1}"
        self.lines.append(f"{ref[1:]}={expression}")
        return ref

    def final(self, source, cardinality="CHOICE_SINGLE", question=None):
        source = "[" + ",".join(source) + "]" if isinstance(source, list) else source
        if question:
            self.lines.append(f"FINAL_QUESTION({source},{cardinality},{j(question)})")
        else:
            self.lines.append(f"FINAL({source},{cardinality})")
        return "\n".join(self.lines)


# Longest lexical heads are preferred, then the first noun in a reviewed NP.
# Generic descriptor phrases are retained in SELECT_SEM, never hidden in category.
HEADS = {
    "sewage treatment plant": "sewage treatment plant",
    "residential area": "residential area",
    "residential unit": "residential building",
    "residential building": "residential building",
    "housing area": "residential area",
    "cluster of houses": "residential area",
    "area with the highest concentration of houses": "residential area",
    "area with the largest cluster of houses": "residential area",
    "larger area of forest": "forest",
    "large expanse of forest": "forest",
    "larger area of greenhouses": "greenhouse area",
    "area of houses": "residential area",
    "multi-unit residential area": "residential area",
    "multi-unit residential building": "residential building",
    "industrial area": "industrial area",
    "shopping mall": "shopping mall",
    "storage tank": "storage tank",
    "signal tower": "signal tower",
    "swimming pool": "swimming pool",
    "tennis court": "tennis court",
    "basketball court": "basketball court",
    "soccer field": "soccer field",
    "football stadium": "stadium",
    "sports field": "sports field",
    "sports area": "sports area",
    "parking lot": "parking lot",
    "parking area": "parking lot",
    "parked area": "parking lot",
    "open-pit mine": "open-pit mine",
    "road intersection": "intersection",
    "round intersection": "roundabout",
    "circular intersection": "roundabout",
    "water channel": "water channel",
    "river channel": "river",
    "body of water": "body of water",
    "open space": "open space",
    "open area": "open space",
    "forest area": "forest",
    "forested area": "forest",
    "green space": "green space",
    "ring road": "ring road",
    "circular road": "circular road",
    "circular track": "track",
    "circular pit": "pit",
    "circular hole": "hole",
    "circular depression": "depression",
    "black hole": "hole",
    "roundabout": "roundabout",
    "interchange": "interchange",
    "greenhouse": "greenhouse",
    "building": "building",
    "house": "house",
    "roof": "building",
    "lake": "lake",
    "pond": "pond",
    "pool": "pool",
    "forest": "forest",
    "stadium": "stadium",
    "playground": "playground",
    "track": "track",
    "racetrack": "racetrack",
    "field": "field",
    "farmland": "farmland",
    "grassland": "grassland",
    "lawn": "lawn",
    "boat": "boat",
    "vessel": "ship",
    "ship": "ship",
    "vehicle": "vehicle",
    "car": "car",
    "truck": "truck",
    "van": "van",
    "container": "container",
    "bridge": "bridge",
    "road": "road",
    "reservoir": "reservoir",
    "quarry": "quarry",
    "mine": "mine",
    "island": "island",
    "helipad": "helipad",
    "shed": "shed",
    "plot of land": "land parcel",
    "patch of grass": "grass patch",
    "solar panel": "solar panel",
    "oil tank": "oil tank",
    "buoy": "buoy",
    "land": "land",
    "water": "water",
    "patch": "patch",
    "clearing": "clearing",
    "terrain": "terrain",
    "object": "object",
    "square": "square",
    "area": "area",
    "river": "river",
    "park": "park",
    "wasteland": "wasteland",
    "space": "open space",
    "grid area": "grid area",
    "grid": "grid area",
}
HEAD_PATTERN = re.compile(
    r"\b("
    + "|".join(
        re.escape(h) + ("es" if h.endswith("s") else "s") + "?"
        for h in sorted(HEADS, key=len, reverse=True)
    )
    + r")\b",
    re.I,
)


# Explicit search instead of a universal NP grammar keeps unusual plural forms
# and compounds visible for author review.
def head(description):
    matches = []
    for word, category in HEADS.items():
        match = re.search(r"\b" + re.escape(word) + r"(?:s|es)?\b", description, re.I)
        if match:
            if word == "square" and not match.group().lower().endswith("s"):
                # "square building/pond" and "square-shaped lake" use square
                # as an intrinsic shape; "six green squares" is a noun.
                continue
            if word == "grid" and re.match(r"\s+building\b", description[match.end() :], re.I):
                continue
            matches.append((match.start(), -len(word), category, match.end()))
    if not matches:
        raise ValueError(f"Unreviewed target head: {description}")
    start, _, category, end = min(matches)
    return category, start, end


COLORS = [
    "red and green",
    "gray and red",
    "emerald green",
    "dark green",
    "light green",
    "deep green",
    "light gray",
    "dark gray",
    "deep gray",
    "dark black",
    "light blue",
    "deep blue",
    "dark blue",
    "gray-white",
    "blue-green",
    "gray-white",
    "cream-colored",
    "brick-red",
    "sky-blue",
    "emerald",
    "white",
    "black",
    "green",
    "gray",
    "grey",
    "blue",
    "red",
    "yellow",
    "orange",
    "brown",
    "purple",
    "pink",
    "silver",
]
SHAPES = [
    "rectangular",
    "square",
    "semicircular",
    "circular",
    "cylindrical",
    "triangular",
    "oval-shaped",
    "L-shaped",
    "U-shaped",
    "H-shaped",
    "T-shaped",
    "C-shaped",
    "Z-shaped",
    "cross-shaped",
    "heart-shaped",
    "fan-shaped",
    "racquet-shaped",
    "trapezoidal",
    "irregular",
    "round",
]
POSITION = [
    ("bottom right", "BOTTOM_RIGHT"),
    ("lower right", "BOTTOM_RIGHT"),
    ("top right", "TOP_RIGHT"),
    ("upper right", "TOP_RIGHT"),
    ("bottom left", "BOTTOM_LEFT"),
    ("lower left", "BOTTOM_LEFT"),
    ("top left", "TOP_LEFT"),
    ("upper left", "TOP_LEFT"),
    ("bottom center", "BOTTOM_CENTER"),
    ("upper center", "TOP_CENTER"),
    ("upper middle", "TOP_CENTER"),
    ("middle left", "CENTER_LEFT"),
    ("center-right", "CENTER_RIGHT"),
    ("central area", "CENTER"),
    ("middle area", "CENTER"),
    ("bottom", "BOTTOM"),
    ("upper part", "TOP"),
    ("top", "TOP"),
    ("left side", "LEFT"),
    ("right side", "RIGHT"),
]


def target_expression(category, attrs=None):
    return (
        "T("
        + j(category)
        + "".join("," + k + "=" + j(v) for k, v in sorted((attrs or {}).items()))
        + ")"
    )


def target(plan, description, cardinality="SINGLE", scope="$image0"):
    description = description.strip(" ,?.")
    if description.lower().startswith("the vertical row of silver sheds"):
        sheds = plan.add(f'LOCATE({scope},T("shed",color="silver"))')
        rows = plan.add(f"GROUP({sheds},ROW)")
        return plan.add(f"SELECT_SEM({rows},null,{j(description)},SINGLE,{scope})")
    if description.lower().startswith("the larger area of greenhouses"):
        greenhouses = plan.add(f'LOCATE({scope},T("greenhouse"))')
        clusters = plan.add(f"GROUP({greenhouses},CLUSTER)")
        return plan.add(f"SELECT_SEM({clusters},null,{j(description)},SINGLE,{scope})")
    category, start, end = head(description)
    prefix = description[:start]
    attrs = {}
    for color in COLORS:
        if re.search(r"\b" + re.escape(color) + r"\b", prefix, re.I):
            attrs["color"] = color
            break
    for shape in SHAPES:
        if re.search(r"\b" + re.escape(shape) + r"\b", prefix, re.I):
            attrs["shape"] = shape
            break
    if re.search(r"\b(?:large|small)\b", prefix, re.I):
        attrs["size"] = "small" if re.search(r"\bsmall\b", prefix, re.I) else "large"
    state_match = re.search(
        r"\b(?:old,? disused|disused|moving|moored|docked|sailing)\b", description, re.I
    )
    if state_match:
        attrs["state"] = state_match.group().lower().replace(",", "")
    if re.search(r"\b(?:striped|patterned|grid)\b", prefix, re.I):
        attrs["pattern"] = (
            "striped"
            if "striped" in prefix.lower()
            else "grid"
            if "grid" in prefix.lower()
            else "patterned"
        )
    for part in ["pointed roof", "sloped roof", "flat roof", "spire", "courtyard"]:
        if part in description.lower() or part.replace(" ", "-") in description.lower():
            attrs["has_part"] = part
            break
    # Object parts / surface patterns are intrinsic. Do not teach SELECT_SEM
    # for an ordinary roof, deck, wall, or marking description.
    part_match = re.search(
        r"\bwith (?:a |the )?(.+?(?:roof|rooftop|deck|exterior wall|vents|yellow spots|white lines))(?=,|\s+(?:in|on|near|next to|above|below|located|relative)\b|$)",
        description,
        re.I,
    )
    if part_match:
        attrs["has_part"] = re.sub(
            r"\b(?:largest|smallest)\s+", "", part_match.group(1), flags=re.I
        )
    roofed = re.search(r"([A-Za-z -]+)-(?:roofed|rooftop)\b", prefix, re.I)
    if roofed and "has_part" not in attrs:
        part_prefix = re.sub(
            r"\b(?:the|a|an|largest|smallest|longest)\b", "", roofed.group(1), flags=re.I
        )
        attrs["has_part"] = part_prefix.strip() + " roof"
    if "pattern" in description.lower() or "character resembling" in description.lower():
        pattern_match = re.search(
            r"\bwith (.+?)(?=\s+(?:in|on|near|located)\b|$)", description, re.I
        )
        if pattern_match:
            attrs["pattern"] = pattern_match.group(1)
    shape_match = re.search(
        r"\bshaped like (.+?)(?=,|\s+(?:in|on|near|located|towards)\b|$)", description, re.I
    )
    if shape_match:
        attrs["shape"] = shape_match.group(1)
    ref = plan.add(f"LOCATE({scope},{target_expression(category, attrs)})")
    # Area rank is valid for largest/smallest spatial footprint, not longest,
    # density, number of vehicles, or detector score.
    rank_order = None
    extreme_direction = None
    primary_size_superlative = re.search(r"\b(largest|smallest)\b", prefix, re.I) or re.search(
        r"\b(?:with the|which is the) (largest|smallest)\b(?=[^;]*?(?:roof|in the residential area))",
        description,
        re.I,
    )
    if primary_size_superlative and cardinality == "SINGLE":
        rank_order = (
            "ASCENDING" if primary_size_superlative.group(1).lower() == "smallest" else "DESCENDING"
        )
    elif re.search(r"\buppermost\b", prefix, re.I) and cardinality == "SINGLE":
        extreme_direction = "TOPMOST"
    # Intrinsic descriptors are not automatically rewrapped as semantic filters.
    residual = description[end:].strip(" ,")
    if state_match:
        residual = re.sub(
            r"\b(?:that is )?(?:old,? disused|disused|moving|moored|docked|sailing)\b",
            "",
            residual,
            flags=re.I,
        ).strip(" ,")
    if part_match:
        residual = residual.replace(part_match.group(0), "").strip(" ,")
    if shape_match:
        residual = residual.replace(shape_match.group(0), "").strip(" ,")
    if "pattern_match" in locals() and pattern_match:
        residual = residual.replace(pattern_match.group(0), "").strip(" ,")
    residual = re.sub(r"^(?:in|of) the (?:image|picture|map)$", "", residual, flags=re.I)
    residual = re.sub(
        r"\b(?:with a |with the |with )?(?:pointed|sloped|flat)(?:-| )roof\b",
        "",
        residual,
        flags=re.I,
    ).strip(" ,")
    residual = re.sub(r"\b(?:with a |with )?courtyard\b", "", residual, flags=re.I).strip(" ,")
    residual_prefix = re.sub(
        r"\b(?:the|a|an|two|three|four|five|six|seven|several|many|large|small|larger|smaller|largest|smallest|uppermost|adjacent|elongated|standard|long|light|dark|deep|irregularly|shaped|roofed|rooftop|slanted|spire|pointed|flat|overall|striped|patterned|grid|old|disused)\b",
        "",
        prefix,
        flags=re.I,
    )
    for intrinsic in COLORS + SHAPES:
        residual_prefix = re.sub(re.escape(intrinsic), "", residual_prefix, flags=re.I)
    # Generic in-image wording does not identify an external condition.
    residual = re.sub(
        r"(?:as seen )?in the (?:image|picture|map)$", "", residual, flags=re.I
    ).strip(" ,")
    if (
        residual
        or residual_prefix.strip(" ,-")
        or re.search(
            r"\b(?:two|three|five|seven|several|adjacent|larger|smaller|longest|longer|long)\b",
            description,
            re.I,
        )
    ):
        query = description
        semantic_cardinality = cardinality
        if rank_order or extreme_direction:
            # Filter the described domain first, then apply the explicit
            # deterministic extremum. Ranking globally before the location
            # filter silently changes "largest on the left" into another task.
            query = re.sub(
                r",?\s+which is the (?:largest|smallest) in\b", " in", query, count=1, flags=re.I
            )
            query = re.sub(r"\b(?:largest|smallest|uppermost)\b", "", query, count=1, flags=re.I)
            query = re.sub(r"\s+", " ", query).strip()
            semantic_cardinality = "MULTI"
        ref = plan.add(f"SELECT_SEM({ref},null,{j(query)},{semantic_cardinality},{scope})")
    if rank_order:
        ref = plan.add(f'SELECT_RANK({ref},null,"bbox_area",1,{rank_order})')
    elif extreme_direction:
        ref = plan.add(f"SELECT_EXTREME({ref},null,{extreme_direction})")
    return ref


# Individually reviewed nonstandard/ambiguous spatial templates. Each pair is
# (subject description, reference description). None references ground truth.
SPATIAL = {
    2: ("the green lawn surrounded by white buildings in the central area", "Rainbow Square"),
    28: (
        "the light green rectangular-roofed building near the ring road, outside the circular brown road and inside the green rectangular soccer field",
        None,
    ),
    29: (
        "the gray rectangular building in the lower area",
        "the sports field with the circular track",
    ),
    30: (
        "the larger circular road towards the bottom left near the edge",
        "the smaller circular road",
    ),
    35: (
        "the square-shaped lake with trees inside it located towards the upper left",
        "the six green squares",
    ),
    44: ("the largest residential area", "the circular road"),
    51: (
        "the building with the gray roof covered in many solar panels",
        "the rectangular green wasteland",
    ),
    61: (
        "the lake surrounded by forest on three sides",
        "the two adjacent buildings with emerald green roofs",
    ),
    78: ("the building with a roof resembling tree leaves", "the largest lake on the left"),
    87: ("the purple square building in the lower left area", "the red rectangular building"),
    88: ("the largest forest area", "the dark gray circular road"),
    97: ("the tennis court with red and green colors", "the largest road intersection on the left"),
    107: (
        "the white rectangular building at the lower right edge",
        "the light blue rectangular building",
    ),
    111: ("the round green object located towards the upper left", "the six green squares"),
    115: (
        "the three gray houses located towards the lower left",
        "the rectangular dark green pond",
    ),
    124: ("the circular pit located towards the upper right", "the rectangular parking lot"),
    148: ("the lake shaped like a clock", "the largest forest area"),
    149: (
        "the rectangular blue swimming pool located slightly above the center of the image",
        "the smaller island",
    ),
    152: (
        "the vertical row of silver sheds located slightly below the center",
        "the black grid area",
    ),
    157: ("the largest orange building", "the green rectangular area"),
    166: ("the blue greenhouse located above the green tennis court", "the residential area"),
    179: ("the standard gray tennis court", "the green rectangular sports field"),
    177: (
        "the yellow building above the river on the right",
        "the building with the red sloped roof",
    ),
    205: ("the green lake located towards the upper left", "the larger forest"),
    210: (
        "the green farmland divided into many regular shapes in the upper left corner",
        "the largest residential area",
    ),
    234: (
        "the largest black rectangular building above the road on the right edge",
        "the seven white storage tanks",
    ),
    247: ("the gray rectangular building", "the circular gray building"),
    291: ("the gray L-shaped building located in the central area", "the circular track"),
    275: (
        "the blue rectangular building below the T-junction at the top right",
        "the building with the blue sloped roof",
    ),
    278: ("the building with the white triangular roof", "the brick-red circular track"),
    280: (
        "the white rectangular building directly below and to the left",
        "the longest gray rectangular building",
    ),
    300: ("the blue container at the top left corner on the road", "the yellow car"),
    301: ("the three green objects located towards the upper right", "the circular roundabout"),
    310: ("the building with a cross-shaped roof", "the largest forest area"),
    311: (
        "the circular black hole located towards the lower left",
        "the irregular gray-white terrain",
    ),
    317: ("the circular depression", "the river"),
    318: (
        "the two gray rectangular greenhouses located towards the upper right",
        "the yellow trapezoidal grassland",
    ),
    335: (
        "the light blue rectangular building at the top left corner",
        "the red rectangular building",
    ),
    355: (
        "the red and green track located towards the upper right",
        "the four vertically aligned houses with white roofs",
    ),
    360: ("the triangular green lake located in the bottom right corner", "the oval-shaped road"),
    378: ("the circular road on the right side", "the racquet-shaped lake"),
    379: ("the brown rectangular building on the right edge", "the white rectangular building"),
    382: ("the oval-shaped roundabout at the top", "the uppermost roundabout"),
    389: ("the black rectangular building", "the wasteland at the bottom right"),
    391: (
        "the green circular object located towards the bottom right",
        "the dark green soccer field",
    ),
    414: (
        "the two green rectangular greenhouses located towards the upper right",
        "the triangular parking lot",
    ),
    420: ("the building with the wavy light gray roof", "the largest forest area"),
    421: ("the largest cluster of houses", "the island in the bottom right corner"),
    435: ("the largest lake", "the rectangular forested area"),
    447: (
        "the circular roundabout located towards the upper left",
        "the gray cross-shaped clearing",
    ),
}


def strip_subject(text):
    text = text.strip(" ,")
    # Leading image/area discourse is distinct from the referent NP. Retain it
    # as a suffix rather than discarding an explicitly supplied search context.
    contexts = []
    while re.match(r"^(?:In|On|Below|Above) (?:the |which )", text) and "," in text:
        context, text = text.split(",", 1)
        if not re.fullmatch(r"In the (?:image|picture|map)", context, re.I):
            contexts.append(context)
        text = text.strip()
    text = re.sub(
        r"^(?:what is the (?:orientation|position|direction|location) of|in which direction is|where is|in what (?:position|direction) is|at which position is|at which position)\s+",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(
        r",?\s+(?:is|are)\s+(?:located|positioned|situated)(?:\s+(?:in|at)\s+which\s+(?:direction|position))?\s*$",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(r",?\s+(?:located|positioned|situated)\s*$", "", text, flags=re.I)
    text = re.sub(r"\s+(?:is|are)\s*$", "", text, flags=re.I)
    return text.strip(" ,") + ("; " + "; ".join(contexts) if contexts else "")


def spatial_pair(index, question):
    if index in SPATIAL:
        return SPATIAL[index]
    match = re.search(r"\s+(?:relative to|in relation to|with respect to)\s+", question, re.I)
    if match:
        return strip_subject(question[: match.start()]), question[match.end() :].strip(" ,?.")
    match = re.search(
        r"\s+(?:is|are)\s+(?:located|positioned|situated)\s+(?:in|at)\s+which\s+(?:direction|position)\s+(?:of|from)\s+",
        question,
        re.I,
    )
    if match:
        return strip_subject(question[: match.start()]), question[match.end() :].strip(" ,?.")
    match = re.search(r"\s+in which direction of\s+", question, re.I)
    if match:
        return strip_subject(question[: match.start()]), question[match.end() :].strip(" ,?.")
    match = re.search(
        r"\s+(?:located\s+)?(?:at which position|in which direction)\s+(?:of|from)\s+",
        question,
        re.I,
    )
    if match:
        return strip_subject(question[: match.start()]), question[match.end() :].strip(" ,?.")
    raise ValueError("Spatial template requires individual review")


# Reviewed count specifications: category, attributes, scope/filter descriptor.
# Explicit alternatives are retained (lake or pond), never silently normalized
# to just one alternative. Context is consumed by SELECT_SEM MULTI before COUNT.
COUNT_SPECS = {
    10: ("car", {"state": "parked"}, "cars parked along the bottom edge of the image"),
    17: (
        "building",
        {"has_part": "courtyard"},
        "buildings with a courtyard surrounded by trees in the center and on the left side of the picture",
    ),
    19: (
        "building",
        {"color": "gray"},
        "buildings with gray roofs in the group of buildings on the left side of the pond at the top of the picture",
    ),
    43: (
        "building",
        {},
        "buildings surrounded by the road in the bottom left corner of the picture",
    ),
    55: ("swimming pool", {}, "swimming pools surrounded by forests in the bottom right corner"),
    62: (
        "building",
        {"color": "orange"},
        "buildings with orange roofs in the town crossed by a river on the right side",
    ),
    68: (
        "building",
        {"color": "gray"},
        "buildings with gray roofs in the town in the upper left corner",
    ),
    80: ("car", {}, "cars in the parking lot on the right side"),
    106: ("car", {}, "cars in the parking lot in the top right corner"),
    109: ("car", {}, "cars in the parking lot at the top left corner"),
    110: ("roundabout", {}, "roundabouts in the top left corner"),
    112: (
        "lake or pond",
        {"shape": "triangular"},
        "triangular lakes or ponds in the top right corner",
    ),
    113: ("oil tank", {}, None),
    123: ("mine", {}, None),
    127: (
        "ship",
        {"size": "large", "state": "docked"},
        "large ships docked in the upper right corner",
    ),
    146: ("ship", {"state": "sailing"}, None),
    147: ("parking lot", {}, "parking lots in the town located towards the bottom center"),
    150: ("boat", {"state": "sailing"}, "boats currently sailing in the middle area"),
    155: ("boat", {"state": "moving"}, "boats moving on the right side"),
    160: (
        "parking lot",
        {},
        "parking lots on the right side of the highway in the bottom right corner",
    ),
    168: (
        "sports field",
        {"color": "green", "shape": "rectangular"},
        "green rectangular sports fields on the right side within the large housing area at the bottom right corner",
    ),
    171: ("roundabout", {}, "round intersections in the top right corner"),
    172: ("lake or pond", {"size": "large"}, "large lakes or ponds in the bottom right corner"),
    186: ("shed", {}, "MARKER"),
    195: ("boat", {}, "boats in the river channel at the center"),
    212: ("boat", {}, "MARKER"),
    219: ("land parcel", {}, "MARKER"),
    221: ("boat", {"state": "docked"}, "boats docked at the shore"),
    226: (
        "roundabout",
        {},
        "circular intersections in the bottom right corner of the large residential area",
    ),
    230: ("vehicle", {"state": "moving"}, "vehicles currently in motion in the top left corner"),
    231: (
        "parking lot",
        {},
        "parking lots in the town on the right side of the river on the right side",
    ),
    233: ("storage tank", {}, None),
    236: ("ship", {"state": "sailing"}, "ships sailing in the upper part"),
    240: ("boat", {}, "boats on the east side of the bridge over the river in the middle"),
    243: ("lake", {"color": "green"}, "green lakes in the middle at the bottom"),
    250: ("parking lot", {}, "parking lots in the town in the bottom right corner"),
    260: ("car", {}, "cars in the parking lot at the bottom right corner"),
    261: (
        "track",
        {"color": "light gray", "shape": "oval"},
        "light gray oval tracks in the middle at the bottom",
    ),
    262: (
        "tennis court",
        {"color": "red"},
        "red tennis courts on the far left side of the large housing area in the top right corner",
    ),
    263: ("car", {}, "cars under the black roof in the bottom left corner"),
    265: ("sports field", {"shape": "rectangular"}, "MARKER"),
    273: ("buoy", {}, None),
    277: ("ship", {}, "ships in the ocean at the bottom right corner"),
    279: ("car", {}, "cars on the north-south road on the left side"),
    281: ("grass patch", {}, "MARKER"),
    284: ("solar panel", {}, "MARKER"),
    321: ("tennis court", {"color": "red"}, "red tennis courts in the bottom right corner"),
    333: ("container", {}, "MARKER"),
    356: ("roundabout", {}, "circular intersections at the bottom of the image"),
    365: ("lake", {"color": "green"}, "green lakes in the forest at the bottom left corner"),
    370: (
        "vehicle",
        {"state": "moving"},
        "vehicles traveling from south to north in the top left corner",
    ),
    380: ("vehicle", {"state": "moving"}, "vehicles traveling on the road on the left side"),
    387: ("island", {}, "islands at the bottom of the picture"),
    412: ("open-pit mine", {}, "open-pit mines in the bottom right corner"),
    416: ("vehicle", {}, "MARKER"),
    425: ("track", {}, "tracks in the large forest in the upper left corner"),
    432: ("boat", {}, "boats in the river at the bottom"),
}

PURE_REGION_COUNTS = {
    10: "BOTTOM",
    110: "TOP_LEFT",
    112: "TOP_RIGHT",
    127: "TOP_RIGHT",
    150: "CENTER",
    155: "RIGHT",
    171: "TOP_RIGHT",
    172: "BOTTOM_RIGHT",
    230: "TOP_LEFT",
    236: "TOP",
    243: "BOTTOM_CENTER",
    261: "BOTTOM_CENTER",
    321: "BOTTOM_RIGHT",
    356: "BOTTOM",
    387: "BOTTOM",
    412: "BOTTOM_RIGHT",
}

# These descriptions specify the referent, rather than asking an unknown
# pairwise relation. Keep a visual residual question; do not answer an already
# supplied "above/near/inside" relation instead of the actual position query.
ABSOLUTE_SPATIAL = {
    2: "the green lawn surrounded by white buildings in the central area below Rainbow Square",
    28: "the light green rectangular-roofed building near the ring road, outside the circular brown road and inside the green rectangular soccer field",
    29: "the gray rectangular building adjacent to the sports field with the circular track in the lower area",
    35: "the square-shaped lake with trees inside it located towards the upper left among the six green squares",
    44: "the largest residential area along the circular road",
    51: "the building with the gray roof covered in many solar panels located to the lower left of the rectangular green wasteland",
    61: "the lake surrounded by forest on three sides near the two adjacent buildings with emerald green roofs",
    111: "the round green object located towards the upper left among the six green squares",
    115: "the three gray houses located towards the lower left in the rectangular dark green pond",
    124: "the circular pit located towards the upper right side in the rectangular parking lot",
    166: "the blue greenhouse located above the green tennis court within the residential area",
    247: "the gray rectangular building directly above the circular gray building",
    389: "the black rectangular building located in the wasteland at the bottom right",
}


def count_plan(index, row):
    p = Plan()
    if index in {5, 411}:
        category, attrs = ("car", {}) if index == 5 else ("signal tower", {"color": "white"})
        # An original colon-separated URI denotes two distinct assets, not a
        # proven side-by-side composite. Require normalization to image0/image1.
        if "image1" not in row["inputs"]:
            raise ValueError(
                "Paired XLRS assets require explicit image0/image1 normalization; refusing invented composite layout"
            )
        counts = []
        for image in ("$image0", "$image1"):
            marker = p.add(f'FIND_MARKER({image},"circle","red")')
            counts.append(p.add(f"COUNT_IMAGE({marker},{target_expression(category, attrs)},true)"))
        difference = p.add(f"ABS_DIFF({counts[0]},{counts[1]})")
        return (
            p.final(difference),
            "xlrs_change_marker_absolute_count",
            "Two explicit temporal image roles; authoritative marker-scoped counts are compared without GT.",
        )
    category, attrs, query = COUNT_SPECS[index]
    t = target_expression(category, attrs)
    if query == "MARKER":
        scope = p.add('FIND_MARKER($image0,"circle","red")')
        ref = p.add(f"COUNT_IMAGE({scope},{t},true)")
        return (
            p.final(ref),
            "xlrs_marker_count",
            "Explicit artificial marker scopes authoritative COUNT.",
        )
    if query is None:
        ref = p.add(f"COUNT_IMAGE($image0,{t},true)")
        return (
            p.final(ref),
            "xlrs_whole_count",
            "Whole-image authoritative count; intrinsic attributes retained.",
        )
    if index in PURE_REGION_COUNTS:
        region = p.add(f"REGION($image0,{PURE_REGION_COUNTS[index]})")
        ref = p.add(f"COUNT_IMAGE({region},{t},true)")
        return (
            p.final(ref),
            "xlrs_region_count",
            "Explicit image position uses deterministic REGION; count remains authoritative.",
        )
    if index == 240:
        boats = p.add(f"LOCATE($image0,{t})")
        bridge = target(p, "the bridge over the river in the middle of the picture")
        selected = p.add(f"SELECT_REL({boats},{bridge},RIGHT_OF)")
        count = p.add(f"COUNT_ENTITIES({selected},{t},false)")
        return (
            p.final(count),
            "xlrs_geometric_filtered_count",
            "East of the explicitly identified bridge is deterministic RIGHT_OF; selected complete boat set feeds COUNT.",
        )
    ref = p.add(f"LOCATE($image0,{t})")
    selected = p.add(f"SELECT_SEM({ref},null,{j(query)},MULTI,$image0)")
    count = p.add(f"COUNT_ENTITIES({selected},{t},false)")
    return (
        p.final(count),
        "xlrs_filtered_count",
        "MULTI semantic filtering consumes the complete stated contextual condition before authoritative COUNT.",
    )


MULTI_COMPLEX = {
    9,
    34,
    40,
    56,
    84,
    86,
    94,
    102,
    103,
    108,
    117,
    118,
    128,
    133,
    135,
    139,
    140,
    151,
    158,
    163,
    164,
    175,
    176,
    187,
    192,
    202,
    204,
    213,
    222,
    228,
    229,
    232,
    245,
    251,
    252,
    264,
    269,
    271,
    276,
    293,
    294,
    298,
    299,
    302,
    303,
    304,
    306,
    307,
    308,
    316,
    320,
    324,
    325,
    328,
    331,
    338,
    340,
    349,
    351,
    352,
    354,
    358,
    361,
    362,
    363,
    366,
    376,
    377,
    381,
    383,
    386,
    393,
    397,
    401,
    405,
    407,
    415,
    419,
    430,
    431,
    436,
    438,
    439,
    440,
    445,
    446,
    448,
}
# Reviewed yes/no propositions and mutually exclusive whole-scene descriptions
# still ask for one judgement, even when the wording contains several factors.
MULTI_COMPLEX.difference_update(
    {
        128,
        175,
        202,
        213,
        228,
        264,
        299,
        306,
        320,
        328,
        349,
        352,
        358,
        361,
        366,
        377,
        381,
        397,
        419,
        436,
        448,
    }
)
# Independent plausible causes/effects have no fixed answer cardinality.
MULTI_COMPLEX.update(
    {65, 69, 75, 130, 181, 194, 207, 211, 242, 266, 268, 286, 288, 290, 313, 364, 392, 422}
)


def build(index, row):
    question = row["question"]
    category = row["metadata"]["source_category"]
    if category.startswith("Counting/"):
        return count_plan(index, row)
    if category.startswith("Object spatial relationship/"):
        if index in ABSOLUTE_SPATIAL:
            p = Plan()
            ref = target(p, ABSOLUTE_SPATIAL[index], "MULTI" if index == 115 else "SINGLE")
            return (
                p.final(ref, question=question),
                "xlrs_absolute_position",
                {
                    "subject": ABSOLUTE_SPATIAL[index],
                    "review": "The descriptor relation is already given; final static residual asks the original location question, without treating that descriptor as an unknown relation.",
                },
            )
        subject, reference = spatial_pair(index, question)
        p = Plan()
        is_plural = bool(
            re.search(r"\b(?:two|three|four|five|six|seven|several|patches)\b", subject, re.I)
        )
        subject_ref = target(p, subject, "MULTI" if is_plural else "SINGLE")
        ref_plural = bool(
            re.search(r"\b(?:two|three|four|five|six|seven|several)\b", reference, re.I)
        )
        reference_ref = target(p, reference, "MULTI" if ref_plural else "SINGLE")
        if index == 355:
            reason = p.add(
                f'VLM_REASON(image=$image0,evidence=[{subject_ref},{reference_ref}],question={j(question)},choices="$choices")'
            )
            return (
                p.final(reason),
                "xlrs_reference_choice_ambiguity",
                {
                    "subject": subject,
                    "reference": reference,
                    "review": "Source wording asks which reference but options name directions; preserve both candidate evidence sets and exact original wording for semantic judgement.",
                },
            )
        relation = p.add(f"RELATION({subject_ref},{reference_ref})")
        return (
            p.final(relation),
            "xlrs_spatial_relation",
            {
                "subject": subject,
                "reference": reference,
                "review": "Question/subject/reference reviewed; set-valued references preserved; no predicted relation or GT copied.",
            },
        )
    p = Plan()
    if index == 229:
        ships = p.add('COUNT_IMAGE($image0,T("ship"),true)')
        buildings = p.add('COUNT_IMAGE($image0,T("building"),true)')
        reason = p.add(
            f'VLM_REASON(image=$image0,evidence=[{ships},{buildings}],question={j(question)},choices="$choices")'
        )
        return (
            p.final(reason, "CHOICE_MULTI"),
            "xlrs_count_based_inference",
            "Question explicitly reasons from numbers of ships and buildings; both authoritative counts remain upstream of visual/semantic inference.",
        )
    if index == 296:
        # This is explicitly a compound count + commercial parking judgement,
        # although source metadata labels it as anomaly reasoning.
        ref = p.add('LOCATE($image0,T("vehicle"))')
        selected = p.add(
            f'SELECT_SEM({ref},null,"vehicles at the transportation hub in the picture",MULTI,$image0)'
        )
        count = p.add(f'COUNT_ENTITIES({selected},T("vehicle"),false)')
        reason = p.add(
            f'VLM_REASON(image=$image0,evidence={count},question={j(question)},choices="$choices")'
        )
        return (
            p.final(reason),
            "xlrs_complex_count_and_context",
            "Compound count retained as authoritative evidence; VLM interprets the commercial parking context.",
        )
    ref = p.add(f'VLM_REASON(image=$image0,question={j(question)},choices="$choices")')
    cardinality = "CHOICE_MULTI" if index in MULTI_COMPLEX else "CHOICE_SINGLE"
    return (
        p.final(ref, cardinality),
        "xlrs_complex_visual_reasoning",
        "Open impact/risk/inference questions allow multiple applicable options; focused causal/suitability requests use a single choice. Original question is preserved verbatim; no predicted answer.",
    )


def main():
    rows = [
        json.loads(line) for line in (BASE / "xlrs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    reviewed_fields = [
        {key: row[key] for key in ("sample_id", "question", "question_type", "choices")}
        for row in rows
    ]
    fingerprint = hashlib.sha256(j(reviewed_fields).encode("utf-8")).hexdigest()
    if fingerprint != REVIEWED_QUESTION_FINGERPRINT:
        raise ValueError(
            "Question/option content or order changed: index-specific Codex rules require a fresh semantic review"
        )
    outputs, failures = [], []
    for index, row in enumerate(rows):
        try:
            row = normalize_reviewed_temporal_inputs(row)
            dsl, rule, notes = build(index, row)
            parsed = parse_taskgraph_dsl(dsl)
            target_graph, valid = validate_candidate(
                parsed,
                inputs=row["inputs"],
                question=row["question"],
                question_type=row["question_type"],
            )
            if not valid.valid:
                raise ValueError(str(valid.model_dump()))
            canonical_dsl = compile_taskgraph_to_dsl(target_graph)
            assert canonicalize_target(parse_taskgraph_dsl(canonical_dsl)) == canonicalize_target(
                target_graph
            )
            outputs.append(
                {
                    "sample_id": row["sample_id"],
                    "dsl": canonical_dsl,
                    "rule_id": rule,
                    "review_notes": notes,
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "index": index,
                    "sample_id": row["sample_id"],
                    "error": str(exc),
                    "question": row["question"],
                }
            )
    directory = BASE / "plans"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "xlrs.jsonl").write_text("".join(j(r) + "\n" for r in outputs), encoding="utf-8")
    (directory / "xlrs_pending.jsonl").write_text(
        "".join(j(r) + "\n" for r in failures), encoding="utf-8"
    )
    print(
        j(
            {
                "input": len(rows),
                "converted": len(outputs),
                "pending": failures,
                "rules": dict(collections.Counter(r["rule_id"] for r in outputs)),
            }
        )
    )


if __name__ == "__main__":
    main()
