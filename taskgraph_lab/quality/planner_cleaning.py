"""Conservative text/graph audit of Planner supervision, without image answers.

Findings are risk flags, not a claim of visual correctness. Only narrow,
information-preserving transformations are automatic; unresolved findings are
excluded from the strict training release and remain in an audit queue.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from typing import Any

from taskgraph_lab.taskgraph.canonicalize import canonicalize_target
from taskgraph_lab.taskgraph.dsl import (
    compile_taskgraph_to_dsl,
    parse_taskgraph_dsl,
    parse_taskgraph_dsl_payload,
)
from taskgraph_lab.taskgraph.dsl.constraint import CanonicalDSLPrefixGrammar
from taskgraph_lab.taskgraph.validator import validate_candidate

VERSION = "planner-quality-clean-v1"
COUNT_OPS = {"COUNT", "COUNT_ENTITIES", "COUNT_IMAGE"}
ALIASES = {
    "cars": "car",
    "vehicles": "vehicle",
    "houses": "house",
    "buildings": "building",
    "ships": "ship",
    "boats": "boat",
    "trucks": "truck",
    "roofs": "roof",
    "structures": "structure",
    "objects": "object",
    "tanks": "tank",
    "gray": "grey",
    "greyish": "grey",
    "reddish": "red",
    "bluish": "blue",
    "greenish": "green",
    "squared": "square",
    "cylinders": "cylindrical",
    "cylinder": "cylindrical",
    "rectangle": "rectangular",
    "triangle": "triangular",
    "round": "circular",
    "circle": "circular",
    "arched": "arch",
    "upper": "top",
    "lower": "bottom",
    "above": "top",
    "below": "bottom",
    "uppermost": "topmost",
    "lowermost": "bottommost",
    "leftmost": "left",
    "rightmost": "right",
    "topmost": "top",
    "bottommost": "bottom",
    "grassland": "grass",
    "grassy": "grass",
    "lawn": "grass",
    "highway": "road",
    "street": "road",
    "trees": "tree",
    "roads": "road",
    "rivers": "river",
    "maples": "maple",
    "leafs": "leaf",
    "leaves": "leaf",
    "surrounding": "around",
    "surrounded": "around",
    "beside": "near",
    "adjacent": "near",
    "next": "near",
    "middle": "center",
    "centre": "center",
    "parallel": "parallel",
    "rows": "row",
    "pools": "pool",
    "fields": "field",
    "pillars": "pillar",
    "flags": "flag",
    "containers": "container",
    "silos": "silo",
    "antennas": "antenna",
    "cranes": "crane",
    "parking": "parking",
    "parked": "parking",
    "park": "parking",
    "crimson": "red",
    "scarlet": "red",
    "brownish": "brown",
    "ovals": "oval",
    "winds": "wind",
    "turbines": "turbine",
}
CRITICAL = set(
    """white black blue red green yellow orange purple pink brown grey silver
    square rectangular triangular cylindrical circular oval striped horizontal vertical
    diagonal parallel maple leaf pattern letter driving moving parking roof road river
    water grass tree shoreline coast pier dock bridge runway railway roundabout container
    crane tower chimney pond pool tennis basketball football track lawn flag flower
    nearest farthest tallest longest shortest wider widest taller highest narrowest
    first second third fourth fifth sixth seventh eighth ninth tenth row cluster
    between around end edge""".split()
)
# A superclass can safely generate candidates for these explicitly named subtypes.
SUPERCLASS = {
    "vehicle": {"car", "van", "truck", "bus"},
    "building": {"house", "warehouse"},
    "ship": {"boat", "yacht"},
}
COUNT_QUESTION = re.compile(
    r"^How many (?P<head>.+?) (?P<verb>are there|are|is there|is|can be seen)\b"
    r"(?P<tail>.*?)\?*$",
    re.I,
)
COMPLEX_COUNT = re.compile(
    r"\b(?:differences?|floors?|stories|storeys|blocks of|groups? of|rows? of|"
    r"types? of|kinds? of|parts? of|sections? of|first|second|third|fourth|fifth|"
    r"largest|smallest|most|least|more|fewer|total|respectively)\b",
    re.I,
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def words(text: str) -> set[str]:
    result = set()
    for token in re.findall(r"[a-z]+", text.casefold()):
        if token not in ALIASES and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        result.add(ALIASES.get(token, token))
    return result


def _values(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(_values(v) for v in value.values())
    if isinstance(value, list):
        return " ".join(_values(v) for v in value)
    return str(value)


def unpack(row: dict) -> tuple[dict, str]:
    messages = row["messages"]
    if [m.get("role") for m in messages] != ["system", "user", "assistant"]:
        raise ValueError("expected exactly system/user/assistant messages")
    if any(not isinstance(m.get("content"), str) for m in messages):
        raise ValueError("all message contents must be strings")
    user = json.loads(messages[1]["content"])
    if any(k in user for k in ("answer", "gt", "ground_truth", "correct_answer")):
        raise ValueError("answer field in Planner input")
    return user, messages[2]["content"]


def question_key(question: str) -> str:
    return " ".join(question.casefold().split())


def identity_key(user: dict) -> str:
    return digest(
        json.dumps(
            [question_key(user["question"]), user["inputs"], user.get("choices")],
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def issue(code: str, detail: str, **kwargs: Any) -> dict:
    return {"code": code, "detail": detail, **kwargs}


def validate(dsl: str, user: dict, sample_id: str) -> dict:
    """Use the actual production bridge as well as lab/parser/decoder checks."""
    from sat_rs_vlm.taskgraph.providers import PlannerRequest, Qwen3VLPlannerProvider
    from sat_rs_vlm.taskgraph.runtime_types import ImageRef

    try:
        graph = canonicalize_target(parse_taskgraph_dsl(dsl))
        _, report = validate_candidate(
            graph,
            inputs=user["inputs"],
            question=user["question"],
            question_type=user["question_type"],
        )
        if not report.valid:
            raise ValueError("; ".join(f"{e.code}: {e.message}" for e in report.errors))
        compiled = compile_taskgraph_to_dsl(graph)
        if canonicalize_target(parse_taskgraph_dsl(compiled)) != graph:
            raise ValueError("canonical DSL roundtrip mismatch")
        grammar = CanonicalDSLPrefixGrammar(image_refs=list(user["inputs"]))
        result = grammar.analyze(compiled)
        if not result.valid_prefix or not result.complete:
            raise ValueError(f"constraint grammar: {result.reason}")
        request = PlannerRequest(
            question=user["question"],
            question_type=user["question_type"],
            choices=tuple(user.get("choices") or []),
            sample_id=sample_id,
            inputs={
                "$" + k.removeprefix("$"): ImageRef(v["uri_or_key"])
                for k, v in user["inputs"].items()
            },
        )
        Qwen3VLPlannerProvider._to_production_graph(compiled, request)
        return {"valid": True, "dsl": compiled, "model_loaded": False}
    except Exception as exc:
        return {"valid": False, "detail": f"{type(exc).__name__}: {exc}", "model_loaded": False}


def _count_repair(graph: dict, user: dict) -> tuple[dict, list[dict]]:
    """Restore omitted conditions only for a simple single-image COUNT path.

    No GROUP/ORDINAL/RANK/RELATION/multiple counts are replaced. The question
    must name the existing count category (or a known subclass). Every condition
    is copied verbatim into a MULTI selector; candidate generation is broad.
    """
    nodes = graph["nodes"]
    counts = [n for n in nodes if n["op"] == "COUNT" and "image" in n["inputs"]]
    if (
        len(counts) != 1
        or len(user["inputs"]) != 1
        or any(n["op"] not in {"REGION", "COUNT"} for n in nodes)
        or graph["final"]["sources"] != ["$" + counts[0]["id"]]
    ):
        return graph, []
    q = user["question"].strip()
    match = COUNT_QUESTION.fullmatch(q)
    if not match or COMPLEX_COUNT.search(q):
        return graph, []
    head, tail = match["head"], match["tail"].strip().rstrip("?").strip()
    if re.search(r"\b(?:that|which)\b", head, re.I):
        return graph, []
    category = counts[0]["params"]["target"]["category"]
    head_words, category_words = words(head), words(category)
    subtype = len(category_words) == 1 and bool(
        SUPERCLASS.get(next(iter(category_words)), set()) & head_words
    )
    if not category_words <= head_words and not subtype:
        return graph, []
    # Pure image-area language is already expressible by REGION. Only add
    # selection when an external object/state/attribute condition is present.
    simple_area = words(
        "in on at the this picture image top bottom left right center "
        "corner side area part region of there visible shown satellite "
        "middle upper lower"
    )
    condition = (words(tail) - simple_area) | ((head_words & CRITICAL) - category_words)
    if not condition:
        return graph, []
    query = "the " + head + (" " + tail if tail else "")
    # Keep intrinsic attributes in SELECT, preventing an old incorrect DINO
    # attribute filter from discarding valid candidates before selection.
    new = deepcopy(graph)
    new["nodes"] = [
        {
            "id": "n1",
            "op": "LOCATE",
            "inputs": {"image": "$image0"},
            "params": {"target": {"category": category}},
        },
        {
            "id": "n2",
            "op": "SELECT",
            "inputs": {"candidates": "$n1", "scope": "$image0"},
            "params": {"mode": "SEMANTIC", "query": query, "selection_type": "MULTI"},
        },
        {
            "id": "n3",
            "op": "COUNT",
            "inputs": {"entities": "$n2"},
            "params": {"target": {"category": category}, "entire": False},
        },
    ]
    new["intent"] = "RELATIONAL_COUNT"
    new["final"]["sources"] = ["$n3"]
    return new, [
        issue(
            "restore_count_conditions",
            "Copy the complete counted referent into "
            "SELECT_SEM MULTI before authoritative COUNT; remove premature crop.",
            query=query,
        )
    ]


def repair(row: dict) -> tuple[dict, list[dict]]:
    """Return a new row. Never mutate the original training/test exports."""
    output = deepcopy(row)
    user, dsl = unpack(row)
    graph = parse_taskgraph_dsl_payload(dsl)
    changes = []
    for node in graph["nodes"]:
        if node["op"] == "SELECT" and node["params"].get("mode") == "RANK":
            if node["params"].get("criterion") in {"size", "area"}:
                node["params"]["criterion"] = "bbox_area"
                changes.append(
                    issue(
                        "canonical_rank_alias",
                        "Normalize supported area/size "
                        "alias to production bbox_area; no new physical metric.",
                    )
                )
    graph, count_changes = _count_repair(graph, user)
    changes += count_changes
    # Explicit source coordinates take priority over any old label coordinates.
    resolution = re.search(r"Image resolution:\s*(\d+)\s*x\s*(\d+)\.", user["question"])
    box = re.search(r"Bounding box:\s*(\[[^\]]+\])", user["question"])
    regions = [n for n in graph["nodes"] if n["op"] == "REGION_FROM_BBOX"]
    if resolution and box and len(regions) == 1:
        supplied = json.loads(box[1])
        size = [int(resolution[1]), int(resolution[2])]
        if len(supplied) != 4 or not (
            0 <= supplied[0] < supplied[2] <= size[0] and 0 <= supplied[1] < supplied[3] <= size[1]
        ):
            raise ValueError("invalid source bounding box")
        if regions[0]["params"].get("bbox") != supplied:
            raise ValueError("label bbox disagrees with the explicit source; requires review")
        if regions[0]["params"].get("image_size") != size:
            regions[0]["params"]["image_size"] = size
            changes.append(
                issue(
                    "explicit_bbox_size",
                    "Preserve source image resolution alongside the already supplied pixel bbox.",
                )
            )
    output["messages"][2]["content"] = compile_taskgraph_to_dsl(canonicalize_target(graph))
    return output, changes


def semantic_findings(user: dict, dsl: str) -> list[dict]:
    """High recall risk audit; uncertain cases must be reviewed, not auto-labelled."""
    graph = parse_taskgraph_dsl_payload(dsl)
    nodes = graph["nodes"]
    q = user["question"]
    qw = words(q)
    evidence = words(_values(nodes))
    for n in nodes:
        if n["params"].get("mode") == "ORDINAL":
            ordinal = {
                1: "first",
                2: "second",
                3: "third",
                4: "fourth",
                5: "fifth",
                6: "sixth",
                7: "seventh",
                8: "eighth",
                9: "ninth",
                10: "tenth",
            }
            evidence.add(ordinal.get(n["params"].get("index"), ""))
    findings = []
    by_ref = {"$" + n["id"]: n for n in nodes}
    for n in nodes:
        if n["op"] != "SELECT":
            continue
        mode = n["params"].get("mode")
        producer = by_ref.get(n["inputs"].get("candidates"))
        if mode in {"ORDINAL", "RANK", "EXTREME"} and producer and producer["op"] == "GROUP":
            findings.append(
                issue(
                    "group_members_not_groups",
                    "GROUP returns original members with group_id metadata; geometric SELECT "
                    "orders members, not group extents. Requires a reviewed planning label.",
                    node=n["id"],
                )
            )
        if mode == "SUBREGION" and n["params"].get("subregion") not in {
            "LEFT_SIDE",
            "RIGHT_SIDE",
            "ABOVE",
            "BELOW",
            "INSIDE",
            "AROUND",
        }:
            findings.append(
                issue(
                    "unsupported_runtime_subregion",
                    "Production SUBREGION returns UNRESOLVED for this enum.",
                    node=n["id"],
                )
            )
    for n in nodes:
        if n["op"] != "COUNT" or "entities" not in n["inputs"]:
            continue
        ref = n["inputs"]["entities"]
        path = []
        seen = set()
        while ref in by_ref and ref not in seen:
            seen.add(ref)
            producer = by_ref[ref]
            path.append(producer)
            if producer["op"] == "SELECT":
                ref = producer["inputs"]["candidates"]
            else:
                break
        consumed = words(_values(path))
        count_attrs = words(_values(n["params"].get("target", {}).get("attributes", {})))
        if count_attrs - consumed:
            findings.append(
                issue(
                    "count_target_not_filter",
                    "COUNT over EntitySet only "
                    "takes its length; target attributes must already be applied "
                    "by LOCATE/SELECT.",
                    node=n["id"],
                    terms=sorted(count_attrs - consumed),
                )
            )
    # A static final residual is intentionally excluded: candidate disambiguation
    # must be represented before ATTRIBUTE/COUNT/RELATION, not silently delegated.
    missing = sorted((qw & CRITICAL) - evidence)
    if missing:
        findings.append(
            issue(
                "condition_not_in_graph",
                "Question modifiers/background "
                "not represented in contributing nodes; may be omitted or "
                "deferred to final VLM.",
                terms=missing,
            )
        )
    if q.lower().startswith("how many") and not any(n["op"] in COUNT_OPS for n in nodes):
        # Floors are an attribute rather than enumeration of ground objects.
        if not (
            re.search(r"\b(?:floors?|stories|storeys)\b", q, re.I)
            and any(n["op"] == "ATTRIBUTE" for n in nodes)
        ):
            findings.append(issue("count_operator_missing", "Enumeration question has no COUNT."))
    if q.lower().startswith("how many") and re.search(
        r"\b(?:blocks|groups|rows|types|kinds|parts|sections) of\b", q, re.I
    ):
        if not any(n["op"] == "GROUP" for n in nodes):
            targets = " ".join(n["params"].get("target", {}).get("category", "") for n in nodes)
            if not re.search(r"\b(?:block|group|row|type|kind|part|section)\b", targets, re.I):
                findings.append(
                    issue(
                        "count_unit_unclear",
                        "Question counts grouped units, "
                        "but label appears to count individual objects.",
                    )
                )
    if re.search(r"\b(?:marked|circled|drawn)\b.*\b(?:circle|box|rectangle)\b", q, re.I):
        if not any(n["op"] == "FIND_MARKER" for n in nodes):
            findings.append(
                issue("marker_not_resolved", "Explicit visual marker has no FIND_MARKER.")
            )
    for node in nodes:
        target = node["params"].get("target", {})
        # Class/attributes are candidate filters and cannot invent absent colors.
        if node["op"] in {"LOCATE", "COUNT"}:
            attrs = words(_values(target.get("attributes", {}))) & CRITICAL
            allowed = qw | words(" ".join(user.get("choices") or []))
            invented = sorted(attrs - allowed)
            if invented:
                findings.append(
                    issue(
                        "unlicensed_target_attribute",
                        "Candidate prefilter has attributes not stated in the question.",
                        node=node["id"],
                        terms=invented,
                    )
                )
        if node["op"] == "SELECT" and node["params"].get("mode") == "SEMANTIC":
            sw = words(node["params"]["query"])
            invented = sorted((sw & CRITICAL) - qw)
            if invented:
                findings.append(
                    issue(
                        "query_adds_condition",
                        "Semantic query introduces modifiers not stated by the question.",
                        node=node["id"],
                        terms=invented,
                    )
                )
    return findings
