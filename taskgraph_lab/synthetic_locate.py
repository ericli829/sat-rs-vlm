"""Structured, image-free planning supervision; never a visual answer audit.

The scene parameters are the source of both language and gold graphs.  Production
Adapter and Planner serialization are reused verbatim.  Existing teacher exports
and their stricter acceptance channel are intentionally not involved.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from sat_rs_vlm.competition.protocol_v2.adapter import CompetitionRequestAdapter
from sat_rs_vlm.competition.protocol_v2.image_package import ResolvedImage
from sat_rs_vlm.competition.protocol_v2.models import AnswerContract, PredictRequest
from sat_rs_vlm.taskgraph.providers import PlannerRequest, Qwen3VLPlannerProvider
from sat_rs_vlm.taskgraph.runtime_types import ImageRef
from taskgraph_lab.taskgraph.canonicalize import canonicalize_target
from taskgraph_lab.taskgraph.dsl.compiler import compile_taskgraph_to_dsl
from taskgraph_lab.taskgraph.dsl.constraint import CanonicalDSLPrefixGrammar
from taskgraph_lab.taskgraph.dsl.parser import parse_taskgraph_dsl
from taskgraph_lab.taskgraph.type_checker import check_types
from taskgraph_lab.taskgraph.validator import validate_candidate

ROOT = Path(__file__).resolve().parents[1]
VERSION = "planner-synthetic-locate-v1"
FAMILIES = ("category", "intrinsic", "region", "extreme", "ordinal", "semantic")
# Deliberately curated, rather than arbitrary category x attribute products.
# Each entry is (natural-language noun phrase, intrinsic TargetSpec attributes).
VOCABULARY = {
    "building": [
        ("white square building", {"color": "white", "shape": "square"}),
        ("large building", {"size": "large"}),
        ("building with a red roof", {"has_part": "red roof"}),
    ],
    "warehouse": [
        ("white rectangular warehouse", {"color": "white", "shape": "rectangular"}),
        ("large warehouse", {"size": "large"}),
    ],
    "airplane": [("white airplane", {"color": "white"}), ("parked airplane", {"state": "parked"})],
    "ship": [("large ship", {"size": "large"}), ("white ship", {"color": "white"})],
    "vehicle": [("red vehicle", {"color": "red"}), ("parked vehicle", {"state": "parked"})],
    "storage tank": [
        ("circular storage tank", {"shape": "circular"}),
        ("white storage tank", {"color": "white"}),
    ],
    "tennis court": [
        ("blue tennis court", {"color": "blue"}),
        ("rectangular tennis court", {"shape": "rectangular"}),
    ],
    "swimming pool": [
        ("rectangular swimming pool", {"shape": "rectangular"}),
        ("blue swimming pool", {"color": "blue"}),
    ],
    "greenhouse": [
        ("long greenhouse", {"shape": "elongated"}),
        ("white greenhouse", {"color": "white"}),
    ],
    "parking area": [
        ("large parking area", {"size": "large"}),
        ("striped parking area", {"pattern": "striped"}),
    ],
    "basketball court": [
        ("red basketball court", {"color": "red"}),
        ("rectangular basketball court", {"shape": "rectangular"}),
    ],
    "football field": [
        ("green football field", {"color": "green"}),
        ("rectangular football field", {"shape": "rectangular"}),
    ],
    "sports field": [
        ("large sports field", {"size": "large"}),
        ("green sports field", {"color": "green"}),
    ],
    "helipad": [
        ("circular helipad", {"shape": "circular"}),
        ("square helipad", {"shape": "square"}),
    ],
    "runway": [
        ("long runway", {"shape": "elongated"}),
        ("runway with white markings", {"pattern": "white markings"}),
    ],
    "bridge": [("long bridge", {"shape": "elongated"}), ("curved bridge", {"shape": "curved"})],
    "tower": [("tall tower", {"size": "tall"}), ("white tower", {"color": "white"})],
    "chimney": [("red chimney", {"color": "red"}), ("tall chimney", {"size": "tall"})],
    "wind turbine": [
        ("white wind turbine", {"color": "white"}),
        ("wind turbine with three blades", {"has_part": "three blades"}),
    ],
    "solar panel": [
        ("blue solar panel", {"color": "blue"}),
        ("rectangular solar panel", {"shape": "rectangular"}),
    ],
    "water tower": [
        ("circular water tower", {"shape": "circular"}),
        ("white water tower", {"color": "white"}),
    ],
    "bus": [("yellow bus", {"color": "yellow"}), ("parked bus", {"state": "parked"})],
    "truck": [
        ("white truck", {"color": "white"}),
        ("truck with a trailer", {"has_part": "trailer"}),
    ],
    "car": [("red car", {"color": "red"}), ("parked car", {"state": "parked"})],
    "cargo ship": [
        ("large cargo ship", {"size": "large"}),
        ("cargo ship with containers", {"has_part": "containers"}),
    ],
    "yacht": [("white yacht", {"color": "white"}), ("small yacht", {"size": "small"})],
    "fishing boat": [
        ("blue fishing boat", {"color": "blue"}),
        ("small fishing boat", {"size": "small"}),
    ],
    "oil tank": [
        ("circular oil tank", {"shape": "circular"}),
        ("white oil tank", {"color": "white"}),
    ],
    "stadium": [("oval stadium", {"shape": "oval"}), ("stadium with a roof", {"has_part": "roof"})],
    "hangar": [
        ("large hangar", {"size": "large"}),
        ("rectangular hangar", {"shape": "rectangular"}),
    ],
    "container": [
        ("red container", {"color": "red"}),
        ("rectangular container", {"shape": "rectangular"}),
    ],
    "dock": [("long dock", {"shape": "elongated"}), ("L-shaped dock", {"shape": "L-shaped"})],
    "roundabout": [
        ("circular roundabout", {"shape": "circular"}),
        ("large roundabout", {"size": "large"}),
    ],
    "train": [("long train", {"size": "long"}), ("red train", {"color": "red"})],
    "railway station": [
        ("large railway station", {"size": "large"}),
        ("railway station with a roof", {"has_part": "roof"}),
    ],
    "antenna": [("circular antenna", {"shape": "circular"}), ("white antenna", {"color": "white"})],
    "silo": [("circular silo", {"shape": "circular"}), ("white silo", {"color": "white"})],
    "industrial building": [
        ("large industrial building", {"size": "large"}),
        ("industrial building with a gray roof", {"has_part": "gray roof"}),
    ],
    "residential building": [
        ("residential building with a red roof", {"has_part": "red roof"}),
        ("white residential building", {"color": "white"}),
    ],
    "transmission tower": [
        ("tall transmission tower", {"size": "tall"}),
        ("gray transmission tower", {"color": "gray"}),
    ],
}
TRAIN_ACTIONS = (
    "Locate the {target}{context}.",
    "Find the {target}{context}.",
    "Identify the {target}{context}.",
    "Give the bounding box of the {target}{context}.",
    "Indicate the position of the {target}{context}.",
    "Determine where the {target} is{context}.",
    "Provide a bounding box for the {target}{context}.",
    "Point out the {target}{context}.",
    "Mark the location of the {target}{context}.",
    "Specify the bounds of the {target}{context}.",
)
VALIDATION_ACTIONS = (
    "Where is the {target}{context}?",
    "What are the bounds of the {target}{context}?",
    "Which part contains the {target}{context}?",
)
CONTEXTS = {
    "train": (
        " in the image",
        " in this satellite image",
        " in the overhead view",
        " in the provided scene",
        " shown here",
    ),
    "validation": (
        " in the satellite photograph",
        " in this aerial scene",
        " in the displayed view",
        " in the supplied image",
        " visible here",
    ),
}
REGIONS = {
    "TOP": "upper half",
    "BOTTOM": "lower half",
    "LEFT": "left half",
    "RIGHT": "right half",
    "CENTER": "central region",
    "TOP_LEFT": "upper-left region",
    "TOP_RIGHT": "upper-right region",
    "BOTTOM_LEFT": "lower-left region",
    "BOTTOM_RIGHT": "lower-right region",
    "TOP_CENTER": "top-center region",
    "BOTTOM_CENTER": "bottom-center region",
    "CENTER_LEFT": "center-left region",
    "CENTER_RIGHT": "center-right region",
}
ORDERS = {
    "LEFT_TO_RIGHT": "left to right",
    "RIGHT_TO_LEFT": "right to left",
    "TOP_TO_BOTTOM": "top to bottom",
    "BOTTOM_TO_TOP": "bottom to top",
}
ORDINALS = {2: "second", 3: "third", 4: "fourth"}
WATER_CATEGORIES = {"ship", "cargo ship", "yacht", "fishing boat", "dock"}


def json_text(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def production_messages(
    question: str, sample_id: str, config: dict, system_prompt: str
) -> list[dict]:
    """Build the actual production messages, with an explicitly symbolic image."""
    size = int(config["symbolic_image_size"])
    request = PredictRequest.model_validate(
        {
            "protocol_version": "2.0",
            "request_id": sample_id,
            "item_id": sample_id,
            "question": question,
            "images": [{"asset_id": "0" * 32, "sha256": "0" * 64, "mime_type": "image/png"}],
            "response_constraint": {"type": "bbox", "coordinate_format": "pixel_xyxy", "length": 4},
            "answer_type": "bbox",
            "image_width": size,
            "image_height": size,
        }
    )
    resolved = (ResolvedImage(Path(config["symbolic_image"]), size, size, request.images[0]),)
    runtime = CompetitionRequestAdapter().adapt(
        request, resolved, AnswerContract.from_request(request)
    )
    planner = PlannerRequest(
        runtime.question,
        runtime.question_type.value,
        runtime.options,
        {"$image0": ImageRef(str(resolved[0].path))},
        sample_id,
    )
    return Qwen3VLPlannerProvider._messages(planner, system_prompt)


def allocate(total: int, weights: dict[str, float]) -> dict[str, int]:
    if (
        total < 1
        or set(weights) != set(FAMILIES)
        or any(v < 0 for v in weights.values())
        or sum(weights.values()) <= 0
    ):
        raise ValueError("positive count and nonnegative weights for all six families required")
    real = {k: total * weights[k] / sum(weights.values()) for k in FAMILIES}
    result = {k: int(v) for k, v in real.items()}
    for key in sorted(FAMILIES, key=lambda k: (-(real[k] - result[k]), FAMILIES.index(k)))[
        : total - sum(result.values())
    ]:
        result[key] += 1
    return result


def render_question(params: dict) -> str:
    category, family = params["category"], params["family"]
    target = category
    if family == "intrinsic":
        target = params["target_phrase"]
    elif family == "region":
        target += " in the " + REGIONS[params["region"]]
    elif family == "extreme":
        target = params["extreme"].lower() + " " + category
    elif family == "ordinal":
        target = ORDINALS[params["ordinal"]] + " " + category + " from " + ORDERS[params["order"]]
    elif family == "semantic":
        target = category + " " + params["condition"]
    actions = TRAIN_ACTIONS if params["split"] == "train" else VALIDATION_ACTIONS
    return actions[params["action_index"]].format(
        target=target, context=CONTEXTS[params["split"]][params["context_index"]]
    )


def build_graph(params: dict) -> dict:
    """No question parsing: only explicit structured rule parameters are read."""
    nodes: list[dict] = []
    scope = "$image0"
    if params["family"] == "region":
        nodes.append(
            {
                "id": "n1",
                "op": "REGION",
                "inputs": {"image": scope},
                "params": {"position": params["region"]},
            }
        )
        scope = "$n1"
    locate_id = f"n{len(nodes) + 1}"
    nodes.append(
        {
            "id": locate_id,
            "op": "LOCATE",
            "inputs": {"image": scope},
            "params": {
                "target": {
                    "category": params["category"],
                    "attributes": params["intrinsic_attributes"],
                }
            },
        }
    )
    source = "$" + locate_id
    select: dict | None = None
    if params["family"] == "extreme":
        select = {"mode": "EXTREME", "direction": params["extreme"]}
    elif params["family"] == "ordinal":
        select = {"mode": "ORDINAL", "index": params["ordinal"], "order": params["order"]}
    elif params["family"] == "semantic":
        select = {
            "mode": "SEMANTIC",
            "query": "the " + params["category"] + " " + params["condition"],
            "selection_type": "SINGLE",
        }
    if select:
        node_id = f"n{len(nodes) + 1}"
        nodes.append(
            {"id": node_id, "op": "SELECT", "inputs": {"candidates": source}, "params": select}
        )
        source = "$" + node_id
    return {
        "intent": "OTHER",
        "nodes": nodes,
        "final": {"sources": [source], "answer_type": "BBOX"},
    }


def _conditions(category: str, split: str) -> tuple[str, ...]:
    if category in WATER_CATEGORIES:
        return (
            (
                "with a pier immediately to its right",
                "with open water below it",
                "beside the shoreline",
                "with a dock above it",
            )
            if split == "train"
            else ("with a pier immediately to its left", "with open water above it")
        )
    if category in {"airplane", "hangar", "runway", "helipad"}:
        return (
            (
                "with a taxiway immediately to its right",
                "with an apron below it",
                "beside a service road",
                "with grass above it",
            )
            if split == "train"
            else ("with a taxiway immediately to its left", "with an apron above it")
        )
    return (
        (
            "with trees immediately to its right",
            "with a road just above it",
            "beside a paved road",
            "with grass immediately below it",
        )
        if split == "train"
        else ("with a road immediately to its left", "with trees above it")
    )


def parameter_pool(family: str, split: str, categories: list[str]) -> list[dict]:
    actions = TRAIN_ACTIONS if split == "train" else VALIDATION_ACTIONS
    result = []
    for category in categories:
        variants: list[dict] = [{}]
        if family == "intrinsic":
            variants = [
                {"target_phrase": phrase, "intrinsic_attributes": attrs}
                for phrase, attrs in VOCABULARY[category]
            ]
        elif family == "region":
            keys = list(REGIONS)[:9] if split == "train" else list(REGIONS)[9:]
            variants = [{"region": key} for key in keys]
        elif family == "extreme":
            variants = [
                {"extreme": key}
                for key in (
                    ("TOPMOST", "LEFTMOST", "RIGHTMOST") if split == "train" else ("BOTTOMMOST",)
                )
            ]
        elif family == "ordinal":
            orders = list(ORDERS)[:3] if split == "train" else list(ORDERS)[3:]
            variants = [
                {"ordinal": i, "order": order} for i, order in itertools.product(ORDINALS, orders)
            ]
        elif family == "semantic":
            variants = [{"condition": condition} for condition in _conditions(category, split)]
        for variant, action, context in itertools.product(
            variants, range(len(actions)), range(len(CONTEXTS[split]))
        ):
            result.append(
                {
                    "family": family,
                    "category": category,
                    "intrinsic_attributes": {},
                    "split": split,
                    "action_index": action,
                    "context_index": context,
                    "template_id": f"{split}:{family}:a{action}:c{context}",
                    "template_family": f"{split}:action{action}",
                    "intended_output_contract": "bbox",
                    **variant,
                }
            )
    return result


def generate_parameters(
    config: dict, excluded_questions: set[str] | None = None
) -> tuple[list[dict], dict]:
    if config["generator_version"] != VERSION or not 0 < config["validation_fraction"] < 1:
        raise ValueError("unsupported generator version or split fraction")
    rng = random.Random(config["seed"])
    categories = sorted(VOCABULARY)
    rng.shuffle(categories)
    # A category never crosses splits. Template action families and spatial rule
    # variants also never cross. Graph skeletons necessarily repeat by design.
    split_categories = {"train": categories[:30], "validation": categories[30:]}
    counts = allocate(config["count"], config["ratios"])
    result = []
    excluded = excluded_questions or set()
    for family in FAMILIES:
        valid_count = round(counts[family] * config["validation_fraction"])
        for split, count in (("train", counts[family] - valid_count), ("validation", valid_count)):
            pool = parameter_pool(family, split, split_categories[split])
            pool = [p for p in pool if render_question(p).strip().casefold() not in excluded]
            rng.shuffle(pool)
            if len(pool) < count:
                raise ValueError(
                    f"{family}/{split}: requested {count}, only {len(pool)} "
                    "unique structured samples available"
                )
            for p in pool[:count]:
                p["seed"] = config["seed"]
                p["generator_version"] = VERSION
                result.append(p)
    rng.shuffle(result)
    return result, split_categories


def validate_sample(
    row: dict,
    config: dict,
    system_prompt: str,
    *,
    every_prefix: bool = False,
    grammar: CanonicalDSLPrefixGrammar | None = None,
) -> dict:
    """Independent label/input/path audit, in addition to compiler validation."""
    params, graph, dsl = row["generation"], row["taskgraph"], row["dsl"]
    if params.get("family") not in FAMILIES or params.get("split") not in CONTEXTS:
        raise ValueError("unknown family or split")
    actions = TRAIN_ACTIONS if params["split"] == "train" else VALIDATION_ACTIONS
    action, context = params["action_index"], params["context_index"]
    if (
        type(action) is not int
        or type(context) is not int
        or not 0 <= action < len(actions)
        or not 0 <= context < len(CONTEXTS[params["split"]])
    ):
        raise ValueError("invalid template indices")
    if (
        params["template_id"] != f"{params['split']}:{params['family']}:a{action}:c{context}"
        or params["template_family"] != f"{params['split']}:action{action}"
    ):
        raise ValueError("incorrect template metadata")
    if (
        params["seed"] != config["seed"]
        or params["generator_version"] != VERSION
        or params["intended_output_contract"] != "bbox"
    ):
        raise ValueError("generation contract/version mismatch")
    metadata = row["metadata"]
    if (
        metadata["generation"] != params
        or metadata["generator_config"] != config
        or metadata["family"] != params["family"]
        or metadata["category"] != params["category"]
        or metadata["template_family"] != params["template_family"]
        or metadata["source"] != "synthetic_programmatic_plan"
    ):
        raise ValueError("metadata disagrees with structured source")
    if params["family"] == "semantic" and params["condition"] not in _conditions(
        params["category"], params["split"]
    ):
        raise ValueError("uncurated semantic condition")
    if row["question"] != render_question(params):
        raise ValueError("question does not match structured language rule")
    if params["category"] not in VOCABULARY:
        raise ValueError("uncurated category")
    nodes, family = graph["nodes"], params["family"]
    expected_ops = (
        ["REGION", "LOCATE"]
        if family == "region"
        else ["LOCATE", "SELECT"]
        if family in {"extreme", "ordinal", "semantic"}
        else ["LOCATE"]
    )
    if [n["op"] for n in nodes] != expected_ops:
        raise ValueError("non-minimal or incorrect operator path")
    locate = next(n for n in nodes if n["op"] == "LOCATE")
    if locate["params"]["target"] != {
        "category": params["category"],
        "attributes": params["intrinsic_attributes"],
    }:
        raise ValueError("category/attribute supervision mismatch")
    if (
        family == "intrinsic"
        and (params["target_phrase"], params["intrinsic_attributes"])
        not in VOCABULARY[params["category"]]
    ):
        raise ValueError("uncurated intrinsic attribute combination")
    if family != "intrinsic" and params["intrinsic_attributes"]:
        raise ValueError("unexpected intrinsic attributes")
    if graph["final"] != {"answer_type": "BBOX", "sources": ["$" + nodes[-1]["id"]]}:
        raise ValueError("bbox must consume exactly its final visual node")
    # Compare each independently enumerated rule field, without deriving labels
    # from the question or trusting only graph == build_graph(params).
    if family == "region" and (
        nodes[0]["params"] != {"position": params["region"]} or locate["inputs"] != {"image": "$n1"}
    ):
        raise ValueError("region rule mismatch")
    if family in {"extreme", "ordinal", "semantic"}:
        selection = nodes[-1]
        expected = (
            {"mode": "EXTREME", "direction": params["extreme"]}
            if family == "extreme"
            else {"mode": "ORDINAL", "index": params["ordinal"], "order": params["order"]}
            if family == "ordinal"
            else {
                "mode": "SEMANTIC",
                "query": "the " + params["category"] + " " + params["condition"],
                "selection_type": "SINGLE",
            }
        )
        if selection["params"] != expected or selection["inputs"] != {"candidates": "$n1"}:
            raise ValueError("selection rule mismatch")
    messages = row["messages"]
    expected_messages = production_messages(row["question"], row["id"], config, system_prompt)
    if messages != [*expected_messages, {"role": "assistant", "content": dsl}]:
        raise ValueError("messages differ from production Planner input + canonical DSL")
    payload = json.loads(messages[1]["content"])
    if (
        metadata["original_question"] != row["question"]
        or metadata["planner_question"] != payload["question"]
    ):
        raise ValueError("original/planner question metadata mismatch")
    if payload["question_type"] != "FREE_FORM" or payload["choices"] is not None:
        raise ValueError("bbox cannot masquerade as a choice task")
    target, report = validate_candidate(
        graph,
        inputs=payload["inputs"],
        question=payload["question"],
        question_type=payload["question_type"],
    )
    if target is None or not report.valid or check_types(target, set(payload["inputs"])).errors:
        raise ValueError(f"schema/type/semantic validation failed: {report.errors}")
    if canonicalize_target(parse_taskgraph_dsl(dsl)) != canonicalize_target(graph):
        raise ValueError("DSL roundtrip changed graph")
    if compile_taskgraph_to_dsl(graph) != dsl:
        raise ValueError("noncanonical DSL")
    grammar = grammar or CanonicalDSLPrefixGrammar(["image0"])
    if not grammar.accepts(dsl):
        raise ValueError("constrained decoding rejects DSL")
    if every_prefix:
        for index in range(len(dsl) + 1):
            if not grammar.analyze(dsl[:index]).valid_prefix:
                raise ValueError(f"constrained decoding rejects prefix at {index}")
    if "answer_audit" in row or "answer_audit" in row.get("metadata", {}):
        raise ValueError("synthetic planning must not claim a visual answer audit")
    return {
        "schema": True,
        "validator": True,
        "type_checker": True,
        "canonical_roundtrip": True,
        "constraint_grammar": True,
        "production_input": True,
        "structured_semantics": True,
        "minimal_path": True,
        "bbox_static_source": True,
        "visual_answer_verified": False,
    }


def make_samples(
    config: dict, *, excluded_questions: set[str] | None = None
) -> tuple[list[dict], dict]:
    system_path = ROOT / "taskgraph_lab/prompts/planner_student_system_prompt.txt"
    system_prompt = system_path.read_text(encoding="utf-8").strip()
    version_files = (
        "taskgraph_lab/taskgraph/schema.py",
        "taskgraph_lab/taskgraph/enums.py",
        "taskgraph_lab/taskgraph/type_checker.py",
        "taskgraph_lab/taskgraph/validator.py",
        "taskgraph_lab/taskgraph/dsl/compiler.py",
        "taskgraph_lab/taskgraph/dsl/parser.py",
        "taskgraph_lab/taskgraph/dsl/constraint.py",
        "taskgraph_lab/prompts/planner_student_system_prompt.txt",
        "src/sat_rs_vlm/competition/protocol_v2/adapter.py",
        "src/sat_rs_vlm/competition/protocol_v2/answers.py",
        "src/sat_rs_vlm/taskgraph/schema.py",
        "src/sat_rs_vlm/taskgraph/runtime_types.py",
        "src/sat_rs_vlm/taskgraph/providers.py",
    )
    hashes = {p: sha256_file(ROOT / p) for p in version_files}
    params_list, groups = generate_parameters(config, excluded_questions)
    grammar = CanonicalDSLPrefixGrammar(["image0"])
    rows = []
    for index, params in enumerate(params_list):
        sample_id = f"synthetic-locate-v1-{index:05d}"
        question = render_question(params)
        graph = canonicalize_target(build_graph(params))
        dsl = compile_taskgraph_to_dsl(graph)
        messages = production_messages(question, sample_id, config, system_prompt)
        messages.append({"role": "assistant", "content": dsl})
        row = {
            "id": sample_id,
            "question": question,
            "generation": params,
            "taskgraph": graph,
            "dsl": dsl,
            "messages": messages,
            "metadata": {
                "source": "synthetic_programmatic_plan",
                "family": params["family"],
                "category": params["category"],
                "template_family": params["template_family"],
                "original_question": question,
                "planner_question": json.loads(messages[1]["content"])["question"],
                "generator_version": VERSION,
                "schema_version": "taskgraph-v1.1",
                "planner_dsl_version": "taskgraph-v1.1-dsl-v1",
                "prompt_version": "taskgraph-v1.1-strict-batch-contract-v3",
                "extension_version": "planner-bbox-final-v1",
                "contract_file_sha256": hashes,
                "generator_config": config,
                "generation": params,
                "synthetic_planning_assumption": (
                    "No image was inspected. The referent is assumed to exist and be unique "
                    "after the stated rule; execution must reject empty, ambiguous, "
                    "or unresolved evidence."
                ),
                "symbolic_image_only": True,
            },
        }
        row["metadata"]["validation"] = validate_sample(row, config, system_prompt, grammar=grammar)
        rows.append(row)
    return rows, groups


def sft_row(row: dict) -> dict:
    return {
        "id": row["id"],
        "messages": row["messages"],
        "task_type": "planner",
        "metadata": row["metadata"],
    }


def quality_statistics(rows: list[dict]) -> dict:
    split_rows = {
        split: [r for r in rows if r["generation"]["split"] == split]
        for split in ("train", "validation")
    }

    def cross(values):
        return len(set(values(split_rows["train"])) & set(values(split_rows["validation"])))

    texts = [r["question"].casefold() for r in rows]

    # Skeleton replaces ALL target/rule language, exposing noun-only near copies.
    def skeleton(r):
        p = r["generation"]
        actions = TRAIN_ACTIONS if p["split"] == "train" else VALIDATION_ACTIONS
        return actions[p["action_index"]].format(
            target="<REFERENT>", context=CONTEXTS[p["split"]][p["context_index"]]
        )

    skeletons = Counter(skeleton(r) for r in rows)
    node_counts = Counter(len(r["taskgraph"]["nodes"]) for r in rows)
    checks = Counter(k for r in rows for k, v in r["metadata"]["validation"].items() if v is True)
    dsl_lengths = [len(r["dsl"]) for r in rows]
    return {
        "total": len(rows),
        "split_counts": {k: len(v) for k, v in split_rows.items()},
        "families": dict(Counter(r["generation"]["family"] for r in rows)),
        "categories": dict(sorted(Counter(r["generation"]["category"] for r in rows).items())),
        "attributes": dict(
            Counter(k for r in rows for k in r["generation"]["intrinsic_attributes"])
        ),
        "operators": dict(Counter(n["op"] for r in rows for n in r["taskgraph"]["nodes"])),
        "select_modes": dict(
            Counter(
                n["params"]["mode"]
                for r in rows
                for n in r["taskgraph"]["nodes"]
                if n["op"] == "SELECT"
            )
        ),
        "node_counts": dict(sorted(node_counts.items())),
        "final_types": {"BBOX": len(rows)},
        "one_node_locate_count": node_counts[1],
        "one_node_locate_fraction": node_counts[1] / len(rows),
        "validation_pass_counts": dict(checks),
        "exact_text_duplicates": len(texts) - len(set(texts)),
        "cross_split_exact_text": cross(lambda rs: (r["question"].casefold() for r in rs)),
        "cross_split_categories": cross(lambda rs: (r["generation"]["category"] for r in rs)),
        "cross_split_template_families": cross(
            lambda rs: (r["generation"]["template_family"] for r in rs)
        ),
        "cross_split_near_template_skeletons": cross(lambda rs: (skeleton(r) for r in rs)),
        "within_split_reused_skeleton_rows": sum(n - 1 for n in skeletons.values()),
        "dsl_characters_mean": sum(dsl_lengths) / len(rows),
        "dsl_characters_max": max(dsl_lengths),
        "structural_split": (
            "Graph skeletons repeat intentionally; category, action-template families, "
            "and available spatial rule values are held out. "
            "One-node LOCATE has no further structural variant to hold out."
        ),
    }
