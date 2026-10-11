from __future__ import annotations

import copy
import json

import pytest
import yaml

from taskgraph_lab.synthetic_locate import (
    FAMILIES,
    ROOT,
    VOCABULARY,
    allocate,
    build_graph,
    generate_parameters,
    make_samples,
    parameter_pool,
    production_messages,
    quality_statistics,
    render_question,
    validate_sample,
)


@pytest.fixture
def config():
    return yaml.safe_load(
        (ROOT / "taskgraph_lab/configs/planner_synthetic_locate_v1.yaml").read_text(
            encoding="utf-8"
        )
    )


@pytest.fixture(scope="module")
def examples():
    config = yaml.safe_load(
        (ROOT / "taskgraph_lab/configs/planner_synthetic_locate_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    config["count"] = 200
    rows, _ = make_samples(config)
    prompt = (
        (ROOT / "taskgraph_lab/prompts/planner_student_system_prompt.txt")
        .read_text(encoding="utf-8")
        .strip()
    )
    return rows, config, prompt


def test_default_distribution_grouping_and_determinism(config):
    first, groups = generate_parameters(config)
    second, same_groups = generate_parameters(config)
    assert first == second and groups == same_groups
    assert len(first) == 2000 and len(VOCABULARY) == 40
    assert set(groups["train"]).isdisjoint(groups["validation"])
    assert allocate(2000, config["ratios"]) == dict(
        category=900, intrinsic=500, region=300, extreme=100, ordinal=100, semantic=100
    )
    assert len({render_question(p) for p in first}) == 2000
    for key in ("category", "template_family"):
        left = {p[key] for p in first if p["split"] == "train"}
        right = {p[key] for p in first if p["split"] == "validation"}
        assert left.isdisjoint(right)


@pytest.mark.parametrize("family", FAMILIES)
def test_each_family_roundtrip_and_every_constraint_prefix(examples, family):
    rows, config, prompt = examples
    row = next(r for r in rows if r["generation"]["family"] == family)
    checks = validate_sample(row, config, prompt, every_prefix=True)
    assert checks["production_input"] and not checks["visual_answer_verified"]
    assert row["dsl"].endswith("BBOX)")
    assert "VLM_REASON" not in row["dsl"]
    assert "answer_audit" not in row and "answer_audit" not in row["metadata"]


def test_production_payload_not_a_handwritten_approximation(examples):
    rows, config, prompt = examples
    row = rows[0]
    assert row["messages"][:2] == production_messages(row["question"], row["id"], config, prompt)
    payload = json.loads(row["messages"][1]["content"])
    assert set(payload) == {"question", "question_type", "choices", "inputs"}
    assert payload["choices"] is None and payload["question_type"] == "FREE_FORM"
    assert payload["question"].startswith(row["question"] + "\nResolve the referred target")
    assert payload["inputs"]["image0"]["type"] == "image"


@pytest.mark.parametrize("change", ["question", "attributes", "select_query", "dead_node", "audit"])
def test_independent_validator_rejects_corruption(examples, change):
    rows, config, prompt = examples
    row = copy.deepcopy(next(r for r in rows if r["generation"]["family"] == "semantic"))
    if change == "question":
        row["question"] = "Count all objects."
    elif change == "attributes":
        row["taskgraph"]["nodes"][0]["params"]["target"]["attributes"] = {"color": "red"}
    elif change == "select_query":
        row["taskgraph"]["nodes"][1]["params"]["query"] = "the wrong target"
    elif change == "dead_node":
        row["taskgraph"]["nodes"].append(copy.deepcopy(row["taskgraph"]["nodes"][0]))
    else:
        row["metadata"]["answer_audit"] = {"valid": True}
    with pytest.raises(ValueError):
        validate_sample(row, config, prompt)


def test_minimal_paths_and_no_hidden_background_in_target():
    for family in FAMILIES:
        params = parameter_pool(family, "train", ["building"])[0]
        graph = build_graph(params)
        locate = next(n for n in graph["nodes"] if n["op"] == "LOCATE")
        assert locate["params"]["target"]["category"] == "building"
        assert len(graph["nodes"]) == (1 if family in {"category", "intrinsic"} else 2)
        if family == "semantic":
            assert locate["params"]["target"]["attributes"] == {}
            assert graph["nodes"][-1]["params"]["query"].endswith(params["condition"])


def test_duplicate_and_split_statistics(examples):
    stats = quality_statistics(examples[0])
    assert stats["one_node_locate_fraction"] == 0.7
    assert stats["exact_text_duplicates"] == 0
    assert stats["cross_split_near_template_skeletons"] == 0
    assert stats["within_split_reused_skeleton_rows"] > 0


def test_old_questions_can_be_explicitly_excluded(config):
    params, _ = generate_parameters(config)
    excluded = {render_question(params[0]).casefold()}
    filtered, _ = generate_parameters(config, excluded)
    assert not excluded.intersection(render_question(p).casefold() for p in filtered)


def test_capacity_and_invalid_ratios_fail_instead_of_repeating(config):
    config["count"] = 100000
    with pytest.raises(ValueError, match="unique structured samples"):
        generate_parameters(config)
    with pytest.raises(ValueError):
        allocate(2000, {"category": 1})


@pytest.mark.parametrize("change", ["family", "split", "template", "seed", "condition"])
def test_validation_rejects_invalid_rule_metadata(examples, change):
    rows, config, prompt = examples
    row = copy.deepcopy(next(r for r in rows if r["generation"]["family"] == "semantic"))
    params = row["generation"]
    if change in {"family", "split"}:
        params[change] = "unknown"
    elif change == "template":
        params["template_family"] = "untracked"
    elif change == "seed":
        params["seed"] += 1
    else:
        params["condition"] = "with an uncurated background condition"
    with pytest.raises(ValueError):
        validate_sample(row, config, prompt)
