from __future__ import annotations

import json
from pathlib import Path

import pytest

from taskgraph_lab.tools.collect_choice_expansion import collect


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _row(sample_id: str, question: str) -> dict:
    return {
        "id": sample_id,
        "task_type": "planner",
        "metadata": {"dataset": "fixture"},
        "messages": [
            {"role": "system", "content": "old prompt"},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": question,
                        "question_type": "MULTIPLE_CHOICE",
                        "choices": ["(A) 1", "(B) 2"],
                        "inputs": {"image0": {"type": "image", "uri_or_key": "fixture.png"}},
                    }
                ),
            },
            {
                "role": "assistant",
                "content": (
                    'INTENT(SIMPLE_COUNT)\nn1=COUNT($image0,T("ship"),true)\n'
                    "FINAL($n1,CHOICE_SINGLE)"
                ),
            },
        ],
    }


def _setup(tmp_path: Path) -> tuple[dict, list[dict], list[dict]]:
    base = [
        _row("base1", "How many ships are visible?"),
        _row("base2", "How many boats are visible?"),
    ]
    test = [_row("test1", "How many buses are visible?")]
    _write(tmp_path / "base.jsonl", base)
    _write(tmp_path / "test.jsonl", test)
    prepared = tmp_path / "prepared"
    seeds = [
        {
            "sample_id": "new1",
            "question": "What color is the building beside the road?",
            "question_type": "MULTIPLE_CHOICE",
            "choices": ["(A) 1", "(B) 2"],
            "inputs": {"image0": {"type": "image", "uri_or_key": "new.png"}},
            "metadata": {"dataset": "fixture"},
        },
        {
            "sample_id": "new2",
            "question": "How many cars are beside the river?",
            "question_type": "MULTIPLE_CHOICE",
            "choices": ["(A) 1", "(B) 2"],
            "inputs": {"image0": {"type": "image", "uri_or_key": "new2.png"}},
            "metadata": {"dataset": "fixture"},
        },
    ]
    _write(prepared / "seed.jsonl", seeds)
    (prepared / "manifest.json").write_text(
        json.dumps({"selected_sample_ids": ["new1", "new2"]}), encoding="utf-8"
    )
    plans = [
        {
            "sample_id": "new1",
            "rule_id": "color-sem-single",
            "dsl": (
                'INTENT(OTHER)\nn1=LOCATE($image0,T("building"))\n'
                'n2=SELECT_SEM($n1,null,"the building beside the road",SINGLE,$image0)\n'
                'FINAL_QUESTION($n2,CHOICE_SINGLE,"What color is the selected building?")'
            ),
        },
        {
            "sample_id": "new2",
            "rule_id": "count-sem-multi",
            "dsl": (
                'INTENT(RELATIONAL_COUNT)\nn1=LOCATE($image0,T("car"))\n'
                'n2=SELECT_SEM($n1,null,"cars beside the river",MULTI,$image0)\n'
                'n3=COUNT_ENTITIES($n2,T("car"),false)\nFINAL($n3,CHOICE_SINGLE)'
            ),
        },
    ]
    _write(prepared / "plans" / "fixture.jsonl", plans)
    answers = {
        sample["sample_id"]: {"answer": "A", "choices": sample["choices"], "source": "fixture"}
        for sample in seeds
    }
    answers.update(
        {
            row["id"]: {"answer": "A", "choices": ["(A) 1", "(B) 2"], "source": "fixture"}
            for row in base
        }
    )
    return (
        {
            "prepared_dir": prepared,
            "output_dir": tmp_path / "output",
            "base_train": tmp_path / "base.jsonl",
            "fixed_test": tmp_path / "test.jsonl",
            "answer_index": answers,
            "target_count": 4,
        },
        plans,
        seeds,
    )


def test_collect_preserves_base_test_and_uses_real_answer_audit(tmp_path: Path) -> None:
    kwargs, _, _ = _setup(tmp_path)
    report = collect(**kwargs)
    assert report["valid"] and report["population_count"] == 4
    assert report["semantic_select"] == {"nodes": 2, "SINGLE": 1, "explicit_scope": 2, "MULTI": 1}
    assert report["fixed_test_preserved_bytes"]
    output = kwargs["output_dir"]
    assert (output / "train.jsonl").read_bytes().startswith(kwargs["base_train"].read_bytes())
    rows = [json.loads(line) for line in (output / "new_train.jsonl").read_text().splitlines()]
    for row in rows:
        assert "source_answer" not in json.dumps(row)
        assert row["metadata"]["source"] == "codex_subagent_choice_plan"
    accepted = [
        json.loads(line) for line in (output / "accepted_plans.jsonl").read_text().splitlines()
    ]
    assert accepted[0]["answer_audit"]["source_answer"] == "A"
    assert accepted[0]["answer_audit"]["valid"] is True


@pytest.mark.parametrize(
    "case", ["missing", "duplicate", "foreign", "bad_dsl", "cardinality", "new_overlap"]
)
def test_collect_rejects_incomplete_or_invalid_plans(tmp_path: Path, case: str) -> None:
    kwargs, plans, seeds = _setup(tmp_path)
    if case == "missing":
        plans.pop()
    elif case == "duplicate":
        plans.append(plans[0])
    elif case == "foreign":
        plans.append({**plans[0], "sample_id": "unknown"})
    elif case == "bad_dsl":
        plans[0]["dsl"] = "not DSL"
    elif case == "cardinality":
        kwargs["answer_index"]["new1"]["answer"] = "AB"
    else:
        seeds[0]["question"] = "How many buses are visible?"
        _write(kwargs["prepared_dir"] / "seed.jsonl", seeds)
    _write(kwargs["prepared_dir"] / "plans" / "fixture.jsonl", plans)
    report = collect(**kwargs, validate_only=True)
    assert not report["valid"] and report["blocking_errors"]
    assert not (kwargs["output_dir"] / "train.jsonl").exists()


def test_collect_rejects_seed_ground_truth_and_manifest_mismatch(tmp_path: Path) -> None:
    kwargs, _, seeds = _setup(tmp_path)
    seeds[0]["metadata"]["answer"] = "A"
    _write(kwargs["prepared_dir"] / "seed.jsonl", seeds)
    with pytest.raises(ValueError, match="ground truth"):
        collect(**kwargs)
    seeds[0]["metadata"].pop("answer")
    _write(kwargs["prepared_dir"] / "seed.jsonl", seeds)
    (kwargs["prepared_dir"] / "manifest.json").write_text(
        '{"selected_sample_ids": []}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="manifest IDs"):
        collect(**kwargs)


def test_collect_revision_rechecks_and_preserves_old_user(tmp_path: Path) -> None:
    kwargs, plans, _ = _setup(tmp_path)
    original = json.loads(kwargs["base_train"].read_text().splitlines()[0])
    plans.append(
        {
            "sample_id": "base1",
            "rule_id": "repair-current-types",
            "dsl": (
                'INTENT(SIMPLE_COUNT)\nn1=COUNT($image0,T("ship"),true)\nFINAL($n1,CHOICE_SINGLE)'
            ),
        }
    )
    _write(kwargs["prepared_dir"] / "plans" / "fixture.jsonl", plans)
    report = collect(**kwargs)
    assert report["revision_count"] == 1
    revised = json.loads((kwargs["output_dir"] / "train.jsonl").read_text().splitlines()[0])
    assert revised["messages"][1] == original["messages"][1]
    assert "SELECT_SEM" in revised["messages"][0]["content"]
    assert report["source_counts"]["base_revised"] == 1


def test_collect_blocks_release_when_historical_current_validation_remains_invalid(
    tmp_path: Path,
) -> None:
    kwargs, _, _ = _setup(tmp_path)
    rows = [json.loads(line) for line in kwargs["base_train"].read_text().splitlines()]
    rows[0]["messages"][2]["content"] = (
        'INTENT(ATTRIBUTE_QUERY)\nn1=LOCATE($image0,T("building"))\nn2=LOCATE($image0,T("road"))\nn3=SELECT_REL($n1,$n2,NEAR)\nn4=ATTRIBUTE($n3,"color")\nFINAL($n4,CHOICE_SINGLE)'
    )
    _write(kwargs["base_train"], rows)
    report = collect(**kwargs)
    assert report["historical_current_validation"]["invalid_count"] == 1
    assert report["historical_current_validation"]["all_release_rows_current_valid"] is False
    assert not report["valid"]
    assert any(
        issue["code"] == "historical_current_validation_failed"
        for issue in report["blocking_errors"]
    )
    assert not (kwargs["output_dir"] / "train.jsonl").exists()
    assert not (kwargs["output_dir"] / "dataset_manifest.json").exists()


def test_collect_allows_release_after_actual_historical_revision(tmp_path: Path) -> None:
    kwargs, plans, _ = _setup(tmp_path)
    rows = [json.loads(line) for line in kwargs["base_train"].read_text().splitlines()]
    valid_dsl = rows[0]["messages"][2]["content"]
    rows[0]["messages"][2]["content"] = "invalid historical DSL"
    _write(kwargs["base_train"], rows)
    plans.append({"sample_id": "base1", "rule_id": "repair-old", "dsl": valid_dsl})
    _write(kwargs["prepared_dir"] / "plans" / "fixture.jsonl", plans)
    report = collect(**kwargs)
    assert report["valid"]
    assert report["historical_current_validation"]["invalid_count"] == 1
    assert report["historical_current_validation"]["remaining_invalid_count"] == 0
    assert report["historical_current_validation"]["all_release_rows_current_valid"]
    assert report["fixed_test_preserved_bytes"]


def test_collect_rejects_lab_legal_production_invalid_rank_until_semantic_revision(
    tmp_path: Path,
) -> None:
    from taskgraph_lab.taskgraph.dsl import parse_taskgraph_dsl

    kwargs, plans, _ = _setup(tmp_path)
    rows = [json.loads(line) for line in kwargs["base_train"].read_text().splitlines()]
    height_dsl = (
        'INTENT(OTHER)\nn1=LOCATE($image0,T("building"))\n'
        'n2=SELECT_RANK($n1,null,"height",1,DESCENDING)\n'
        'FINAL_QUESTION($n2,CHOICE_SINGLE,"What color is the selected building?")'
    )
    assert parse_taskgraph_dsl(height_dsl).nodes[1].params["criterion"] == "height"
    rows[0]["messages"][2]["content"] = height_dsl
    _write(kwargs["base_train"], rows)
    blocked = collect(**kwargs)
    old = blocked["historical_current_validation"]["invalid_records"]["base1"]
    assert "production_graph_rejected" in old["errors"]
    assert old["production_validation"]["valid"] is False
    assert "height" in old["production_validation"]["detail"]
    assert not blocked["valid"]
    assert not (kwargs["output_dir"] / "train.jsonl").exists()
    plans.append(
        {
            "sample_id": "base1",
            "rule_id": "production-safe-height-semantic",
            "dsl": (
                'INTENT(OTHER)\nn1=LOCATE($image0,T("building"))\n'
                'n2=SELECT_SEM($n1,null,"the tallest building",SINGLE,$image0)\n'
                'FINAL_QUESTION($n2,CHOICE_SINGLE,"What color is the selected building?")'
            ),
        }
    )
    _write(kwargs["prepared_dir"] / "plans" / "fixture.jsonl", plans)
    released = collect(**kwargs)
    assert released["valid"]
    assert released["historical_current_validation"]["remaining_invalid_count"] == 0
    accepted = [
        json.loads(line)
        for line in (kwargs["output_dir"] / "accepted_plans.jsonl").read_text().splitlines()
    ]
    assert all(row["production_validation"]["valid"] for row in accepted)


def test_collect_new_plan_production_gate_is_not_bypassed_by_lab_validity(tmp_path: Path) -> None:
    kwargs, plans, _ = _setup(tmp_path)
    plans[0]["dsl"] = (
        'INTENT(OTHER)\nn1=LOCATE($image0,T("building"))\n'
        'n2=SELECT_RANK($n1,null,"height",1,DESCENDING)\n'
        'FINAL_QUESTION($n2,CHOICE_SINGLE,"What color is the selected building?")'
    )
    _write(kwargs["prepared_dir"] / "plans" / "fixture.jsonl", plans)
    blocked = collect(**kwargs)
    audits = [
        json.loads(line)
        for line in (kwargs["output_dir"] / "plan_audit.jsonl").read_text().splitlines()
    ]
    first = next(row for row in audits if row["sample_id"] == "new1")
    assert not first["valid"] and first["errors"] == ["production_graph_rejected"]
    assert first["production_validation"]["valid"] is False
    assert not blocked["valid"]
    assert not (kwargs["output_dir"] / "train.jsonl").exists()


@pytest.mark.parametrize(
    "uri",
    [
        "C:/images/one.tif",
        "https://example.org/a.tif:b.tif",
        "other/a.tif:other/b.tif",
        "HRSCD-1/a.txt:HRSCD-2/b.txt",
    ],
)
def test_input_normalization_preserves_non_hrscd_inputs(uri: str) -> None:
    from taskgraph_lab.tools.choice_source_inputs import normalize_hrscd_inputs

    inputs = {"image0": {"type": "image", "uri_or_key": uri}}
    assert normalize_hrscd_inputs(inputs) == inputs


@pytest.mark.parametrize("suffix", ["tif", "png"])
def test_input_normalization_splits_known_hrscd_image_pairs(suffix: str) -> None:
    from taskgraph_lab.tools.choice_source_inputs import normalize_hrscd_inputs

    first, second = f"HRSCD-1/a.{suffix}", f"HRSCD-2/b.{suffix}"
    inputs = {"image0": {"type": "image", "uri_or_key": first + ":" + second}}
    assert normalize_hrscd_inputs(inputs) == {
        "image0": {"type": "image", "uri_or_key": first},
        "image1": {"type": "image", "uri_or_key": second},
    }


def test_hrscd_input_change_requires_audited_revision_and_preserves_source(tmp_path: Path) -> None:
    kwargs, plans, _ = _setup(tmp_path)
    rows = [json.loads(line) for line in kwargs["base_train"].read_text().splitlines()]
    original = json.loads(rows[0]["messages"][1]["content"])
    original["question"] = "How many ships differ between the two times?"
    original["inputs"]["image0"]["uri_or_key"] = "HRSCD-2012/a.tif:HRSCD-2015/b.tif"
    rows[0]["messages"][1]["content"] = json.dumps(original)
    _write(kwargs["base_train"], rows)
    original_bytes = kwargs["base_train"].read_bytes()
    blocked = collect(**kwargs)
    assert not blocked["valid"]
    assert any(
        issue["code"] == "normalized_input_requires_revision"
        for issue in blocked["blocking_errors"]
    )
    assert not (kwargs["output_dir"] / "train.jsonl").exists()
    plans.append(
        {
            "sample_id": "base1",
            "rule_id": "hrscd-pair-revision",
            "dsl": (
                'INTENT(OTHER)\nn1=COUNT($image0,T("ship"),true)\n'
                'n2=COUNT($image1,T("ship"),true)\nn3=ABS_DIFF($n1,$n2)\n'
                "FINAL($n3,CHOICE_SINGLE)"
            ),
        }
    )
    _write(kwargs["prepared_dir"] / "plans" / "fixture.jsonl", plans)
    report = collect(**kwargs)
    assert report["valid"] and report["revision_count"] == 1
    revised = json.loads((kwargs["output_dir"] / "train.jsonl").read_text().splitlines()[0])
    user = json.loads(revised["messages"][1]["content"])
    assert user["question"] == original["question"] and user["choices"] == original["choices"]
    assert user["inputs"] == {
        "image0": {"type": "image", "uri_or_key": "HRSCD-2012/a.tif"},
        "image1": {"type": "image", "uri_or_key": "HRSCD-2015/b.tif"},
    }
    assert revised["metadata"]["original_inputs"] == original["inputs"]
    assert kwargs["base_train"].read_bytes() == original_bytes
    assert report["fixed_test_preserved_bytes"]


@pytest.mark.parametrize(
    ("question", "kind", "index", "category", "selection"),
    [
        (
            "Where is the square building with a green top in the picture?",
            "position",
            193,
            "building",
            "SINGLE",
        ),
        (
            "Where is the narrow river crossing bridge in the picture?",
            "position",
            273,
            "bridge",
            "SINGLE",
        ),
        ("Where are the white and pink roofs in the picture?", "position", 390, "roof", "MULTI"),
        (
            "What color are the walls of the small square house next to the train track?",
            "color",
            32,
            "house",
            "SINGLE",
        ),
        (
            "How many white-roofed buildings are there on the top right side of the picture?",
            "count",
            268,
            "building",
            "MULTI",
        ),
    ],
)
def test_reviewed_mme_converter_preserves_target_roles(
    question: str,
    kind: str,
    index: int,
    category: str,
    selection: str,
) -> None:
    from taskgraph_lab.taskgraph.dsl import parse_taskgraph_dsl_payload
    from taskgraph_lab.tools.codex_choice_mme_b import build_plan

    row = {
        "sample_id": f"fixture-{index}",
        "question": question,
        "question_type": "MULTIPLE_CHOICE",
        "choices": ["(A) one", "(B) two"],
        "inputs": {"image0": {"type": "image", "uri_or_key": "mock.png"}},
        "metadata": {"source_category": kind},
    }
    graph = parse_taskgraph_dsl_payload(build_plan(row, index)["dsl"])
    locates = [node for node in graph["nodes"] if node["op"] == "LOCATE"]
    assert locates[0]["params"]["target"]["category"] == category
    selects = [node for node in graph["nodes"] if node["op"] == "SELECT"]
    assert selects[0]["params"]["selection_type"] == selection
    if index == 268:
        assert locates[0]["params"]["target"]["attributes"] == {"has_part": "white roof"}
    if index == 390:
        assert "color" not in locates[0]["params"]["target"]["attributes"]


def test_reviewed_mme_converter_groups_configuration_and_preserves_extreme_candidate_set() -> None:
    from taskgraph_lab.taskgraph.dsl import parse_taskgraph_dsl_payload
    from taskgraph_lab.tools.codex_choice_mme_b import build_plan

    base = {
        "sample_id": "fixture-group",
        "question_type": "MULTIPLE_CHOICE",
        "choices": ["(A) top", "(B) bottom"],
        "inputs": {"image0": {"type": "image", "uri_or_key": "mock.png"}},
    }
    group = {
        **base,
        "question": "Where is the U-shape formed by connected red circular structures?",
        "metadata": {"source_category": "position"},
    }
    graph = parse_taskgraph_dsl_payload(build_plan(group, 40)["dsl"])
    assert [node["op"] for node in graph["nodes"]] == ["LOCATE", "GROUP", "SELECT"]
    assert graph["nodes"][2]["inputs"]["candidates"] == "$n2"
    extreme = {
        **base,
        "question": "What color are the wings of the rightmost one of the three "
        "adjacent planes in the upper left corner of the picture?",
        "metadata": {"source_category": "color"},
    }
    graph = parse_taskgraph_dsl_payload(build_plan(extreme, 98)["dsl"])
    semantic = next(
        node
        for node in graph["nodes"]
        if node["op"] == "SELECT" and node["params"]["mode"] == "SEMANTIC"
    )
    assert semantic["params"]["selection_type"] == "MULTI"
    assert semantic["params"]["query"].startswith("the three adjacent planes")
    assert graph["nodes"][-1]["params"] == {"mode": "EXTREME", "direction": "RIGHTMOST"}
