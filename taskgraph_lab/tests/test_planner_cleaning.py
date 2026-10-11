from __future__ import annotations

import json
from copy import deepcopy

import pytest

from taskgraph_lab.quality.planner_cleaning import (
    identity_key,
    repair,
    semantic_findings,
    unpack,
    validate,
)
from taskgraph_lab.taskgraph.dsl import parse_taskgraph_dsl_payload


def row(question: str, dsl: str, choices: list[str] | None = None) -> dict:
    user = {
        "question": question,
        "question_type": "MULTIPLE_CHOICE",
        "choices": choices or ["(A) 1", "(B) 2"],
        "inputs": {"image0": {"type": "image", "uri_or_key": "fixture.png"}},
    }
    return {
        "id": "fixture",
        "messages": [
            {"role": "system", "content": "old contract"},
            {"role": "user", "content": json.dumps(user)},
            {"role": "assistant", "content": dsl},
        ],
    }


def codes(user: dict, dsl: str) -> set[str]:
    return {f["code"] for f in semantic_findings(user, dsl)}


def test_restore_count_parking_condition_without_inventing_crop() -> None:
    source = row(
        "How many cars are in the parking lot at the top left of the picture?",
        'n1=REGION($image0,TOP_LEFT)\nn2=COUNT($n1,T("car"),false)\nFINAL($n2,CHOICE_SINGLE)',
    )
    before = deepcopy(source)
    cleaned, changes = repair(source)
    user, dsl = unpack(cleaned)
    graph = parse_taskgraph_dsl_payload(dsl)
    assert source == before
    assert [n["op"] for n in graph["nodes"]] == ["LOCATE", "SELECT", "COUNT"]
    select = graph["nodes"][1]
    assert select["params"]["selection_type"] == "MULTI"
    assert select["params"]["query"] == "the cars in the parking lot at the top left of the picture"
    assert graph["nodes"][0]["inputs"] == {"image": "$image0"}
    assert validate(dsl, user, "fixture")["valid"]
    assert "restore_count_conditions" in {c["code"] for c in changes}
    assert not semantic_findings(user, dsl)


def test_plain_area_count_not_needlessly_rewritten() -> None:
    source = row(
        "How many cars are there in the top left of the picture?",
        'n1=REGION($image0,TOP_LEFT)\nn2=COUNT($n1,T("car"),false)\nFINAL($n2,CHOICE_SINGLE)',
    )
    _, changes = repair(source)
    assert not changes


@pytest.mark.parametrize(
    "question",
    [
        "How many blocks of brick red buildings are there in the lower right area?",
        "How many rows of cars are there in the parking lot?",
        "How many differences are there in the number of cars in two images?",
        "How many cars that are parked on the road are there?",
    ],
)
def test_complex_count_never_automatically_replanned(question: str) -> None:
    source = row(question, 'n1=COUNT($image0,T("building"),true)\nFINAL($n1,CHOICE_SINGLE)')
    _, changes = repair(source)
    assert not changes


def test_group_count_unit_is_flagged_not_silently_changed() -> None:
    source = row(
        "How many blocks of brick red buildings are there in the lower right area?",
        'n1=COUNT($image0,T("building",color="brick red"),true)\nFINAL($n1,CHOICE_SINGLE)',
    )
    assert "count_unit_unclear" in codes(*unpack(source))


def test_entity_count_cannot_apply_target_attributes() -> None:
    source = row(
        "How many red cars are there?",
        'n1=LOCATE($image0,T("car"))\n'
        'n2=COUNT($n1,T("car",color="red"),false)\nFINAL($n2,CHOICE_SINGLE)',
    )
    assert "count_target_not_filter" in codes(*unpack(source))
    selected = row(
        "How many red cars are there?",
        'n1=LOCATE($image0,T("car"))\n'
        'n2=SELECT_SEM($n1,null,"the red cars",MULTI,$image0)\n'
        'n3=COUNT_ENTITIES($n2,T("car",color="red"),false)\nFINAL($n3,CHOICE_SINGLE)',
    )
    assert "count_target_not_filter" not in codes(*unpack(selected))


def test_missing_conditions_in_residual_are_still_risks() -> None:
    source = row(
        "What color is the vehicle driving on the road?",
        'n1=LOCATE($image0,T("vehicle"))\n'
        'FINAL_QUESTION($n1,CHOICE_SINGLE,"What color is the vehicle driving on the road?")',
    )
    assert "condition_not_in_graph" in codes(*unpack(source))


def test_ordinal_numeric_index_and_plural_attributes_do_not_false_flag() -> None:
    source = row(
        "What color is the second building with maple leaf patterns?",
        'n1=LOCATE($image0,T("building",pattern="maple leaf pattern"))\n'
        "n2=SELECT_ORD($n1,null,2,LEFT_TO_RIGHT)\n"
        'n3=ATTRIBUTE($n2,"color")\nFINAL($n3,CHOICE_SINGLE)',
    )
    assert not semantic_findings(*unpack(source))


def test_semantic_query_inventing_nearest_is_flagged() -> None:
    source = row(
        "What color is the car?",
        'n1=LOCATE($image0,T("car"))\n'
        'n2=SELECT_SEM($n1,null,"the nearest car",SINGLE,$image0)\n'
        'n3=ATTRIBUTE($n2,"color")\nFINAL($n3,CHOICE_SINGLE)',
    )
    assert "query_adds_condition" in codes(*unpack(source))


def test_bbox_source_resolution_restored_without_changing_coordinates() -> None:
    source = row(
        "Determine the color of the object. Image resolution: 100 x 80. "
        "Bounding box: [10, 20, 30, 40].",
        'n1=REGION_BBOX($image0,[10,20,30,40])\nn2=ATTRIBUTE($n1,"color")\n'
        "FINAL($n2,CHOICE_SINGLE)",
    )
    cleaned, changes = repair(source)
    graph = parse_taskgraph_dsl_payload(unpack(cleaned)[1])
    assert graph["nodes"][0]["params"] == {"bbox": [10, 20, 30, 40], "image_size": [100, 80]}
    assert changes[0]["code"] == "explicit_bbox_size"


def test_bbox_disagreement_requires_review() -> None:
    source = row(
        "Image resolution: 100 x 80. Bounding box: [10, 20, 30, 40].",
        "n1=REGION_BBOX($image0,[11,20,30,40])\nn2=CLASSIFY($n1)\nFINAL($n2,CHOICE_SINGLE)",
    )
    with pytest.raises(ValueError, match="disagrees"):
        repair(source)


def test_rank_alias_is_normalized_not_interpreted_as_height() -> None:
    source = row(
        "Find the largest building.",
        'n1=LOCATE($image0,T("building"))\n'
        'n2=SELECT_RANK($n1,null,"size",1,DESCENDING)\nn3=ATTRIBUTE($n2,"color")\n'
        "FINAL($n3,CHOICE_SINGLE)",
    )
    cleaned, changes = repair(source)
    assert '"bbox_area"' in unpack(cleaned)[1]
    assert changes[0]["code"] == "canonical_rank_alias"


def test_same_question_on_different_images_is_not_an_identity_duplicate() -> None:
    u, _ = unpack(
        row("How many cars?", 'n1=COUNT($image0,T("car"),true)\nFINAL($n1,CHOICE_SINGLE)')
    )
    other = deepcopy(u)
    other["inputs"]["image0"]["uri_or_key"] = "other.png"
    assert identity_key(u) != identity_key(other)


def test_group_ordinals_do_not_select_group_extents() -> None:
    source = row(
        "What color is the second row of cars?",
        'n1=LOCATE($image0,T("car"))\nn2=GROUP($n1,ROW)\n'
        "n3=SELECT_ORD($n2,null,2,TOP_TO_BOTTOM)\n"
        'n4=ATTRIBUTE($n3,"color")\nFINAL($n4,CHOICE_SINGLE)',
    )
    assert "group_members_not_groups" in codes(*unpack(source))


def test_schema_subregion_enum_without_runtime_support_is_quarantined() -> None:
    source = row(
        "How many cars are on both sides of the road?",
        'n1=LOCATE($image0,T("road"))\nn2=SELECT_SUBREGION($n1,null,BOTH_SIDES)\n'
        'n3=COUNT_IMAGE($n2,T("car"),false)\nFINAL($n3,CHOICE_SINGLE)',
    )
    assert "unsupported_runtime_subregion" in codes(*unpack(source))


def test_release_prevents_holdout_leakage_and_preserves_originals(tmp_path, monkeypatch) -> None:
    from taskgraph_lab.tools import clean_planner_dataset as tool

    data = tmp_path / "taskgraph_lab/data"
    source = data / "source"
    source.mkdir(parents=True)
    prompt = tmp_path / "taskgraph_lab/prompts/planner_student_system_prompt.txt"
    prompt.parent.mkdir(parents=True)
    prompt.write_text("current contract", encoding="utf8")
    same = 'n1=COUNT($image0,T("car"),true)\nFINAL($n1,CHOICE_SINGLE)'
    train = row("How many cars are there?", same)
    test = deepcopy(train)
    test["id"] = "holdout"
    paths = {"choice_train": source / "train.jsonl", "choice_test": source / "test.jsonl"}
    for path, value in zip(paths.values(), [train, test], strict=True):
        path.write_text(json.dumps(value) + "\n", encoding="utf8")
    before = {p: p.read_bytes() for p in paths.values()}
    monkeypatch.setattr(tool, "ROOT", tmp_path)
    monkeypatch.setattr(tool, "DATA", data)
    monkeypatch.setattr(tool, "SOURCES", paths)
    monkeypatch.setattr(tool, "OUTPUT", data / "clean")
    monkeypatch.setattr(tool, "REVIEWS", tmp_path / "reviews")
    tool.audit(0, 10)
    summary = tool.release()
    assert summary["combined_train_count"] == 0
    assert summary["retained_counts"]["choice_test"] == 1
    assert summary["finding_counts"]["heldout_question_overlap"] == 1
    assert {p: p.read_bytes() for p in paths.values()} == before
    assert (tool.OUTPUT / "original_test_reference.jsonl").read_bytes() == before[
        paths["choice_test"]
    ]
    with pytest.raises(FileExistsError):
        tool.release()
