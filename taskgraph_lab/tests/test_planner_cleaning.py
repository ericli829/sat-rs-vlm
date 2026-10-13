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


def test_legacy_count_repair_preserves_existing_region_for_semantic_review() -> None:
    source = row(
        "How many cars are in the parking lot at the top left of the picture?",
        'n1=REGION($image0,TOP_LEFT)\nn2=COUNT($n1,T("car"),false)\nFINAL($n2,CHOICE_SINGLE)',
    )
    before = deepcopy(source)
    cleaned, changes = repair(source)
    user, dsl = unpack(cleaned)
    graph = parse_taskgraph_dsl_payload(dsl)
    assert source == before
    assert [n["op"] for n in graph["nodes"]] == ["REGION", "COUNT"]
    assert graph["nodes"][0]["params"]["position"] == "TOP_LEFT"
    assert graph["nodes"][1]["inputs"]["image"] == "$n1"
    assert validate(dsl, user, "fixture")["valid"]
    assert not changes
    assert "condition_not_in_graph" in {f["code"] for f in semantic_findings(user, dsl)}


def test_plain_area_count_not_needlessly_rewritten() -> None:
    source = row(
        "How many cars are there in the top left of the picture?",
        'n1=REGION($image0,TOP_LEFT)\nn2=COUNT($n1,T("car"),false)\nFINAL($n2,CHOICE_SINGLE)',
    )
    _, changes = repair(source)
    assert not changes


def test_absolute_region_expressed_only_by_select_sem_is_a_review_finding() -> None:
    source = row(
        "How many ships are in the upper-right corner of the picture?",
        'n1=LOCATE($image0,T("ship"))\n'
        'n2=SELECT_SEM($n1,null,"ships in the upper-right corner",MULTI,$image0)\n'
        'n3=COUNT_ENTITIES($n2,T("ship"),false)\nFINAL($n3,CHOICE_SINGLE)',
    )
    assert "absolute_region_missing" in codes(*unpack(source))


def test_legacy_count_repair_does_not_create_whole_image_selector_for_absolute_area() -> None:
    source = row(
        "How many red cars are in the bottom-right corner of the picture?",
        'n1=COUNT($image0,T("car",color="red"),false)\nFINAL($n1,CHOICE_SINGLE)',
    )
    cleaned, changes = repair(source)
    assert unpack(cleaned)[1] == unpack(source)[1]
    assert not changes
    assert "absolute_region_missing" in codes(*unpack(cleaned))


def test_relative_left_of_reference_is_not_misclassified_as_image_region() -> None:
    source = row(
        "How many cars are on the left side of the building?",
        'n1=LOCATE($image0,T("building"))\n'
        'n2=LOCATE($image0,T("car"))\n'
        'n3=SELECT_REL($n2,$n1,LEFT_OF)\n'
        'n4=COUNT_ENTITIES($n3,T("car"),false)\nFINAL($n4,CHOICE_SINGLE)',
    )
    findings = codes(*unpack(source))
    assert "absolute_region_missing" not in findings
    assert "relative_relation_requires_operator_review" not in findings


def test_missing_relative_relation_is_flagged_without_inserting_region() -> None:
    source = row(
        "How many cars are to the left of the building?",
        'n1=COUNT($image0,T("car"),false)\nFINAL($n1,CHOICE_SINGLE)',
    )
    findings = codes(*unpack(source))
    assert "relative_relation_requires_operator_review" in findings
    assert "absolute_region_missing" not in findings


def test_legacy_count_repair_does_not_turn_relative_relation_into_select_sem() -> None:
    source = row(
        "How many cars are to the left of the building?",
        'n1=COUNT($image0,T("car"),false)\nFINAL($n1,CHOICE_SINGLE)',
    )
    cleaned, changes = repair(source)
    assert unpack(cleaned)[1] == unpack(source)[1]
    assert not changes


def test_absolute_target_region_feeds_the_target_search_branch() -> None:
    source = row(
        "How many cars are in the upper-left corner of the picture?",
        'n1=REGION($image0,TOP_LEFT)\n'
        'n2=COUNT_IMAGE($n1,T("car"),false)\nFINAL($n2,CHOICE_SINGLE)',
    )
    user, dsl = unpack(source)
    graph = parse_taskgraph_dsl_payload(dsl)
    assert [n["op"] for n in graph["nodes"]] == ["REGION", "COUNT"]
    assert graph["nodes"][1]["inputs"]["image"] == "$n1"
    assert "absolute_region_missing" not in codes(user, dsl)


@pytest.mark.parametrize(
    ("phrase", "position"),
    [
        ("the top of the image", "TOP"),
        ("the bottom of the image", "BOTTOM"),
        ("the left side of the image", "LEFT"),
        ("the right side of the image", "RIGHT"),
        ("the center of the image", "CENTER"),
        ("the top-left corner of the image", "TOP_LEFT"),
        ("the top-right corner of the image", "TOP_RIGHT"),
        ("the bottom-left corner of the image", "BOTTOM_LEFT"),
        ("the bottom-right corner of the image", "BOTTOM_RIGHT"),
    ],
)
def test_nine_common_absolute_regions_map_to_region(phrase: str, position: str) -> None:
    source = row(
        f"How many cars are in {phrase}?",
        f'n1=REGION($image0,{position})\n'
        'n2=COUNT_IMAGE($n1,T("car"),false)\nFINAL($n2,CHOICE_SINGLE)',
    )
    user, dsl = unpack(source)
    graph = parse_taskgraph_dsl_payload(dsl)
    assert graph["nodes"][0]["params"]["position"] == position
    assert "absolute_region_missing" not in codes(user, dsl)


def test_reference_object_region_does_not_crop_the_counted_target() -> None:
    source = row(
        "How many cars are to the left of the building in the bottom-right corner of the picture?",
        'n1=REGION($image0,BOTTOM_RIGHT)\n'
        'n2=LOCATE($n1,T("building"))\n'
        'n3=LOCATE($image0,T("car"))\n'
        'n4=SELECT_REL($n3,$n2,LEFT_OF)\n'
        'n5=COUNT_ENTITIES($n4,T("car"),false)\nFINAL($n5,CHOICE_SINGLE)',
    )
    user, dsl = unpack(source)
    graph = parse_taskgraph_dsl_payload(dsl)
    assert graph["nodes"][1]["inputs"]["image"] == "$n1"
    assert graph["nodes"][2]["inputs"]["image"] == "$image0"
    assert graph["nodes"][3]["inputs"] == {"candidates": "$n3", "reference": "$n2"}
    assert graph["nodes"][4]["inputs"]["entities"] == "$n4"
    assert "absolute_region_missing" not in codes(user, dsl)


def test_marker_bbox_and_count_operator_contracts() -> None:
    marker = row(
        "How many cars are inside the red circle?",
        'n1=FIND_MARKER($image0,"circle","red")\n'
        'n2=COUNT_IMAGE($n1,T("car"),false)\nFINAL($n2,CHOICE_SINGLE)',
    )
    marker_graph = parse_taskgraph_dsl_payload(unpack(marker)[1])
    assert [n["op"] for n in marker_graph["nodes"]] == ["FIND_MARKER", "COUNT"]
    assert marker_graph["nodes"][1]["inputs"]["image"] == "$n1"

    filtered = row(
        "How many red cars are there?",
        'n1=LOCATE($image0,T("car"))\n'
        'n2=SELECT_SEM($n1,null,"red cars",MULTI,$image0)\n'
        'n3=COUNT_ENTITIES($n2,T("car"),false)\nFINAL($n3,CHOICE_SINGLE)',
    )
    filtered_graph = parse_taskgraph_dsl_payload(unpack(filtered)[1])
    assert filtered_graph["nodes"][-1]["op"] == "COUNT"
    assert filtered_graph["nodes"][-1]["inputs"]["entities"] == "$n2"

    whole_image = row(
        "How many ships are in the entire image?",
        'n1=COUNT_IMAGE($image0,T("ship"),false)\nFINAL($n1,CHOICE_SINGLE)',
    )
    whole_graph = parse_taskgraph_dsl_payload(unpack(whole_image)[1])
    assert whole_graph["nodes"][0]["op"] == "COUNT"
    assert whole_graph["nodes"][0]["inputs"]["image"] == "$image0"


def test_prompt_a_remains_active_and_strict_contract_is_candidate_only() -> None:
    from pathlib import Path

    prompt_dir = Path(__file__).parents[1] / "prompts"
    active = (prompt_dir / "planner_student_system_prompt.txt").read_text(encoding="utf-8")
    rollback = (prompt_dir / "planner_student_system_prompt_a_backup.txt").read_text(encoding="utf-8")
    candidate = (prompt_dir / "planner_student_system_prompt_spatial_strict_v2_candidate.txt").read_text(encoding="utf-8")
    assert active == rollback
    assert "do not express that position only in SELECT_SEM" not in active
    assert "do not express that position only in SELECT_SEM" in candidate
    assert "If the position describes a reference object" in candidate


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
