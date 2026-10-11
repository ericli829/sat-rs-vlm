from __future__ import annotations

import json
from copy import deepcopy

import pytest

from taskgraph_lab.taskgraph.dsl import parse_taskgraph_dsl_payload
from taskgraph_lab.tools.choice_source_inputs import normalize_hrscd_inputs
from taskgraph_lab.tools.codex_choice_bbox import TEMPLATES
from taskgraph_lab.tools.codex_choice_bbox import compile_reviewed as bbox_plan
from taskgraph_lab.tools.codex_choice_legacy import compile_reviewed as legacy_plan


def sample(question: str, sample_id: str = "fixture") -> dict:
    return {
        "sample_id": sample_id,
        "question": question,
        "question_type": "MULTIPLE_CHOICE",
        "choices": ["(A) blue", "(B) red"],
        "inputs": {"image0": {"type": "image", "uri_or_key": "fixture.png"}},
    }


@pytest.mark.parametrize("template,definition", TEMPLATES.items())
def test_explicit_bbox_template_keeps_supplied_coordinates(template, definition) -> None:
    row = sample(template + " Image resolution: 3000 x 4000. Bounding box: [10, 20, 30, 50].")
    graph = parse_taskgraph_dsl_payload(bbox_plan(row)["dsl"])
    assert graph["nodes"][0]["params"] == {"bbox": [10, 20, 30, 50], "image_size": [3000, 4000]}
    assert graph["nodes"][1]["op"] == definition[1]
    assert graph["final"]["answer_type"] == "CHOICE_SINGLE"


@pytest.mark.parametrize("box", ["[10, 20, 10, 30]", "[10, 20, 4000, 5000]", "[30, 20, 10, 50]"])
def test_bbox_rejects_invalid_supplied_coordinates(box: str) -> None:
    row = sample(next(iter(TEMPLATES)) + f" Image resolution: 3000 x 4000. Bounding box: {box}.")
    with pytest.raises(ValueError, match="invalid supplied"):
        bbox_plan(row)


def test_bbox_fails_closed_on_unreviewed_question() -> None:
    with pytest.raises(ValueError, match="unreviewed source template"):
        bbox_plan(
            sample(
                "Guess the correct answer. Image resolution: 3000 x 4000. "
                "Bounding box: [1, 2, 3, 4]."
            )
        )


@pytest.mark.parametrize("extension", ["tif", "tiff", "png", "jpg", "jpeg"])
def test_pair_normalization_preserves_images_and_original(extension: str) -> None:
    value = {
        "image0": {
            "type": "image",
            "uri_or_key": f"HRSCD-4/first.{extension}:HRSCD-5/second.{extension}",
        }
    }
    original = deepcopy(value)
    normalized = normalize_hrscd_inputs(value)
    assert normalized == {
        "image0": {"type": "image", "uri_or_key": f"HRSCD-4/first.{extension}"},
        "image1": {"type": "image", "uri_or_key": f"HRSCD-5/second.{extension}"},
    }
    assert value == original
    assert normalize_hrscd_inputs(normalized) == normalized


@pytest.mark.parametrize(
    "uri",
    [
        "D:/images/one.tif",
        "https://host/one.tif",
        "a.tif:b.tif",
        "HRSCD-4/a.tif",
        "HRSCD-4/a.tif:other/b.tif",
    ],
)
def test_pair_normalization_does_not_guess_arbitrary_colons(uri: str) -> None:
    value = {"image0": {"type": "image", "uri_or_key": uri}}
    assert normalize_hrscd_inputs(value) == value


def color_row(plural: bool = False) -> dict:
    row = sample(
        "What color are the houses on the right side of the brown land in the upper right corner?"
        if plural
        else "What color is the car at the end of the second road on the left?",
        "mme_rs_perception_remote_sensing_color_2019"
        if plural
        else "mme_rs_perception_remote_sensing_color_2868",
    )
    row["old_dsl"] = (
        'INTENT(ATTRIBUTE_QUERY)\nn1=LOCATE($image0,T("road"))\n'
        "n2=SELECT_ORD($n1,null,2,LEFT_TO_RIGHT)\n"
        f"n3=LOCATE($image0,T({json.dumps('house' if plural else 'car')}))\n"
        'n4=SELECT_REL($n3,$n2,NEAR)\nn5=ATTRIBUTE($n4,"color")\nFINAL($n5,CHOICE_SINGLE)'
    )
    return row


def test_legacy_singular_color_keeps_ordinal_reference_and_full_query() -> None:
    row = color_row()
    original = deepcopy(row)
    graph = parse_taskgraph_dsl_payload(legacy_plan(row)["dsl"])
    selectors = [n for n in graph["nodes"] if n["op"] == "SELECT"]
    assert {n["params"]["mode"] for n in selectors} == {"ORDINAL", "SEMANTIC"}
    semantic = next(n for n in selectors if n["params"]["mode"] == "SEMANTIC")
    assert semantic["params"]["selection_type"] == "SINGLE"
    assert semantic["params"]["query"] == "the car at the end of the second road on the left"
    assert semantic["inputs"]["reference"]
    assert graph["nodes"][-1]["op"] == "ATTRIBUTE"
    assert row == original


def test_legacy_plural_color_does_not_invent_singleton_attribute() -> None:
    graph = parse_taskgraph_dsl_payload(legacy_plan(color_row(plural=True))["dsl"])
    semantic = next(n for n in graph["nodes"] if n["params"].get("mode") == "SEMANTIC")
    assert semantic["params"]["selection_type"] == "MULTI"
    assert not any(n["op"] == "ATTRIBUTE" for n in graph["nodes"])
    assert graph["final"]["question"].startswith("What color are the houses")


def test_legacy_paired_count_has_independent_inputs_and_no_fictitious_panels() -> None:
    row = sample(
        "How many differences are there in the number of farms within the areas marked "
        "by red circles in the two images?",
        "xlrs_001922",
    )
    row["inputs"]["image0"]["uri_or_key"] = "HRSCD-4/first.tif:HRSCD-4/second.tif"
    row["old_dsl"] = (
        'INTENT(SIMPLE_COUNT)\nn1=COUNT($image0,T("red circle"),true)\nFINAL($n1,CHOICE_SINGLE)'
    )
    graph = parse_taskgraph_dsl_payload(legacy_plan(row)["dsl"])
    assert sum(n["op"] == "COUNT" for n in graph["nodes"]) == 2
    assert graph["nodes"][-1]["op"] == "ABS_DIFF"
    assert {n["inputs"]["image"] for n in graph["nodes"] if n["op"] == "FIND_MARKER"} == {
        "$image0",
        "$image1",
    }
    assert all(n["op"] != "REGION" for n in graph["nodes"])
