from pathlib import Path

import pytest

from sat_rs_vlm.evaluation.semantic.extractors import (
    extract_semantic_facts as evaluation_extract,
)
from sat_rs_vlm.semantics import RuleBasedQueryParser, TaskSpec, extract_semantic_facts
from sat_rs_vlm.semantics.ontology import load_ontology

ONTOLOGY = Path("configs/eval/semantic/remote_sensing_ontology.json")


@pytest.fixture(scope="module")
def parser() -> RuleBasedQueryParser:
    return RuleBasedQueryParser(load_ontology(ONTOLOGY))


@pytest.mark.parametrize(
    ("question", "operation", "attribute"),
    [
        ("How many airplanes are visible?", "count", None),
        ("Are there ships in the image?", "existence", None),
        ("What color is the aircraft?", "attribute", "color"),
        ("Where is the airport?", "position", None),
        ("Locate the bridge.", "grounding", None),
        ("What category is the vehicle?", "category", None),
    ],
)
def test_rules_first_operation_parsing(
    parser: RuleBasedQueryParser,
    question: str,
    operation: str,
    attribute: str | None,
) -> None:
    task = parser.parse(question)
    assert task.operation == operation
    if attribute is not None:
        assert attribute in task.attributes


def test_parser_reuses_ontology_objects_relations_and_spatial_scope(
    parser: RuleBasedQueryParser,
) -> None:
    task = parser.parse("How many planes are visible in the upper-left part?")
    assert task.targets == ("aircraft",)
    assert task.spatial_scope == "upper_left"
    assert task.multi_instance is True

    relation = parser.parse("Is the ship north of the harbor?")
    assert relation.operation == "relation"
    assert relation.relations[0].to_dict() == {
        "subject": "ship",
        "predicate": "north_of",
        "object": "harbor",
    }


def test_parser_distinguishes_center_right_from_center(
    parser: RuleBasedQueryParser,
) -> None:
    task = parser.parse("ships in the center-right area")
    assert task.targets == ("ship",)
    assert task.spatial_scope == "center_right"

    green_land = parser.parse("green land at the bottom of the image")
    assert green_land.targets == ("green_land",)
    assert green_land.spatial_scope == "lower"

    chinese = parser.parse("中间左侧的船只")
    assert chinese.targets == ("ship",)
    assert chinese.spatial_scope == "center_left"

    trees = parser.parse("trees in the lower-left corner")
    assert trees.targets == ("tree",)
    assert trees.spatial_scope == "lower_left"


def test_unresolved_question_is_explicit(parser: RuleBasedQueryParser) -> None:
    task = parser.parse("Could this be important?")
    assert task.operation == "unknown"
    assert "operation_unresolved" in task.warnings


def test_given_bbox_is_absolute_xyxy_and_invalid_bbox_is_not_guessed(
    parser: RuleBasedQueryParser,
) -> None:
    task = parser.parse("What color is the vehicle in bbox [10, 20, 110, 220]?")
    assert task.given_bbox == (10.0, 20.0, 110.0, 220.0)
    assert task.scope == "given_bbox"

    unresolved = parser.parse("Locate the bridge in bbox [10, 20, 5, 6].")
    assert unresolved.given_bbox is None
    assert "given_bbox_unresolved" in unresolved.warnings


def test_task_spec_validation_rejects_unknown_schema_values() -> None:
    with pytest.raises(ValueError, match="unsupported TaskSpec operation"):
        TaskSpec(raw_question="question", operation="invented")
    with pytest.raises(ValueError, match="non-degenerate"):
        TaskSpec(
            raw_question="question",
            operation="grounding",
            given_bbox=(1.0, 1.0, 1.0, 2.0),
        )


def test_evaluation_wrapper_preserves_common_semantic_behavior() -> None:
    ontology = load_ontology(ONTOLOGY)
    text = "Two aircraft are north of the harbor and one ship appeared."
    assert evaluation_extract(text, ontology) == extract_semantic_facts(text, ontology)
