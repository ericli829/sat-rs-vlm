from __future__ import annotations

from pathlib import Path

from taskgraph_lab.taskgraph.prompt_contract import prompt_operator_contract_issues
from taskgraph_lab.taskgraph.validator import validate_candidate

PROMPT = (Path(__file__).parents[1] / "prompts/system_prompt.txt").read_text(encoding="utf-8")


def _single_node(op: str, role: str, *, params: dict, answer_type: str) -> dict:
    return {
        "intent": "OTHER",
        "nodes": [
            {
                "id": "n1",
                "op": op,
                "inputs": {role: "$image0"},
                "params": params,
            }
        ],
        "final": {"sources": ["$n1"], "answer_type": answer_type},
    }


def test_prompt_operator_roles_and_types_match_runtime_registry() -> None:
    assert prompt_operator_contract_issues(PROMPT) == []
    assert "CLASSIFY\nInputs:\n  input: Region | Entity | ImageRef" in PROMPT
    assert "MOTION\nInputs:\n  input: Region | Entity | EntitySet" in PROMPT


def test_prompt_locks_down_benchmark_failure_modes() -> None:
    required_rules = (
        "Never answer a count with VLM_REASON",
        'Never write `{"input": ...}` for COUNT',
        "COUNT(entities=boats, target=boat, entire=false)",
        "Do not relabel a task as COMPLEX_REASONING or OTHER merely to evade",
        '"Where in the picture is A?" with quadrant choices -> OTHER',
        'If it says "nearest", include a contributing RANK',
        "Did I preserve every explicit landmark, relation, ordinal/rank, grouping",
        "question_type is coarse transport metadata",
        "COUNT(image=<selected region>, entire=false) is valid",
    )
    for rule in required_rules:
        assert rule in PROMPT


def test_classify_and_motion_contracts_initially_validate() -> None:
    classify, classify_validation = validate_candidate(
        _single_node(
            "CLASSIFY", "input", params={"label_space": ["urban", "rural"]}, answer_type="LABEL"
        ),
        inputs={"image0": {}},
    )
    motion, motion_validation = validate_candidate(
        {
            "intent": "MOTION_QUERY",
            "nodes": [
                {
                    "id": "n1",
                    "op": "REGION_FROM_BBOX",
                    "inputs": {"image": "$image0"},
                    "params": {"bbox": [0, 0, 10, 10]},
                },
                {
                    "id": "n2",
                    "op": "MOTION",
                    "inputs": {"input": "$n1"},
                    "params": {},
                },
            ],
            "final": {"sources": ["$n2"], "answer_type": "BOOLEAN"},
        },
        inputs={"image0": {}},
        question="Is the object in bounding box [0, 0, 10, 10] moving?",
        question_type="BOOLEAN",
    )
    assert classify is not None and classify_validation.valid
    assert motion is not None and motion_validation.valid


def test_explicit_category_alternative_loss_is_a_warning_not_hard_failure() -> None:
    collapsed = {
        "intent": "SIMPLE_COUNT",
        "nodes": [
            {
                "id": "n1",
                "op": "COUNT",
                "inputs": {"image": "$image0"},
                "params": {
                    "target": {"category": "ship", "attributes": {}},
                    "entire": True,
                },
            }
        ],
        "final": {"sources": ["$n1"], "answer_type": "INTEGER"},
    }
    _, validation = validate_candidate(
        collapsed,
        inputs={"image0": {}},
        question="How many ships or yachts are there?",
        question_type="INTEGER",
    )
    assert validation.valid
    assert "explicit_alternative_loss" in {warning.code for warning in validation.warnings}

    faithful = collapsed | {
        "nodes": [
            {
                **collapsed["nodes"][0],
                "params": {
                    "target": {"category": "ship or yacht", "attributes": {}},
                    "entire": True,
                },
            }
        ]
    }
    _, faithful_validation = validate_candidate(
        faithful,
        inputs={"image0": {}},
        question="How many ships or yachts are there?",
        question_type="INTEGER",
    )
    assert "explicit_alternative_loss" not in {
        warning.code for warning in faithful_validation.warnings
    }
    assert "ships or yachts" in PROMPT
