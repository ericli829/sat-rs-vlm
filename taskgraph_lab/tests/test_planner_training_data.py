from __future__ import annotations

import json
from pathlib import Path

from taskgraph_lab.training.planner_collator import PlannerTextDataCollator
from taskgraph_lab.training.planner_dataset import PlannerSFTDataset, file_sha256


def _accepted(sample_id: str, *, answer_type: str = "CHOICE_SINGLE") -> dict:
    return {
        "sample_id": sample_id,
        "bucket": "accepted",
        "sample": {
            "sample_id": sample_id,
            "question": "How many ships are visible?",
            "question_type": "MULTIPLE_CHOICE_SINGLE",
            "choices": ["(A) One", "(B) Two"],
            "inputs": {"image0": {"type": "image", "uri_or_key": "ship.png"}},
            "metadata": {"dataset": "fixture"},
        },
        "taskgraph": {
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
            "final": {"sources": ["$n1"], "answer_type": answer_type},
        },
        "runtime_validation": {"valid": True},
        "answer_audit": {"valid": True, "source_answer": "A"},
    }


def test_planner_loader_normalizes_cardinality_metadata_and_hides_answers(tmp_path: Path) -> None:
    source = tmp_path / "accepted.jsonl"
    source.write_text(
        "\n".join(json.dumps(_accepted(f"sample-{index}")) for index in range(3)) + "\n",
        encoding="utf-8",
    )
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Return only canonical Planner DSL.", encoding="utf-8")

    dataset = PlannerSFTDataset(source, system_prompt=prompt)

    assert len(dataset) == 3
    user = json.loads(dataset[0]["messages"][1]["content"])
    assert user["question_type"] == "MULTIPLE_CHOICE"
    assert "source_answer" not in json.dumps(dataset[0])
    assert dataset[0]["messages"][2]["content"].endswith("FINAL($n1,CHOICE_SINGLE)")


def test_planner_split_is_deterministic_and_manifested(tmp_path: Path) -> None:
    source = tmp_path / "accepted.jsonl"
    source.write_text(
        "\n".join(json.dumps(_accepted(f"sample-{index}")) for index in range(10)) + "\n",
        encoding="utf-8",
    )
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Return only canonical Planner DSL.", encoding="utf-8")
    dataset = PlannerSFTDataset(source, system_prompt=prompt)

    first = dataset.write_splits(tmp_path / "first", validation_fraction=0.2, seed=7)
    second = dataset.write_splits(tmp_path / "second", validation_fraction=0.2, seed=7)

    assert first["population_count"] == 10
    assert first["splits"]["train"]["count"] == 8
    assert first["splits"]["validation"]["count"] == 2
    assert (
        first["splits"]["validation"]["sample_ids"] == second["splits"]["validation"]["sample_ids"]
    )
    assert first["splits"]["train"]["sha256"] == file_sha256(Path(first["splits"]["train"]["path"]))


def test_text_collator_omits_absent_visual_inputs() -> None:
    messages = [
        [
            {"role": "system", "content": "planner"},
            {"role": "user", "content": "question"},
        ]
    ]
    assert PlannerTextDataCollator._process_vision_info(messages) == (None, None)
