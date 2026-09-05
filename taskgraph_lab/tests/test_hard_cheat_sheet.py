from __future__ import annotations

import json
from pathlib import Path

import pytest

from taskgraph_lab.retrieval.hard_cheat_sheet import (
    CheatSheetRetriever,
    compose_cheat_sheet_prompt,
    route_hard_intent,
)
from taskgraph_lab.taskgraph.dsl import CanonicalDSLPrefixGrammar, parse_taskgraph_dsl
from taskgraph_lab.tools.build_planner_cheat_sheet import build_bank
from taskgraph_lab.tools.evaluate_qwen3vl_planner import _rag_messages


def _example(
    example_id: str,
    *,
    intent: str = "RELATIONAL_COUNT",
    question: str = "How many ships are near the harbor?",
    relation_depth: int = 1,
) -> dict:
    return {
        "example_id": example_id,
        "source_split": "train",
        "intent": intent,
        "question": question,
        "metadata": {
            "relations": ["NEAR"],
            "relation_depth": relation_depth,
            "node_count": 4,
            "operators": ["LOCATE", "LOCATE", "SELECT", "COUNT"],
            "ordinal_signal": False,
            "rank_signal": False,
        },
        "dsl": (
            "INTENT(RELATIONAL_COUNT)\n"
            'n1=LOCATE($image0,T("ship"))\n'
            'n2=LOCATE($image0,T("harbor"))\n'
            "n3=SELECT_REL($n1,$n2,NEAR)\n"
            'n4=COUNT_ENTITIES($n3,T("ship"),false)\n'
            "FINAL($n4,INTEGER)"
        ),
    }


def _planner_row(sample_id: str, *, bucket: str = "accepted") -> dict:
    example = _example(sample_id)
    return {
        "id": sample_id,
        "messages": [
            {"role": "system", "content": "Return DSL only."},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": example["question"],
                        "question_type": "INTEGER",
                        "choices": None,
                        "inputs": {"image0": {"type": "image", "uri_or_key": "x.png"}},
                    }
                ),
            },
            {"role": "assistant", "content": example["dsl"]},
        ],
        "metadata": {
            "intent": "RELATIONAL_COUNT",
            "dataset": "fixture",
            "source_bucket": bucket,
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_router_triggers_hard_intents_but_not_simple_questions() -> None:
    assert route_hard_intent("How many ships are inside the harbor near the road?") == (
        "RELATIONAL_COUNT"
    )
    assert route_hard_intent("Where is the ship relative to the bridge?") == "OBJECT_RELATION"
    assert route_hard_intent("Does housing have an impact on farmland?") == (
        "COMPLEX_REASONING"
    )
    assert route_hard_intent("How many cars are on the right side of the picture?") is None
    assert route_hard_intent("How many airplanes are visible?") is None
    assert route_hard_intent("What color is the roof?") is None


def test_only_train_examples_are_searchable() -> None:
    invalid = _example("heldout")
    invalid["source_split"] = "test"
    with pytest.raises(ValueError, match="only accepts train"):
        CheatSheetRetriever([invalid])


def test_bm25_and_metadata_rerank_are_deterministic() -> None:
    examples = [
        _example("rel", relation_depth=2),
        _example(
            "route",
            intent="ROUTE_PLANNING",
            question="Find the driving route from the school to the bridge",
        ),
        _example(
            "object",
            intent="OBJECT_RELATION",
            question="Where is the ship relative to the road?",
        ),
    ]
    retriever = CheatSheetRetriever(examples)
    first, _ = retriever.retrieve(
        "How many ships are near the harbor inside the road?",
        top_k=2,
        intent="RELATIONAL_COUNT",
    )
    second, _ = retriever.retrieve(
        "How many ships are near the harbor inside the road?",
        top_k=2,
        intent="RELATIONAL_COUNT",
    )
    assert [item.log_record() for item in first] == [item.log_record() for item in second]
    assert first[0].example["example_id"] == "rel"
    assert first[0].metadata_score > first[1].metadata_score


def test_top_two_prompt_format_and_parser_contract() -> None:
    retriever = CheatSheetRetriever([_example("one"), _example("two"), _example("three")])
    retrieved, _ = retriever.retrieve("How many ships are near the harbor?", top_k=2)
    prompt = compose_cheat_sheet_prompt(
        "Existing instruction.",
        rule_cards="COUNT must consume SELECT_REL.",
        retrieved=retrieved,
    )
    assert "[CHEAT SHEET RULES]" in prompt
    assert "[RETRIEVED EXAMPLE 1]" in prompt
    assert "[RETRIEVED EXAMPLE 2]" in prompt
    assert "[RETRIEVED EXAMPLE 3]" not in prompt
    assert "Do not copy object names" in prompt
    for item in retrieved:
        parse_taskgraph_dsl(item.example["dsl"])
        assert CanonicalDSLPrefixGrammar(["image0"]).accepts(item.example["dsl"])


def test_empty_bank_and_retrieval_failure_fall_back_safely() -> None:
    empty = CheatSheetRetriever([])
    assert empty.retrieve("hard question", top_k=2)[0] == []

    class BrokenRetriever:
        def retrieve(self, *_: object, **__: object) -> object:
            raise RuntimeError("index unavailable")

    messages, log = _rag_messages(
        [{"role": "system", "content": "DSL only"}, {"role": "user", "content": "q"}],
        question="How many ships are near the road?",
        routed_intent="RELATIONAL_COUNT",
        retriever=BrokenRetriever(),  # type: ignore[arg-type]
        rule_cards="",
        top_k=2,
    )
    assert messages[0]["content"] == "DSL only"
    assert log["rag_used"] is False
    assert log["retrieved_example_ids"] == []
    assert log["retrieval_error"].startswith("RuntimeError:")


def test_logging_records_retrieved_example_ids() -> None:
    retriever = CheatSheetRetriever([_example("one"), _example("two")])
    _, log = _rag_messages(
        [{"role": "system", "content": "DSL only"}, {"role": "user", "content": "q"}],
        question="How many ships are near the harbor?",
        routed_intent="RELATIONAL_COUNT",
        retriever=retriever,
        rule_cards="",
        top_k=2,
    )
    assert log["rag_used"] is True
    assert log["retrieved_example_ids"] == ["one", "two"]
    assert all("bm25_score" in item for item in log["retrieval_scores"])


def test_bank_builder_rejects_test_overlap_and_skips_nonaccepted(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    heldout = tmp_path / "test.jsonl"
    rules = tmp_path / "rules.txt"
    rules.write_text("One concise rule.", encoding="utf-8")
    _write_jsonl(
        train,
        [_planner_row("train-good"), _planner_row("not-accepted", bucket="rejected")],
    )
    _write_jsonl(heldout, [])
    manifest = build_bank(
        train_file=train,
        heldout_files=[heldout],
        output_dir=tmp_path / "bank",
        rule_cards=rules,
        quotas={"RELATIONAL_COUNT": 10},
    )
    assert manifest["selected_count"] == 1
    assert manifest["example_ids"] == ["train-good"]
    selected = CheatSheetRetriever.from_jsonl(tmp_path / "bank" / "examples.jsonl")
    assert [item["example_id"] for item in selected.examples] == ["train-good"]

    overlap_train = tmp_path / "overlap-train.jsonl"
    overlap_test = tmp_path / "overlap-test.jsonl"
    _write_jsonl(overlap_train, [_planner_row("leak")])
    _write_jsonl(overlap_test, [{"id": "leak"}])
    with pytest.raises(ValueError, match="train/heldout overlap"):
        build_bank(
            train_file=overlap_train,
            heldout_files=[overlap_test],
            output_dir=tmp_path / "overlap-bank",
            rule_cards=rules,
            quotas={"RELATIONAL_COUNT": 1},
        )
