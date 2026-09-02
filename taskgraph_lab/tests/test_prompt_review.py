from __future__ import annotations

import json
from pathlib import Path

from taskgraph_lab.tools.build_prompt_review import build_review, markdown


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def test_review_deduplicates_prompt_and_sample_input(tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("single prompt", encoding="utf-8")
    outputs = tmp_path / "outputs"
    sample = {
        "sample_id": "s1",
        "question": "How many ships?",
        "question_type": "INTEGER",
        "choices": None,
        "inputs": {"image0": {"type": "image", "uri_or_key": "x.png"}},
        "metadata": {"dataset": "fixture"},
    }
    raw = {
        "sample_id": "s1",
        "status": "generated",
        "sample": sample,
        "candidate_text": "{}",
        "validation": {"errors": []},
        "provider_trace": {"usage": {}, "latency_ms": 1.0},
    }
    accepted = {
        "sample_id": "s1",
        "target": {"intent": "OTHER", "nodes": [], "final": {}},
        "validation": {"errors": []},
    }
    for name in ("low", "disabled"):
        _write_jsonl(outputs / "raw" / f"{name}.jsonl", [raw])
        _write_jsonl(outputs / "valid" / f"{name}.jsonl", [accepted])
    review = build_review(
        prompt,
        [
            ("thinking_low", outputs / "raw" / "low.jsonl"),
            ("disabled", outputs / "raw" / "disabled.jsonl"),
        ],
    )
    assert review["prompt"]["text"] == "single prompt"
    assert len(review["samples"]) == 1
    assert set(review["samples"][0]["outcomes"]) == {"thinking_low", "disabled"}
    rendered = markdown(review)
    assert rendered.count("single prompt") == 1
