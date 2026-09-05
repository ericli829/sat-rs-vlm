from __future__ import annotations

import json

from sat_rs_vlm.data.rs_merger_expert import build_counting_expert_data
from sat_rs_vlm.utils.jsonl import read_jsonl, write_jsonl


def _row(sample_id: str, image: str, answer: str, task: str = "counting"):
    return {
        "id": sample_id,
        "task_type": task,
        "metadata": {"dataset": "VRSBench", "split": "train"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "How many small vehicles?"},
                ],
            },
            {"role": "assistant", "content": answer},
        ],
    }


def test_counting_data_protects_images_and_keeps_unparsed_answers(tmp_path):
    source = tmp_path / "train.jsonl"
    write_jsonl(
        source,
        [
            _row("a", "vrs/a.png", "2"),
            _row("b", "vrs/protected.png", "7"),
            _row("c", "vrs/c.png", "many objects"),
            _row("d", "vrs/d.png", "4", task="detection"),
        ],
    )
    source_manifest = tmp_path / "source_manifest.json"
    source_manifest.write_text(json.dumps({"canonical": True}), encoding="utf-8")
    tiers = []
    for name in ("e1", "e2", "e3"):
        tier = tmp_path / f"{name}.jsonl"
        write_jsonl(tier, [_row(f"{name}_p", "vrs/protected.png", "7")])
        tiers.append(tier)
    result = build_counting_expert_data(
        source,
        tmp_path / "output",
        protected_tiers=tiers,
        source_manifest=source_manifest,
    )
    output_rows = list(read_jsonl(result["train"]))
    assert [row["id"] for row in output_rows] == ["a", "c"]
    audit = result["audit"]
    assert audit["source_rows"] == 4
    assert audit["counting_rows_before_image_exclusion"] == 3
    assert audit["rows_removed_by_image_overlap"] == 1
    assert audit["final_train_rows"] == 2
    assert audit["answer_parse_rate"] == 0.5
    assert audit["count_bin_population"]["0-2"] == 1


def test_duplicate_sample_ids_are_reported_not_silently_dropped(tmp_path):
    source = tmp_path / "train.jsonl"
    write_jsonl(source, [_row("same", "a.png", "1"), _row("same", "b.png", "3")])
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    tiers = []
    for name in ("e1", "e2", "e3"):
        tier = tmp_path / f"{name}.jsonl"
        write_jsonl(tier, [_row(name, f"protected/{name}.png", "0")])
        tiers.append(tier)
    result = build_counting_expert_data(
        source, tmp_path / "out", protected_tiers=tiers, source_manifest=manifest
    )
    assert result["audit"]["duplicate_sample_ids"] == ["same"]
