import json

import pytest

from sat_rs_vlm.evaluation.counting_focused_tier import (
    build_counting_focused_tier,
    build_counting_focused_tier_v2,
)
from sat_rs_vlm.evaluation.tiers import file_sha256
from sat_rs_vlm.utils.jsonl import write_jsonl


def _row(sample_id: str, task: str, answer: str) -> dict:
    return {
        "id": sample_id,
        "task_type": task,
        "messages": [{"role": "assistant", "content": answer}],
        "metadata": {"fixture": True},
    }


def _qa_row(sample_id: str, task: str, question: str, answer: str) -> dict:
    return {
        "id": sample_id,
        "task_type": task,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": question}]},
            {"role": "assistant", "content": answer},
        ],
        "metadata": {"fixture": True},
    }


def _unified_manifest(path, e1, e2) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "2.0",
                "tier_version": "unified-v2",
                "tiers": {
                    "E1": {
                        "path": str(e1),
                        "sha256": file_sha256(e1),
                        "sample_count": sum(1 for _ in e1.open(encoding="utf-8")),
                    },
                    "E2": {
                        "path": str(e2),
                        "sha256": file_sha256(e2),
                        "sample_count": sum(1 for _ in e2.open(encoding="utf-8")),
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def test_counting_focused_selection_and_manifest(tmp_path):
    e1 = tmp_path / "e1.jsonl"
    e2 = tmp_path / "e2.jsonl"
    write_jsonl(
        e1,
        [
            _row("guard-caption", "captioning", "a field"),
            _row("duplicate", "captioning", "keep e2"),
            _row("excluded-count", "counting", "99"),
        ],
    )
    write_jsonl(
        e2,
        [
            _row("count-1", "counting", "1"),
            _row("count-8", "counting", "8"),
            _row("duplicate", "counting", "3"),
            _row("ignored-caption", "captioning", "not selected"),
        ],
    )
    output = tmp_path / "e_count_v1.jsonl"
    manifest_path = tmp_path / "e_count_v1_manifest.json"
    manifest = build_counting_focused_tier(
        e1_path=e1, e2_path=e2, output_path=output, manifest_path=manifest_path
    )
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in rows] == ["count-1", "count-8", "duplicate", "guard-caption"]
    assert all(row["id"] != "excluded-count" for row in rows)
    assert all(row["id"] != "ignored-caption" for row in rows)
    assert len({row["id"] for row in rows}) == len(rows)
    assert manifest["duplicate_removal_count"] == 1
    assert manifest["source_overlap_count"] == 1
    assert manifest["excluded_e1_counting_count"] == 1
    assert manifest["counting_count_bin_distribution"] == {
        "0-2": 1,
        "3-5": 1,
        "6-10": 1,
        "11+": 0,
    }
    assert manifest["total_rows"] == 4
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["final_tier_sha256"]


def test_counting_focused_build_is_deterministic(tmp_path):
    e1 = tmp_path / "e1.jsonl"
    e2 = tmp_path / "e2.jsonl"
    write_jsonl(e1, [_row("a", "scene", "urban")])
    write_jsonl(e2, [_row("b", "counting", "4")])
    first = build_counting_focused_tier(
        e1_path=e1,
        e2_path=e2,
        output_path=tmp_path / "one.jsonl",
        manifest_path=tmp_path / "one.json",
    )
    second = build_counting_focused_tier(
        e1_path=e1,
        e2_path=e2,
        output_path=tmp_path / "two.jsonl",
        manifest_path=tmp_path / "two.json",
    )
    assert first["final_tier_sha256"] == second["final_tier_sha256"]
    assert (tmp_path / "one.jsonl").read_bytes() == (tmp_path / "two.jsonl").read_bytes()


def test_e_count_v2_uses_all_unified_counting_and_e1_guard(tmp_path):
    e1 = tmp_path / "e1.jsonl"
    e2 = tmp_path / "e2.jsonl"
    source_manifest = tmp_path / "evaluation_tiers_manifest.json"
    write_jsonl(
        e1,
        [
            _qa_row("guard-change", "change_detection", "What changed?", "none"),
            _qa_row("guard-scene", "scene_classification", "Which scene?", "urban"),
            _qa_row("excluded-e1-count", "counting", "How many cars?", "9"),
        ],
    )
    write_jsonl(
        e2,
        [
            _qa_row("formal-a", "counting", "How many cars are visible?", "1"),
            _qa_row("formal-b", "counting", "How many ships are visible?", "4"),
            _qa_row(
                "formal-comparative", "counting", "Are there more cars than ships?", "2"
            ),
            _qa_row("formal-caption", "captioning", "Describe it", "scene"),
        ],
    )
    _unified_manifest(source_manifest, e1, e2)
    first_output = tmp_path / "e_count_v2_one.jsonl"
    first = build_counting_focused_tier_v2(
        e1_path=e1,
        e2_path=e2,
        source_manifest_path=source_manifest,
        output_path=first_output,
        manifest_path=tmp_path / "e_count_v2_one.json",
        expected_exact_cardinality_valid_count=2,
    )
    second_output = tmp_path / "e_count_v2_two.jsonl"
    second = build_counting_focused_tier_v2(
        e1_path=e1,
        e2_path=e2,
        source_manifest_path=source_manifest,
        output_path=second_output,
        manifest_path=tmp_path / "e_count_v2_two.json",
        expected_exact_cardinality_valid_count=2,
    )
    rows = [json.loads(line) for line in first_output.read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in rows] == [
        "formal-a",
        "formal-b",
        "formal-comparative",
        "guard-change",
        "guard-scene",
    ]
    assert first["tier_version"] == "unified-v2"
    assert first["raw_counting_count"] == 3
    assert first["exact_cardinality_valid_count"] == 2
    assert first["exact_cardinality_valid_sample_ids"] == ["formal-a", "formal-b"]
    assert first["exact_cardinality_valid_sample_ids_sha256"]
    assert first["per_task_counts"]["change_detection"] == 1
    assert first["sources"]["E1"]["sha256"] == file_sha256(e1)
    assert first["sources"]["E2"]["sha256"] == file_sha256(e2)
    assert first["final_tier_sha256"] == second["final_tier_sha256"]
    assert first_output.read_bytes() == second_output.read_bytes()


def test_e_count_v2_rejects_legacy_even_with_forged_version(tmp_path):
    legacy = tmp_path / "data" / "evaluation" / "tiers"
    legacy.mkdir(parents=True)
    e1 = legacy / "e1_quick.jsonl"
    e2 = legacy / "e2_standard.jsonl"
    write_jsonl(e1, [_qa_row("a", "scene_classification", "Which?", "urban")])
    write_jsonl(e2, [_qa_row("b", "counting", "How many?", "1")])
    source_manifest = tmp_path / "manifest.json"
    _unified_manifest(source_manifest, e1, e2)
    with pytest.raises(ValueError, match="refuses legacy"):
        build_counting_focused_tier_v2(
            e1_path=e1,
            e2_path=e2,
            source_manifest_path=source_manifest,
            output_path=tmp_path / "out.jsonl",
            manifest_path=tmp_path / "out.json",
            expected_exact_cardinality_valid_count=1,
        )


def test_e_count_v2_fails_when_formal_valid_population_drifts(tmp_path):
    e1 = tmp_path / "e1.jsonl"
    e2 = tmp_path / "e2.jsonl"
    source_manifest = tmp_path / "manifest.json"
    write_jsonl(e1, [_qa_row("guard", "captioning", "Describe", "scene")])
    write_jsonl(e2, [_qa_row("count", "counting", "How many cars?", "2")])
    _unified_manifest(source_manifest, e1, e2)
    with pytest.raises(ValueError, match="expected=327, actual=1"):
        build_counting_focused_tier_v2(
            e1_path=e1,
            e2_path=e2,
            source_manifest_path=source_manifest,
            output_path=tmp_path / "out.jsonl",
            manifest_path=tmp_path / "out.json",
        )
