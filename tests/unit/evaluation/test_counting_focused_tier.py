import json

from sat_rs_vlm.evaluation.counting_focused_tier import build_counting_focused_tier
from sat_rs_vlm.utils.jsonl import write_jsonl


def _row(sample_id: str, task: str, answer: str) -> dict:
    return {
        "id": sample_id,
        "task_type": task,
        "messages": [{"role": "assistant", "content": answer}],
        "metadata": {"fixture": True},
    }


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
