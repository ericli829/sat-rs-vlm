"""Build the fixed E_COUNT_V1 evaluation tier without mutating E1/E2."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from sat_rs_vlm.data.object_adapter_v0 import count_bin, extract_answer
from sat_rs_vlm.data.task_protocol import parse_count
from sat_rs_vlm.evaluation.tiers import file_sha256
from sat_rs_vlm.utils.jsonl import read_jsonl, write_jsonl

COUNTING_FOCUSED_TIER = "E_COUNT_V1"
DEFAULT_COUNTING_FOCUSED_FILE = "data/evaluation/tiers/e_count_v1.jsonl"
DEFAULT_COUNTING_FOCUSED_MANIFEST = "data/evaluation/tiers/e_count_v1_manifest.json"


def _canonical_sample_id(row: dict[str, Any]) -> str:
    sample_id = str(row.get("id", "")).strip()
    if not sample_id:
        raise ValueError("Every E1/E2 row must have a non-empty canonical sample id in id")
    return sample_id


def _counting_bins(rows: list[dict[str, Any]]) -> dict[str, int]:
    bins = Counter({"0-2": 0, "3-5": 0, "6-10": 0, "11+": 0})
    unknown = 0
    for row in rows:
        parsed = parse_count(extract_answer(row)).value
        if parsed is None:
            unknown += 1
        else:
            bins[count_bin(parsed)] += 1
    result = {name: bins[name] for name in ("0-2", "3-5", "6-10", "11+")}
    if unknown:
        result["unknown"] = unknown
    return result


def build_counting_focused_tier(
    *,
    e1_path: str | Path,
    e2_path: str | Path,
    output_path: str | Path = DEFAULT_COUNTING_FOCUSED_FILE,
    manifest_path: str | Path = DEFAULT_COUNTING_FOCUSED_MANIFEST,
) -> dict[str, Any]:
    """Select E2 counting plus E1 non-counting rows in a deterministic order."""

    e1 = Path(e1_path)
    e2 = Path(e2_path)
    output = Path(output_path)
    manifest_file = Path(manifest_path)
    e1_rows = [dict(row) for row in read_jsonl(e1)]
    e2_rows = [dict(row) for row in read_jsonl(e2)]
    e2_counting = [row for row in e2_rows if str(row.get("task_type", "")).lower() == "counting"]
    e1_guard = [row for row in e1_rows if str(row.get("task_type", "")).lower() != "counting"]
    e1_ids = {_canonical_sample_id(row) for row in e1_rows}
    e2_ids = {_canonical_sample_id(row) for row in e2_rows}
    source_overlap_ids = sorted(e1_ids & e2_ids)

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicate_ids: list[str] = []
    for row in [*e2_counting, *e1_guard]:
        sample_id = _canonical_sample_id(row)
        if sample_id in seen:
            duplicate_ids.append(sample_id)
            continue
        seen.add(sample_id)
        selected.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, selected)
    per_task = Counter(str(row.get("task_type", "unknown")) for row in selected)
    counting_rows = [row for row in selected if str(row.get("task_type", "")).lower() == "counting"]
    manifest = {
        "schema_version": "1.0",
        "tier_name": COUNTING_FOCUSED_TIER,
        "path": output.as_posix(),
        "selection_rule": {
            "counting": "all E2 rows with task_type == 'counting'",
            "non_counting_guard": "all E1 rows with task_type != 'counting'",
            "deduplication": "canonical id field 'id'; E2 counting rows have priority",
        },
        "sources": {
            "E1": {"path": e1.as_posix(), "sha256": file_sha256(e1), "row_count": len(e1_rows)},
            "E2": {"path": e2.as_posix(), "sha256": file_sha256(e2), "row_count": len(e2_rows)},
        },
        "total_rows": len(selected),
        "per_task_counts": dict(sorted(per_task.items())),
        "counting_count_bin_distribution": _counting_bins(counting_rows),
        "duplicate_removal_count": len(duplicate_ids),
        "duplicate_ids": duplicate_ids,
        "source_overlap_count": len(source_overlap_ids),
        "source_overlap_ids": source_overlap_ids,
        "excluded_e1_counting_count": sum(
            str(row.get("task_type", "")).lower() == "counting" for row in e1_rows
        ),
        "final_tier_sha256": file_sha256(output),
    }
    manifest_file.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
