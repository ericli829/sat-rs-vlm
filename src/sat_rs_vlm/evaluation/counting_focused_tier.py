"""Build immutable counting-focused tiers without mutating their source tiers."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sat_rs_vlm.data.object_adapter_v0 import count_bin, extract_answer, extract_prompt
from sat_rs_vlm.data.task_protocol import parse_count
from sat_rs_vlm.evaluation.counting_protocol import classify_counting_predictions
from sat_rs_vlm.evaluation.tiers import file_sha256
from sat_rs_vlm.utils.jsonl import read_jsonl, write_jsonl

COUNTING_FOCUSED_TIER = "E_COUNT_V1"
DEFAULT_COUNTING_FOCUSED_FILE = "data/evaluation/tiers/e_count_v1.jsonl"
DEFAULT_COUNTING_FOCUSED_MANIFEST = "data/evaluation/tiers/e_count_v1_manifest.json"
COUNTING_FOCUSED_TIER_V2 = "E_COUNT_V2"
UNIFIED_TIER_VERSION = "unified-v2"
DEFAULT_UNIFIED_TIER_ROOT = "data/evaluation/tiers_v2"
DEFAULT_UNIFIED_E1_FILE = f"{DEFAULT_UNIFIED_TIER_ROOT}/e1_quick.jsonl"
DEFAULT_UNIFIED_E2_FILE = f"{DEFAULT_UNIFIED_TIER_ROOT}/e2_standard.jsonl"
DEFAULT_UNIFIED_MANIFEST = f"{DEFAULT_UNIFIED_TIER_ROOT}/evaluation_tiers_manifest.json"
DEFAULT_COUNTING_FOCUSED_V2_FILE = f"{DEFAULT_UNIFIED_TIER_ROOT}/e_count_v2.jsonl"
DEFAULT_COUNTING_FOCUSED_V2_MANIFEST = (
    f"{DEFAULT_UNIFIED_TIER_ROOT}/e_count_v2_manifest.json"
)
FORMAL_R1_EXACT_CARDINALITY_COUNT = 327


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


def _ordered_ids_sha256(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _is_legacy_tier_path(path: Path) -> bool:
    normalized = path.as_posix().lower().rstrip("/")
    return "/data/evaluation/tiers/" in f"/{normalized}/"


def _validate_unified_source(
    *, e1: Path, e2: Path, source_manifest: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if _is_legacy_tier_path(e1) or _is_legacy_tier_path(e2):
        raise ValueError(
            "E_COUNT_V2 refuses legacy data/evaluation/tiers; use unified-v2 tiers_v2"
        )
    payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    if payload.get("tier_version") != UNIFIED_TIER_VERSION:
        raise ValueError(
            "E_COUNT_V2 requires tier_version=unified-v2; "
            f"got {payload.get('tier_version')!r}"
        )
    tiers = payload.get("tiers")
    if not isinstance(tiers, dict):
        raise ValueError("Unified tier manifest is missing its tiers mapping")
    records: dict[str, dict[str, Any]] = {}
    for name, path in (("E1", e1), ("E2", e2)):
        record = tiers.get(name)
        if not isinstance(record, dict):
            raise ValueError(f"Unified tier manifest is missing {name}")
        recorded_path = Path(str(record.get("path", "")))
        if recorded_path.resolve() != path.resolve():
            raise ValueError(
                f"{name} path does not match unified manifest: "
                f"manifest={recorded_path}, requested={path}"
            )
        actual_sha = file_sha256(path)
        if record.get("sha256") != actual_sha:
            raise ValueError(
                f"{name} SHA256 mismatch: manifest={record.get('sha256')}, actual={actual_sha}"
            )
        records[name] = record
    return payload, records["E1"], records["E2"]


def _eligibility_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": _canonical_sample_id(row),
            "task_type": "counting",
            "question": extract_prompt(row),
            "reference": extract_answer(row),
            # Reference-as-prediction isolates eligibility from model behavior.
            "prediction": extract_answer(row),
        }
        for row in rows
    ]


def build_counting_focused_tier_v2(
    *,
    e1_path: str | Path = DEFAULT_UNIFIED_E1_FILE,
    e2_path: str | Path = DEFAULT_UNIFIED_E2_FILE,
    source_manifest_path: str | Path = DEFAULT_UNIFIED_MANIFEST,
    output_path: str | Path = DEFAULT_COUNTING_FOCUSED_V2_FILE,
    manifest_path: str | Path = DEFAULT_COUNTING_FOCUSED_V2_MANIFEST,
    expected_exact_cardinality_valid_count: int = FORMAL_R1_EXACT_CARDINALITY_COUNT,
) -> dict[str, Any]:
    """Build E_COUNT_V2 from all unified-v2 E2 counting and E1 guard rows."""

    e1 = Path(e1_path)
    e2 = Path(e2_path)
    source_manifest = Path(source_manifest_path)
    output = Path(output_path)
    manifest_file = Path(manifest_path)
    source_payload, e1_record, e2_record = _validate_unified_source(
        e1=e1, e2=e2, source_manifest=source_manifest
    )
    e1_rows = [dict(row) for row in read_jsonl(e1)]
    e2_rows = [dict(row) for row in read_jsonl(e2)]
    if int(e1_record.get("sample_count", -1)) != len(e1_rows):
        raise ValueError("E1 row count does not match unified manifest")
    if int(e2_record.get("sample_count", -1)) != len(e2_rows):
        raise ValueError("E2 row count does not match unified manifest")
    e2_counting = [
        row for row in e2_rows if str(row.get("task_type", "")).strip().lower() == "counting"
    ]
    e1_guard = [
        row
        for row in e1_rows
        if str(row.get("task_type", "")).strip().lower() != "counting"
    ]
    classified = classify_counting_predictions(_eligibility_rows(e2_counting))
    valid_count = int(classified["diagnostics"]["valid_cardinality_rows"])
    exact_cardinality_ids = [
        _canonical_sample_id(row) for row in classified["valid_rows"]
    ]
    if valid_count != int(expected_exact_cardinality_valid_count):
        raise ValueError(
            "Formal R1 exact-cardinality population mismatch: "
            f"expected={expected_exact_cardinality_valid_count}, actual={valid_count}"
        )

    selected: list[dict[str, Any]] = []
    duplicate_ids: list[str] = []
    seen: set[str] = set()
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
    e1_per_task = Counter(str(row.get("task_type", "unknown")) for row in e1_rows)
    e2_per_task = Counter(str(row.get("task_type", "unknown")) for row in e2_rows)
    e1_source_ids = [_canonical_sample_id(row) for row in e1_rows]
    e2_source_ids = [_canonical_sample_id(row) for row in e2_rows]
    counting_ids = [_canonical_sample_id(row) for row in e2_counting]
    guard_ids = [_canonical_sample_id(row) for row in e1_guard]
    sample_ids = [_canonical_sample_id(row) for row in selected]
    manifest = {
        "schema_version": "2.0",
        "tier_name": COUNTING_FOCUSED_TIER_V2,
        "tier_version": UNIFIED_TIER_VERSION,
        "path": output.as_posix(),
        "selection_rule": {
            "counting": "all unified-v2 E2 rows with task_type == 'counting'",
            "non_counting_guard": "all unified-v2 E1 rows with task_type != 'counting'",
            "metric_eligibility": (
                "exact-cardinality eligibility is applied by the formal evaluator; "
                "raw counting rows are retained"
            ),
            "deduplication": "canonical id field 'id'; E2 counting rows have priority",
        },
        "sources": {
            "tier_manifest": {
                "path": source_manifest.as_posix(),
                "sha256": file_sha256(source_manifest),
                "tier_version": source_payload["tier_version"],
            },
            "E1": {
                "path": e1.as_posix(),
                "sha256": file_sha256(e1),
                "formal_r1_sha256": e1_record.get("formal_r1_sha256"),
                "tier_version": source_payload["tier_version"],
                "row_count": len(e1_rows),
                "sample_ids_sha256": _ordered_ids_sha256(e1_source_ids),
                "sample_ids": e1_source_ids,
                "per_task_counts": dict(sorted(e1_per_task.items())),
            },
            "E2": {
                "path": e2.as_posix(),
                "sha256": file_sha256(e2),
                "formal_r1_sha256": e2_record.get("formal_r1_sha256"),
                "tier_version": source_payload["tier_version"],
                "row_count": len(e2_rows),
                "sample_ids_sha256": _ordered_ids_sha256(e2_source_ids),
                "sample_ids": e2_source_ids,
                "per_task_counts": dict(sorted(e2_per_task.items())),
            },
        },
        "raw_counting_count": len(e2_counting),
        "exact_cardinality_valid_count": valid_count,
        "exact_cardinality_excluded_count": len(e2_counting) - valid_count,
        "exact_cardinality_diagnostics": classified["diagnostics"],
        "exact_cardinality_valid_sample_ids_sha256": _ordered_ids_sha256(
            exact_cardinality_ids
        ),
        "exact_cardinality_valid_sample_ids": exact_cardinality_ids,
        "counting_sample_ids_sha256": _ordered_ids_sha256(counting_ids),
        "counting_sample_ids": counting_ids,
        "guard_sample_ids_sha256": _ordered_ids_sha256(guard_ids),
        "guard_sample_ids": guard_ids,
        "sample_ids_sha256": _ordered_ids_sha256(sample_ids),
        "sample_ids": sample_ids,
        "total_rows": len(selected),
        "per_task_counts": dict(sorted(per_task.items())),
        "counting_count_bin_distribution": _counting_bins(e2_counting),
        "duplicate_removal_count": len(duplicate_ids),
        "duplicate_ids": duplicate_ids,
        "excluded_e1_counting_count": len(e1_rows) - len(e1_guard),
        "final_tier_sha256": file_sha256(output),
    }
    manifest_file.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


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
