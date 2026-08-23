"""Build one self-contained Unified-v2 evaluation bundle atomically.

The builder never consumes the repo's materialized tier JSONL files.  It first
constructs E1/E2/E3 from the configured population in a sibling temporary
directory, derives E_COUNT_V2 from those newly-created files, validates every
benchmark invariant, and only then swaps the completed directory into place.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sat_rs_vlm.evaluation.counting_focused_tier import (  # noqa: E402
    build_counting_focused_tier_v2,
)
from sat_rs_vlm.evaluation.tier_builder import build_unified_evaluation_tiers  # noqa: E402
from sat_rs_vlm.evaluation.tiers import canonical_jsonl_sha256, file_sha256  # noqa: E402
from sat_rs_vlm.utils.jsonl import read_jsonl  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_TIER_VERSION = "unified-v2"
EXPECTED_E1_ROWS = 593
EXPECTED_E2_ROWS = 3000
EXPECTED_ECOUNT_ROWS = 882
EXPECTED_EXACT_VALID = 327


def _ordered_ids_sha256(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        "\n".join(str(row["id"]) for row in rows).encode("utf-8")
    ).hexdigest()


def _asset_record(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "row_count": len(rows),
        "raw_sha256": file_sha256(path),
        "canonical_jsonl_sha256": canonical_jsonl_sha256(path),
        "ordered_sample_ids_sha256": _ordered_ids_sha256(rows),
        "sample_ids": [str(row["id"]) for row in rows],
        "per_task_counts": dict(
            sorted(Counter(str(row.get("task_type", "unknown")) for row in rows).items())
        ),
    }


def _source_record(path: Path) -> dict[str, str]:
    return {
        "path": path.as_posix(),
        "raw_sha256": file_sha256(path),
        "canonical_jsonl_sha256": canonical_jsonl_sha256(path),
    }


def _resolve_config_paths(
    config_path: Path, payload: dict[str, Any], project_root: Path
) -> tuple[list[Path], list[Path]]:
    data = dict(payload.get("data", {}))
    source_files = [Path(value).expanduser() for value in data.get("source_files", [])]
    train_files = [Path(value).expanduser() for value in data.get("train_files", [])]
    source_files = [path if path.is_absolute() else project_root / path for path in source_files]
    train_files = [path if path.is_absolute() else project_root / path for path in train_files]
    if not source_files or not train_files:
        raise ValueError("Unified-v2 config must declare data.source_files and data.train_files")
    return source_files, train_files


def _semantic_json(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def compare_existing_ecount(
    old_path: Path, new_path: Path, new_manifest: dict[str, Any]
) -> dict[str, Any]:
    """Return a migration report; callers must reject any semantic drift."""

    old_rows = [dict(row) for row in read_jsonl(old_path)]
    new_rows = [dict(row) for row in read_jsonl(new_path)]
    old_ids = [str(row["id"]) for row in old_rows]
    new_ids = [str(row["id"]) for row in new_rows]
    first_order_difference: dict[str, Any] | None = None
    for index, (old_id, new_id) in enumerate(zip(old_ids, new_ids, strict=False)):
        if old_id != new_id:
            first_order_difference = {"index": index, "old": old_id, "new": new_id}
            break
    if first_order_difference is None and len(old_ids) != len(new_ids):
        index = min(len(old_ids), len(new_ids))
        first_order_difference = {
            "index": index,
            "old": old_ids[index] if index < len(old_ids) else None,
            "new": new_ids[index] if index < len(new_ids) else None,
        }
    old_by_id = {str(row["id"]): row for row in old_rows}
    new_by_id = {str(row["id"]): row for row in new_rows}
    old_manifest_path = old_path.with_name("e_count_v2_manifest.json")
    old_manifest = (
        json.loads(old_manifest_path.read_text(encoding="utf-8"))
        if old_manifest_path.is_file()
        else {}
    )
    report = {
        "semantic_benchmark_unchanged": (
            old_ids == new_ids
            and all(
                _semantic_json(old_by_id[sample_id]) == _semantic_json(new_by_id[sample_id])
                for sample_id in old_ids
            )
        ),
        "old": {
            "total_rows": len(old_rows),
            "raw_sha256": file_sha256(old_path),
            "canonical_jsonl_sha256": canonical_jsonl_sha256(old_path),
            "task_distribution": dict(
                sorted(
                    Counter(str(row.get("task_type", "unknown")) for row in old_rows).items()
                )
            ),
            "counting_raw_count": sum(
                str(row.get("task_type", "")).lower() == "counting" for row in old_rows
            ),
            "exact_cardinality_valid_count": old_manifest.get("exact_cardinality_valid_count"),
        },
        "new": {
            "total_rows": len(new_rows),
            "raw_sha256": file_sha256(new_path),
            "canonical_jsonl_sha256": canonical_jsonl_sha256(new_path),
            "task_distribution": dict(
                sorted(
                    Counter(str(row.get("task_type", "unknown")) for row in new_rows).items()
                )
            ),
            "counting_raw_count": sum(
                str(row.get("task_type", "")).lower() == "counting" for row in new_rows
            ),
            "exact_cardinality_valid_count": new_manifest.get("exact_cardinality_valid_count"),
        },
        "id_set_difference": {
            "only_old": sorted(set(old_ids) - set(new_ids)),
            "only_new": sorted(set(new_ids) - set(old_ids)),
        },
        "first_order_difference": first_order_difference,
        "row_semantic_differences": [
            sample_id
            for sample_id in sorted(set(old_by_id) & set(new_by_id))
            if _semantic_json(old_by_id[sample_id]) != _semantic_json(new_by_id[sample_id])
        ][:100],
        "exact_valid_id_difference": {
            "only_old": sorted(
                set(old_manifest.get("exact_cardinality_valid_sample_ids", []))
                - set(new_manifest.get("exact_cardinality_valid_sample_ids", []))
            ),
            "only_new": sorted(
                set(new_manifest.get("exact_cardinality_valid_sample_ids", []))
                - set(old_manifest.get("exact_cardinality_valid_sample_ids", []))
            ),
        },
    }
    return report


def _atomic_replace_directory(staged: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup: Path | None = None
    try:
        if destination.exists():
            backup = destination.with_name(f".{destination.name}.previous")
            if backup.exists():
                shutil.rmtree(backup)
            os.replace(destination, backup)
        os.replace(staged, destination)
    except Exception:
        if destination.exists() and destination != staged:
            shutil.rmtree(destination)
        if backup is not None and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    else:
        if backup is not None and backup.exists():
            shutil.rmtree(backup)


def prepare_bundle(
    config_path: str | Path,
    output_root: str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
    allow_benchmark_migration: bool = False,
) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    root = Path(project_root).resolve()
    destination = Path(output_root).expanduser()
    if not destination.is_absolute():
        destination = (root / destination).resolve()
    frozen_root = (root / "data/evaluation/tiers_v2").resolve()
    try:
        destination.relative_to(frozen_root)
    except ValueError:
        pass
    else:
        raise ValueError("Bundle output must be external to tracked data/evaluation/tiers_v2")

    payload = dict(yaml.safe_load(config_file.read_text(encoding="utf-8")) or {})
    source_files, train_files = _resolve_config_paths(config_file, payload, root)
    staging_parent = destination.parent
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=staging_parent))
    migration_report_path = destination.with_name(f"{destination.name}.migration_diff.json")
    try:
        unified = build_unified_evaluation_tiers(
            config_file,
            project_root=root,
            output_dir_override=staging,
        )
        if unified.get("tier_version") != EXPECTED_TIER_VERSION:
            raise ValueError("Unified bundle tier_version must be unified-v2")
        ecount_manifest = build_counting_focused_tier_v2(
            e1_path=staging / "e1_quick.jsonl",
            e2_path=staging / "e2_standard.jsonl",
            source_manifest_path=staging / "evaluation_tiers_manifest.json",
            output_path=staging / "e_count_v2.jsonl",
            manifest_path=staging / "e_count_v2_manifest.json",
            expected_exact_cardinality_valid_count=EXPECTED_EXACT_VALID,
        )
        tiers = {
            name: [dict(row) for row in read_jsonl(staging / filename)]
            for name, filename in {
                "E1": "e1_quick.jsonl",
                "E2": "e2_standard.jsonl",
                "E3": "e3_full.jsonl",
                "E_COUNT_V2": "e_count_v2.jsonl",
            }.items()
        }
        if len(tiers["E1"]) != EXPECTED_E1_ROWS or len(tiers["E2"]) != EXPECTED_E2_ROWS:
            raise ValueError(
                f"Unified tier row invariant failed: E1={len(tiers['E1'])}, E2={len(tiers['E2'])}"
            )
        e1_ids = {str(row["id"]) for row in tiers["E1"]}
        e2_ids = {str(row["id"]) for row in tiers["E2"]}
        e3_ids = {str(row["id"]) for row in tiers["E3"]}
        if not e1_ids <= e2_ids or not e2_ids <= e3_ids:
            raise ValueError("Unified bundle subset invariant failed")
        if len(tiers["E_COUNT_V2"]) != EXPECTED_ECOUNT_ROWS:
            raise ValueError(f"E_COUNT_V2 row invariant failed: {len(tiers['E_COUNT_V2'])}")
        if int(ecount_manifest["exact_cardinality_valid_count"]) != EXPECTED_EXACT_VALID:
            raise ValueError("E_COUNT_V2 exact-cardinality invariant failed")

        tracked_ecount = root / "data/evaluation/tiers_v2/e_count_v2.jsonl"
        migration_report: dict[str, Any] | None = None
        if tracked_ecount.is_file():
            migration_report = compare_existing_ecount(
                tracked_ecount,
                staging / "e_count_v2.jsonl",
                ecount_manifest,
            )
            (staging / "e_count_v2_migration_report.json").write_text(
                json.dumps(migration_report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            if not migration_report["semantic_benchmark_unchanged"]:
                migration_report_path.write_text(
                    json.dumps(migration_report, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                if not allow_benchmark_migration:
                    raise ValueError(
                        "E_COUNT_V2 semantic benchmark migration requires manual approval; "
                        f"see {migration_report_path}"
                    )

        unified_manifest_path = staging / "evaluation_tiers_manifest.json"
        bundle_manifest = {
            "schema_version": "1.0",
            "tier_version": EXPECTED_TIER_VERSION,
            "config": {"path": config_file.as_posix(), "raw_sha256": file_sha256(config_file)},
            "evaluation_population_source": [_source_record(path) for path in source_files],
            "training_population_source": [_source_record(path) for path in train_files],
            "unified_manifest_sha256": file_sha256(unified_manifest_path),
            "tiers": {
                name: _asset_record(
                    staging / filename,
                    tiers[name],
                )
                for name, filename in {
                    "E1": "e1_quick.jsonl",
                    "E2": "e2_standard.jsonl",
                    "E3": "e3_full.jsonl",
                    "E_COUNT_V2": "e_count_v2.jsonl",
                }.items()
            },
            "E_COUNT_V2": {
                "exact_cardinality_valid_count": int(
                    ecount_manifest["exact_cardinality_valid_count"]
                ),
                "exact_cardinality_valid_sample_ids_sha256": ecount_manifest[
                    "exact_cardinality_valid_sample_ids_sha256"
                ],
            },
            "invariants": {
                "E1_rows": EXPECTED_E1_ROWS,
                "E2_rows": EXPECTED_E2_ROWS,
                "E_COUNT_V2_rows": EXPECTED_ECOUNT_ROWS,
                "E1_subset_E2": True,
                "E2_subset_E3": True,
                "train_eval_leakage": 0,
                "portable_image_paths_validated": True,
                "semantic_benchmark_unchanged": (
                    True
                    if migration_report is None
                    else bool(migration_report["semantic_benchmark_unchanged"])
                ),
                "benchmark_migration_approved": bool(
                    allow_benchmark_migration
                    and migration_report is not None
                    and not migration_report["semantic_benchmark_unchanged"]
                ),
            },
        }
        if migration_report is not None:
            bundle_manifest["benchmark_migration"] = {
                "tracked_ecount_path": tracked_ecount.as_posix(),
                "semantic_benchmark_unchanged": bool(
                    migration_report["semantic_benchmark_unchanged"]
                ),
                "approved": bool(
                    allow_benchmark_migration
                    and not migration_report["semantic_benchmark_unchanged"]
                ),
                "report_path": migration_report_path.as_posix(),
                "report": migration_report,
            }
        (staging / "evaluation_bundle_manifest.json").write_text(
            json.dumps(bundle_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _atomic_replace_directory(staging, destination)
        return bundle_manifest
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/eval/evaluation_tiers_v2.yaml")
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--allow-benchmark-migration", action="store_true")
    args = parser.parse_args()
    result = prepare_bundle(
        args.config,
        args.output_root,
        project_root=args.project_root,
        allow_benchmark_migration=args.allow_benchmark_migration,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
