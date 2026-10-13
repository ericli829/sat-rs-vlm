"""Back up and audit runtime route memory into a strict 2B-correct whitelist."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _image_hashes(row: dict[str, Any]) -> tuple[str, ...] | None:
    values = row.get("image_sha256s")
    if values is None:
        values = [row.get("image_sha256")]
    if not isinstance(values, list | tuple) or not values:
        return None
    if any(not isinstance(value, str) or not SHA256.fullmatch(value) for value in values):
        return None
    return tuple(values)


def _memory_key(row: dict[str, Any]) -> tuple[tuple[str, ...], str] | None:
    images = _image_hashes(row)
    question = row.get("question_sha256")
    if images is None or not isinstance(question, str) or not SHA256.fullmatch(question):
        return None
    return images, question


def _verified(row: dict[str, Any]) -> tuple[bool, str]:
    if row.get("mode") != "DIRECT_VLM":
        return False, "not_direct_vlm"
    if row.get("model_role") != "semantic_2b":
        return False, "missing_or_non_2b_model_role"
    if row.get("correctness_status") != "verified_correct":
        return False, "missing_verified_correctness"
    proof = row.get("correctness_evidence")
    if not isinstance(proof, dict):
        return False, "missing_independent_evaluation_evidence"
    if proof.get("sample_id") != row.get("sample_id"):
        return False, "evaluation_sample_identity_mismatch"
    if proof.get("correct") is not True:
        return False, "evaluation_does_not_explicitly_mark_correct"
    path_value = proof.get("evaluation_path")
    digest = proof.get("evaluation_sha256")
    if not isinstance(path_value, str) or not isinstance(digest, str) or not SHA256.fullmatch(digest):
        return False, "missing_evaluation_file_hash"
    evidence_path = Path(path_value).expanduser()
    if not evidence_path.is_file():
        return False, "evaluation_file_unavailable"
    try:
        actual = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    except OSError:
        return False, "evaluation_file_unreadable"
    if actual != digest:
        return False, "evaluation_file_hash_mismatch"
    if _memory_key(row) is None:
        return False, "invalid_image_or_question_hash"
    return True, "verified_2b_correct"


def audit_memory(source: Path, output_dir: Path, whitelist_path: Path) -> dict[str, Any]:
    source = source.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    backup = output_dir / "original_memory.jsonl"
    shutil.copy2(source, backup)
    raw_rows = [line for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [json.loads(line) for line in raw_rows]
    ids = [str(row.get("sample_id") or "").strip() for row in rows]
    id_counts = Counter(ids)
    duplicate_ids = {key for key, count in id_counts.items() if key and count > 1}
    key_counts: dict[tuple[tuple[str, ...], str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        key = _memory_key(row)
        if key is not None:
            key_counts[key].append(index)
    duplicate_keys = {key for key, indexes in key_counts.items() if len(indexes) > 1}

    retained: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    indexes: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        keep, reason = _verified(row)
        key = _memory_key(row)
        if str(row.get("sample_id") or "").strip() in duplicate_ids:
            keep, reason = False, "duplicate_sample_id"
        elif key in duplicate_keys:
            keep, reason = False, "duplicate_or_conflicting_hash_key"
        elif keep and row.get("mode") != "DIRECT_VLM":
            keep, reason = False, "verified_model_route_not_direct_vlm"
        reasons[reason] += 1
        indexes.append(
            {
                "sample_id": row.get("sample_id"),
                "mode": row.get("mode"),
                "source": row.get("source"),
                "image_sha256_count": len(_image_hashes(row) or ()),
                "question_sha256_present": bool(row.get("question_sha256")),
                "review_status": "RETAINED" if keep else "EXCLUDED",
                "reason": reason,
            }
        )
        if keep:
            retained.append(row)

    whitelist_path.parent.mkdir(parents=True, exist_ok=True)
    whitelist_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in retained),
        encoding="utf-8",
    )
    index_path = output_dir / "audit_index.jsonl"
    index_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in indexes),
        encoding="utf-8",
    )
    digest = hashlib.sha256(backup.read_bytes()).hexdigest()
    summary = {
        "source": str(source),
        "source_sha256": digest,
        "backup": str(backup),
        "backup_sha256": digest,
        "original_count": len(rows),
        "verified_2b_correct_count": len(retained),
        "final_whitelist_count": len(retained),
        "direct_vlm_count": sum(row.get("mode") == "DIRECT_VLM" for row in rows),
        "taskgraph_count": sum(row.get("mode") == "TASKGRAPH_UHR" for row in rows),
        "missing_model_role_count": sum("model_role" not in row for row in rows),
        "missing_correctness_status_count": sum("correctness_status" not in row for row in rows),
        "missing_correctness_evidence_count": sum("correctness_evidence" not in row for row in rows),
        "duplicate_sample_id_groups": len(duplicate_ids),
        "duplicate_or_conflicting_hash_key_groups": len(duplicate_keys),
        "exclusion_reasons": dict(sorted(reasons.items())),
        "audit_index": str(index_path),
        "whitelist": str(whitelist_path.resolve()),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("configs/taskgraph/taskgraph_qwen_test.jsonl"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/taskgraph/runtime_memory_audit_20261013")
    )
    parser.add_argument(
        "--whitelist", type=Path, default=Path("configs/taskgraph/taskgraph_qwen_verified.jsonl")
    )
    args = parser.parse_args()
    print(json.dumps(audit_memory(args.input, args.output_dir, args.whitelist), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
