"""Build reproducible stratified MME-RealWorld TaskGraph evaluation sets."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--subtask", default="Remote Sensing")
    parser.add_argument("--archive-size", type=int, default=200)
    parser.add_argument("--eval-size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260903)
    return parser.parse_args()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"{path} must contain a JSON list of objects")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _allocate(total: int, counts: Counter[str]) -> dict[str, int]:
    if total < 0:
        raise ValueError("sample size must be non-negative")
    available = sum(counts.values())
    if total > available:
        raise ValueError(f"requested {total} samples, but only {available} are available")
    if not counts:
        if total:
            raise ValueError("cannot sample from empty strata")
        return {}
    raw = {key: total * count / available for key, count in counts.items()}
    quotas = {key: min(counts[key], int(value)) for key, value in raw.items()}
    remainder = total - sum(quotas.values())
    order = sorted(
        counts,
        key=lambda key: (raw[key] - int(raw[key]), counts[key] - quotas[key], key),
        reverse=True,
    )
    while remainder:
        progress = False
        for key in order:
            if quotas[key] < counts[key]:
                quotas[key] += 1
                remainder -= 1
                progress = True
                if not remainder:
                    break
        if not progress:
            raise ValueError("unable to allocate all requested samples")
    return quotas


def _normalize(row: dict[str, Any], role: str, seed: int) -> dict[str, Any]:
    required = ("Question_id", "Image", "Text", "Answer choices", "Ground truth", "Category")
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"row {row.get('Question_id', '<unknown>')} is missing {missing}")
    category = str(row["Category"]).strip().lower()
    question_id = str(row["Question_id"])
    image = str(row["Image"])
    return {
        "sample_id": question_id,
        "dataset": "MME_RealWorld_RS",
        "task_category": category,
        "question": str(row["Text"]),
        "image_paths": [image],
        "options": list(row["Answer choices"]),
        "question_type": "MULTIPLE_CHOICE_SINGLE",
        "metadata": {
            "source_question_id": question_id,
            "source_dataset": row.get("Dataset"),
            "source_task": row.get("Task"),
            "source_subtask": row.get("Subtask"),
            "source_category": row.get("Category"),
            "source_question_type": row.get("Question Type"),
            "ground_truth": str(row["Ground truth"]),
            "image_manifest_path": image,
            "stratification_key": category,
            "sampling_role": role,
            "sampling_seed": seed,
        },
    }


def _sample(
    rows: list[dict[str, Any]],
    size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    counts = Counter(str(row["Category"]).strip().lower() for row in rows)
    quotas = _allocate(size, counts)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for category in sorted(quotas):
        pool = [row for row in rows if str(row["Category"]).strip().lower() == category]
        rng.shuffle(pool)
        selected.extend(pool[: quotas[category]])
    rng.shuffle(selected)
    return selected, dict(sorted(quotas.items()))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = _parse_args()
    if args.archive_size < 0 or args.eval_size < 0:
        raise SystemExit("archive and evaluation sizes must be non-negative")
    if not args.input.is_file():
        raise SystemExit(f"input file does not exist: {args.input}")
    rows = _read_rows(args.input)
    filtered = [row for row in rows if row.get("Subtask") == args.subtask]
    if not filtered:
        raise SystemExit(f"no rows found for subtask {args.subtask!r}")
    ids = [str(row.get("Question_id", "")) for row in filtered]
    if len(ids) != len(set(ids)):
        raise SystemExit("filtered rows contain duplicate Question_id values")
    if args.image_root:
        missing_images = [
            str(row["Image"])
            for row in filtered
            if not (args.image_root / str(row["Image"])).is_file()
        ]
        if missing_images:
            raise SystemExit(
                f"{len(missing_images)} filtered rows have missing images; "
                f"first={missing_images[0]}"
            )

    archive_raw, archive_counts = _sample(filtered, args.archive_size, args.seed)
    archive_ids = {str(row["Question_id"]) for row in archive_raw}
    remaining = [row for row in filtered if str(row["Question_id"]) not in archive_ids]
    phase1_raw, phase1_counts = _sample(remaining, args.eval_size, args.seed + 1)
    phase1_ids = {str(row["Question_id"]) for row in phase1_raw}
    if archive_ids & phase1_ids:
        raise SystemExit("archive and phase1 samples overlap")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    subtask_slug = args.subtask.lower().replace(" ", "_")
    archive_path = output_dir / (
        f"mme_{subtask_slug}_archive_{len(archive_raw)}.jsonl"
    )
    phase1_path = output_dir / (
        f"mme_{subtask_slug}_phase1_{len(phase1_raw)}.jsonl"
    )
    manifest_path = output_dir / "mme_sampling_manifest.json"
    archive_rows = [_normalize(row, "resource_monitor_archive", args.seed) for row in archive_raw]
    phase1_rows = [_normalize(row, "formal_evaluation_phase1", args.seed + 1) for row in phase1_raw]
    _write_jsonl(archive_path, archive_rows)
    _write_jsonl(phase1_path, phase1_rows)
    manifest = {
        "manifest_version": "mme-taskgraph-stratified-sampling-v1",
        "source": {
            "path": str(args.input.resolve()),
            "sha256": _sha256(args.input),
            "total_rows": len(rows),
            "filtered_subtask": args.subtask,
            "filtered_rows": len(filtered),
            "strata_counts": dict(
                sorted(
                    Counter(
                        str(row["Category"]).strip().lower() for row in filtered
                    ).items()
                )
            ),
        },
        "sampling": {
            "method": "proportional_stratified_without_replacement",
            "stratification_key": "Category",
            "archive_size": len(archive_rows),
            "archive_seed": args.seed,
            "archive_counts": archive_counts,
            "phase1_size": len(phase1_rows),
            "phase1_seed": args.seed + 1,
            "phase1_counts": phase1_counts,
            "archive_phase1_id_overlap": 0,
            "remaining_rows_after_archive": len(remaining),
        },
        "outputs": {
            "archive_jsonl": str(archive_path.resolve()),
            "phase1_jsonl": str(phase1_path.resolve()),
            "manifest_json": str(manifest_path.resolve()),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
