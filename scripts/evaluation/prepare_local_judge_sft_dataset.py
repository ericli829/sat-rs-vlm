"""Create a deterministic development-only SFT dataset for the local text judge."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sat_rs_vlm.evaluation.change_judge import LOCAL_JUDGE_SYSTEM_PROMPT  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260819)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_gold(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    required = {
        "audit_id",
        "caption",
        "human_change_label",
        "human_changed_objects",
        "human_change_directions",
        "human_annotation_confidence",
        "label_source",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"gold CSV is missing required columns: {sorted(required)}")
    ids: set[str] = set()
    captions: set[str] = set()
    for row in rows:
        audit_id, caption = row["audit_id"].strip(), row["caption"].strip()
        label = row["human_change_label"].strip()
        if not audit_id or audit_id in ids or not caption or caption in captions:
            raise ValueError("gold CSV must contain unique non-empty audit IDs and captions")
        if label not in {"0", "1", "U"}:
            raise ValueError(f"invalid human_change_label for {audit_id}: {label!r}")
        ids.add(audit_id)
        captions.add(caption)
    return rows


def _quality_weight(row: dict[str, str]) -> float:
    confidence = row["human_annotation_confidence"].strip()
    source = row["label_source"].strip()
    if source == "annotator_agreement" and confidence == "high":
        return 1.0
    if confidence == "low":
        return 0.6
    return 0.8


def _sample(row: dict[str, str]) -> dict[str, Any]:
    caption = row["caption"].strip()
    return {
        "id": row["audit_id"].strip(),
        "messages": [
            {"role": "system", "content": LOCAL_JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Caption to classify:\n<caption>\n{caption}\n</caption>",
            },
            {"role": "assistant", "content": row["human_change_label"].strip()},
        ],
        "target_label": row["human_change_label"].strip(),
        "quality_weight": _quality_weight(row),
        "metadata": {
            "split_origin": "levir_fine_semantic_development_v1",
            "objects": row["human_changed_objects"].strip().split("|"),
            "directions": row["human_change_directions"].strip().split("|"),
            "confidence": row["human_annotation_confidence"].strip(),
            "label_source": row["label_source"].strip(),
        },
    }


def split_supervised_rows(
    rows: list[dict[str, str]], *, validation_ratio: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Stratify 0/1 rows; retain scarce U rows outside model-selection metrics."""

    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be in (0, 1)")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["human_change_label"].strip()].append(row)
    rng = random.Random(seed)
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    uncertain: list[dict[str, Any]] = []
    for label, group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda row: row["audit_id"])
        rng.shuffle(ordered)
        if label == "U":
            uncertain.extend(_sample(row) for row in ordered)
            continue
        num_validation = max(1, round(len(ordered) * validation_ratio))
        validation.extend(_sample(row) for row in ordered[:num_validation])
        train.extend(_sample(row) for row in ordered[num_validation:])
    train.sort(key=lambda row: str(row["id"]))
    validation.sort(key=lambda row: str(row["id"]))
    uncertain.sort(key=lambda row: str(row["id"]))
    return train, validation, uncertain


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    source, output_dir = args.gold_csv.resolve(), args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"output directory must be empty or absent: {output_dir}")
    try:
        train, validation, uncertain = split_supervised_rows(
            _read_gold(source), validation_ratio=args.validation_ratio, seed=args.seed
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "local_judge_train.jsonl", train)
    _write_jsonl(output_dir / "local_judge_validation.jsonl", validation)
    _write_jsonl(output_dir / "local_judge_uncertain_excluded.jsonl", uncertain)
    manifest = {
        "schema_version": "1.0",
        "dataset": "levir_cc_local_text_judge_sft_v1",
        "source_gold_csv": str(source),
        "source_gold_sha256": _sha256(source),
        "source_split": "development_only",
        "holdout_used": False,
        "seed": args.seed,
        "validation_ratio": args.validation_ratio,
        "counts": {
            "train": len(train),
            "validation": len(validation),
            "uncertain_excluded": len(uncertain),
            "train_labels": dict(Counter(row["target_label"] for row in train)),
            "validation_labels": dict(Counter(row["target_label"] for row in validation)),
        },
        "output_protocol": "exactly_one_of_0_1_U",
        "notes": [
            "The deployed local judge remains a text-only post-hoc classifier.",
            "Fine semantic fields are metadata for slicing and error analysis, not model output.",
            "The single U example is excluded from optimization and model selection metrics.",
        ],
    }
    (output_dir / "sft_dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved development-only local-judge SFT data to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
