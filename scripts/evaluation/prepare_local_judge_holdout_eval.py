"""Convert a locked human Caption audit into local-judge evaluation JSONL."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
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
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def convert_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Create evaluation records, preserving U as an unscored audit label."""

    required = {"audit_id", "caption", "human_gold_label", "label_source"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"holdout CSV is missing columns: {sorted(required)}")
    converted: list[dict[str, Any]] = []
    ids: set[str] = set()
    for index, row in enumerate(rows, start=2):
        sample_id = row["audit_id"].strip()
        caption = row["caption"].strip()
        label = row["human_gold_label"].strip()
        if not sample_id or sample_id in ids or not caption or label not in {"0", "1", "U"}:
            raise ValueError(f"invalid row at input line {index}")
        converted.append(
            {
                "id": sample_id,
                "messages": [
                    {"role": "system", "content": LOCAL_JUDGE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Caption to classify:\n<caption>\n{caption}\n</caption>",
                    },
                    {"role": "assistant", "content": label},
                ],
                "target_label": label,
                "quality_weight": 1.0,
                "metadata": {
                    "split_origin": "levir_caption_semantic_holdout_v1.7",
                    "label_source": row["label_source"].strip(),
                    "scoring_status": "unscored_uncertain_reference" if label == "U" else "scored",
                },
            }
        )
        ids.add(sample_id)
    return converted


def main() -> int:
    args = parse_args()
    source, output_dir = args.gold_csv.resolve(), args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"output directory must be empty or absent: {output_dir}")
    with source.open(encoding="utf-8-sig", newline="") as file:
        source_rows = list(csv.DictReader(file))
    try:
        converted = convert_rows(source_rows)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "local_judge_holdout_eval.jsonl"
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in converted), encoding="utf-8"
    )
    manifest = {
        "schema_version": "1.0",
        "purpose": "locked_local_judge_holdout_evaluation",
        "source_gold_csv": str(source),
        "source_sha256": _sha256(source),
        "num_samples": len(converted),
        "label_distribution": dict(Counter(row["target_label"] for row in converted)),
        "scored_labels": ["0", "1"],
        "unscored_reference_label": "U",
        "prompt_source": "LOCAL_JUDGE_SYSTEM_PROMPT",
        "output_file": output_path.name,
        "training_eligible": False,
        "notes": [
            "This dataset is locked for final local-judge validation and must not be used "
            "for tuning.",
            "U references are retained for audit but excluded from binary accuracy denominators.",
        ],
    }
    (output_dir / "holdout_eval_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved locked holdout evaluation data to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
