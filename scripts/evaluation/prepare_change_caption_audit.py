"""Prepare a stratified, blind, two-annotator LEVIR caption audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sat_rs_vlm.evaluation.parsers import parse_change_prediction  # noqa: E402

_CONTRAST = re.compile(r"\b(?:but|however|although|though|while|except|yet)\b", re.I)
_NEGATION = re.compile(r"\b(?:no|not|nothing|unchanged|same|similar|identical|without)\b", re.I)
_CAPTURE_ARTIFACT = re.compile(
    r"\b(?:light(?:ing)?|brightness|shadow|resolution|blur|color|colour|angle|"
    r"viewpoint|zoom|crop|season|weather|exposure|contrast|camera)\b",
    re.I,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        "--judged-predictions",
        dest="predictions",
        type=Path,
        required=True,
        help="Full free-caption prediction JSONL used as the sampling population.",
    )
    parser.add_argument(
        "--comparison-judged",
        type=Path,
        help="Optional judged subset; every old/new disagreement is forced into the audit.",
    )
    parser.add_argument(
        "--local-judge-results",
        type=Path,
        help="Optional judged JSONL for the selected population, stored only in the answer key.",
    )
    parser.add_argument(
        "--exclude-captions",
        type=Path,
        help="CSV/JSON/JSONL audit whose captions must be excluded from sampling.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260813)
    return parser.parse_args()


def caption_key(caption: str) -> str:
    return " ".join(caption.lower().split())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _read_caption_keys(path: Path) -> set[str]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
    elif path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        rows = payload.get("rows", [])
    else:
        rows = _read_jsonl(path)
    return {
        caption_key(str(row.get("caption", row.get("prediction", ""))))
        for row in rows
        if str(row.get("caption", row.get("prediction", ""))).strip()
    }


def _strata(caption: str) -> list[str]:
    parsed = parse_change_prediction(caption)
    strata = ["rule_no_change" if parsed.value == 0 else "default_change"]
    if _CONTRAST.search(caption) and _NEGATION.search(caption):
        strata.append("negation_or_contrast")
    if _CAPTURE_ARTIFACT.search(caption):
        strata.append("capture_artifact")
    if len(caption.split()) >= 40:
        strata.append("long_caption")
    return strata


def _unique_captions(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        caption = str(row.get("prediction", ""))
        key = caption_key(caption)
        if not key:
            continue
        item = unique.setdefault(
            key,
            {
                "caption": caption,
                "occurrences": 0,
                "example_ids": [],
                "example_row": row,
                "strata": _strata(caption),
            },
        )
        item["occurrences"] += 1
        if len(item["example_ids"]) < 3:
            item["example_ids"].append(str(row.get("id", "")))
    return unique


def _disagreement_keys(rows: list[dict[str, Any]]) -> set[str]:
    keys: set[str] = set()
    for row in rows:
        decision = row.get("prediction_changeflag")
        old = parse_change_prediction(str(row.get("prediction", ""))).value
        if decision in {0, 1} and old in {0, 1} and decision != old:
            keys.add(caption_key(str(row.get("prediction", ""))))
    return keys


def build_audit_rows(
    rows: list[dict[str, Any]],
    sample_size: int,
    seed: int,
    *,
    disagreement_keys: set[str] | None = None,
    local_judgments: dict[str, dict[str, Any]] | None = None,
    excluded_caption_keys: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Select unique captions with fixed strata and forced disagreements."""

    if sample_size < 1:
        raise ValueError("sample size must be positive")
    unique = _unique_captions(rows)
    excluded_caption_keys = excluded_caption_keys or set()
    unique = {key: item for key, item in unique.items() if key not in excluded_caption_keys}
    disagreement_keys = disagreement_keys or set()
    local_judgments = local_judgments or {}
    rng = random.Random(seed)
    selected: dict[str, str] = {}

    forced = sorted(key for key in disagreement_keys if key in unique)
    rng.shuffle(forced)
    for key in forced:
        selected[key] = "forced_old_new_disagreement"

    quotas = {
        "negation_or_contrast": 60,
        "capture_artifact": 45,
        "long_caption": 60,
        "rule_no_change": 55,
        "default_change": 80,
    }
    for stratum, quota in quotas.items():
        candidates = [
            key for key, item in unique.items() if stratum in item["strata"] and key not in selected
        ]
        rng.shuffle(candidates)
        for key in candidates[:quota]:
            if len(selected) >= sample_size:
                break
            selected[key] = f"stratum:{stratum}"

    remaining = [key for key in unique if key not in selected]
    rng.shuffle(remaining)
    for key in remaining:
        if len(selected) >= sample_size:
            break
        selected[key] = "random_fill"

    selected_items = list(selected.items())[:sample_size]
    rng.shuffle(selected_items)
    result: list[dict[str, Any]] = []
    for index, (key, reason) in enumerate(selected_items, start=1):
        item = unique[key]
        example_id = item["example_ids"][0]
        old = parse_change_prediction(item["caption"])
        judged = local_judgments.get(example_id, {})
        result.append(
            {
                "audit_id": f"caption-{index:04d}",
                "caption": item["caption"],
                "occurrences": item["occurrences"],
                "human_caption_semantic_label": None,
                "human_note": None,
                "allowed_labels": {
                    "0": "no relevant permanent-structure change",
                    "1": "relevant permanent-structure change",
                    "U": "caption meaning is genuinely ambiguous",
                },
                "_source_row": item["example_row"],
                "_answer_key": {
                    "caption_sha256": hashlib.sha256(key.encode("utf-8")).hexdigest(),
                    "selection_reason": reason,
                    "strata": item["strata"],
                    "old_parser_decision": old.value,
                    "old_parser_mode": old.match_type,
                    "local_judge_decision": judged.get("prediction_changeflag"),
                    "local_judge_source": judged.get("binary_prediction_source"),
                    "example_ids": item["example_ids"],
                },
            }
        )
    return result


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        fields = [
            "audit_id",
            "caption",
            "occurrences",
            "human_caption_semantic_label",
            "human_note",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


def main() -> int:
    args = parse_args()
    source = args.predictions.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"output directory must be empty or absent: {output_dir}")
    rows = _read_jsonl(source)
    comparison_rows = (
        _read_jsonl(args.comparison_judged.resolve()) if args.comparison_judged else []
    )
    disagreement_keys = _disagreement_keys(comparison_rows)
    judged_rows = (
        _read_jsonl(args.local_judge_results.resolve()) if args.local_judge_results else []
    )
    local_judgments = {str(row.get("id", "")): row for row in judged_rows}
    excluded_caption_keys = (
        _read_caption_keys(args.exclude_captions.resolve()) if args.exclude_captions else set()
    )
    prepared = build_audit_rows(
        rows,
        args.sample_size,
        args.seed,
        disagreement_keys=disagreement_keys,
        local_judgments=local_judgments,
        excluded_caption_keys=excluded_caption_keys,
    )
    blind_rows = [
        {key: value for key, value in row.items() if not key.startswith("_")} for row in prepared
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "schema_version": "1.1",
        "task": "blind_levir_caption_semantic_audit",
        "protocol": "levir_cc_permanent_structure_change_v1",
        "seed": args.seed,
        "num_audit_rows": len(blind_rows),
        "warning": "Label only caption meaning. Do not inspect images, references, or answer key.",
    }
    for annotator in ("annotator_a", "annotator_b"):
        _write_json(
            output_dir / f"{annotator}.json", {**common, "annotator": annotator, "rows": blind_rows}
        )
        _write_csv(output_dir / f"{annotator}.csv", blind_rows)

    answer_rows = [{"audit_id": row["audit_id"], **row["_answer_key"]} for row in prepared]
    _write_json(
        output_dir / "answer_key.json",
        {"schema_version": "1.1", "protocol": common["protocol"], "rows": answer_rows},
    )
    with (output_dir / "audit_source_predictions.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as file:
        for row in prepared:
            file.write(json.dumps(row["_source_row"], ensure_ascii=False) + "\n")
    manifest = {
        **common,
        "source_file": str(source),
        "source_sha256": _sha256(source),
        "num_source_rows": len(rows),
        "num_unique_captions": len(_unique_captions(rows)),
        "comparison_file": str(args.comparison_judged.resolve())
        if args.comparison_judged
        else None,
        "comparison_sha256": _sha256(args.comparison_judged.resolve())
        if args.comparison_judged
        else None,
        "num_forced_disagreement_captions": len(disagreement_keys),
        "local_judge_results": str(args.local_judge_results.resolve())
        if args.local_judge_results
        else None,
        "local_judge_results_sha256": _sha256(args.local_judge_results.resolve())
        if args.local_judge_results
        else None,
        "exclude_captions_file": str(args.exclude_captions.resolve())
        if args.exclude_captions
        else None,
        "exclude_captions_sha256": _sha256(args.exclude_captions.resolve())
        if args.exclude_captions
        else None,
        "num_excluded_caption_keys": len(excluded_caption_keys),
        "files": [
            "annotator_a.json",
            "annotator_a.csv",
            "annotator_b.json",
            "annotator_b.csv",
            "answer_key.json",
            "audit_source_predictions.jsonl",
        ],
    }
    _write_json(output_dir / "audit_manifest.json", manifest)
    print(f"Saved {len(blind_rows)} blind audit rows to: {output_dir}")
    print(f"Forced disagreement captions: {len(disagreement_keys)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
