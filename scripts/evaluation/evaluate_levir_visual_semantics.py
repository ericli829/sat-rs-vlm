"""Evaluate LEVIR-CC generated Captions against image-audited semantic gold."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sat_rs_vlm.evaluation.visual_semantics import (  # noqa: E402
    aggregate_visual_semantic_metrics,
    extract_visual_semantics,
    parse_gold_semantics,
    sample_semantic_metrics,
)

IMPLEMENTATION_VERSION = "levir-visual-semantics-evaluator-v1.2"
REQUIRED_PROMPT_FIELDS = (
    "prompt_text_verbatim",
    "image_t1_role",
    "image_t2_role",
    "input_image_order",
    "do_sample",
    "temperature",
    "top_p",
    "max_new_tokens",
    "num_beams",
    "output_postprocessing",
)
REQUIRED_MODEL_FIELDS = (
    "model_id",
    "adapter_id",
    "quantization",
    "code_version",
)
REQUIRED_REPRODUCIBILITY_FIELDS = REQUIRED_PROMPT_FIELDS + REQUIRED_MODEL_FIELDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-csv", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-profile", required=True)
    parser.add_argument(
        "--generation-manifest",
        type=Path,
        required=True,
        help="Declared prompt/image-order/generation manifest for this exact predictions file.",
    )
    parser.add_argument(
        "--allow-incomplete-historical-manifest",
        action="store_true",
        help=(
            "Allow only a manifest explicitly marked historical_incomplete_exploratory. "
            "Results remain hash-bound but are not formal reproducible evaluations."
        ),
    )
    parser.add_argument(
        "--verify-image-paths",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require each gold image_t1_path/image_t2_path to exist locally (default: true).",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_gold(path: Path, *, verify_image_paths: bool) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    required = {
        "sample_id",
        "image_t1_path",
        "image_t2_path",
        "gold_change_label",
        "gold_changed_objects",
        "gold_change_directions",
        "gold_change_events",
        "annotation_confidence",
        "label_source",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"gold CSV is missing required columns: {sorted(required)}")
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        sample_id = row["sample_id"].strip()
        if not sample_id or sample_id in result:
            raise ValueError("gold CSV requires unique non-empty sample_id values")
        for field in ("image_t1_path", "image_t2_path"):
            raw_path = row[field].strip()
            if not raw_path:
                raise ValueError(f"gold CSV {sample_id}: {field} must be non-empty")
            image_path = Path(raw_path)
            resolved_image = image_path if image_path.is_absolute() else path.parent / image_path
            if verify_image_paths and not resolved_image.is_file():
                raise ValueError(
                    f"gold CSV {sample_id}: declared {field} is not a readable local file: "
                    f"{resolved_image}"
                )
        parse_gold_semantics(row)
        result[sample_id] = row
    return result


def _read_predictions(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("id", "")).strip()
            if not sample_id or sample_id in rows or not isinstance(row.get("prediction"), str):
                raise ValueError(f"invalid prediction at {path}:{line_number}")
            rows[sample_id] = row
    return rows


def _stored_binary_decision(row: dict[str, Any]) -> tuple[int | None, str]:
    """Reuse the established LEVIR binary judge; never infer it from Caption keywords.

    The detailed visual-semantic evaluator extracts objects/directions/events
    from the Caption itself, but binary change/no-change is deliberately a
    separate, cached decision.  If a server-rule or offline local-judge result
    is absent, the sample remains unresolved rather than being silently routed
    through a second keyword parser.
    """

    raw = row.get("prediction_changeflag")
    if type(raw) is int and raw in {0, 1}:
        source = str(row.get("binary_prediction_source", "")).strip()
        if not source:
            judge = row.get("change_judge")
            if isinstance(judge, dict):
                source = str(judge.get("source", "")).strip()
        return raw, source or "stored_binary_decision"
    judge = row.get("change_judge")
    if isinstance(judge, dict):
        value = judge.get("value")
        if type(value) is int and value in {0, 1}:
            source = str(judge.get("source", "")).strip()
            return value, source or "stored_change_judge"
    return None, "unresolved_missing_binary_judge"


def _manifest_value(payload: dict[str, Any], field: str) -> Any:
    """Read a field from the documented manifest or the inference run manifest."""

    aliases = {"prompt_text_verbatim": "prompt_text"}
    candidates = (field, aliases.get(field, field))
    for candidate in candidates:
        if candidate in payload:
            return payload[candidate]
    generation = payload.get("generation")
    if isinstance(generation, dict):
        for candidate in candidates:
            if candidate in generation:
                return generation[candidate]
    reproducibility = payload.get("prompt_reproducibility")
    if isinstance(reproducibility, dict):
        declared = reproducibility.get("declared")
        if isinstance(declared, dict):
            for candidate in candidates:
                if candidate in declared:
                    return declared[candidate]
    return None


def _load_generation_manifest(
    path: Path,
    *,
    predictions_path: Path,
    prompt_profile: str,
    allow_incomplete_historical_manifest: bool,
) -> dict[str, Any]:
    """Validate formal prompt disclosure and bind it to the prediction bytes."""

    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("generation manifest root must be an object")
    missing = [
        field
        for field in REQUIRED_REPRODUCIBILITY_FIELDS
        if _manifest_value(payload, field) is None
    ]
    reproducibility_status = str(payload.get("reproducibility_status", "complete")).strip()
    is_historical_incomplete = reproducibility_status == "historical_incomplete_exploratory"
    if missing and not (allow_incomplete_historical_manifest and is_historical_incomplete):
        raise ValueError(f"generation manifest is missing required prompt fields: {missing}")
    if allow_incomplete_historical_manifest and not is_historical_incomplete:
        raise ValueError(
            "--allow-incomplete-historical-manifest requires "
            "reproducibility_status=historical_incomplete_exploratory"
        )
    declared_profile = _manifest_value(payload, "prompt_profile")
    if declared_profile is None:
        declared_profile = _manifest_value(payload, "generation_profile_name")
    if str(declared_profile or "").strip() != prompt_profile:
        raise ValueError(
            "--prompt-profile must exactly match the declared generation manifest profile: "
            f"{prompt_profile!r} != {declared_profile!r}"
        )
    expected_hash = payload.get("predictions_sha256")
    outputs = payload.get("outputs")
    if expected_hash is None and isinstance(outputs, dict):
        expected_hash = outputs.get("predictions_sha256")
    actual_hash = _sha256(predictions_path)
    if not isinstance(expected_hash, str) or expected_hash != actual_hash:
        raise ValueError(
            "generation manifest predictions_sha256 must match the exact --predictions file"
        )
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "prompt_profile": prompt_profile,
        "reproducibility_status": reproducibility_status,
        "missing_reproducibility_fields": missing,
        "declared_fields": {
            field: _manifest_value(payload, field) for field in REQUIRED_REPRODUCIBILITY_FIELDS
        },
    }


def evaluate(
    gold_rows: dict[str, dict[str, str]], predictions: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return scored rows and audit-only rows after strict ID alignment."""

    if gold_rows.keys() - predictions.keys():
        missing = sorted(gold_rows.keys() - predictions.keys())
        raise ValueError(f"predictions missing {len(missing)} gold IDs: {missing[:20]}")
    scored: list[dict[str, Any]] = []
    audit_only: list[dict[str, Any]] = []
    for sample_id, gold_row in sorted(gold_rows.items()):
        gold = parse_gold_semantics(gold_row)
        prediction_row = predictions[sample_id]
        caption_semantics = extract_visual_semantics(str(prediction_row["prediction"]))
        stored_binary, binary_source = _stored_binary_decision(prediction_row)
        # Preserve the Caption-extracted objects/directions/events, but binary
        # change detection is valid only when a prior decision was persisted.
        prediction = replace(caption_semantics, change_label=stored_binary)
        row = {
            "id": sample_id,
            "caption": prediction_row["prediction"],
            "image_t1_path": gold_row["image_t1_path"].strip(),
            "image_t2_path": gold_row["image_t2_path"].strip(),
            "gold": (
                {
                    "change_label": gold.change_label,
                    "objects": sorted(gold.objects),
                    "directions": sorted(gold.directions),
                    "events": [
                        {"object": object_name, "direction": direction}
                        for object_name, direction in sorted(gold.events)
                    ],
                }
                if gold is not None
                else {"change_label": "U"}
            ),
            "prediction": prediction.to_dict(),
            "caption_semantics": caption_semantics.to_dict(),
            "binary_decision": {
                "value": prediction.change_label,
                "source": binary_source,
                "used_stored_judge": stored_binary is not None,
                "status": "resolved" if stored_binary is not None else "unresolved",
            },
            "annotation_confidence": gold_row["annotation_confidence"].strip(),
            "label_source": gold_row["label_source"].strip(),
        }
        if gold is None:
            audit_only.append(row)
            continue
        row["sample_metrics"] = sample_semantic_metrics(prediction, gold)
        scored.append(row)
    return scored, audit_only


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"output directory must be empty or absent: {output_dir}")
    try:
        gold_path, predictions_path = args.gold_csv.resolve(), args.predictions.resolve()
        generation_manifest = _load_generation_manifest(
            args.generation_manifest.resolve(),
            predictions_path=predictions_path,
            prompt_profile=args.prompt_profile,
            allow_incomplete_historical_manifest=args.allow_incomplete_historical_manifest,
        )
        scored, audit_only = evaluate(
            _read_gold(gold_path, verify_image_paths=args.verify_image_paths),
            _read_predictions(predictions_path),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"visual semantic evaluation failed: {exc}") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "visual_semantic_evaluated_predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in scored),
        encoding="utf-8",
    )
    (output_dir / "visual_semantic_audit_only.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in audit_only),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "1.0",
        "implementation_version": IMPLEMENTATION_VERSION,
        "metric_label": "internal_rule_extracted_caption_vs_visual_gold",
        "result_scope": (
            "historical_exploratory"
            if generation_manifest["reproducibility_status"] == "historical_incomplete_exploratory"
            else "formal_reproducible"
        ),
        "prompt_profile": args.prompt_profile,
        "generation_manifest": generation_manifest,
        "num_gold_rows": len(scored) + len(audit_only),
        "num_scored_rows": len(scored),
        "num_audit_only_uncertain_rows": len(audit_only),
        "binary_decision_source_distribution": dict(
            sorted(Counter(row["binary_decision"]["source"] for row in scored).items())
        ),
        "binary_decision_status_distribution": dict(
            sorted(Counter(row["binary_decision"]["status"] for row in scored).items())
        ),
        "metrics": aggregate_visual_semantic_metrics(scored),
        "limitations": [
            "Gold labels are image-audited, but prediction facts are deterministically extracted "
            "from generated Caption text.",
            "This measures semantic correctness expressed by the Caption, not unexpressed image "
            "understanding.",
            "Binary change/no-change metrics use only persisted server-rule or offline local-"
            "judge decisions. Captions without a persisted binary decision are unresolved and "
            "reported through coverage rather than keyword fallback.",
            *(
                [
                    "Historical exploration only: the original prompt and/or generation "
                    "settings were not fully recorded, so this result must not be presented "
                    "as a formal reproducible comparison."
                ]
                if generation_manifest["reproducibility_status"]
                == "historical_incomplete_exploratory"
                else []
            ),
        ],
    }
    (output_dir / "visual_semantic_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": "1.0",
        "implementation_version": IMPLEMENTATION_VERSION,
        "run_time_utc": datetime.now(timezone.utc).isoformat(),
        "gold_csv": str(gold_path),
        "gold_sha256": _sha256(gold_path),
        "predictions": str(predictions_path),
        "predictions_sha256": _sha256(predictions_path),
        "prompt_profile": args.prompt_profile,
        "result_scope": summary["result_scope"],
        "generation_manifest": generation_manifest,
        "verify_image_paths": args.verify_image_paths,
        "outputs": [
            "visual_semantic_evaluated_predictions.jsonl",
            "visual_semantic_audit_only.jsonl",
            "visual_semantic_summary.json",
        ],
        "remote_write_performed": False,
    }
    (output_dir / "visual_semantic_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved visual semantic evaluation to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
