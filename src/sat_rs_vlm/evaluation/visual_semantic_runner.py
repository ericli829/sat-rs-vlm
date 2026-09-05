"""Unified runner for the image-audited LEVIR visual-semantic profile."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .visual_semantics import (
    aggregate_visual_semantic_metrics,
    extract_visual_semantics,
    parse_gold_semantics,
    sample_semantic_metrics,
)

IMPLEMENTATION_VERSION = "levir-visual-semantics-evaluator-v1.3"
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


@dataclass(frozen=True)
class VisualSemanticEvaluationResult:
    scored_rows: tuple[dict[str, Any], ...]
    audit_only_rows: tuple[dict[str, Any], ...]
    summary: dict[str, Any]
    manifest: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_visual_semantic_gold(
    path: str | Path,
    *,
    verify_image_paths: bool = True,
) -> dict[str, dict[str, str]]:
    """Read and validate image-audited gold annotations."""

    gold_path = Path(path).expanduser().resolve()
    with gold_path.open(encoding="utf-8-sig", newline="") as file:
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
        raise ValueError(
            "visual semantic gold CSV is missing required columns: "
            f"{sorted(required)}"
        )
    result: dict[str, dict[str, str]] = {}
    for raw_row in rows:
        row = {str(key): str(value or "") for key, value in raw_row.items()}
        sample_id = row["sample_id"].strip()
        if not sample_id or sample_id in result:
            raise ValueError("visual semantic gold CSV requires unique non-empty sample_id values")
        for field in ("image_t1_path", "image_t2_path"):
            declared = row[field].strip()
            if not declared:
                raise ValueError(f"visual semantic gold {sample_id}: {field} must be non-empty")
            image_path = Path(declared)
            resolved_image = (
                image_path if image_path.is_absolute() else gold_path.parent / image_path
            )
            if verify_image_paths and not resolved_image.is_file():
                raise ValueError(
                    f"visual semantic gold {sample_id}: declared {field} is not a readable "
                    f"local file: {resolved_image}"
                )
        parse_gold_semantics(row)
        result[sample_id] = row
    return result


def read_prediction_outputs(path: str | Path) -> list[dict[str, Any]]:
    """Read prediction JSONL for the standalone visual-semantic CLI."""

    prediction_path = Path(path).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with prediction_path.open(encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"invalid prediction at {prediction_path}:{line_number}")
            sample_id = str(payload.get("id", "")).strip()
            if not sample_id or sample_id in seen or not isinstance(payload.get("prediction"), str):
                raise ValueError(f"invalid prediction at {prediction_path}:{line_number}")
            seen.add(sample_id)
            rows.append(dict(payload))
    if not rows:
        raise ValueError(f"prediction JSONL contains no records: {prediction_path}")
    return rows


def _stored_binary_decision(row: Mapping[str, Any]) -> tuple[int | None, str]:
    raw = row.get("prediction_changeflag")
    if type(raw) is not int:
        raw = row.get("predicted_changeflag")
    if type(raw) is int and raw in {0, 1}:
        source = str(row.get("binary_prediction_source", "")).strip()
        judge = row.get("change_judge")
        if not source and isinstance(judge, Mapping):
            source = str(judge.get("source", "")).strip()
        return raw, source or "stored_binary_decision"
    judge = row.get("change_judge")
    if isinstance(judge, Mapping):
        value = judge.get("value")
        if type(value) is int and value in {0, 1}:
            return value, str(judge.get("source", "")).strip() or "stored_change_judge"
    return None, "unresolved_missing_binary_judge"


def _manifest_value(payload: Mapping[str, Any], field: str) -> Any:
    aliases = {"prompt_text_verbatim": "prompt_text"}
    candidates = (field, aliases.get(field, field))
    for candidate in candidates:
        if candidate in payload:
            return payload[candidate]
    for section_name in ("generation", "prompt_reproducibility"):
        section = payload.get(section_name)
        if not isinstance(section, Mapping):
            continue
        declared = section.get("declared") if section_name == "prompt_reproducibility" else section
        if not isinstance(declared, Mapping):
            continue
        for candidate in candidates:
            if candidate in declared:
                return declared[candidate]
    return None


def load_generation_manifest(
    path: str | Path | None,
    *,
    predictions_path: Path | None,
    prompt_profile: str | None,
    allow_incomplete_historical_manifest: bool = False,
) -> dict[str, Any]:
    """Validate optional reproducibility metadata without weakening gold provenance."""

    if path is None:
        if allow_incomplete_historical_manifest:
            raise ValueError(
                "historical incomplete mode requires a generation manifest"
            )
        return {
            "path": None,
            "sha256": None,
            "prompt_profile": prompt_profile,
            "reproducibility_status": "auxiliary_unbound",
            "missing_reproducibility_fields": list(REQUIRED_REPRODUCIBILITY_FIELDS),
            "declared_fields": {field: None for field in REQUIRED_REPRODUCIBILITY_FIELDS},
        }
    manifest_path = Path(path).expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, Mapping):
        raise ValueError("visual semantic generation manifest root must be an object")
    missing = [
        field
        for field in REQUIRED_REPRODUCIBILITY_FIELDS
        if _manifest_value(payload, field) is None
    ]
    reproducibility_status = str(
        payload.get("reproducibility_status", "complete")
    ).strip()
    historical = reproducibility_status == "historical_incomplete_exploratory"
    if missing and not (allow_incomplete_historical_manifest and historical):
        raise ValueError(
            "visual semantic generation manifest is missing required fields: " + str(missing)
        )
    if allow_incomplete_historical_manifest and not historical:
        raise ValueError(
            "allow_incomplete_historical_manifest requires "
            "reproducibility_status=historical_incomplete_exploratory"
        )
    declared_profile = _manifest_value(payload, "prompt_profile")
    if declared_profile is None:
        declared_profile = _manifest_value(payload, "generation_profile_name")
    if prompt_profile is None or str(declared_profile or "").strip() != prompt_profile:
        raise ValueError(
            "visual semantic prompt_profile must match the generation manifest: "
            f"{prompt_profile!r} != {declared_profile!r}"
        )
    if predictions_path is not None:
        expected_hash = payload.get("predictions_sha256")
        outputs = payload.get("outputs")
        if expected_hash is None and isinstance(outputs, Mapping):
            expected_hash = outputs.get("predictions_sha256")
        actual_hash = _sha256(predictions_path)
        if not isinstance(expected_hash, str) or expected_hash != actual_hash:
            raise ValueError(
                "generation manifest predictions_sha256 must match the exact predictions file"
            )
    return {
        "path": str(manifest_path),
        "sha256": _sha256(manifest_path),
        "prompt_profile": prompt_profile,
        "reproducibility_status": reproducibility_status,
        "missing_reproducibility_fields": missing,
        "declared_fields": {
            field: _manifest_value(payload, field) for field in REQUIRED_REPRODUCIBILITY_FIELDS
        },
    }


def evaluate_visual_semantics(
    outputs: Iterable[Mapping[str, Any]],
    *,
    gold_csv: str | Path,
    predictions_path: str | Path | None = None,
    generation_manifest_path: str | Path | None = None,
    prompt_profile: str | None = None,
    allow_incomplete_historical_manifest: bool = False,
    verify_image_paths: bool = True,
) -> VisualSemanticEvaluationResult:
    """Evaluate generated captions against image-audited semantic gold."""

    gold_path = Path(gold_csv).expanduser().resolve()
    prediction_path = (
        Path(predictions_path).expanduser().resolve() if predictions_path is not None else None
    )
    gold_rows = read_visual_semantic_gold(gold_path, verify_image_paths=verify_image_paths)
    prediction_rows: dict[str, dict[str, Any]] = {}
    for output in outputs:
        sample_id = str(output.get("id", "")).strip()
        if sample_id:
            if sample_id in prediction_rows:
                raise ValueError(f"duplicate prediction id: {sample_id}")
            prediction_rows[sample_id] = dict(output)
    missing = sorted(set(gold_rows) - set(prediction_rows))
    if missing:
        raise ValueError(
            f"predictions missing {len(missing)} visual semantic gold IDs: {missing[:20]}"
        )
    generation_manifest = load_generation_manifest(
        generation_manifest_path,
        predictions_path=prediction_path,
        prompt_profile=prompt_profile,
        allow_incomplete_historical_manifest=allow_incomplete_historical_manifest,
    )
    scored: list[dict[str, Any]] = []
    audit_only: list[dict[str, Any]] = []
    for sample_id, gold_row in sorted(gold_rows.items()):
        gold = parse_gold_semantics(gold_row)
        prediction_row = prediction_rows[sample_id]
        caption_semantics = extract_visual_semantics(str(prediction_row["prediction"]))
        stored_binary, binary_source = _stored_binary_decision(prediction_row)
        prediction = replace(caption_semantics, change_label=stored_binary)
        row: dict[str, Any] = {
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
        else:
            row["sample_metrics"] = sample_semantic_metrics(prediction, gold)
            scored.append(row)
    result_scope = (
        "historical_exploratory"
        if generation_manifest["reproducibility_status"] == "historical_incomplete_exploratory"
        else "formal_reproducible_auxiliary"
        if generation_manifest["reproducibility_status"] == "complete"
        else "auxiliary_unbound"
    )
    summary = {
        "schema_version": "1.1",
        "implementation_version": IMPLEMENTATION_VERSION,
        "metric_label": "research_auxiliary_visual_semantic_profile",
        "official_benchmark_metric": False,
        "result_scope": result_scope,
        "gold_source": "image_audited_annotation",
        "caption_reference_used_as_visual_gold": False,
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
            "Gold labels are image-audited, while prediction facts are deterministically "
            "extracted from generated caption text.",
            "This measures semantic correctness expressed by the caption, not unexpressed "
            "image understanding.",
            "Binary change metrics use only persisted binary judge decisions; missing decisions "
            "remain unresolved and are reported through coverage.",
            "This auxiliary profile is separate from official LEVIR-CC benchmark metrics.",
        ],
    }
    manifest = {
        "schema_version": "1.1",
        "implementation_version": IMPLEMENTATION_VERSION,
        "run_time_utc": datetime.now(timezone.utc).isoformat(),
        "gold_csv": str(gold_path),
        "gold_sha256": _sha256(gold_path),
        "predictions": str(prediction_path) if prediction_path is not None else None,
        "predictions_sha256": _sha256(prediction_path) if prediction_path is not None else None,
        "prompt_profile": prompt_profile,
        "result_scope": result_scope,
        "official_benchmark_metric": False,
        "generation_manifest": generation_manifest,
        "verify_image_paths": verify_image_paths,
        "remote_write_performed": False,
    }
    return VisualSemanticEvaluationResult(
        tuple(scored),
        tuple(audit_only),
        summary,
        manifest,
    )


def write_visual_semantic_evaluation(
    result: VisualSemanticEvaluationResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write the auxiliary report files and return their paths."""

    destination = Path(output_dir).expanduser().resolve()
    outputs = {
        "visual_semantic_evaluated_predictions": destination
        / "visual_semantic_evaluated_predictions.jsonl",
        "visual_semantic_audit_only": destination / "visual_semantic_audit_only.jsonl",
        "visual_semantic_summary": destination / "visual_semantic_summary.json",
        "visual_semantic_manifest": destination / "visual_semantic_manifest.json",
    }
    destination.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("visual_semantic_evaluated_predictions", result.scored_rows),
        ("visual_semantic_audit_only", result.audit_only_rows),
    ):
        outputs[name].write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    outputs["visual_semantic_summary"].write_text(
        json.dumps(result.summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = dict(result.manifest)
    manifest["outputs"] = {name: str(path) for name, path in outputs.items()}
    outputs["visual_semantic_manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return outputs
