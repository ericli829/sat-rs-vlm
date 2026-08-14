"""Prediction JSONL 的独立评测执行、聚合和安全输出。"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from sat_rs_vlm.evaluation.extended_metrics import (
    box_iou,
    caption_scores,
    cider_d_single_reference_approx_scores,
    corpus_bleu_single_reference_approx,
    generalized_box_iou,
    latency_statistics,
    mean,
    metric_value,
    normalized_center_distance,
    text_task_scores,
)
from sat_rs_vlm.evaluation.metrics import score_sample
from sat_rs_vlm.evaluation.parsers import parse_change_prediction, parse_count, parse_grounding
from sat_rs_vlm.evaluation.protocols import (
    ProtocolResolution,
    coordinate_format_for_record,
    load_contract,
    manifest_coordinate_format,
    resolve_protocol,
)
from sat_rs_vlm.evaluation.records import (
    EvaluationError,
    InputValidationError,
    PredictionRecord,
    read_prediction_jsonl,
)
from sat_rs_vlm.evaluation.semantic.runner import SemanticEvaluator

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SEMANTIC_CONTRACT = (
    PROJECT_ROOT / "configs" / "eval" / "semantic" / "semantic_contract.json"
)
DEFAULT_SEMANTIC_ONTOLOGY = (
    PROJECT_ROOT / "configs" / "eval" / "semantic" / "remote_sensing_ontology.json"
)


@dataclass(frozen=True)
class EvaluatedRow:
    output: dict[str, Any]
    protocol: str
    kind: str
    protocol_status: str


@dataclass(frozen=True)
class LatencyContext:
    semantics: str
    eval_batch_size: int | None
    group_by_task: bool | None
    status: str
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantics": self.semantics,
            "eval_batch_size": self.eval_batch_size,
            "group_by_task": self.group_by_task,
            "status": self.status,
            "note": self.note,
        }


def resolve_latency_context(
    semantics: str,
    eval_batch_size: int | None,
    group_by_task: bool | None,
) -> LatencyContext:
    allowed = {"unresolved", "single_sample", "batch_amortized_per_sample"}
    if semantics not in allowed:
        raise EvaluationError(
            f"latency semantics must be one of {sorted(allowed)}, got: {semantics}"
        )
    if eval_batch_size is not None and eval_batch_size < 1:
        raise EvaluationError("eval batch size must be a positive integer")
    if semantics == "single_sample":
        if eval_batch_size not in {None, 1}:
            raise EvaluationError("single_sample latency requires eval batch size 1")
        return LatencyContext(
            semantics=semantics,
            eval_batch_size=1,
            group_by_task=group_by_task,
            status="resolved",
            note="Each latency value represents one independently generated sample.",
        )
    if semantics == "batch_amortized_per_sample":
        status = "resolved" if eval_batch_size is not None else "incomplete"
        note = (
            "Each row stores total batch generation time divided by batch sample count; "
            "P50/P95 describe recorded batch-amortized values, not independent request latency."
        )
        if eval_batch_size is None:
            note += " Evaluation batch size was not supplied."
        return LatencyContext(
            semantics=semantics,
            eval_batch_size=eval_batch_size,
            group_by_task=group_by_task,
            status=status,
            note=note,
        )
    return LatencyContext(
        semantics=semantics,
        eval_batch_size=eval_batch_size,
        group_by_task=group_by_task,
        status="unresolved",
        note=(
            "Latency provenance was not supplied; do not compare these values across "
            "different evaluation batch configurations."
        ),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_size(metadata: dict[str, Any]) -> tuple[int, int] | None:
    width = metadata.get("image_width", metadata.get("width"))
    height = metadata.get("image_height", metadata.get("height"))
    if isinstance(width, (int, float)) and isinstance(height, (int, float)):
        return int(width), int(height)
    return None


def _base_output(record: PredictionRecord, resolution: ProtocolResolution) -> dict[str, Any]:
    output = dict(record.raw)
    output.update(
        {
            "eval_protocol": resolution.name,
            "metric_profile": resolution.metric_profile,
        }
    )
    return output


def _evaluate_grounding(
    record: PredictionRecord,
    resolution: ProtocolResolution,
    coordinate_format: str | None,
    *,
    strict: bool,
) -> EvaluatedRow:
    image_size = _image_size(record.metadata)
    reference = parse_grounding(
        record.reference,
        coordinate_format=coordinate_format,
        image_size=image_size,
    )
    reference_error = reference.parse_error or reference.coordinate_error
    if not reference.parse_ok or not reference.coordinate_valid or reference.bbox is None:
        message = f"sample {record.id}: invalid Grounding reference: {reference_error}"
        if strict:
            raise InputValidationError(message)
        output = _base_output(record, resolution)
        output.update(
            {
                "parsed_prediction": None,
                "parse_ok": False,
                "parse_error": f"invalid_reference:{reference_error}",
                "sample_metrics": {},
            }
        )
        return EvaluatedRow(output, resolution.name, resolution.kind, "data_error")

    prediction = parse_grounding(
        record.prediction,
        coordinate_format=coordinate_format,
        image_size=image_size,
    )
    comparable = prediction.parse_ok and prediction.coordinate_valid and prediction.bbox is not None
    if comparable:
        assert prediction.bbox is not None
        assert reference.bbox is not None
        iou = box_iou(prediction.bbox, reference.bbox)
        giou = generalized_box_iou(prediction.bbox, reference.bbox)
        center_distance = normalized_center_distance(prediction.bbox, reference.bbox)
    else:
        iou = 0.0
        giou = -1.0
        center_distance = 1.0
    label_match = bool(
        prediction.parse_ok
        and prediction.label is not None
        and reference.label is not None
        and prediction.label == reference.label
    )
    parse_error = prediction.parse_error or prediction.coordinate_error
    parsed_prediction = (
        {"label": prediction.label, "bbox": list(prediction.bbox)}
        if prediction.parse_ok and prediction.bbox is not None
        else None
    )
    output = _base_output(record, resolution)
    output.update(
        {
            "parsed_prediction": parsed_prediction,
            "parse_ok": prediction.parse_ok,
            "parse_error": parse_error,
            "sample_metrics": {
                "valid_json": prediction.valid_json,
                "parse_success": prediction.parse_ok,
                "valid_coordinate": prediction.coordinate_valid,
                "coordinate_error": prediction.coordinate_error,
                "label_match": label_match,
                "iou": iou,
                "generalized_iou": giou,
                "normalized_center_distance": center_distance,
                "correct_at_0_5": iou >= 0.5,
                "correct_at_0_7": iou >= 0.7,
                "label_and_iou_correct_at_0_5": label_match and iou >= 0.5,
            },
        }
    )
    return EvaluatedRow(output, resolution.name, resolution.kind, resolution.status)


def _evaluate_counting(
    record: PredictionRecord,
    resolution: ProtocolResolution,
    *,
    strict: bool,
) -> EvaluatedRow:
    reference = parse_count(record.reference)
    if reference.value is None:
        message = f"sample {record.id}: invalid Counting reference: {reference.reason}"
        if strict:
            raise InputValidationError(message)
        output = _base_output(record, resolution)
        output.update(
            {
                "parsed_prediction": None,
                "parse_ok": False,
                "parse_error": f"invalid_reference:{reference.reason}",
                "sample_metrics": {},
            }
        )
        return EvaluatedRow(output, resolution.name, resolution.kind, "data_error")

    prediction = parse_count(record.prediction)
    predicted_value = prediction.value
    parsed = predicted_value is not None
    error = abs(predicted_value - reference.value) if predicted_value is not None else None
    signed_error = predicted_value - reference.value if predicted_value is not None else None
    absolute_percentage_error = (
        error / abs(reference.value) if error is not None and reference.value != 0 else None
    )
    output = _base_output(record, resolution)
    output.update(
        {
            "parsed_prediction": predicted_value,
            "parse_ok": parsed,
            "parse_error": prediction.reason,
            "sample_metrics": {
                "number_parse_success": parsed,
                "exact_count_correct": bool(parsed and error == 0),
                "within_1_correct": bool(parsed and error is not None and error <= 1),
                "absolute_error": error,
                "squared_error": error * error if error is not None else None,
                "signed_error": signed_error,
                "absolute_percentage_error": absolute_percentage_error,
            },
        }
    )
    return EvaluatedRow(output, resolution.name, resolution.kind, resolution.status)


def _evaluate_text(
    record: PredictionRecord,
    resolution: ProtocolResolution,
    *,
    strict: bool,
) -> EvaluatedRow:
    if not record.reference.strip() and strict:
        raise InputValidationError(f"sample {record.id}: text reference is empty")
    scores = text_task_scores(record.prediction, record.reference)
    parsed = bool(record.prediction.strip())
    output = _base_output(record, resolution)
    output.update(
        {
            "parsed_prediction": record.prediction.strip() if parsed else None,
            "parse_ok": parsed,
            "parse_error": None if parsed else "empty_prediction",
            "sample_metrics": scores,
        }
    )
    return EvaluatedRow(output, resolution.name, resolution.kind, resolution.status)


def _evaluate_caption(
    record: PredictionRecord,
    resolution: ProtocolResolution,
    *,
    strict: bool,
) -> EvaluatedRow:
    if not record.reference.strip() and strict:
        raise InputValidationError(f"sample {record.id}: caption reference is empty")
    parsed = bool(record.prediction.strip())
    output = _base_output(record, resolution)
    output.update(
        {
            "parsed_prediction": record.prediction.strip() if parsed else None,
            "parse_ok": parsed,
            "parse_error": None if parsed else "empty_prediction",
            "sample_metrics": caption_scores(record.prediction, record.reference),
        }
    )
    return EvaluatedRow(output, resolution.name, resolution.kind, resolution.status)


def _evaluate_change_caption(
    record: PredictionRecord,
    resolution: ProtocolResolution,
    *,
    strict: bool,
) -> EvaluatedRow:
    if not record.reference.strip() and strict:
        raise InputValidationError(f"sample {record.id}: change caption reference is empty")
    raw_changeflag = record.metadata.get("changeflag")
    changeflag_valid = type(raw_changeflag) is int and raw_changeflag in {0, 1}
    if strict and not changeflag_valid:
        raise InputValidationError(
            f"sample {record.id}: metadata.changeflag must be integer 0 or 1"
        )
    reference_changeflag = cast(int, raw_changeflag) if changeflag_valid else None
    parsed = parse_change_prediction(record.prediction)
    binary_correct = (
        parsed.value == reference_changeflag
        if parsed.value is not None and reference_changeflag is not None
        else None
    )
    scores = caption_scores(record.prediction, record.reference)
    scores.update(
        {
            "changeflag_valid": changeflag_valid,
            "binary_parse_success": parsed.value is not None,
            "binary_correct": binary_correct,
        }
    )
    output = _base_output(record, resolution)
    output.update(
        {
            "parsed_prediction": parsed.normalized_text or None,
            "parse_ok": parsed.value is not None,
            "parse_error": parsed.reason,
            "reference_changeflag": reference_changeflag,
            "predicted_changeflag": parsed.value,
            "binary_correct": binary_correct,
            "sample_metrics": scores,
        }
    )
    return EvaluatedRow(output, resolution.name, resolution.kind, resolution.status)


def _evaluate_unimplemented(
    record: PredictionRecord,
    resolution: ProtocolResolution,
) -> EvaluatedRow:
    output = _base_output(record, resolution)
    output.update(
        {
            "parsed_prediction": None,
            "parse_ok": None,
            "parse_error": "protocol_not_implemented",
            "sample_metrics": {},
        }
    )
    return EvaluatedRow(output, resolution.name, resolution.kind, resolution.status)


def evaluate_record(
    record: PredictionRecord,
    contract: dict[str, Any],
    manifest_format: str | None,
    *,
    strict: bool,
) -> tuple[EvaluatedRow, list[str]]:
    resolution = resolve_protocol(record, contract)
    if resolution.kind == "visual_grounding":
        coordinate_format, warnings = coordinate_format_for_record(record, manifest_format)
        return (
            _evaluate_grounding(
                record,
                resolution,
                coordinate_format,
                strict=strict,
            ),
            warnings,
        )
    if resolution.kind == "counting":
        return _evaluate_counting(record, resolution, strict=strict), []
    if resolution.kind == "text":
        return _evaluate_text(record, resolution, strict=strict), []
    if resolution.kind == "caption":
        return _evaluate_caption(record, resolution, strict=strict), []
    if resolution.kind == "change_caption":
        return _evaluate_change_caption(record, resolution, strict=strict), []
    return _evaluate_unimplemented(record, resolution), []


def _common_metrics(rows: list[EvaluatedRow]) -> dict[str, dict[str, Any]]:
    total = len(rows)
    outputs = [row.output for row in rows]
    parse_values = [row["parse_ok"] for row in outputs if row.get("parse_ok") is not None]
    metrics = {
        "num_samples": metric_value(total, num_samples=total),
        "empty_prediction_rate": metric_value(
            sum(not str(row.get("prediction", "")).strip() for row in outputs) / total
            if total
            else None,
            num_samples=total,
        ),
        "average_generation_character_length": metric_value(
            mean(len(str(row.get("prediction", ""))) for row in outputs),
            num_samples=total,
        ),
    }
    metrics["parse_success_rate"] = metric_value(
        mean(parse_values),
        num_samples=len(parse_values),
        status="ok" if parse_values else "not_available",
        note="Unimplemented protocols are excluded from this denominator.",
    )
    return metrics


def _group_summary(rows: list[EvaluatedRow]) -> dict[str, Any]:
    metrics = _common_metrics(rows)
    kinds = {row.kind for row in rows}
    status = "ok"
    if kinds == {"unimplemented"}:
        status = "registered_not_implemented"
    elif len(kinds) > 1:
        status = "mixed_protocol_kinds"

    if kinds == {"visual_grounding"}:
        samples = [row.output["sample_metrics"] for row in rows]
        total = len(samples)
        repository_ious = [sample["iou"] for sample in samples if sample.get("valid_coordinate")]
        metrics.update(
            {
                "valid_json_rate": metric_value(
                    mean(sample["valid_json"] for sample in samples),
                    num_samples=total,
                    note=(
                        "Legacy name: this means JSON object extraction. Prefer "
                        "json_object_extraction_rate."
                    ),
                ),
                "json_object_extraction_rate": metric_value(
                    mean(sample["valid_json"] for sample in samples), num_samples=total
                ),
                "grounding_parse_success_rate": metric_value(
                    mean(sample["parse_success"] for sample in samples), num_samples=total
                ),
                "repository_valid_detection_rate": metric_value(
                    mean(sample["parse_success"] for sample in samples),
                    num_samples=total,
                    note="Compatible with sat-rs-vlm repository valid_json_rate.",
                ),
                "valid_coordinate_rate": metric_value(
                    mean(sample["valid_coordinate"] for sample in samples), num_samples=total
                ),
                "label_exact_match_rate": metric_value(
                    mean(sample["label_match"] for sample in samples), num_samples=total
                ),
                "continuous_mean_iou": metric_value(
                    mean(sample["iou"] for sample in samples),
                    num_samples=total,
                    note="Strict internal profile: invalid predictions contribute IoU=0.",
                ),
                "continuous_mean_generalized_iou": metric_value(
                    mean(sample["generalized_iou"] for sample in samples),
                    num_samples=total,
                    note="Invalid predictions contribute GIoU=-1 in the strict internal profile.",
                ),
                "mean_normalized_center_distance": metric_value(
                    mean(sample["normalized_center_distance"] for sample in samples),
                    num_samples=total,
                    note=(
                        "Lower is better; distance is divided by the unit-square diagonal. "
                        "Invalid predictions contribute 1."
                    ),
                ),
                "repository_mean_iou_on_valid": metric_value(
                    mean(repository_ious),
                    num_samples=len(repository_ious),
                    status="ok" if repository_ious else "not_available",
                    note=(
                        "Repository-compatible profile: invalid-coordinate predictions "
                        "are excluded from this mean."
                    ),
                ),
                "continuous_acc_at_0_5": metric_value(
                    mean(sample["correct_at_0_5"] for sample in samples), num_samples=total
                ),
                "continuous_acc_at_0_7": metric_value(
                    mean(sample["correct_at_0_7"] for sample in samples), num_samples=total
                ),
                "label_and_iou_accuracy_at_0_5": metric_value(
                    mean(sample["label_and_iou_correct_at_0_5"] for sample in samples),
                    num_samples=total,
                ),
            }
        )
    elif kinds == {"counting"}:
        samples = [row.output["sample_metrics"] for row in rows]
        total = len(samples)
        absolute_errors = [
            float(sample["absolute_error"])
            for sample in samples
            if sample.get("absolute_error") is not None
        ]
        squared_errors = [
            float(sample["squared_error"])
            for sample in samples
            if sample.get("squared_error") is not None
        ]
        signed_errors = [
            float(sample["signed_error"])
            for sample in samples
            if sample.get("signed_error") is not None
        ]
        percentage_errors = [
            float(sample["absolute_percentage_error"])
            for sample in samples
            if sample.get("absolute_percentage_error") is not None
        ]
        parsed = len(absolute_errors)
        mean_squared_error = mean(squared_errors)
        metrics.update(
            {
                "number_parse_success_rate": metric_value(
                    mean(sample["number_parse_success"] for sample in samples), num_samples=total
                ),
                "exact_count_accuracy": metric_value(
                    mean(sample["exact_count_correct"] for sample in samples), num_samples=total
                ),
                "accuracy_within_1": metric_value(
                    mean(sample["within_1_correct"] for sample in samples), num_samples=total
                ),
                "mae_on_parsed": metric_value(
                    mean(absolute_errors),
                    num_samples=parsed,
                    status="ok" if parsed else "not_available",
                ),
                "rmse_on_parsed": metric_value(
                    math.sqrt(mean_squared_error) if mean_squared_error is not None else None,
                    num_samples=parsed,
                    status="ok" if parsed else "not_available",
                ),
                "mean_signed_error_on_parsed": metric_value(
                    mean(signed_errors),
                    num_samples=len(signed_errors),
                    status="ok" if signed_errors else "not_available",
                    note="Positive means systematic over-counting; negative means under-counting.",
                ),
                "mape_on_parsed_nonzero_reference": metric_value(
                    mean(percentage_errors),
                    num_samples=len(percentage_errors),
                    status="ok" if percentage_errors else "not_available",
                    note="References equal to zero are excluded.",
                ),
            }
        )
    elif kinds == {"text"}:
        samples = [row.output["sample_metrics"] for row in rows]
        total = len(samples)
        metrics.update(
            {
                "exact_match": metric_value(
                    mean(sample["exact_match"] for sample in samples), num_samples=total
                ),
                "micro_normalized_accuracy": metric_value(
                    mean(sample["normalized_exact_match"] for sample in samples),
                    num_samples=total,
                ),
                "keyword_hit_diagnostic": metric_value(
                    mean(sample["keyword_hit"] for sample in samples),
                    num_samples=total,
                    note="Diagnostic only; not a primary accuracy metric.",
                ),
                "token_f1": metric_value(
                    mean(sample["token_f1"] for sample in samples),
                    num_samples=total,
                    note=(
                        "Internal lexical-overlap diagnostic; not official VRSBench "
                        "semantic accuracy."
                    ),
                ),
                "normalized_edit_similarity": metric_value(
                    mean(sample["normalized_edit_similarity"] for sample in samples),
                    num_samples=total,
                    note="Internal character-edit similarity after repository text normalization.",
                ),
            }
        )
        typed: dict[str, list[bool]] = defaultdict(list)
        for row in rows:
            qa_type = str(row.output.get("metadata", {}).get("qa_type", "")).strip()
            if qa_type:
                typed[qa_type].append(bool(row.output["sample_metrics"]["normalized_exact_match"]))
        if typed:
            type_scores = [float(mean(values) or 0.0) for values in typed.values()]
            metrics["macro_qa_type_accuracy"] = metric_value(
                mean(type_scores), num_samples=len(type_scores)
            )
    elif kinds == {"caption"}:
        samples = [row.output["sample_metrics"] for row in rows]
        total = len(samples)
        predictions = [str(row.output.get("prediction", "")) for row in rows]
        references = [str(row.output.get("reference", "")) for row in rows]
        for key in (
            "bleu_1_approx",
            "bleu_2_approx",
            "bleu_3_approx",
            "bleu_4_approx",
            "rouge_l_f1_approx",
            "meteor_exact_approx",
            "chrf_approx",
            "cider_d_single_reference_approx",
            "prediction_token_count",
            "reference_token_count",
            "length_ratio",
        ):
            values = [sample.get(key) for sample in samples]
            available = sum(value is not None for value in values)
            metrics[f"average_{key}" if key.endswith("token_count") else key] = metric_value(
                mean(values),
                num_samples=available,
                status="ok" if available else "not_available",
                note=(
                    "Lightweight repository-compatible approximation; not an official score."
                    if "approx" in key
                    else None
                ),
            )
        for order in range(1, 5):
            metrics[f"corpus_bleu_{order}_single_reference_approx"] = metric_value(
                corpus_bleu_single_reference_approx(predictions, references, order),
                num_samples=total,
                note=(
                    "Corpus-level clipped BLEU with brevity penalty and the local tokenizer; "
                    "single-reference internal profile, not an official score."
                ),
            )
    elif kinds == {"change_caption"}:
        samples = [row.output["sample_metrics"] for row in rows]
        total = len(samples)
        valid_flags = sum(bool(sample.get("changeflag_valid")) for sample in samples)
        parsed = sum(bool(sample.get("binary_parse_success")) for sample in samples)
        tp = tn = fp = fn = 0
        comparable = 0
        positive_samples: list[dict[str, Any]] = []
        positive_rows: list[EvaluatedRow] = []
        for row in rows:
            reference_flag = row.output.get("reference_changeflag")
            predicted_flag = row.output.get("predicted_changeflag")
            if reference_flag == 1:
                positive_samples.append(row.output["sample_metrics"])
                positive_rows.append(row)
            if reference_flag not in {0, 1} or predicted_flag not in {0, 1}:
                continue
            comparable += 1
            if reference_flag == 1 and predicted_flag == 1:
                tp += 1
            elif reference_flag == 0 and predicted_flag == 0:
                tn += 1
            elif reference_flag == 0 and predicted_flag == 1:
                fp += 1
            else:
                fn += 1

        def ratio(numerator: int, denominator: int) -> float | None:
            return numerator / denominator if denominator else None

        accuracy = ratio(tp + tn, comparable)
        change_precision = ratio(tp, tp + fp)
        change_recall = ratio(tp, tp + fn)
        specificity = ratio(tn, tn + fp)
        f1_denominator = 2 * tp + fp + fn
        change_f1 = ratio(2 * tp, f1_denominator)
        balanced_accuracy = (
            (change_recall + specificity) / 2
            if change_recall is not None and specificity is not None
            else None
        )
        negative_predictive_value = ratio(tn, tn + fn)
        mcc_denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        matthews_correlation = (tp * tn - fp * fn) / mcc_denominator if mcc_denominator else None
        expected_accuracy = (
            ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (comparable * comparable)
            if comparable
            else None
        )
        cohen_kappa = (
            (accuracy - expected_accuracy) / (1 - expected_accuracy)
            if accuracy is not None and expected_accuracy is not None and expected_accuracy != 1
            else None
        )
        metrics.update(
            {
                "changeflag_valid_rate": metric_value(ratio(valid_flags, total), num_samples=total),
                "binary_parse_success_rate": metric_value(ratio(parsed, total), num_samples=total),
                "binary_accuracy": metric_value(
                    accuracy,
                    num_samples=comparable,
                    status="ok" if comparable else "not_available",
                ),
                "balanced_accuracy": metric_value(
                    balanced_accuracy,
                    num_samples=comparable,
                    status="ok" if balanced_accuracy is not None else "not_available",
                ),
                "change_precision": metric_value(
                    change_precision,
                    num_samples=tp + fp,
                    status="ok" if change_precision is not None else "not_available",
                ),
                "change_recall": metric_value(
                    change_recall,
                    num_samples=tp + fn,
                    status="ok" if change_recall is not None else "not_available",
                ),
                "change_f1": metric_value(
                    change_f1,
                    num_samples=comparable,
                    status="ok" if change_f1 is not None else "not_available",
                ),
                "no_change_recall_specificity": metric_value(
                    specificity,
                    num_samples=tn + fp,
                    status="ok" if specificity is not None else "not_available",
                ),
                "negative_predictive_value": metric_value(
                    negative_predictive_value,
                    num_samples=tn + fn,
                    status="ok" if negative_predictive_value is not None else "not_available",
                ),
                "matthews_correlation_coefficient": metric_value(
                    matthews_correlation,
                    num_samples=comparable,
                    status="ok" if matthews_correlation is not None else "not_available",
                ),
                "cohen_kappa": metric_value(
                    cohen_kappa,
                    num_samples=comparable,
                    status="ok" if cohen_kappa is not None else "not_available",
                ),
                "change_prevalence": metric_value(
                    ratio(tp + fn, comparable), num_samples=comparable
                ),
                "predicted_change_rate": metric_value(
                    ratio(tp + fp, comparable), num_samples=comparable
                ),
                "false_positive_rate": metric_value(
                    ratio(fp, fp + tn),
                    num_samples=fp + tn,
                    status="ok" if fp + tn else "not_available",
                ),
                "false_negative_rate": metric_value(
                    ratio(fn, fn + tp),
                    num_samples=fn + tp,
                    status="ok" if fn + tp else "not_available",
                ),
                "true_positives": metric_value(tp, num_samples=comparable),
                "true_negatives": metric_value(tn, num_samples=comparable),
                "false_positives": metric_value(fp, num_samples=comparable),
                "false_negatives": metric_value(fn, num_samples=comparable),
            }
        )
        for key in (
            "bleu_1_approx",
            "bleu_2_approx",
            "bleu_3_approx",
            "bleu_4_approx",
            "rouge_l_f1_approx",
            "meteor_exact_approx",
            "chrf_approx",
            "cider_d_single_reference_approx",
        ):
            all_values = [sample.get(key) for sample in samples]
            positive_values = [sample.get(key) for sample in positive_samples]
            metrics[key] = metric_value(
                mean(all_values),
                num_samples=sum(value is not None for value in all_values),
                note="Lightweight approximation; not an official LEVIR-CC score.",
            )
            metrics[f"positive_change_{key}"] = metric_value(
                mean(positive_values),
                num_samples=sum(value is not None for value in positive_values),
                status="ok" if positive_values else "not_available",
                note="Computed only where reference metadata.changeflag=1.",
            )
        predictions = [str(row.output.get("prediction", "")) for row in rows]
        references = [str(row.output.get("reference", "")) for row in rows]
        positive_predictions = [str(row.output.get("prediction", "")) for row in positive_rows]
        positive_references = [str(row.output.get("reference", "")) for row in positive_rows]
        for order in range(1, 5):
            key = f"corpus_bleu_{order}_single_reference_approx"
            metrics[key] = metric_value(
                corpus_bleu_single_reference_approx(predictions, references, order),
                num_samples=total,
                note="Single-reference internal corpus BLEU; not an official LEVIR-CC score.",
            )
            metrics[f"positive_change_{key}"] = metric_value(
                corpus_bleu_single_reference_approx(
                    positive_predictions, positive_references, order
                )
                if positive_rows
                else None,
                num_samples=len(positive_rows),
                status="ok" if positive_rows else "not_available",
                note="Computed only where metadata.changeflag=1.",
            )
    return {"status": status, "metrics": metrics}


def _extend_corpus_caption_metrics(rows: list[EvaluatedRow]) -> None:
    """Attach corpus-IDF-dependent CIDEr-D approximations before aggregation."""

    grouped: dict[str, list[EvaluatedRow]] = defaultdict(list)
    for row in rows:
        if row.kind in {"caption", "change_caption"}:
            grouped[row.protocol].append(row)
    for protocol_rows in grouped.values():
        predictions = [str(row.output.get("prediction", "")) for row in protocol_rows]
        references = [str(row.output.get("reference", "")) for row in protocol_rows]
        scores = cider_d_single_reference_approx_scores(predictions, references)
        for row, score in zip(protocol_rows, scores, strict=True):
            row.output["sample_metrics"]["cider_d_single_reference_approx"] = score


def _repository_native_score(row: EvaluatedRow) -> dict[str, float | bool | None]:
    """委托现有 v2 helper 生成兼容诊断，避免维护第二份同义评分代码。"""

    output = row.output
    return cast(
        dict[str, float | bool | None],
        score_sample(
            str(output.get("task_type", "unknown")),
            str(output.get("prediction", "")),
            str(output.get("reference", "")),
            change_detection_as_text=True,
        ),
    )


def _repository_native_summary(
    rows: list[EvaluatedRow],
    upstream: dict[str, Any],
    latency_context: LatencyContext,
) -> dict[str, Any]:
    """Build a side-by-side summary compatible with repository v2_task_metrics."""

    grouped: dict[str, list[EvaluatedRow]] = defaultdict(list)
    for row in rows:
        grouped[str(row.output.get("task_type", "unknown"))].append(row)
    empty = sum(not str(row.output.get("prediction", "")).strip() for row in rows)
    latencies = [
        float(row.output["inference_latency_ms"])
        for row in rows
        if row.output.get("inference_latency_ms") is not None
    ]
    by_task: dict[str, Any] = {}
    metric_names = {
        "valid_json": "valid_json_rate",
        "parsable": "parsable_rate",
        "iou": "mean_iou",
    }
    for task, task_rows in sorted(grouped.items()):
        scores = [_repository_native_score(row) for row in task_rows]
        task_metrics: dict[str, Any] = {
            "num_samples": len(task_rows),
            "empty_prediction_rate": sum(
                not str(row.output.get("prediction", "")).strip() for row in task_rows
            )
            / len(task_rows),
            "average_generation_length": mean(
                len(str(row.output.get("prediction", ""))) for row in task_rows
            ),
        }
        keys = sorted({key for score in scores for key in score})
        for key in keys:
            numeric = [
                value
                for value in (score.get(key) for score in scores)
                if isinstance(value, (int, float, bool))
            ]
            task_metrics[metric_names.get(key, key)] = mean(numeric)
        by_task[task] = task_metrics
    return {
        "profile": "repository_native_v2",
        "schema_version": "1.0",
        "metrics_version": "v2_task_metrics",
        "upstream_repository": upstream,
        "latency_context": latency_context.to_dict(),
        "overall": {
            "num_samples": len(rows),
            "empty_predictions": empty,
            "empty_prediction_rate": empty / len(rows) if rows else None,
            "inference_latency_ms": mean(latencies),
        },
        "by_task": by_task,
        "note": (
            "Metric aggregation preserves the repository-native v2 compatibility profile; "
            "strict internal and semantic profiles remain separate."
        ),
    }


def _build_summary(
    rows: list[EvaluatedRow],
    *,
    contract: dict[str, Any],
    input_errors: list[dict[str, Any]],
    warnings: list[str],
    semantic_summary: dict[str, Any] | None,
    latency_context: LatencyContext,
) -> dict[str, Any]:
    by_task_rows: dict[str, list[EvaluatedRow]] = defaultdict(list)
    by_protocol_rows: dict[str, list[EvaluatedRow]] = defaultdict(list)
    for row in rows:
        by_task_rows[str(row.output.get("task_type", "unknown"))].append(row)
        by_protocol_rows[row.protocol].append(row)

    latencies = [
        float(row.output["inference_latency_ms"])
        for row in rows
        if row.output.get("inference_latency_ms") is not None
    ]
    latency = latency_statistics(latencies)
    overall_metrics = _common_metrics(rows)
    latency_samples = len(latencies)
    for name in ("mean", "p50", "p95", "min", "max"):
        overall_metrics[f"latency_ms_{name}"] = metric_value(
            latency[name],
            num_samples=latency_samples,
            status="ok" if latency_samples else "not_available",
            note=latency_context.note,
        )
    overall_metrics["latency_sample_count"] = metric_value(latency_samples, num_samples=len(rows))

    by_qa_type_rows: dict[str, list[EvaluatedRow]] = defaultdict(list)
    for row in rows:
        if row.kind != "text":
            continue
        qa_type = str(row.output.get("metadata", {}).get("qa_type", "")).strip()
        if qa_type:
            by_qa_type_rows[qa_type].append(row)

    summary = {
        "schema_version": "1.5",
        "implementation_version": str(contract["implementation_version"]),
        "contract_version": str(contract["contract_version"]),
        "contract_status": str(contract.get("contract_status", "unknown")),
        "overall": {
            "metrics": overall_metrics,
            "latency_context": latency_context.to_dict(),
            "task_distribution": dict(
                sorted(Counter(str(row.output.get("task_type", "unknown")) for row in rows).items())
            ),
            "protocol_distribution": dict(sorted(Counter(row.protocol for row in rows).items())),
        },
        "by_task": {
            name: _group_summary(task_rows) for name, task_rows in sorted(by_task_rows.items())
        },
        "by_protocol": {
            name: _group_summary(protocol_rows)
            for name, protocol_rows in sorted(by_protocol_rows.items())
        },
        "by_qa_type": {
            name: _group_summary(type_rows) for name, type_rows in sorted(by_qa_type_rows.items())
        },
        "unimplemented_protocols": sorted(
            {
                row.protocol
                for row in rows
                if row.kind == "unimplemented" or "not_implemented" in row.protocol_status
            }
        ),
        "input_errors": input_errors,
        "warnings": sorted(set(warnings)),
        "repository_compatibility": _repository_native_summary(
            rows,
            dict(contract.get("upstream_repository", {})),
            latency_context,
        ),
    }
    if semantic_summary is not None:
        summary["semantic"] = semantic_summary
    grounding_rows = [row for row in rows if row.kind == "visual_grounding"]
    text_rows = [row for row in rows if row.kind == "text"]
    caption_rows = [row for row in rows if row.kind in {"caption", "change_caption"}]

    def coverage(selected: list[EvaluatedRow], predicate: Any, note: str) -> dict[str, Any]:
        available = sum(bool(predicate(row.output)) for row in selected)
        return metric_value(
            available / len(selected) if selected else None,
            num_samples=len(selected),
            status="ok" if selected and available == len(selected) else "data_insufficient",
            note=note,
        )

    summary["p0_data_availability"] = {
        "grounding_is_unique_coverage": coverage(
            grounding_rows,
            lambda output: isinstance(output.get("metadata", {}).get("is_unique"), bool),
            "Required for official Unique/Non-Unique Grounding stratification.",
        ),
        "vqa_question_coverage": coverage(
            text_rows,
            lambda output: bool(str(output.get("metadata", {}).get("question", "")).strip()),
            "Required for question-aware semantic judging such as the VRSBench judge profile.",
        ),
        "caption_multi_reference_coverage": coverage(
            caption_rows,
            lambda output: (
                isinstance(output.get("references"), list) and len(output.get("references", [])) > 1
            ),
            "Required for paper-comparable multi-reference caption metrics, especially LEVIR-CC.",
        ),
        "resource_benchmark": {
            "value": None,
            "label": "internal",
            "status": "not_available_from_predictions",
            "num_samples": 0,
            "note": (
                "Parameters, file size, peak VRAM and throughput require benchmark_report.json; "
                "they are not inferred from predictions.jsonl."
            ),
        },
    }
    return summary


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def validate_output_directory(
    output_dir: Path,
    protected_repository: Path | None = None,
) -> None:
    output = output_dir.resolve()
    protected = protected_repository.resolve() if protected_repository is not None else None
    if protected is not None and _is_within(output, protected):
        raise EvaluationError(
            f"output directory must not be inside protected repository: {protected}"
        )
    if output.exists():
        if not output.is_dir():
            raise EvaluationError(f"output path exists and is not a directory: {output}")
        if any(output.iterdir()):
            raise EvaluationError(f"output directory already exists and is not empty: {output}")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def run_evaluation(
    predictions_path: str | Path,
    output_dir: str | Path,
    *,
    contract_path: str | Path,
    manifest_path: str | Path | None = None,
    strict: bool = True,
    protected_repository: str | Path | None = None,
    semantic_enabled: bool = True,
    semantic_contract_path: str | Path | None = None,
    semantic_ontology_path: str | Path | None = None,
    latency_semantics: str = "unresolved",
    eval_batch_size: int | None = None,
    group_by_task: bool | None = None,
    evaluation_tier: str | None = None,
    evaluation_tier_sha256: str | None = None,
) -> dict[str, Path]:
    """只读 Prediction JSONL，并将全部产物写到受保护仓库之外。"""

    predictions = Path(predictions_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    contract_file = Path(contract_path).expanduser().resolve()
    manifest_file = Path(manifest_path).expanduser().resolve() if manifest_path else None
    protected = (
        Path(protected_repository).expanduser().resolve()
        if protected_repository is not None
        else None
    )
    semantic_contract_file = (
        Path(semantic_contract_path).expanduser().resolve()
        if semantic_contract_path
        else DEFAULT_SEMANTIC_CONTRACT
    )
    semantic_ontology_file = (
        Path(semantic_ontology_path).expanduser().resolve()
        if semantic_ontology_path
        else DEFAULT_SEMANTIC_ONTOLOGY
    )
    latency_context = resolve_latency_context(
        latency_semantics,
        eval_batch_size,
        group_by_task,
    )
    validate_output_directory(destination, protected)

    input_hash_before = _sha256(predictions)
    contract = load_contract(contract_file)
    semantic_evaluator: SemanticEvaluator | None = None
    if semantic_enabled:
        try:
            semantic_evaluator = SemanticEvaluator(
                semantic_contract_file,
                semantic_ontology_file,
            )
        except (OSError, ValueError) as exc:
            raise EvaluationError(f"failed to load semantic evaluation: {exc}") from exc
    manifest_format, warnings = manifest_coordinate_format(manifest_file)
    if latency_context.status != "resolved":
        warnings.append(latency_context.note)
    records, input_errors = read_prediction_jsonl(predictions, strict=strict)

    evaluated: list[EvaluatedRow] = []
    for record in records:
        try:
            evaluated_row, row_warnings = evaluate_record(
                record,
                contract,
                manifest_format,
                strict=strict,
            )
        except InputValidationError as exc:
            if strict:
                raise
            input_errors.append(
                {
                    "line_number": record.line_number,
                    "id": record.id,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            continue
        if semantic_evaluator is not None:
            semantic_evaluator.extend_output(evaluated_row.output)
        evaluated.append(evaluated_row)
        warnings.extend(row_warnings)
    if not evaluated:
        raise EvaluationError("no records remained after evaluation")

    input_hash_after = _sha256(predictions)
    if input_hash_before != input_hash_after:
        raise EvaluationError("predictions file changed while it was being evaluated")

    _extend_corpus_caption_metrics(evaluated)
    evaluated_outputs = [row.output for row in evaluated]
    semantic_summary = (
        semantic_evaluator.build_summary(evaluated_outputs)
        if semantic_evaluator is not None
        else None
    )
    summary = _build_summary(
        evaluated,
        contract=contract,
        input_errors=input_errors,
        warnings=warnings,
        semantic_summary=semantic_summary,
        latency_context=latency_context,
    )
    if evaluation_tier is not None:
        summary["evaluation_tier"] = evaluation_tier
        summary["evaluation_tier_sha256"] = evaluation_tier_sha256
    outputs = {
        "evaluated_predictions": destination / "evaluated_predictions.jsonl",
        "metrics": destination / "metrics.json",
        "summary": destination / "summary.json",
        "manifest": destination / "evaluation_manifest.json",
    }
    manifest = {
        "schema_version": "1.5",
        "implementation_version": str(contract["implementation_version"]),
        "contract_version": str(contract["contract_version"]),
        "upstream_repository": dict(contract.get("upstream_repository", {})),
        "run_time_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "input_file": str(predictions),
        "input_sha256": input_hash_before,
        "input_samples": len(records),
        "evaluated_samples": len(evaluated),
        "contract_file": str(contract_file),
        "contract_sha256": _sha256(contract_file),
        "dataset_manifest": str(manifest_file) if manifest_file else None,
        "evaluation_tier": evaluation_tier,
        "evaluation_tier_sha256": evaluation_tier_sha256,
        "semantic_evaluation_enabled": semantic_evaluator is not None,
        "semantic_contract_file": (
            str(semantic_contract_file) if semantic_evaluator is not None else None
        ),
        "semantic_contract_sha256": (
            _sha256(semantic_contract_file) if semantic_evaluator is not None else None
        ),
        "semantic_ontology_file": (
            str(semantic_ontology_file) if semantic_evaluator is not None else None
        ),
        "semantic_ontology_sha256": (
            _sha256(semantic_ontology_file) if semantic_evaluator is not None else None
        ),
        "strict": strict,
        "latency_context": latency_context.to_dict(),
        "repository_compatibility_profile": "repository_native_v2",
        "output_files": {name: str(path) for name, path in outputs.items()},
        "remote_write_performed": False,
        "protected_repository": str(protected) if protected is not None else None,
    }

    destination.mkdir(parents=True, exist_ok=True)
    _write_jsonl(outputs["evaluated_predictions"], evaluated_outputs)
    _write_json(outputs["metrics"], summary)
    _write_json(outputs["summary"], summary)
    _write_json(outputs["manifest"], manifest)
    return outputs
