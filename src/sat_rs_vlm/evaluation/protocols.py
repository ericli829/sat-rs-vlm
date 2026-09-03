"""根据当前 sat-rs-vlm 字段派生评测协议。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from sat_rs_vlm.evaluation.records import EvaluationError, PredictionRecord


@dataclass(frozen=True)
class ProtocolResolution:
    name: str
    kind: str
    status: str
    metric_profile: str | None
    reason: str
    metric_label: str
    provenance: dict[str, Any]


def _normalized_dataset(value: Any) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _metadata_text(record: PredictionRecord, *keys: str) -> str:
    for key in keys:
        value = record.metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def load_contract(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise EvaluationError(f"evaluation contract does not exist: {path}")
    try:
        text = path.read_text(encoding="utf-8")
        payload = (
            yaml.safe_load(text) if path.suffix.lower() in {".yaml", ".yml"} else json.loads(text)
        )
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise EvaluationError(f"invalid evaluation contract: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("protocols"), dict):
        raise EvaluationError("evaluation contract must contain a protocols object")
    if not str(payload.get("contract_version", "")).strip():
        raise EvaluationError("evaluation contract is missing contract_version")
    return payload


def _protocol_name(record: PredictionRecord) -> tuple[str, str]:
    explicit = record.raw.get("eval_protocol")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip(), "explicit eval_protocol"

    dataset = _normalized_dataset(record.metadata.get("dataset", ""))
    source_task = str(record.metadata.get("source_task", "")).strip().lower()
    task = record.task_type
    if dataset == "levircc" and task == "change_detection":
        return "levir_cc_change_caption", "LEVIR-CC change detection caption"
    if dataset in {"vrsbench", "rsbench"}:
        if task == "detection" and source_task == "referring":
            return "vrsbench_visual_grounding", "VRSBench detection + referring"
        if task == "counting":
            return "vrsbench_counting", "VRSBench counting task"
        if task == "captioning":
            return "vrsbench_detailed_caption", "VRSBench captioning task"
        if task in {"vqa", "scene_classification"}:
            return "vrsbench_open_vqa", "VRSBench text question task"

    if dataset in {
        "mmerealworldrs",
        "mmerealworldremotesensing",
    }:
        return "mme_realworld_rs_mcq", "MME-RealWorld-RS dataset"
    if dataset in {"mmerealworld", "mmerealworldcn"}:
        subtask = _metadata_text(record, "official_subtask", "subtask")
        if _normalized_dataset(subtask) == "remotesensing":
            return "mme_realworld_rs_mcq", "MME-RealWorld Remote Sensing subtask"

    if dataset in {"xlrs", "xlrsbench", "xlrsbenchlite"}:
        category = _metadata_text(record, "official_category", "category")
        normalized_category = _normalized_dataset(category)
        if task in {"detection", "visual_grounding"}:
            return "xlrs_visual_grounding", "XLRS visual grounding task"
        if task == "captioning":
            return "xlrs_caption", "XLRS detailed caption task"
        if normalized_category == "landuseclassificationoveralllanduseclassification":
            return "xlrs_vqa_multiselect", "XLRS Overall Land Use Classification"
        if task in {"vqa", "scene_classification", "counting"}:
            return "xlrs_vqa_mcq", "XLRS VQA task"

    generic = {
        "detection": "generic_single_target_grounding_internal",
        "counting": "generic_counting_internal",
        "vqa": "generic_text_internal",
        "scene_classification": "generic_text_internal",
        "captioning": "generic_captioning_internal",
        "change_detection": "generic_change_captioning_internal",
        "segmentation": "segmentation_not_implemented",
    }
    return generic.get(task, "unknown_not_implemented"), "generic task_type fallback"


def resolve_protocol(
    record: PredictionRecord,
    contract: dict[str, Any],
) -> ProtocolResolution:
    name, reason = _protocol_name(record)
    spec = contract["protocols"].get(name)
    if not isinstance(spec, dict):
        return ProtocolResolution(
            name=name,
            kind="unimplemented",
            status="protocol_unresolved",
            metric_profile=None,
            reason=f"{reason}; protocol is absent from contract",
            metric_label="internal",
            provenance={},
        )
    return ProtocolResolution(
        name=name,
        kind=str(spec.get("kind", "unimplemented")),
        status=str(spec.get("status", "registered_not_implemented")),
        metric_profile=(
            str(spec["metric_profile"]) if spec.get("metric_profile") is not None else None
        ),
        reason=reason,
        metric_label=str(
            spec.get("metric_label", contract.get("default_metric_label", "internal"))
        ),
        provenance=(
            dict(spec["official_protocol"])
            if isinstance(spec.get("official_protocol"), dict)
            else {}
        ),
    )


def manifest_coordinate_format(path: Path | None) -> tuple[str | None, list[str]]:
    """把当前仓库 manifest.coordinate_range 转成显式坐标格式。"""

    if path is None:
        return None, []
    if not path.is_file():
        raise EvaluationError(f"dataset manifest does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"invalid dataset manifest JSON: {exc}") from exc
    coordinate_range = payload.get("coordinate_range")
    if coordinate_range == [0, 1] or coordinate_range == [0.0, 1.0]:
        return "normalized_0_1", []
    if coordinate_range == [0, 100] or coordinate_range == [0.0, 100.0]:
        return "percent_0_100", []
    if coordinate_range == [0, 1000] or coordinate_range == [0.0, 1000.0]:
        return "scaled_0_1000", []
    return None, [f"unsupported manifest coordinate_range: {coordinate_range}"]


def coordinate_format_for_record(
    record: PredictionRecord,
    manifest_format: str | None,
) -> tuple[str | None, list[str]]:
    warnings: list[str] = []
    metadata_format = record.metadata.get("bbox_target_format")
    if isinstance(metadata_format, str) and metadata_format.strip():
        normalized = metadata_format.strip()
        if manifest_format and normalized != manifest_format:
            warnings.append(
                f"sample {record.id}: bbox_target_format={normalized} differs from "
                f"manifest={manifest_format}; metadata takes precedence"
            )
        return normalized, warnings
    return manifest_format, warnings
