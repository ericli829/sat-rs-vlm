"""Deterministic E1/E2/E3 evaluation-tier construction.

The module does not evaluate a model. It validates the evaluation population,
derives task-aware strata, freezes nested JSONL assets, and records enough
distribution and checksum information to audit sampling and prevent leakage.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

from sat_rs_vlm.data.qwen3vl_dataset import sample_to_messages
from sat_rs_vlm.data.prompt_templates import (
    CAPTION_INSTRUCTION,
    CHANGE_DESCRIPTION_INSTRUCTION,
    strengthen_instruction,
)
from sat_rs_vlm.data.task_protocol import counting_json, parse_count, parse_detection
from sat_rs_vlm.data.vrsbench import counting_instruction, scene_instruction
from sat_rs_vlm.evaluation.config import (
    EvaluationTierBuildConfig,
    EvaluationTierSourceConfig,
)
from sat_rs_vlm.training.config import BBoxAreaThresholdConfig
from sat_rs_vlm.training.data_statistics import bbox_area_bucket
from sat_rs_vlm.utils.jsonl import read_jsonl

TIER_FILES = {
    "E1": "e1_quick.jsonl",
    "E2": "e2_standard.jsonl",
    "E3": "e3_full.jsonl",
}
SUPPORTED_TASKS = {
    "captioning",
    "detection",
    "counting",
    "scene_classification",
    "vqa",
    "change_detection",
}


class EvaluationTierError(ValueError):
    """Raised when tier inputs violate identity, schema, or leakage invariants."""


@dataclass(frozen=True)
class TierSample:
    """A normalized sample plus its deterministic sampling dimensions."""

    row: dict[str, Any]
    dataset: str
    task: str
    subtype: str
    detection_size: str | None = None
    detection_class: str | None = None
    count_bucket: str | None = None
    qa_type: str | None = None
    changeflag: str | None = None

    @property
    def sample_id(self) -> str:
        return str(self.row["id"])

    @property
    def stratum(self) -> tuple[str, str, str]:
        return (self.dataset, self.task, self.subtype)


def file_sha256(path: str | Path) -> str:
    """Return the SHA256 of a file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assistant_text(messages: Sequence[Mapping[str, Any]]) -> str:
    for message in reversed(messages):
        if str(message.get("role", "")).lower() != "assistant":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            return " ".join(
                str(item.get("text", "")).strip()
                for item in content
                if isinstance(item, Mapping) and item.get("type") == "text"
            ).strip()
    return ""


def _portable_image_path(value: str, prefix: str) -> str:
    """Convert source-specific absolute paths into a shared DATA_ROOT-relative path."""

    normalized_prefix = prefix.replace("\\", "/").strip("/")
    raw = str(value).replace("\\", "/")
    parts = [part for part in raw.split("/") if part and not part.endswith(":")]
    lower = [part.lower() for part in parts]
    if "images" in lower:
        start = len(lower) - 1 - lower[::-1].index("images")
        relative_parts = parts[start:]
    elif PureWindowsPath(value).is_absolute() or Path(value).is_absolute():
        relative_parts = [parts[-1]] if parts else []
    else:
        relative_parts = parts
    relative = "/".join(relative_parts)
    if normalized_prefix and not relative.lower().startswith(normalized_prefix.lower() + "/"):
        return f"{normalized_prefix}/{relative}"
    return relative


def _normalize_messages(
    row: Mapping[str, Any],
    *,
    image_prefix: str,
) -> list[dict[str, Any]]:
    messages = sample_to_messages(dict(row))
    normalized: list[dict[str, Any]] = []
    image_count = 0
    for message in messages:
        copied = dict(message)
        content = copied.get("content")
        if isinstance(content, list):
            normalized_content: list[Any] = []
            for item in content:
                if isinstance(item, Mapping) and item.get("type") == "image":
                    image_item = dict(item)
                    image_item["image"] = _portable_image_path(
                        str(image_item.get("image", "")), image_prefix
                    )
                    image_count += 1
                    normalized_content.append(image_item)
                else:
                    normalized_content.append(item)
            copied["content"] = normalized_content
        normalized.append(copied)
    if image_count == 0:
        raise EvaluationTierError(f"Sample {row.get('id')} has no image content")
    return normalized


def _user_text(messages: Sequence[Mapping[str, Any]]) -> str:
    for message in messages:
        if str(message.get("role", "")).lower() != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            return " ".join(
                str(item.get("text", "")).strip()
                for item in content
                if isinstance(item, Mapping) and item.get("type") == "text"
            ).strip()
    return ""


def _replace_message_text(
    messages: list[dict[str, Any]],
    *,
    role: str,
    text: str,
) -> None:
    for message in messages:
        if message.get("role") != role:
            continue
        content = message.get("content")
        if role == "assistant" or isinstance(content, str):
            message["content"] = text
            return
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    item["text"] = text
                    return


def counting_bucket(value: int) -> str:
    """Bucket non-negative ground-truth counts while retaining dense-object cases."""

    if value < 0:
        raise EvaluationTierError("Counting ground truth must be non-negative")
    if value <= 4:
        return str(value)
    if value <= 9:
        return "5-9"
    return "10+"


def _build_tier_sample(
    row: Mapping[str, Any],
    source: EvaluationTierSourceConfig,
    thresholds: BBoxAreaThresholdConfig,
) -> TierSample:
    sample_id = str(row.get("id", "")).strip()
    task = str(row.get("task_type", "")).strip().lower()
    if not sample_id:
        raise EvaluationTierError("Evaluation sample has an empty id")
    if task not in SUPPORTED_TASKS:
        raise EvaluationTierError(f"Sample {sample_id} has unsupported task_type={task!r}")
    messages = _normalize_messages(row, image_prefix=source.image_prefix)
    answer = _assistant_text(messages)
    if not answer:
        raise EvaluationTierError(f"Sample {sample_id} has no assistant reference")
    metadata = dict(row.get("metadata", {}))
    dataset = str(metadata.get("dataset") or source.name).strip()
    if not dataset:
        raise EvaluationTierError(f"Sample {sample_id} has no dataset/source")
    metadata["dataset"] = dataset
    metadata.setdefault("training_source", dataset)
    normalized_row = {
        "id": sample_id,
        "messages": messages,
        "task_type": task,
        "metadata": metadata,
    }

    instruction = _user_text(messages)
    if task == "change_detection":
        _replace_message_text(messages, role="user", text=CHANGE_DESCRIPTION_INSTRUCTION)
    elif task == "captioning":
        _replace_message_text(messages, role="user", text=CAPTION_INSTRUCTION)
    elif task in {"detection", "counting", "scene_classification"}:
        _replace_message_text(
            messages,
            role="user",
            text=strengthen_instruction(task, instruction),
        )

    detection_size = detection_class = count_name = qa_type = changeflag = None
    if task == "detection":
        parsed = parse_detection(answer, coordinate_format="normalized_0_1")
        if parsed is None or not parsed.valid_coordinate_range:
            raise EvaluationTierError(f"Sample {sample_id} has invalid detection ground truth")
        x_min, y_min, x_max, y_max = parsed.bbox
        area = (x_max - x_min) * (y_max - y_min)
        detection_size = bbox_area_bucket(area, thresholds)
        detection_class = parsed.label
        metadata.setdefault("bbox_target_format", "normalized_0_1")
        subtype = f"bbox={detection_size}|class={detection_class}"
    elif task == "counting":
        parsed_count = parse_count(answer)
        # VRSBench contains qualitative quantity answers (for example
        # ``Multiple`` or ``Same number``) that the existing Evaluation v1.5
        # accepts but cannot score as numeric counting. Keep them in an explicit
        # diagnostic bucket instead of guessing a count or changing task labels.
        if parsed_count.value is None:
            # Reuse the established VRSBench conversion contract: qualitative
            # quantity answers are VQA, because strict counting evaluation needs
            # an integer reference. This is a protocol normalization, not text
            # based qa_type inference; metadata.qa_type remains authoritative.
            task = "vqa"
            qa_type = str(metadata.get("qa_type", "unknown")).strip().lower() or "unknown"
            metadata["counting_unresolved"] = True
            subtype = f"qa_type={qa_type}"
            normalized_row["task_type"] = task
        else:
            count_name = counting_bucket(parsed_count.value)
            structured_answer = counting_json(parsed_count.value)
            if structured_answer is None:
                raise EvaluationTierError(f"Sample {sample_id} count normalization failed")
            _replace_message_text(messages, role="assistant", text=structured_answer)
            _replace_message_text(
                messages,
                role="user",
                text=counting_instruction(instruction),
            )
            subtype = f"count={count_name}"
    elif task == "scene_classification":
        qa_type = str(metadata.get("qa_type", "unknown")).strip().lower() or "unknown"
        _replace_message_text(messages, role="user", text=scene_instruction(instruction))
        subtype = f"qa_type={qa_type}"
    elif task == "vqa":
        qa_type = str(metadata.get("qa_type", "unknown")).strip().lower() or "unknown"
        subtype = f"qa_type={qa_type}"
    elif task == "change_detection":
        raw_changeflag = metadata.get("changeflag")
        if raw_changeflag not in {0, 1, False, True}:
            raise EvaluationTierError(
                f"Sample {sample_id} has invalid metadata.changeflag={raw_changeflag!r}"
            )
        changeflag = str(int(raw_changeflag))
        subtype = f"changeflag={changeflag}"
    else:
        subtype = "all"
    return TierSample(
        row=normalized_row,
        dataset=dataset,
        task=task,
        subtype=subtype,
        detection_size=detection_size,
        detection_class=detection_class,
        count_bucket=count_name,
        qa_type=qa_type,
        changeflag=changeflag,
    )


def load_population(
    sources: Sequence[EvaluationTierSourceConfig],
    thresholds: BBoxAreaThresholdConfig,
    *,
    project_root: Path,
) -> tuple[list[TierSample], list[dict[str, Any]], set[str], list[dict[str, str]]]:
    """Load legal validation samples and reject duplicate or train-overlapping IDs."""

    population: list[TierSample] = []
    by_id: dict[str, TierSample] = {}
    source_records: list[dict[str, Any]] = []
    train_ids: set[str] = set()
    excluded_rows: list[dict[str, str]] = []
    for source in sources:
        eval_path = _resolve_path(source.eval_file, project_root)
        if not eval_path.is_file():
            raise FileNotFoundError(f"Evaluation source does not exist: {eval_path}")
        source_count = 0
        source_input_count = 0
        source_excluded = 0
        for row in read_jsonl(eval_path):
            source_input_count += 1
            raw_id = str(row.get("id", "")).strip()
            if raw_id and raw_id in by_id:
                raise EvaluationTierError(f"Duplicate evaluation id: {raw_id}")
            try:
                sample = _build_tier_sample(row, source, thresholds)
            except EvaluationTierError as exc:
                excluded_rows.append(
                    {
                        "id": raw_id or "<empty>",
                        "source": source.name,
                        "reason": str(exc),
                    }
                )
                source_excluded += 1
                continue
            if sample.sample_id in by_id:
                raise EvaluationTierError(f"Duplicate evaluation id: {sample.sample_id}")
            by_id[sample.sample_id] = sample
            population.append(sample)
            source_count += 1
        train_path: Path | None = None
        source_train_ids: set[str] = set()
        if source.train_file:
            train_path = _resolve_path(source.train_file, project_root)
            if not train_path.is_file():
                raise FileNotFoundError(f"Training source does not exist: {train_path}")
            for row in read_jsonl(train_path):
                sample_id = str(row.get("id", "")).strip()
                if sample_id:
                    source_train_ids.add(sample_id)
            train_ids.update(source_train_ids)
        source_records.append(
            {
                "name": source.name,
                "path": source.manifest_path or source.eval_file,
                "sha256": file_sha256(eval_path),
                "sample_count": source_count,
                "input_sample_count": source_input_count,
                "excluded_invalid_sample_count": source_excluded,
                "train_path": (
                    source.manifest_path.replace("val", "train")
                    if source.manifest_path and source.train_file
                    else source.train_file
                ),
                "train_sha256": file_sha256(train_path) if train_path else None,
                "train_sample_count": len(source_train_ids),
                "image_prefix": source.image_prefix,
            }
        )
    overlap = set(by_id) & train_ids
    if overlap:
        examples = sorted(overlap)[:10]
        raise EvaluationTierError(
            f"Training/evaluation ID leakage detected: count={len(overlap)}, examples={examples}"
        )
    return population, source_records, train_ids, excluded_rows


def _resolve_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root / path


def _stable_rank(sample_id: str, seed: int) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()
    return digest, sample_id


def _allocate_quotas(sizes: Mapping[tuple[str, str, str], int], target: int) -> dict[Any, int]:
    """Allocate sqrt-population quotas with one-per-stratum coverage where possible."""

    total = sum(sizes.values())
    target = min(target, total)
    keys = sorted(sizes)
    quotas = {key: 0 for key in keys}
    if target <= 0:
        return quotas
    if target >= len(keys):
        for key in keys:
            quotas[key] = 1
    else:
        ranked = sorted(keys, key=lambda key: (-sizes[key], key))
        for key in ranked[:target]:
            quotas[key] = 1
        return quotas
    remaining = target - sum(quotas.values())
    while remaining:
        candidates = [key for key in keys if quotas[key] < sizes[key]]
        if not candidates:
            break
        weights = {key: math.sqrt(sizes[key]) for key in candidates}
        weight_sum = sum(weights.values())
        ideal = {key: remaining * weights[key] / weight_sum for key in candidates}
        grants = {
            key: min(sizes[key] - quotas[key], int(math.floor(ideal[key])))
            for key in candidates
        }
        granted = sum(grants.values())
        if granted:
            for key, value in grants.items():
                quotas[key] += value
            remaining -= granted
            continue
        key = min(
            candidates,
            key=lambda item: (
                quotas[item] / math.sqrt(sizes[item]),
                -sizes[item],
                item,
            ),
        )
        quotas[key] += 1
        remaining -= 1
    return quotas


def stratified_select(
    population: Sequence[TierSample],
    target: int,
    *,
    seed: int,
    required_ids: Sequence[str] = (),
) -> list[TierSample]:
    """Select an exact nested tier while preserving required historical IDs."""

    if target < len(required_ids):
        raise EvaluationTierError(
            f"Target {target} is smaller than required fixed set ({len(required_ids)})"
        )
    by_id = {sample.sample_id: sample for sample in population}
    if len(by_id) != len(population):
        raise EvaluationTierError("Population IDs must be unique")
    missing = [sample_id for sample_id in required_ids if sample_id not in by_id]
    if missing:
        raise EvaluationTierError(
            f"Historical E1 IDs are missing from population: count={len(missing)}, "
            f"examples={missing[:10]}"
        )
    exact_target = min(target, len(population))
    selected_ids = set(required_ids)
    selected = [by_id[sample_id] for sample_id in required_ids]
    if len(selected) == exact_target:
        return selected
    groups: dict[tuple[str, str, str], list[TierSample]] = defaultdict(list)
    for sample in population:
        groups[sample.stratum].append(sample)
    for group in groups.values():
        group.sort(key=lambda sample: _stable_rank(sample.sample_id, seed))
    quotas = _allocate_quotas({key: len(value) for key, value in groups.items()}, exact_target)

    for key in sorted(groups):
        if len(selected) >= exact_target:
            break
        current = sum(1 for sample in selected if sample.stratum == key)
        need = min(max(0, quotas[key] - current), exact_target - len(selected))
        for sample in groups[key]:
            if need == 0:
                break
            if sample.sample_id in selected_ids:
                continue
            selected.append(sample)
            selected_ids.add(sample.sample_id)
            need -= 1
    if len(selected) < exact_target:
        candidates = sorted(
            (sample for sample in population if sample.sample_id not in selected_ids),
            key=lambda sample: _stable_rank(sample.sample_id, seed ^ 0xE2E3),
        )
        for sample in candidates[: exact_target - len(selected)]:
            selected.append(sample)
            selected_ids.add(sample.sample_id)
    if len(selected) != exact_target:
        raise EvaluationTierError(
            f"Unable to construct exact tier: expected={exact_target}, actual={len(selected)}"
        )
    return selected


def _counter(samples: Sequence[TierSample], getter: Any) -> dict[str, int]:
    return dict(sorted(Counter(str(getter(sample)) for sample in samples).items()))


def distribution(samples: Sequence[TierSample]) -> dict[str, Any]:
    """Return all report-visible dataset, task, and subtype distributions."""

    return {
        "dataset": _counter(samples, lambda sample: sample.dataset),
        "task": _counter(samples, lambda sample: sample.task),
        "dataset_task": _counter(
            samples, lambda sample: f"{sample.dataset}|{sample.task}"
        ),
        "stratum": _counter(samples, lambda sample: "|".join(sample.stratum)),
        "qa_type": _counter(
            [sample for sample in samples if sample.qa_type is not None],
            lambda sample: sample.qa_type,
        ),
        "detection_size": _counter(
            [sample for sample in samples if sample.detection_size is not None],
            lambda sample: sample.detection_size,
        ),
        "detection_class": _counter(
            [sample for sample in samples if sample.detection_class is not None],
            lambda sample: sample.detection_class,
        ),
        "count_bucket": _counter(
            [sample for sample in samples if sample.count_bucket is not None],
            lambda sample: sample.count_bucket,
        ),
        "levir_changeflag": _counter(
            [sample for sample in samples if sample.changeflag is not None],
            lambda sample: sample.changeflag,
        ),
    }


def _sampling_fractions(
    population_distribution: Mapping[str, Any],
    tier_distribution: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for dimension in ("dataset", "task", "dataset_task", "stratum"):
        population_counts = dict(population_distribution.get(dimension, {}))
        tier_counts = dict(tier_distribution.get(dimension, {}))
        result[dimension] = {
            key: {
                "population_count": count,
                "tier_count": int(tier_counts.get(key, 0)),
                "sampling_fraction": int(tier_counts.get(key, 0)) / count,
            }
            for key, count in sorted(population_counts.items())
        }
    return result


def _write_jsonl(path: Path, samples: Iterable[TierSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for sample in samples:
            handle.write(
                json.dumps(sample.row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )


def _content_hashes(samples: Sequence[TierSample]) -> dict[str, str]:
    """Hash canonical rows by ID so nested-tier content equality is auditable."""

    return {
        sample.sample_id: hashlib.sha256(
            json.dumps(
                sample.row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        for sample in samples
    }


def _read_fixed_ids(path: Path | None) -> list[str]:
    if path is None:
        return []
    if not path.is_file():
        raise FileNotFoundError(f"Existing E1 ID file does not exist: {path}")
    ids = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()]
    ids = [sample_id for sample_id in ids if sample_id]
    if len(ids) != len(set(ids)):
        raise EvaluationTierError("Existing E1 ID file contains duplicate IDs")
    return ids


def _legacy_sample_to_row(
    sample: Mapping[str, Any],
    current: TierSample,
) -> dict[str, Any]:
    """Restore historical E1 task/prompt/reference while retaining current metadata."""

    sample_id = str(sample.get("id", "")).strip()
    task = str(sample.get("task_type", "")).strip().lower()
    question = str(sample.get("question", "")).strip()
    reference = str(sample.get("reference", "")).strip()
    images_value = sample.get("images", [])
    if not sample_id or task not in SUPPORTED_TASKS or not question or not reference:
        raise EvaluationTierError(f"Invalid historical E1 sample manifest row: {sample_id}")
    if not isinstance(images_value, list) or not images_value:
        raise EvaluationTierError(f"Historical E1 sample {sample_id} has no images")
    portable_images: list[str] = []
    for image in images_value:
        raw = str(image).replace("\\", "/")
        parts = [part for part in raw.split("/") if part]
        lower = [part.lower() for part in parts]
        if "vrsbench" in lower:
            start = lower.index("vrsbench")
            portable_images.append("/".join(parts[start:]))
        elif "levir-cc" in lower:
            start = lower.index("levir-cc")
            portable_images.append("/".join(parts[start:]))
        else:
            portable_images.append(_portable_image_path(raw, current.dataset))
    content: list[dict[str, str]] = [
        {"type": "image", "image": image} for image in portable_images
    ]
    content.append({"type": "text", "text": question})
    metadata = dict(current.row.get("metadata", {}))
    metadata["legacy_fixed_e1_content"] = True
    metadata["legacy_source_task_type"] = current.task
    metadata.setdefault("training_source", current.dataset)
    if task == "detection":
        metadata.setdefault("bbox_target_format", "normalized_0_1")
    return {
        "id": sample_id,
        "messages": [
            {"role": "user", "content": content},
            {"role": "assistant", "content": reference},
        ],
        "task_type": task,
        "metadata": metadata,
    }


def _apply_historical_e1_content(
    population: Sequence[TierSample],
    samples_path: Path | None,
    fixed_ids: Sequence[str],
    thresholds: BBoxAreaThresholdConfig,
) -> list[TierSample]:
    if samples_path is None:
        return list(population)
    if not samples_path.is_file():
        raise FileNotFoundError(f"Existing E1 samples file does not exist: {samples_path}")
    by_id = {sample.sample_id: sample for sample in population}
    replacements: dict[str, TierSample] = {}
    for legacy in read_jsonl(samples_path):
        sample_id = str(legacy.get("id", "")).strip()
        current = by_id.get(sample_id)
        if current is None:
            raise EvaluationTierError(f"Historical E1 sample is absent from population: {sample_id}")
        row = _legacy_sample_to_row(legacy, current)
        source = EvaluationTierSourceConfig(
            name=current.dataset,
            eval_file=str(samples_path),
            image_prefix="",
        )
        replacement = _build_tier_sample(row, source, thresholds)
        # Historical sample manifests are already protocol-normalized. Restore
        # their exact question/reference after deriving the stratification fields.
        replacement = TierSample(
            row=row,
            dataset=replacement.dataset,
            task=replacement.task,
            subtype=replacement.subtype,
            detection_size=replacement.detection_size,
            detection_class=replacement.detection_class,
            count_bucket=replacement.count_bucket,
            qa_type=replacement.qa_type,
            changeflag=replacement.changeflag,
        )
        replacements[sample_id] = replacement
    if set(replacements) != set(fixed_ids):
        missing = sorted(set(fixed_ids) - set(replacements))
        extra = sorted(set(replacements) - set(fixed_ids))
        raise EvaluationTierError(
            f"Historical E1 samples/IDs mismatch: missing={missing[:10]}, extra={extra[:10]}"
        )
    return [replacements.get(sample.sample_id, sample) for sample in population]


def build_evaluation_tiers(
    config: EvaluationTierBuildConfig,
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    """Build nested fixed tiers and return the persisted audit manifest."""

    config.validate_semantics()
    root = Path(project_root).resolve()
    population, source_records, train_ids, excluded_rows = load_population(
        config.sources,
        config.bbox_area_thresholds,
        project_root=root,
    )
    fixed_path = (
        _resolve_path(config.existing_e1.ids_file, root)
        if config.existing_e1.ids_file
        else None
    )
    fixed_ids = _read_fixed_ids(fixed_path)
    fixed_samples_path = (
        _resolve_path(config.existing_e1.samples_file, root)
        if config.existing_e1.samples_file
        else None
    )
    if config.existing_e1.required and not fixed_ids:
        raise EvaluationTierError("existing_e1.required=true but no fixed IDs were loaded")
    population = _apply_historical_e1_content(
        population,
        fixed_samples_path,
        fixed_ids,
        config.bbox_area_thresholds,
    )
    e1_target = int(config.tiers["E1"].target_samples or 0)
    e2_target = int(config.tiers["E2"].target_samples or 0)
    if fixed_ids and len(fixed_ids) != min(e1_target, len(population)):
        raise EvaluationTierError(
            f"Existing E1 contains {len(fixed_ids)} IDs but E1 target is {e1_target}"
        )
    e1 = stratified_select(population, e1_target, seed=config.seed, required_ids=fixed_ids)
    e2 = stratified_select(
        population,
        e2_target,
        seed=config.seed,
        required_ids=[sample.sample_id for sample in e1],
    )
    e3 = stratified_select(
        population,
        len(population),
        seed=config.seed,
        required_ids=[sample.sample_id for sample in e2],
    )
    tier_samples = {"E1": e1, "E2": e2, "E3": e3}
    output_dir = _resolve_path(config.output.directory, root)
    configured_names = {
        "E1": config.output.e1_file,
        "E2": config.output.e2_file,
        "E3": config.output.e3_file,
    }
    paths = {name: output_dir / configured_names[name] for name in tier_samples}
    for name, samples in tier_samples.items():
        _write_jsonl(paths[name], samples)

    population_dist = distribution(population)
    tier_records: dict[str, Any] = {}
    for name, samples in tier_samples.items():
        tier_dist = distribution(samples)
        tier_records[name] = {
            "path": paths[name].relative_to(root).as_posix(),
            "sample_count": len(samples),
            "sha256": file_sha256(paths[name]),
            "sample_ids": [sample.sample_id for sample in samples],
            "dataset_distribution": tier_dist["dataset"],
            "task_distribution": tier_dist["task"],
            "qa_type_distribution": tier_dist["qa_type"],
            "detection_size_distribution": tier_dist["detection_size"],
            "detection_class_distribution": tier_dist["detection_class"],
            "count_bucket_distribution": tier_dist["count_bucket"],
            "levir_changeflag_distribution": tier_dist["levir_changeflag"],
            "distribution": tier_dist,
            "sampling_fraction": _sampling_fractions(population_dist, tier_dist),
        }
    e1_ids = {sample.sample_id for sample in e1}
    e2_ids = {sample.sample_id for sample in e2}
    e3_ids = {sample.sample_id for sample in e3}
    e1_content = _content_hashes(e1)
    e2_content = _content_hashes(e2)
    e3_content = _content_hashes(e3)
    same_id_content = all(
        e1_content[sample_id] == e2_content[sample_id] == e3_content[sample_id]
        for sample_id in e1_ids
    ) and all(e2_content[sample_id] == e3_content[sample_id] for sample_id in e2_ids)
    manifest = {
        "schema_version": config.schema_version,
        "seed": config.seed,
        "description": {
            "E1": "Quick stratified diagnostic evaluation set.",
            "E2": "Standard stratified diagnostic evaluation set and training default.",
            "E3": "Full legal evaluation population.",
            "interpretation": (
                "E1/E2 are stratified diagnostic evaluation sets. "
                "E3 is the full evaluation population."
            ),
        },
        "source_files": source_records,
        "population_sample_count": len(population),
        "excluded_invalid_samples": excluded_rows,
        "excluded_invalid_sample_count": len(excluded_rows),
        "population_distribution": population_dist,
        "bbox_area_thresholds": config.bbox_area_thresholds.model_dump(),
        "count_bucket_definition": {
            "0": "0",
            "1": "1",
            "2": "2",
            "3": "3",
            "4": "4",
            "5-9": "5 through 9",
            "10+": "10 or greater",
            "unresolved": "reference accepted by the dataset but not parseable as an integer",
        },
        "existing_e1": {
            "origin": config.existing_e1.origin,
            "ids_file": (
                fixed_path.relative_to(root).as_posix() if fixed_path is not None else None
            ),
            "sample_count": len(fixed_ids),
            "sha256": file_sha256(fixed_path) if fixed_path is not None else None,
            "samples_file": (
                fixed_samples_path.relative_to(root).as_posix()
                if fixed_samples_path is not None
                else None
            ),
            "samples_sha256": (
                file_sha256(fixed_samples_path) if fixed_samples_path is not None else None
            ),
            "preserved_exactly": bool(fixed_ids),
        },
        "tiers": tier_records,
        "invariants": {
            "E1_subset_of_E2": e1_ids < e2_ids,
            "E2_subset_of_E3": e2_ids < e3_ids,
            "tier_ids_unique": all(
                len(samples) == len({sample.sample_id for sample in samples})
                for samples in tier_samples.values()
            ),
            "same_id_content_sha256": same_id_content,
            "train_eval_intersection_count": len(e3_ids & train_ids),
            "train_eval_disjoint": not bool(e3_ids & train_ids),
        },
        "excluded_from_training": {
            "all_tier_ids_file": "this manifest: tiers.*.sample_ids",
            "evaluation_ids_count": len(e3_ids),
            "legacy_fixed_evaluation_ids_count": len(fixed_ids),
            "note": "All E1/E2/E3 IDs are evaluation-only and must be excluded from mining/replay/H1.",
        },
    }
    if not manifest["invariants"]["E1_subset_of_E2"]:
        raise EvaluationTierError("E1 must be a strict subset of E2")
    if not manifest["invariants"]["E2_subset_of_E3"]:
        raise EvaluationTierError("E2 must be a strict subset of E3")
    if not manifest["invariants"]["same_id_content_sha256"]:
        raise EvaluationTierError("A shared sample ID has different content across tiers")
    manifest_path = output_dir / config.output.manifest_file
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def tier_metadata(
    tier: str,
    manifest_path: str | Path,
    *,
    project_root: str | Path,
) -> tuple[Path, str]:
    """Resolve and verify one frozen tier from its manifest."""

    normalized = tier.upper()
    if normalized not in TIER_FILES:
        raise EvaluationTierError(f"Unknown evaluation tier {tier!r}; choose E1, E2, or E3")
    root = Path(project_root).resolve()
    path = _resolve_path(manifest_path, root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    record = dict(payload.get("tiers", {}).get(normalized, {}))
    tier_path = _resolve_path(str(record.get("path", "")), root)
    expected = str(record.get("sha256", ""))
    if not tier_path.is_file() or not expected:
        raise EvaluationTierError(f"Tier {normalized} is missing from manifest {path}")
    actual = file_sha256(tier_path)
    if actual != expected:
        raise EvaluationTierError(
            f"Tier {normalized} checksum mismatch: expected={expected}, actual={actual}"
        )
    return tier_path, actual
