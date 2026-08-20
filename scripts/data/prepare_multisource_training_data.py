"""Build portable Qwen3-VL JSONL files from multiple remote-sensing datasets."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from sat_rs_vlm.configuration.environment import expand_environment
from sat_rs_vlm.data.cyclic_training import (
    assert_no_evaluation_leakage,
    combine_source_rounds,
    load_protected_e3_ids,
    partition_group_variants,
    partition_task_population,
    partition_task_population_evenly,
    sha256_file,
    top_up_source_to_pattern,
    validate_cycle_coverage,
)
from sat_rs_vlm.data.prompt_templates import strengthen_answer, strengthen_instruction
from sat_rs_vlm.data.stage_a_v2 import build_canonical_training_population
from sat_rs_vlm.utils.jsonl import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare portable multi-source Qwen3-VL training data."
    )
    parser.add_argument("--config", required=True, help="Multi-source YAML configuration.")
    parser.add_argument(
        "--include-source",
        action="append",
        default=[],
        help="Include only this source name. Repeat to include multiple sources.",
    )
    parser.add_argument("--train-output", default=None)
    parser.add_argument("--validation-output", default=None)
    parser.add_argument("--report-output", default=None)
    parser.add_argument(
        "--round-index",
        type=int,
        default=0,
        help="Zero-based replay round used for deterministic sample rotation.",
    )
    parser.add_argument(
        "--full-evaluation-only",
        action="store_true",
        help=(
            "Build the complete legal evaluation population without applying "
            "validation_samples limits or preparing training output."
        ),
    )
    parser.add_argument("--evaluation-output", default=None)
    parser.add_argument("--evaluation-report-output", default=None)
    parser.add_argument("--build-cycle", action="store_true")
    parser.add_argument(
        "--build-population",
        action="store_true",
        help="Build the Stage-A v2 canonical legal training population.",
    )
    parser.add_argument("--population-output-dir", default=None)
    parser.add_argument("--cycle-index", type=int, default=0)
    parser.add_argument("--cycle-output-dir", default=None)
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = dict(yaml.safe_load(handle) or {})
    return dict(expand_environment(data, environ=os.environ, allow_unresolved=False))


def _message_images(row: dict[str, Any]) -> list[str]:
    images: list[str] = []
    for message in list(row.get("messages", [])):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if item.get("type") == "image":
                images.append(str(item.get("image", "")))
    return images


def _to_qwen_row(
    row: dict[str, Any],
    *,
    source_name: str,
    instruction_override: str | None,
    prompt_profile: str = "canonical",
    strengthen_existing_messages: bool = False,
) -> dict[str, Any]:
    sample_id = str(row.get("id", "")).strip()
    task_type = str(row.get("task_type", "unknown")).strip().lower()
    if not sample_id:
        raise ValueError(f"Source {source_name} contains a sample without an id")

    metadata = dict(row.get("metadata", {}))
    metadata.setdefault("dataset", source_name)
    metadata["training_source"] = source_name

    if "messages" in row:
        messages = [dict(message) for message in list(row["messages"])]
        if strengthen_existing_messages:
            messages = _strengthen_existing_messages(messages, task_type, prompt_profile)
    else:
        missing = [key for key in ("images", "instruction", "answer") if key not in row]
        if missing:
            raise ValueError(f"Sample {sample_id} is missing fields: {missing}")
        instruction = instruction_override or str(row["instruction"])
        content = [
            {"type": "image", "image": str(image_path)} for image_path in list(row["images"])
        ]
        content.append(
            {
                "type": "text",
                "text": strengthen_instruction(task_type, instruction, profile=prompt_profile),
            }
        )
        messages = [
            {"role": "user", "content": content},
            {
                "role": "assistant",
                "content": strengthen_answer(task_type, row["answer"], profile=prompt_profile),
            },
        ]

    return {
        "id": sample_id,
        "messages": messages,
        "task_type": task_type,
        "metadata": metadata,
    }


def _path_parts_after_images(raw_path: str) -> tuple[str, ...] | None:
    parts = tuple(part for part in raw_path.replace("\\", "/").split("/") if part)
    for index, part in enumerate(parts):
        if part.lower() == "images":
            return parts[index:]
    return None


def _resolve_source_image(raw_path: str, source_root: Path) -> Path:
    normalized = raw_path.replace("\\", "/")
    candidates: list[Path] = []
    direct = Path(raw_path).expanduser()
    if direct.is_absolute():
        candidates.append(direct)
    candidates.append(source_root.joinpath(*[part for part in normalized.split("/") if part]))
    marker_parts = _path_parts_after_images(raw_path)
    if marker_parts is not None:
        candidates.append(source_root.joinpath(*marker_parts))

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    attempted = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Image does not exist: {raw_path}; attempted: {attempted}")


def _portable_image_path(image_path: Path, common_image_root: Path) -> str:
    try:
        return image_path.relative_to(common_image_root).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"Image {image_path} is outside common_image_root {common_image_root}"
        ) from exc


def _rewrite_images(
    row: dict[str, Any],
    *,
    source_root: Path,
    common_image_root: Path,
    path_cache: dict[str, str],
) -> dict[str, Any]:
    rewritten_messages: list[dict[str, Any]] = []
    image_count = 0
    for message in list(row["messages"]):
        message_copy = dict(message)
        content = message.get("content")
        if not isinstance(content, list):
            rewritten_messages.append(message_copy)
            continue
        rewritten_content: list[dict[str, Any]] = []
        for item in content:
            item_copy = dict(item)
            if item_copy.get("type") == "image":
                raw_path = str(item_copy.get("image", ""))
                if raw_path not in path_cache:
                    resolved = _resolve_source_image(raw_path, source_root)
                    path_cache[raw_path] = _portable_image_path(resolved, common_image_root)
                item_copy["image"] = path_cache[raw_path]
                image_count += 1
            rewritten_content.append(item_copy)
        message_copy["content"] = rewritten_content
        rewritten_messages.append(message_copy)

    if image_count == 0:
        raise ValueError(f"Sample {row['id']} does not contain an image")
    if row["task_type"] == "change_detection" and image_count != 2:
        raise ValueError(
            f"Change-detection sample {row['id']} must contain exactly two images; "
            f"found {image_count}"
        )
    output = dict(row)
    output["messages"] = rewritten_messages
    return output


def _load_source_split(
    source: dict[str, Any],
    split_key: str,
    common_image_root: Path,
) -> list[dict[str, Any]]:
    source_name = str(source["name"])
    source_root = Path(str(source["image_root"])).expanduser().resolve()
    jsonl_path = Path(str(source[split_key])).expanduser()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source image_root does not exist: {source_root}")
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"Source JSONL does not exist: {jsonl_path}")

    path_cache: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    for source_row in read_jsonl(jsonl_path):
        normalized = _to_qwen_row(
            source_row,
            source_name=source_name,
            instruction_override=source.get("instruction_override"),
            prompt_profile=str(source.get("prompt_profile", "canonical")),
            strengthen_existing_messages=bool(source.get("strengthen_existing_messages", False)),
        )
        rows.append(
            _rewrite_images(
                normalized,
                source_root=source_root,
                common_image_root=common_image_root,
                path_cache=path_cache,
            )
        )
    if not rows:
        raise ValueError(f"Source JSONL is empty: {jsonl_path}")
    return rows


def _sample_validation_rows(
    rows: list[dict[str, Any]],
    *,
    limit: int | None,
    group_by_images: bool,
    seed: int,
) -> list[dict[str, Any]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    if group_by_images:
        unique_rows: list[dict[str, Any]] = []
        seen_images: set[tuple[str, ...]] = set()
        for row in shuffled:
            image_key = tuple(_message_images(row))
            if image_key in seen_images:
                continue
            seen_images.add(image_key)
            unique_rows.append(row)
        shuffled = unique_rows
    return shuffled[:limit] if limit is not None else shuffled


def _sample_group_balanced_rows(
    rows: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    if count > len(rows):
        raise ValueError(f"Requested {count} samples from a pool of {len(rows)}")
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(_message_images(row)), []).append(row)
    rng = random.Random(seed)
    group_keys = list(groups)
    rng.shuffle(group_keys)
    for group_rows in groups.values():
        rng.shuffle(group_rows)

    selected: list[dict[str, Any]] = []
    depth = 0
    while len(selected) < count:
        made_progress = False
        for group_key in group_keys:
            group_rows = groups[group_key]
            if depth >= len(group_rows):
                continue
            selected.append(group_rows[depth])
            made_progress = True
            if len(selected) == count:
                break
        if not made_progress:
            break
        depth += 1
    if len(selected) != count:
        raise ValueError(f"Could only select {len(selected)} of {count} grouped samples")
    return selected


def _sample_task_quotas(
    rows: list[dict[str, Any]],
    quotas: dict[str, int],
    *,
    seed: int,
    group_by_images: bool,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["task_type"]), []).append(row)
    missing = sorted(set(quotas).difference(grouped))
    if missing:
        raise ValueError(f"Training quota tasks are missing from source: {missing}")

    selected: list[dict[str, Any]] = []
    for task_index, (task, requested) in enumerate(quotas.items()):
        count = int(requested)
        if count < 1:
            raise ValueError(f"Training quota must be positive for task {task}")
        pool = grouped[task]
        if count > len(pool):
            raise ValueError(
                f"Training quota for {task} requests {count}, but only {len(pool)} exist"
            )
        if group_by_images:
            chosen = _sample_group_balanced_rows(
                pool,
                count=count,
                seed=seed + task_index,
            )
        else:
            chosen = list(pool)
            random.Random(seed + task_index).shuffle(chosen)
            chosen = chosen[:count]
        selected.extend(chosen)
    return selected


def _rotate_group_variants(
    rows: list[dict[str, Any]],
    *,
    variants_per_group: int,
    round_index: int,
) -> list[dict[str, Any]]:
    if variants_per_group < 1:
        raise ValueError("training_samples_per_image_group must be positive")
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(_message_images(row)), []).append(row)

    selected: list[dict[str, Any]] = []
    for image_key in sorted(groups):
        group_rows = sorted(groups[image_key], key=lambda row: str(row["id"]))
        if variants_per_group > len(group_rows):
            raise ValueError(
                f"Image group {image_key} has {len(group_rows)} samples, "
                f"but {variants_per_group} were requested"
            )
        offset = (round_index * variants_per_group) % len(group_rows)
        selected.extend(
            group_rows[(offset + index) % len(group_rows)] for index in range(variants_per_group)
        )
    return selected


def _sample_training_rows(
    rows: list[dict[str, Any]],
    source: dict[str, Any],
    *,
    seed: int,
    round_index: int,
) -> list[dict[str, Any]]:
    selected = list(rows)
    raw_quotas = source.get("training_task_quotas")
    if raw_quotas:
        quotas = {str(task): int(count) for task, count in dict(raw_quotas).items()}
        selected = _sample_task_quotas(
            selected,
            quotas,
            seed=seed + round_index * 1_000,
            group_by_images=bool(source.get("training_group_by_images", True)),
        )
    variants = source.get("training_samples_per_image_group")
    if variants is not None:
        selected = _rotate_group_variants(
            selected,
            variants_per_group=int(variants),
            round_index=round_index,
        )
    random.Random(seed + round_index * 10_000).shuffle(selected)
    return selected


def _assert_unique_ids(rows: list[dict[str, Any]], split: str) -> None:
    counts = Counter(str(row["id"]) for row in rows)
    duplicates = sorted(sample_id for sample_id, count in counts.items() if count > 1)
    if duplicates:
        preview = ", ".join(duplicates[:5])
        raise ValueError(f"{split} contains duplicate sample ids: {preview}")


def prepare_multisource_data(
    config: dict[str, Any],
    *,
    include_sources: set[str] | None = None,
    train_output: str | Path | None = None,
    validation_output: str | Path | None = None,
    report_output: str | Path | None = None,
    round_index: int = 0,
) -> dict[str, Any]:
    if round_index < 0:
        raise ValueError("round_index must be non-negative")
    common_image_root = Path(str(config["common_image_root"])).expanduser().resolve()
    if not common_image_root.is_dir():
        raise FileNotFoundError(f"common_image_root does not exist: {common_image_root}")

    output_config = dict(config.get("output", {}))
    train_path = Path(str(train_output or output_config["train_file"]))
    validation_path = Path(str(validation_output or output_config["validation_file"]))
    report_path = Path(str(report_output or output_config["report_file"]))
    seed = int(config.get("seed", 42))

    selected_sources = [
        dict(source)
        for source in list(config.get("sources", []))
        if not include_sources or str(source.get("name", "")).lower() in include_sources
    ]
    if not selected_sources:
        raise ValueError("No data sources were selected")

    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    source_report: dict[str, Any] = {}
    for source_index, source in enumerate(selected_sources):
        name = str(source["name"])
        source_train_all = _load_source_split(source, "train_file", common_image_root)
        source_train = _sample_training_rows(
            source_train_all,
            source,
            seed=seed + source_index,
            round_index=round_index,
        )
        source_validation_all = _load_source_split(source, "validation_file", common_image_root)
        validation_limit = source.get("validation_samples")
        source_validation = _sample_validation_rows(
            source_validation_all,
            limit=int(validation_limit) if validation_limit is not None else None,
            group_by_images=bool(source.get("validation_group_by_images", False)),
            seed=seed + source_index,
        )
        train_rows.extend(source_train)
        validation_rows.extend(source_validation)
        source_report[name] = {
            "train_samples_available": len(source_train_all),
            "train_samples_selected": len(source_train),
            "validation_samples_available": len(source_validation_all),
            "validation_samples_selected": len(source_validation),
            "task_distribution": dict(
                sorted(Counter(row["task_type"] for row in source_train).items())
            ),
        }

    _assert_unique_ids(train_rows, "train")
    _assert_unique_ids(validation_rows, "validation")
    train_ids = {str(row["id"]) for row in train_rows}
    overlap = sorted(train_ids.intersection(str(row["id"]) for row in validation_rows))
    if overlap:
        raise ValueError(f"Train/validation id overlap: {', '.join(overlap[:5])}")

    random.Random(seed).shuffle(train_rows)
    random.Random(seed + 10_000).shuffle(validation_rows)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(train_path, train_rows)
    write_jsonl(validation_path, validation_rows)

    report = {
        "valid": True,
        "common_image_root": str(common_image_root),
        "round_index": round_index,
        "train_file": str(train_path),
        "validation_file": str(validation_path),
        "train_samples": len(train_rows),
        "validation_samples": len(validation_rows),
        "task_distribution": dict(sorted(Counter(row["task_type"] for row in train_rows).items())),
        "sources": source_report,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def prepare_full_evaluation_population(
    config: dict[str, Any],
    *,
    include_sources: set[str] | None = None,
    evaluation_output: str | Path | None = None,
    report_output: str | Path | None = None,
) -> dict[str, Any]:
    """构建完整合法多源评测 population，不改变历史训练采样行为。

    对 ``validation_group_by_images=true`` 的数据源，每组图像只保留一个由固定
    seed 选择的 reference case。LEVIR-CC 因而以 image pair 为评测单位，而不是
    把同一 image pair 的五条 caption 当作五个独立 case。所有图像仍通过现有
    ``_load_source_split`` 路径修复和 portable-path 校验链路。
    """

    common_image_root = Path(str(config["common_image_root"])).expanduser().resolve()
    if not common_image_root.is_dir():
        raise FileNotFoundError(f"common_image_root does not exist: {common_image_root}")
    output_config = dict(config.get("output", {}))
    evaluation_config = dict(config.get("evaluation", {}))
    destination_value = (
        evaluation_output
        or evaluation_config.get("full_population_file")
        or output_config.get("full_evaluation_file")
    )
    report_value = (
        report_output
        or evaluation_config.get("full_population_report")
        or output_config.get("full_evaluation_report")
    )
    if destination_value is None or report_value is None:
        raise ValueError(
            "Full evaluation preparation requires --evaluation-output and "
            "--evaluation-report-output, or matching evaluation/output config fields"
        )
    destination = Path(str(destination_value))
    report_path = Path(str(report_value))
    seed = int(config.get("seed", 42))
    selected_sources = [
        dict(source)
        for source in list(config.get("sources", []))
        if not include_sources or str(source.get("name", "")).lower() in include_sources
    ]
    if not selected_sources:
        raise ValueError("No data sources were selected")

    population: list[dict[str, Any]] = []
    source_report: dict[str, Any] = {}
    train_ids: set[str] = set()
    for source_index, source in enumerate(selected_sources):
        name = str(source["name"])
        validation_rows = _load_source_split(source, "validation_file", common_image_root)
        group_by_images = bool(source.get("validation_group_by_images", False))
        selected = _sample_validation_rows(
            validation_rows,
            limit=None,
            group_by_images=group_by_images,
            seed=seed + source_index,
        )
        raw_train_path = Path(str(source["train_file"])).expanduser()
        train_ids.update(str(row.get("id", "")) for row in read_jsonl(raw_train_path))
        available_image_groups = {tuple(_message_images(row)) for row in validation_rows}
        population.extend(selected)
        source_report[name] = {
            "validation_rows_available": len(validation_rows),
            "unique_image_groups": len(available_image_groups),
            "evaluation_cases_selected": len(selected),
            "evaluation_unit": "image_pair" if group_by_images else "sample",
            "group_by_images": group_by_images,
            "reference_selection": (
                "one_deterministic_reference_per_image_group"
                if group_by_images
                else "every_validation_row"
            ),
            "task_distribution": dict(
                sorted(Counter(row["task_type"] for row in selected).items())
            ),
        }

    _assert_unique_ids(population, "full evaluation population")
    population_ids = {str(row["id"]) for row in population}
    overlap = sorted(population_ids.intersection(train_ids))
    if overlap:
        raise ValueError(f"Train/evaluation id overlap: {', '.join(overlap[:5])}")
    population.sort(key=lambda row: str(row["id"]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(destination, population)
    report = {
        "schema_version": "2.0",
        "valid": True,
        "common_image_root": str(common_image_root),
        "evaluation_file": str(destination),
        "evaluation_samples": len(population),
        "unique_ids": len(population_ids),
        "train_evaluation_overlap": 0,
        "task_distribution": dict(sorted(Counter(row["task_type"] for row in population).items())),
        "dataset_distribution": dict(
            sorted(
                Counter(
                    str(dict(row.get("metadata", {})).get("dataset", "unknown"))
                    for row in population
                ).items()
            )
        ),
        "sources": source_report,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _strengthen_existing_messages(
    messages: list[dict[str, Any]], task_type: str, prompt_profile: str
) -> list[dict[str, Any]]:
    """加固已转换 messages 的输出协议，同时保留图片与对话结构。"""

    normalized: list[dict[str, Any]] = []
    for message in messages:
        copied = dict(message)
        role = str(copied.get("role", ""))
        content = copied.get("content")
        if role == "user" and isinstance(content, list):
            items = [dict(item) for item in content]
            text_indices = [index for index, item in enumerate(items) if item.get("type") == "text"]
            if text_indices:
                index = text_indices[-1]
                items[index]["text"] = strengthen_instruction(
                    task_type, str(items[index].get("text", "")), profile=prompt_profile
                )
            copied["content"] = items
        elif role == "user" and isinstance(content, str):
            copied["content"] = strengthen_instruction(task_type, content, profile=prompt_profile)
        elif role == "assistant" and isinstance(content, str):
            copied["content"] = strengthen_answer(task_type, content, profile=prompt_profile)
        elif role == "assistant" and isinstance(content, list):
            items = [dict(item) for item in content]
            for item in items:
                if item.get("type") == "text":
                    item["text"] = strengthen_answer(
                        task_type, item.get("text", ""), profile=prompt_profile
                    )
            copied["content"] = items
        normalized.append(copied)
    return normalized


def _distribution(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    if field == "source":
        values = (
            str(dict(row.get("metadata", {})).get("training_source", "unknown")) for row in rows
        )
    else:
        values = (str(row.get("task_type", "unknown")) for row in rows)
    return dict(sorted(Counter(values).items()))


def prepare_canonical_population(
    config: dict[str, Any],
    *,
    population_output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """复用正式 normalization 链路构建 Stage-A v2 canonical population。"""

    common_root = Path(str(config["common_image_root"])).expanduser().resolve()
    if not common_root.is_dir():
        raise FileNotFoundError(f"common_image_root does not exist: {common_root}")
    population_config = dict(config.get("population", {}))
    output_value = population_output_dir or population_config.get("output_dir")
    if not output_value:
        raise ValueError(
            "Population preparation requires population.output_dir or " "--population-output-dir"
        )
    protected_manifest = population_config.get("protected_evaluation_manifest")
    if not protected_manifest:
        raise ValueError("population.protected_evaluation_manifest is required")

    seed = int(config.get("seed", 42))
    source_rows: dict[str, list[dict[str, Any]]] = {}
    source_inputs: dict[str, dict[str, Any]] = {}
    prompt_profiles: dict[str, str] = {}
    validation_rows: list[dict[str, Any]] = []
    for source_index, source_value in enumerate(list(config.get("sources", []))):
        source = dict(source_value)
        name = str(source["name"])
        source_rows[name] = _load_source_split(source, "train_file", common_root)
        train_file = Path(str(source["train_file"])).expanduser()
        source_inputs[name] = {
            "train_file": str(train_file),
            "train_file_sha256": sha256_file(train_file),
            "image_root": str(Path(str(source["image_root"])).expanduser()),
        }
        prompt_profiles[name] = str(source.get("prompt_profile", "canonical"))
        validation_all = _load_source_split(source, "validation_file", common_root)
        validation_limit = source.get("validation_samples")
        validation_rows.extend(
            _sample_validation_rows(
                validation_all,
                limit=int(validation_limit) if validation_limit is not None else None,
                group_by_images=bool(source.get("validation_group_by_images", False)),
                seed=seed + source_index,
            )
        )
    return build_canonical_training_population(
        source_rows,
        validation_rows,
        output_dir=str(output_value),
        protected_evaluation_manifest=str(protected_manifest),
        seed=seed,
        source_inputs=source_inputs,
        prompt_profiles=prompt_profiles,
    )


def prepare_training_cycle(
    config: dict[str, Any],
    *,
    cycle_index: int = 0,
    cycle_output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """构建一个可审计的 full-coverage cycle，并一次性写出所有 round。"""

    selection_mode = str(config.get("training_selection_mode", ""))
    if selection_mode not in {
        "cyclic_full_coverage",
        "balanced_cyclic_full_coverage",
    }:
        raise ValueError(
            "--build-cycle requires a full-coverage cyclic selection mode; "
            "legacy_round_sampling remains available through the historical command"
        )
    if cycle_index < 0:
        raise ValueError("cycle_index must be non-negative")
    common_root = Path(str(config["common_image_root"])).expanduser().resolve()
    if not common_root.is_dir():
        raise FileNotFoundError(f"common_image_root does not exist: {common_root}")
    cycle_config = dict(config.get("cycle", {}))
    output_value = cycle_output_dir or cycle_config.get("output_dir")
    if not output_value:
        raise ValueError("Cyclic preparation requires cycle.output_dir or --cycle-output-dir")
    output_dir = Path(str(output_value))
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 42))

    source_rounds: dict[str, list[list[dict[str, Any]]]] = {}
    source_population: dict[str, list[dict[str, Any]]] = {}
    source_configs: dict[str, dict[str, Any]] = {}
    source_seeds: dict[str, int] = {}
    validation_rows: list[dict[str, Any]] = []
    levir_report: dict[str, Any] = {}
    for source_index, source_value in enumerate(list(config.get("sources", []))):
        source = dict(source_value)
        name = str(source["name"])
        population = _load_source_split(source, "train_file", common_root)
        source_population[name] = population
        source_seed = seed + source_index
        source_configs[name] = source
        source_seeds[name] = source_seed
        if selection_mode == "balanced_cyclic_full_coverage":
            num_rounds = int(cycle_config.get("num_rounds", 0))
            if num_rounds < 1:
                raise ValueError("balanced_cyclic_full_coverage requires cycle.num_rounds")
            source_rounds[name] = partition_task_population_evenly(
                population,
                num_rounds,
                seed=source_seed,
                cycle_index=cycle_index,
            )
        elif source.get("training_task_quotas"):
            source_rounds[name] = partition_task_population(
                population,
                {
                    str(task): int(size)
                    for task, size in dict(source["training_task_quotas"]).items()
                },
                seed=source_seed,
                cycle_index=cycle_index,
            )
        elif source.get("training_samples_per_image_group") is not None:
            rounds, variant_report = partition_group_variants(
                population,
                variants_per_round=int(source["training_samples_per_image_group"]),
                seed=source_seed,
                cycle_index=cycle_index,
                image_key=lambda row: tuple(_message_images(row)),
            )
            source_rounds[name] = rounds
            levir_report[name] = variant_report
        else:
            source_rounds[name] = [population]

        source_validation_all = _load_source_split(source, "validation_file", common_root)
        validation_limit = source.get("validation_samples")
        validation_rows.extend(
            _sample_validation_rows(
                source_validation_all,
                limit=int(validation_limit) if validation_limit is not None else None,
                group_by_images=bool(source.get("validation_group_by_images", False)),
                seed=source_seed,
            )
        )

    target_round_count = max((len(rounds) for rounds in source_rounds.values()), default=0)
    for name, source in source_configs.items():
        if source.get("training_samples_per_image_group") is None:
            continue
        if len(source_rounds[name]) == target_round_count:
            continue
        spread_rounds, variant_report = partition_group_variants(
            source_population[name],
            variants_per_round=int(source["training_samples_per_image_group"]),
            seed=source_seeds[name],
            cycle_index=cycle_index,
            image_key=lambda row: tuple(_message_images(row)),
            target_rounds=target_round_count,
        )
        source_rounds[name] = spread_rounds
        levir_report[name] = variant_report

    population = [row for name in sorted(source_population) for row in source_population[name]]
    _assert_unique_ids(population, "cyclic training population")
    _assert_unique_ids(validation_rows, "cyclic validation")
    protected_path = cycle_config.get("protected_evaluation_manifest")
    if not protected_path:
        raise ValueError("cycle.protected_evaluation_manifest is required")
    leakage = assert_no_evaluation_leakage(
        population,
        load_protected_e3_ids(str(protected_path)),
    )

    base_rounds = combine_source_rounds(
        source_rounds,
        seed=seed,
        cycle_index=cycle_index,
    )
    coverage = validate_cycle_coverage(population, base_rounds)
    if not coverage["valid"]:
        raise ValueError(f"Full-cycle coverage validation failed: {coverage}")
    rounds = base_rounds
    replay_report: dict[str, Any] = {"enabled": False, "replay_exposures_added": 0}
    replay_config = dict(cycle_config.get("replay_short_source", {}))
    if bool(replay_config.get("enabled", False)):
        replay_source = str(replay_config["source"])
        reference_source = str(replay_config["reference_source"])
        if replay_source not in source_population:
            raise ValueError(f"Unknown replay source: {replay_source}")
        if reference_source not in source_population:
            raise ValueError(f"Unknown replay reference source: {reference_source}")
        rounds, replay_report = top_up_source_to_pattern(
            base_rounds,
            source_population[replay_source],
            list(cycle_config.get("source_batch_pattern", [])),
            replay_source=replay_source,
            reference_source=reference_source,
            seed=seed,
            cycle_index=cycle_index,
        )
    validation_rows.sort(key=lambda row: str(row["id"]))
    validation_path = output_dir / "validation.jsonl"
    write_jsonl(validation_path, validation_rows)

    round_entries: list[dict[str, Any]] = []
    for round_index, rows in enumerate(rounds):
        train_path = output_dir / f"round_{round_index:03d}_train.jsonl"
        report_path = output_dir / f"round_{round_index:03d}_report.json"
        write_jsonl(train_path, rows)
        replay_rows = [
            row for row in rows if bool(dict(row.get("metadata", {})).get("cycle_replay"))
        ]
        report = {
            "schema_version": "1.0",
            "cycle_index": cycle_index,
            "round_index": round_index,
            "sample_count": len(rows),
            "base_sample_count": len(rows) - len(replay_rows),
            "replay_exposure_count": len(replay_rows),
            "unique_id_count": len({str(row["id"]) for row in rows}),
            "source_distribution": _distribution(rows, "source"),
            "task_distribution": _distribution(rows, "task"),
            "train_file": str(train_path),
            "sha256": sha256_file(train_path),
        }
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        round_entries.append({**report, "report_file": str(report_path)})

    scheduled_population = [row for round_rows in rounds for row in round_rows]
    scheduled_source_distribution = _distribution(scheduled_population, "source")
    manifest = {
        "schema_version": "1.0",
        "training_selection_mode": selection_mode,
        "seed": seed,
        "cycle_index": cycle_index,
        "num_rounds": len(rounds),
        "validation_file": str(validation_path),
        "validation_sha256": sha256_file(validation_path),
        "source_population_counts": {
            name: len(rows) for name, rows in sorted(source_population.items())
        },
        "source_scheduling": {
            "level": "batch",
            "preference_pattern": list(cycle_config.get("source_batch_pattern", [])),
            "exhaustion_policy": "coverage_first",
            "base_population_counts": _distribution(population, "source"),
            "expected_exposure_counts": scheduled_source_distribution,
            "expected_exposure_ratio": {
                source: count / max(1, len(scheduled_population))
                for source, count in scheduled_source_distribution.items()
            },
            "tail_may_deviate_from_pattern": True,
            "replay": replay_report,
        },
        "task_population_counts": _distribution(population, "task"),
        "rounds": round_entries,
        "global": coverage,
        "source_coverage": {
            name: validate_cycle_coverage(
                rows,
                [
                    [
                        row
                        for row in round_rows
                        if dict(row.get("metadata", {})).get("training_source") == name
                    ]
                    for round_rows in base_rounds
                ],
            )
            for name, rows in sorted(source_population.items())
        },
        "task_coverage": {
            task: validate_cycle_coverage(
                [row for row in population if row.get("task_type") == task],
                [
                    [row for row in round_rows if row.get("task_type") == task]
                    for round_rows in base_rounds
                ],
            )
            for task in sorted({str(row.get("task_type")) for row in population})
        },
        "levir": levir_report,
        "protected_evaluation": {
            "manifest": str(protected_path),
            **leakage,
        },
    }
    manifest_path = output_dir / "cycle_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {**manifest, "cycle_manifest": str(manifest_path)}


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    include_sources = {name.lower() for name in args.include_source} or None
    selected_modes = sum(
        bool(value)
        for value in (args.build_cycle, args.build_population, args.full_evaluation_only)
    )
    if selected_modes > 1:
        raise ValueError(
            "--build-cycle, --build-population, and --full-evaluation-only are exclusive"
        )
    if args.build_population:
        if include_sources:
            raise ValueError("--include-source is not supported by population preparation")
        report = prepare_canonical_population(
            config,
            population_output_dir=args.population_output_dir,
        )
    elif args.build_cycle:
        if include_sources:
            raise ValueError("--include-source is not supported by full-cycle preparation")
        report = prepare_training_cycle(
            config,
            cycle_index=args.cycle_index,
            cycle_output_dir=args.cycle_output_dir,
        )
    elif args.full_evaluation_only:
        report = prepare_full_evaluation_population(
            config,
            include_sources=include_sources,
            evaluation_output=args.evaluation_output,
            report_output=args.evaluation_report_output,
        )
    else:
        report = prepare_multisource_data(
            config,
            include_sources=include_sources,
            train_output=args.train_output,
            validation_output=args.validation_output,
            report_output=args.report_output,
            round_index=args.round_index,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
