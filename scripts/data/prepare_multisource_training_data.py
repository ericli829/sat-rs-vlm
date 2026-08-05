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
from sat_rs_vlm.data.prompt_templates import strengthen_answer, strengthen_instruction
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
    else:
        missing = [key for key in ("images", "instruction", "answer") if key not in row]
        if missing:
            raise ValueError(f"Sample {sample_id} is missing fields: {missing}")
        instruction = instruction_override or str(row["instruction"])
        content = [
            {"type": "image", "image": str(image_path)}
            for image_path in list(row["images"])
        ]
        content.append(
            {
                "type": "text",
                "text": strengthen_instruction(task_type, instruction),
            }
        )
        messages = [
            {"role": "user", "content": content},
            {
                "role": "assistant",
                "content": strengthen_answer(task_type, row["answer"]),
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
            group_rows[(offset + index) % len(group_rows)]
            for index in range(variants_per_group)
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
        source_validation_all = _load_source_split(
            source, "validation_file", common_image_root
        )
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
        "task_distribution": dict(
            sorted(Counter(row["task_type"] for row in train_rows).items())
        ),
        "sources": source_report,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    include_sources = {name.lower() for name in args.include_source} or None
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
