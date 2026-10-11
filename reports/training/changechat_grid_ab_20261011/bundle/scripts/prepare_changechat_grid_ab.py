#!/usr/bin/env python3
"""Build a deterministic 2B ChangeChat grid-first train/validation bundle.

The script is CPU-only. It quarantines grid-positive pairs whose associated
global changeflag is zero, reserves a pair-grouped grid dev split, samples
single-turn ChangeChat support rows, and adds a stratified VRSBench replay set.
It never loads a model or initializes CUDA.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

GRID_LABELS = {
    f"{vertical}-{horizontal}"
    for vertical in ("top", "center", "bottom")
    for horizontal in ("left", "center", "right")
}
GRID_ANSWER_RE = re.compile(
    r"roads\s*:\s*\[(.*?)\]\s*\.?\s*buildings\s*:\s*\[(.*?)\]",
    re.IGNORECASE | re.DOTALL,
)
GRID_TOKEN_RE = re.compile(r"['\"]([^'\"]+)['\"]")
BUCKETS = ("both_empty", "roads_only", "buildings_only", "both_nonempty")
AUX_BUCKETS = ("binary", "count", "caption", "open_qa")
AUX_FRACTIONS = {"binary": 0.35, "count": 0.35, "caption": 0.20, "open_qa": 0.10}
AUX_TASK_TYPES = {
    "binary": "change_detection",
    "count": "counting",
    "caption": "captioning",
    "open_qa": "vqa",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def pair_key(row: dict[str, Any]) -> tuple[str, str]:
    paths = row.get("image") or row.get("images")
    if not isinstance(paths, list) or len(paths) != 2:
        raise ValueError(f"Expected an ordered two-image pair in row {row.get('id')!r}")
    normalized = tuple(str(value).replace("\\", "/").strip("/") for value in paths)
    return normalized[0], normalized[1]


def pair_id(key: tuple[str, str]) -> str:
    return "|".join(key)


def conversations(row: dict[str, Any]) -> list[dict[str, Any]]:
    values = row.get("conversations")
    if not isinstance(values, list):
        raise ValueError(f"Missing conversations in ChangeChat row {row.get('id')!r}")
    return values


def message_text(message: dict[str, Any]) -> str:
    return str(message.get("value", message.get("content", "")))


def parse_grid(answer: str) -> tuple[set[str], set[str]]:
    match = GRID_ANSWER_RE.search(answer)
    if match is None:
        raise ValueError(f"Cannot parse grid answer: {answer!r}")

    def parse_cells(raw: str) -> set[str]:
        if not raw.strip():
            return set()
        values = [value.strip() for value in GRID_TOKEN_RE.findall(raw)]
        residue = GRID_TOKEN_RE.sub("", raw).strip(" \t,\r\n")
        if residue or not values:
            raise ValueError(f"Malformed grid cell list: {raw!r}")
        invalid = sorted(set(values).difference(GRID_LABELS))
        if invalid or len(values) != len(set(values)):
            raise ValueError(f"Invalid or duplicated grid labels: {invalid or values!r}")
        return set(values)

    return parse_cells(match.group(1)), parse_cells(match.group(2))


def grid_class(roads: set[str], buildings: set[str]) -> str:
    if roads and buildings:
        return "both_nonempty"
    if roads:
        return "roads_only"
    if buildings:
        return "buildings_only"
    return "both_empty"


def apportion(total: int, weights: dict[str, float]) -> dict[str, int]:
    weight_sum = sum(weights.values())
    exact = {key: total * value / weight_sum for key, value in weights.items()}
    counts = {key: math.floor(value) for key, value in exact.items()}
    remainder = total - sum(counts.values())
    order = sorted(weights, key=lambda key: (-(exact[key] - counts[key]), key))
    for key in order[:remainder]:
        counts[key] += 1
    return counts


def split_by_stratum(
    rows: list[tuple[dict[str, Any], tuple[str, str], set[str], set[str], str]],
    fraction: float,
    seed: int,
) -> tuple[list[Any], list[Any], dict[str, dict[str, int]]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for item in rows:
        grouped[item[4]].append(item)
    validation_total = int(round(len(rows) * fraction))
    quotas = apportion(
        validation_total,
        {key: float(len(grouped.get(key, []))) for key in BUCKETS},
    )
    rng = random.Random(seed)
    train: list[Any] = []
    validation: list[Any] = []
    stats: dict[str, dict[str, int]] = {}
    for key in BUCKETS:
        values = sorted(grouped.get(key, []), key=lambda item: pair_id(item[1]))
        rng.shuffle(values)
        n_val = quotas[key]
        validation.extend(values[:n_val])
        train.extend(values[n_val:])
        stats[key] = {"all": len(values), "validation": n_val, "train": len(values) - n_val}
    return train, validation, stats


def make_changechat_row(
    raw: dict[str, Any],
    *,
    source_task: str,
    training_source: str,
    category: str,
    pair: tuple[str, str],
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    conv = conversations(raw)
    user = next((item for item in conv if item.get("from") == "human"), None)
    assistant = next((item for item in reversed(conv) if item.get("from") == "gpt"), None)
    if user is None or assistant is None:
        raise ValueError(f"Missing human/gpt turns in ChangeChat row {raw.get('id')!r}")
    instruction = re.sub(r"<image>\s*", "", message_text(user)).strip()
    metadata: dict[str, Any] = {
        "dataset": "ChangeChat-105k",
        "split": "train",
        "training_source": training_source,
        "source_task": source_task,
        "source_row_id": raw.get("id"),
        "pair_id": pair_id(pair),
        "sample_category": category,
    }
    if raw.get("changeflag") in (0, 1):
        metadata["changeflag"] = int(raw["changeflag"])
    if extra_metadata:
        metadata.update(extra_metadata)
    return {
        "id": f"changechat-{raw.get('id', 'unknown')}-{category}",
        "images": [f"LEVIR-CC/images/{value}" for value in pair],
        "instruction": instruction,
        "answer": message_text(assistant),
        "task_type": "grid_localization" if category.startswith("grid_") else AUX_TASK_TYPES[category],
        "metadata": metadata,
    }


def aux_bucket(question: str) -> str:
    text = re.sub(r"^\s*(<image>\s*)+", "", question, flags=re.IGNORECASE).strip().lower()
    if "briefly describe the changes" in text:
        return "caption"
    if re.search(r"\bhow many\b|\bcount\b|\bnumber of\b", text):
        return "count"
    if (
        "yes or no" in text
        or "judge whether" in text
        or re.match(r"^(are|is|has|have|did|do|does)\b", text)
    ):
        return "binary"
    return "open_qa"


def jsonl_rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc


def vrs_task(row: dict[str, Any]) -> str:
    task = str(row.get("task_type", "unknown")).strip().lower()
    return task or "unknown"


def vrs_images(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    if isinstance(row.get("images"), list):
        values.extend(str(item) for item in row["images"])

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("type") == "image" and isinstance(value.get("image"), str):
                values.append(value["image"])
            for key, child in value.items():
                if key != "image":
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(row.get("messages", []))
    return list(dict.fromkeys(values))


def normalize_vrs_image(value: str) -> str:
    path = value.replace("\\", "/")
    if path.startswith("/") or re.match(r"^[A-Za-z]:/", path):
        return path
    if path.startswith("VRSBench/"):
        return path
    return f"VRSBench/{path.lstrip('/')}"


def rewrite_vrs_images(row: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(row)
    if isinstance(result.get("images"), list):
        result["images"] = [normalize_vrs_image(str(value)) for value in result["images"]]

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("type") == "image" and isinstance(value.get("image"), str):
                value["image"] = normalize_vrs_image(value["image"])
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(result.get("messages", []))
    metadata = result.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        result["metadata"] = metadata
    metadata["training_source"] = "vrsbench_replay"
    return result


def resolve_image(value: str, image_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else image_root / Path(*PurePosixPath(value).parts)


def row_images(row: dict[str, Any]) -> list[str]:
    if isinstance(row.get("images"), list):
        return [str(value) for value in row["images"]]
    return vrs_images(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--changechat-train", required=True, type=Path)
    parser.add_argument("--vrs-train", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-fraction", type=float, default=0.10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not 0.0 < args.dev_fraction < 0.5:
        parser.error("--dev-fraction must be greater than 0 and less than 0.5")
    if not args.changechat_train.is_file() or not args.vrs_train.is_file():
        raise FileNotFoundError("Both ChangeChat and VRSBench training files must exist")
    if not args.image_root.is_dir():
        raise FileNotFoundError(f"Image root does not exist: {args.image_root}")
    if not args.overwrite and (args.output_dir / "data/train.jsonl").exists():
        raise FileExistsError(f"Output already exists: {args.output_dir / 'data/train.jsonl'}")

    raw_rows = load_json(args.changechat_train)
    if not isinstance(raw_rows, list):
        raise ValueError("ChangeChat training JSON must be a top-level list")

    flags: dict[tuple[str, str], set[int]] = defaultdict(set)
    row_keys: dict[int, tuple[str, str]] = {}
    for raw in raw_rows:
        key = pair_key(raw)
        row_keys[id(raw)] = key
        try:
            flag = int(raw.get("changeflag"))
        except (TypeError, ValueError):
            continue
        if flag in (0, 1):
            flags[key].add(flag)

    grid_rows: list[tuple[dict[str, Any], tuple[str, str], set[str], set[str], str]] = []
    conflict_rows: list[tuple[dict[str, Any], tuple[str, str], set[str], set[str], str, str]] = []
    malformed_grid: list[str] = []
    for raw in raw_rows:
        conv = conversations(raw)
        question = message_text(conv[0]).lower() if conv else ""
        if "3x3 grid" not in question:
            continue
        key = row_keys[id(raw)]
        try:
            roads, buildings = parse_grid(message_text(conv[-1]))
        except ValueError as exc:
            malformed_grid.append(f"{raw.get('id')}: {exc}")
            continue
        label = grid_class(roads, buildings)
        pair_flags = sorted(flags.get(key, set()))
        reason = ""
        if pair_flags == [0] and label != "both_empty":
            reason = "grid_positive_but_global_changeflag_zero"
        elif len(pair_flags) > 1:
            reason = "inconsistent_global_changeflag_for_pair"
        item = (raw, key, roads, buildings, label)
        if reason:
            conflict_rows.append((*item, reason))
        else:
            grid_rows.append(item)
    if malformed_grid:
        raise ValueError("Malformed grid annotations; refusing to silently skip: " + malformed_grid[0])
    if not grid_rows:
        raise ValueError("No usable grid rows found")

    grid_train_items, grid_dev_items, split_stats = split_by_stratum(
        grid_rows, args.dev_fraction, args.seed
    )
    train_pairs = {item[1] for item in grid_train_items}
    if len(train_pairs) != len(grid_train_items):
        raise ValueError("Duplicate grid rows for an image pair; resolve before training")

    train_grid: list[dict[str, Any]] = []
    for raw, key, roads, buildings, label in grid_train_items:
        train_grid.append(
            make_changechat_row(
                raw,
                source_task="grid_localization",
                training_source="changechat_grid",
                category="grid_train",
                pair=key,
                extra_metadata={
                    "grid_class": label,
                    "road_cells": sorted(roads),
                    "building_cells": sorted(buildings),
                },
            )
        )
    dev_grid: list[dict[str, Any]] = []
    for raw, key, roads, buildings, label in grid_dev_items:
        dev_grid.append(
            make_changechat_row(
                raw,
                source_task="grid_localization",
                training_source="changechat_grid_dev",
                category="grid_dev",
                pair=key,
                extra_metadata={
                    "grid_class": label,
                    "road_cells": sorted(roads),
                    "building_cells": sorted(buildings),
                },
            )
        )

    aux_candidates: dict[str, dict[tuple[str, str], list[dict[str, Any]]]] = {
        key: defaultdict(list) for key in AUX_BUCKETS
    }
    seen_aux_records: set[tuple[tuple[str, str], str, str]] = set()
    aux_duplicate_rows = 0
    for raw in raw_rows:
        key = row_keys[id(raw)]
        if key not in train_pairs:
            continue
        conv = conversations(raw)
        if len(conv) != 2 or conv[0].get("from") != "human" or conv[1].get("from") != "gpt":
            continue
        question, answer = message_text(conv[0]), message_text(conv[1])
        if "3x3 grid" in question.lower():
            continue
        duplicate_key = (key, question.strip(), answer.strip())
        if duplicate_key in seen_aux_records:
            aux_duplicate_rows += 1
            continue
        seen_aux_records.add(duplicate_key)
        aux_candidates[aux_bucket(question)][key].append(raw)

    total_aux = int(round(len(train_grid) * (0.20 / 0.70)))
    aux_quotas = apportion(
        total_aux, {key: AUX_FRACTIONS[key] for key in AUX_BUCKETS}
    )
    aux_rng = random.Random(args.seed + 1)
    aux_rows: list[dict[str, Any]] = []
    used_pairs: set[tuple[str, str]] = set()
    aux_available: dict[str, int] = {}
    for category in AUX_BUCKETS:
        candidates = aux_candidates[category]
        aux_available[category] = len(candidates)
        pair_keys = [key for key in candidates if key not in used_pairs]
        pair_keys.sort(key=pair_id)
        aux_rng.shuffle(pair_keys)
        quota = aux_quotas[category]
        if len(pair_keys) < quota:
            raise ValueError(
                f"Insufficient unique-pair samples for {category}: "
                f"need {quota}, have {len(pair_keys)}"
            )
        for key in pair_keys[:quota]:
            used_pairs.add(key)
            choices = candidates[key]
            raw = choices[aux_rng.randrange(len(choices))]
            aux_rows.append(
                make_changechat_row(
                    raw,
                    source_task=category,
                    training_source="changechat_aux",
                    category=category,
                    pair=key,
                )
            )

    vrs_counts: Counter[str] = Counter()
    for row in jsonl_rows(args.vrs_train):
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        split = str(metadata.get("split", "train")).lower()
        if split != "train" or not vrs_images(row):
            continue
        vrs_counts[vrs_task(row)] += 1
    total_vrs = int(round(len(train_grid) * (0.10 / 0.70)))
    vrs_quotas = apportion(total_vrs, {key: float(value) for key, value in vrs_counts.items()})
    vrs_rng = random.Random(args.seed + 2)
    vrs_reservoirs: dict[str, list[dict[str, Any]]] = {key: [] for key in vrs_quotas}
    vrs_seen: Counter[str] = Counter()
    for row in jsonl_rows(args.vrs_train):
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        if str(metadata.get("split", "train")).lower() != "train" or not vrs_images(row):
            continue
        category = vrs_task(row)
        quota = vrs_quotas.get(category, 0)
        if quota <= 0:
            continue
        vrs_seen[category] += 1
        reservoir = vrs_reservoirs[category]
        if len(reservoir) < quota:
            reservoir.append(row)
        else:
            index = vrs_rng.randrange(vrs_seen[category])
            if index < quota:
                reservoir[index] = row
    vrs_rows = [rewrite_vrs_images(row) for category in sorted(vrs_reservoirs) for row in vrs_reservoirs[category]]
    if len(vrs_rows) != total_vrs:
        raise ValueError(f"VRS replay sample count mismatch: expected {total_vrs}, got {len(vrs_rows)}")

    train_rows = train_grid + aux_rows + vrs_rows
    random.Random(args.seed + 3).shuffle(train_rows)
    dev_rows = dev_grid
    review_rows: list[dict[str, Any]] = []
    for raw, key, roads, buildings, label, reason in conflict_rows:
        review_rows.append(
            make_changechat_row(
                raw,
                source_task="grid_localization_review",
                training_source="quarantine",
                category="grid_review",
                pair=key,
                extra_metadata={
                    "grid_class": label,
                    "road_cells": sorted(roads),
                    "building_cells": sorted(buildings),
                    "global_changeflags_for_pair": sorted(flags.get(key, set())),
                    "quarantine_reason": reason,
                },
            )
        )

    missing_images: list[str] = []
    for row in train_rows + dev_rows + review_rows:
        for image in row_images(row):
            resolved = resolve_image(image, args.image_root)
            if not resolved.is_file():
                missing_images.append(str(resolved))
                if len(missing_images) >= 25:
                    break
        if len(missing_images) >= 25:
            break
    if missing_images:
        raise FileNotFoundError(
            "Image references are missing under image_root (first paths): "
            + ", ".join(missing_images[:10])
        )

    data_dir = args.output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_rows = {
        "train.jsonl": train_rows,
        "grid_dev.jsonl": dev_rows,
        "grid_review.jsonl": review_rows,
        "grid_train.jsonl": train_grid,
        "changechat_aux.jsonl": aux_rows,
        "vrs_replay.jsonl": vrs_rows,
    }
    for name, rows in output_rows.items():
        write_jsonl(data_dir / name, rows)

    source_mix = Counter(str(row.get("metadata", {}).get("training_source", "unknown")) for row in train_rows)
    grid_label_counts = Counter(item[4] for item in grid_rows)
    grid_train_counts = Counter(str(row["metadata"]["grid_class"]) for row in train_grid)
    grid_dev_counts = Counter(str(row["metadata"]["grid_class"]) for row in dev_grid)
    aux_selected_counts = Counter(str(row["metadata"]["sample_category"]) for row in aux_rows)
    vrs_selected_counts = Counter(vrs_task(row) for row in vrs_rows)
    file_hashes = {name: sha256_file(data_dir / name) for name in output_rows}
    manifest = {
        "schema_version": "1.0",
        "purpose": "qwen3vl_2b_changechat_grid_first_ab",
        "seed": args.seed,
        "dev_fraction": args.dev_fraction,
        "image_root": str(args.image_root),
        "changechat_source": str(args.changechat_train),
        "changechat_source_sha256": sha256_file(args.changechat_train),
        "vrs_source": str(args.vrs_train),
        "vrs_source_sha256": sha256_file(args.vrs_train),
        "counts": {
            "changechat_grid_valid_before_quarantine": len(grid_rows) + len(conflict_rows),
            "changechat_grid_quarantine": len(conflict_rows),
            "changechat_grid_train": len(train_grid),
            "changechat_grid_dev": len(dev_grid),
            "changechat_aux": len(aux_rows),
            "vrs_replay": len(vrs_rows),
            "combined_train": len(train_rows),
            "grid_parse_errors": len(malformed_grid),
            "duplicate_aux_rows_ignored": aux_duplicate_rows,
            "image_references_missing": 0,
        },
        "quarantine_reasons": dict(Counter(str(row[5]) for row in conflict_rows)),
        "grid_label_counts_after_quarantine": dict(sorted(grid_label_counts.items())),
        "grid_train_label_counts": dict(sorted(grid_train_counts.items())),
        "grid_dev_label_counts": dict(sorted(grid_dev_counts.items())),
        "grid_pair_split_by_stratum": split_stats,
        "auxiliary": {
            "target_counts": aux_quotas,
            "available_unique_pair_counts": aux_available,
            "selected_counts": dict(sorted(aux_selected_counts.items())),
        },
        "vrs_replay": {
            "available_task_counts": dict(sorted(vrs_counts.items())),
            "target_counts": vrs_quotas,
            "selected_counts": dict(sorted(vrs_selected_counts.items())),
        },
        "combined_source_counts": dict(sorted(source_mix.items())),
        "combined_source_fractions": {
            key: value / len(train_rows) for key, value in sorted(source_mix.items())
        },
        "estimated_optimizer_steps_at_effective_batch_16": math.ceil(len(train_rows) / 16),
        "sha256": file_hashes,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
