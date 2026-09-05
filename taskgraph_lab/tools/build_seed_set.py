from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from taskgraph_lab.datasets.base import NormalizedSample
from taskgraph_lab.datasets.mme_rs import iter_mme_rs
from taskgraph_lab.datasets.xlrs import iter_xlrs

COUNT_HINT = re.compile(r"\b(?:how many|number of|count|quantity)\b", re.IGNORECASE)
ABSOLUTE_REGION = re.compile(
    r"\b(?:top|bottom|upper|lower|left|right|center|middle)(?:[- ](?:left|right|center|middle))?\b",
    re.IGNORECASE,
)
RELATION = re.compile(
    r"\b(?:left of|right of|above|below|near|next to|inside|outside|between|around|both sides)\b",
    re.IGNORECASE,
)


def structure_category(sample: NormalizedSample) -> str:
    text = f"{sample.question} {sample.metadata.get('source_category', '')}".lower()
    image_count = len(sample.inputs)
    if (
        image_count >= 2
        and COUNT_HINT.search(text)
        and re.search(r"difference|change|between", text)
    ):
        return "two_image_abs_diff"
    if "route" in text or "driving" in text and ("from" in text and "to" in text):
        return "route_planning"
    bbox = "bounding box" in text or "bbox" in text or "reference box" in text
    if bbox:
        if re.search(r"motion|moving|stationary", text):
            return "bbox_motion"
        if re.search(r"class|category|type of object|identify", text):
            return "bbox_classify"
        return "bbox_attribute"
    if re.search(
        r"\b(?:red|blue|green|yellow|colored|coloured) (?:circle|box|rectangle|border)", text
    ):
        return "marker_region"
    if (
        re.search(r"\b(?:why|cause|reason|explain)\b", text)
        and len(ABSOLUTE_REGION.findall(text)) >= 2
    ):
        return "multi_region_complex_reasoning"
    if COUNT_HINT.search(text):
        if re.search(r"\b(?:row|column|cluster)\b", text):
            return "group_row_column_cluster"
        if re.search(
            r"\b(?:first|second|third|fourth|fifth)\b.*\b(?:top|bottom|left|right)\b", text
        ):
            return "ordinal"
        if re.search(r"\b(?:largest|smallest|nearest|farthest|second largest)\b", text):
            return "rank_superlative"
        if re.search(r"\b(?:leftmost|rightmost|topmost|bottommost)\b", text):
            return "extreme"
        if RELATION.search(text):
            return "nested_relational_count"
        if ABSOLUTE_REGION.search(text):
            return "absolute_region_count"
        return "entire_image_count"
    if re.search(r"\b(?:relation|relative position|where is .* relative)\b", text):
        return "relation_query"
    if re.search(r"select all|multiple labels|land uses|land-use types", text):
        return "multi_label_classification"
    if re.search(r"\b(?:classify|classification|land use|scene type|what type)\b", text):
        return "single_label_classification"
    if len(ABSOLUTE_REGION.findall(text)) >= 2:
        return "multi_region_complex_reasoning"
    return "other"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_normalized(path: Path) -> Iterable[NormalizedSample]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield NormalizedSample.model_validate_json(line)
                except Exception as exc:
                    raise ValueError(f"invalid normalized row {path}:{line_number}: {exc}") from exc


def build_seed_set(
    samples: Iterable[NormalizedSample],
    config: dict[str, Any],
    *,
    excluded_ids: set[str] | None = None,
) -> tuple[list[NormalizedSample], dict[str, Any]]:
    max_total = int(config.get("max_total", 250))
    seed = int(config.get("seed", 0))
    per_dataset = {
        str(key): int(value) for key, value in dict(config.get("per_dataset") or {}).items()
    }
    per_category = {
        str(key): int(value) for key, value in dict(config.get("per_category") or {}).items()
    }
    include_categories = {
        str(value) for value in list(config.get("include_categories") or [])
    }
    candidates: list[tuple[NormalizedSample, str]] = []
    seen: set[str] = set()
    excluded = excluded_ids or set()
    excluded_candidate_count = 0
    category_filtered_count = 0
    for sample in samples:
        if sample.sample_id in seen:
            raise ValueError(f"duplicate seed candidate sample_id: {sample.sample_id}")
        seen.add(sample.sample_id)
        if sample.sample_id in excluded:
            excluded_candidate_count += 1
            continue
        category = structure_category(sample)
        if include_categories and category not in include_categories:
            category_filtered_count += 1
            continue
        candidates.append((sample, category))

    def rank(item: tuple[NormalizedSample, str]) -> tuple[str, str]:
        sample, _ = item
        digest = hashlib.sha256(f"{seed}:{sample.sample_id}".encode()).hexdigest()
        return digest, sample.sample_id

    candidates.sort(key=rank)
    selected: list[tuple[NormalizedSample, str]] = []
    selected_ids: set[str] = set()
    dataset_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()

    def can_take(item: tuple[NormalizedSample, str]) -> bool:
        sample, category = item
        dataset = str(sample.metadata.get("dataset", "unknown"))
        dataset_limit = per_dataset.get(dataset, max_total)
        category_limit = per_category.get(category, max_total)
        return (
            len(selected) < max_total
            and dataset_counts[dataset] < dataset_limit
            and category_counts[category] < category_limit
        )

    def take(item: tuple[NormalizedSample, str]) -> None:
        sample, category = item
        selected.append(item)
        selected_ids.add(sample.sample_id)
        dataset_counts[str(sample.metadata.get("dataset", "unknown"))] += 1
        category_counts[category] += 1

    for category in per_category:
        for item in candidates:
            if item[1] == category and item[0].sample_id not in selected_ids and can_take(item):
                take(item)
    for item in candidates:
        if item[0].sample_id not in selected_ids and can_take(item):
            take(item)
    selected.sort(key=lambda item: item[0].sample_id)
    manifest = {
        "seed": seed,
        "max_total": max_total,
        "excluded_id_count": len(excluded),
        "excluded_candidate_count": excluded_candidate_count,
        "include_categories": sorted(include_categories),
        "category_filtered_count": category_filtered_count,
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "dataset_distribution": dict(sorted(dataset_counts.items())),
        "category_distribution": dict(sorted(category_counts.items())),
        "configured_per_dataset": per_dataset,
        "configured_per_category": per_category,
        "sample_ids": [item[0].sample_id for item in selected],
    }
    return [item[0] for item in selected], manifest


def _load_excluded_ids(paths: Iterable[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise TypeError(f"{path}:{line_number} exclusion row must be an object")
                sample = payload.get("sample")
                sample_id = payload.get("sample_id", payload.get("id"))
                if sample_id is None and isinstance(sample, dict):
                    sample_id = sample.get("sample_id")
                value = str(sample_id or "").strip()
                if not value:
                    raise ValueError(f"{path}:{line_number} exclusion row has no sample id")
                excluded.add(value)
    return excluded


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a deterministic structure-coverage seed set"
    )
    parser.add_argument("--input", type=Path, action="append", default=[])
    parser.add_argument("--xlrs-json", type=Path)
    parser.add_argument("--mme-json", type=Path)
    parser.add_argument(
        "--exclude",
        type=Path,
        action="append",
        default=[],
        help="JSONL whose sample_id/id values must not enter the new seed",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.input and args.xlrs_json is None and args.mme_json is None:
        parser.error("provide --input and/or --xlrs-json/--mme-json")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("seed sampling config must be a mapping")

    def samples() -> Iterable[NormalizedSample]:
        for path in args.input:
            yield from _iter_normalized(path)
        if args.xlrs_json is not None:
            yield from iter_xlrs(args.xlrs_json)
        if args.mme_json is not None:
            yield from iter_mme_rs(args.mme_json)

    excluded_ids = _load_excluded_ids(args.exclude)
    selected, manifest = build_seed_set(samples(), config, excluded_ids=excluded_ids)
    sources = [*args.input]
    if args.xlrs_json is not None:
        sources.append(args.xlrs_json)
    if args.mme_json is not None:
        sources.append(args.mme_json)
    manifest["sources"] = [
        {"path": str(path.resolve()), "sha256": _sha256(path.resolve())} for path in sources
    ]
    manifest["exclusions"] = [
        {"path": str(path.resolve()), "sha256": _sha256(path.resolve())}
        for path in args.exclude
    ]
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    with (output / "seed.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for sample in selected:
            handle.write(json.dumps(sample.model_dump(mode="json"), ensure_ascii=False) + "\n")
    (output / "seed_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    distribution = {
        "dataset_distribution": manifest["dataset_distribution"],
        "category_distribution": manifest["category_distribution"],
    }
    (output / "category_distribution.json").write_text(
        json.dumps(distribution, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps({"output_dir": str(output.resolve()), **manifest}, ensure_ascii=False, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
