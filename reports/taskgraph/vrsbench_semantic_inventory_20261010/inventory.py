"""Read existing VRSBench only; mine review candidates, never training labels."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
DATA = Path("F:/VIT-data/VRSBench")
OUT = Path(__file__).parent
PATTERNS = {
    "background": (
        r"\b(?:trees?|grass|green area|green land|vegetation|forest|roads?|river|water|"
        r"parking (?:area|lot)|runway)\b"
    ),
    "relation": (
        r"\b(?:to (?:the )?(?:left|right) of|(?:above|below|behind|beside|between|near|"
        r"next to|adjacent to|surrounded by|in front of|closest to|nearest to|bordered by))\b"
    ),
    "color": r"\b(?:white|black|red|green|blue|gray|grey|yellow|brown|dark|bright|silver|orange)\b",
    "shape": (
        r"\b(?:square|rectangular|rectangle|circular|round|triangular|triangle|oval|"
        r"l-shaped|u-shaped)\b"
    ),
    "size": r"\b(?:small|large|big|tiny|long|short)\b",
    "extreme": (
        r"\b(?:left[- ]?most|right[- ]?most|top[- ]?most|bottom[- ]?most|uppermost|"
        r"lowermost|largest|smallest)\b"
    ),
    "ordinal": r"\b(?:second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b",
    "absolute_position": (
        r"\b(?:top[- ](?:left|right)|bottom[- ](?:left|right)|upper[- ](?:left|right)|"
        r"lower[- ](?:left|right)|center|centre|corner|middle)\b"
    ),
}
COMPILED = {key: re.compile(value, re.I) for key, value in PATTERNS.items()}


def main() -> None:
    stats: dict = {"source_root": str(DATA), "splits": {}, "heuristic_patterns": PATTERNS}
    stats["scope"] = (
        "Full scan of the existing converted train/val JSONL; keyword flags are "
        "overlapping review hints, not gold TaskGraphs. Selected examples are "
        "cross-checked against original per-image annotations."
    )
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    images: dict[str, set[str]] = {}
    same_class: Counter = Counter()
    for split in ("train", "val"):
        source = ROOT / f"data/processed/rs_{split}.jsonl"
        tasks: Counter = Counter()
        qa_types: Counter = Counter()
        tags_by_kind: dict[str, Counter] = defaultdict(Counter)
        seen: set[str] = set()
        labels: Counter = Counter()
        quality: Counter = Counter()
        caption_images: Counter = Counter()
        for line_no, line in enumerate(source.open(encoding="utf-8"), 1):
            row = json.loads(line)
            meta = row["metadata"]
            assert meta["dataset"] == "VRSBench"
            seen.add(meta["source_image"])
            tasks[row["task_type"]] += 1
            if meta.get("qa_type"):
                qa_types[meta["qa_type"]] += 1
            if row["task_type"] == "captioning":
                caption_images[meta["source_image"]] += 1
                continue
            kind = "referring" if meta["source_task"] == "referring" else "qa"
            question = row["instruction"]
            label, bbox = None, None
            if kind == "referring":
                match = re.fullmatch(
                    r'Locate the object described as: "(.*)"\. '
                    r"Return its class and normalized bounding box\.",
                    question,
                )
                assert match, row["id"]
                question = match[1]
                answer = json.loads(row["answer"])
                label, bbox = answer["label"], answer["bbox"]
                labels[label] += 1
                same_class[split, meta["source_image"], label] += 1
                quality["converted_bbox_clipped"] += bool(meta.get("coordinate_clipped"))
                raw_bbox = meta.get("bbox_raw", bbox)
                quality["raw_bbox_outside_unit_square"] += any(
                    value < 0 or value > 1 for value in raw_bbox
                )
                quality["converted_bbox_nonpositive_area"] += (
                    bbox[0] >= bbox[2] or bbox[1] >= bbox[3]
                )
            matched = {name: pattern.findall(question) for name, pattern in COMPILED.items()}
            tags = {name for name, values in matched.items() if values}
            if matched["background"] and matched["relation"]:
                tags.add("background_relation")
            if len(matched["relation"]) >= 2:
                tags.add("multiple_relation_mentions")
            if "background_relation" in tags and (
                len(matched["relation"]) >= 2
                or len(tags & {"color", "shape", "size", "absolute_position"}) >= 2
            ):
                tags.add("compound_background_candidate")
            if row["task_type"] == "counting" and matched["relation"]:
                tags.add("relational_count_candidate")
            tags_by_kind[kind].update(tags)
            record = {
                "id": row["id"],
                "split": split,
                "kind": kind,
                "question": question,
                "original_instruction": row["instruction"],
                "answer": row["answer"],
                "qa_type": meta.get("qa_type"),
                "label": label,
                "bbox_normalized": bbox,
                "tags": sorted(tags),
                "source_file": str(source),
                "source_line": line_no,
                "image_relative": row["images"][0],
                "source_annotation": meta["source_annotation"],
                "object_id": meta.get("object_id"),
                "question_id": meta.get("question_id"),
                "review_status": "keyword_candidate_not_gold_plan",
            }
            for tag in tags & {
                "compound_background_candidate",
                "multiple_relation_mentions",
                "background_relation",
                "extreme",
                "ordinal",
                "relational_count_candidate",
            }:
                buckets[split, kind + "." + tag].append(record)
        stats["splits"][split] = {
            "source": str(source),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "rows": sum(tasks.values()),
            "unique_images": len(seen),
            "task_counts": tasks,
            "qa_types": qa_types,
            "heuristic_counts": tags_by_kind,
            "referring_labels": labels,
            "quality_flags": quality,
            "duplicate_caption_images": {
                name: count for name, count in caption_images.items() if count > 1
            },
        }
        images[split] = seen
    stats["image_name_overlap"] = len(images["train"] & images["val"])
    parent_groups = {
        split: {Path(image).stem.split("_")[0] for image in names}
        for split, names in images.items()
    }
    stats["parent_prefix_overlap"] = len(parent_groups["train"] & parent_groups["val"])
    stats["parent_prefix_overlap_note"] = (
        "Filename-prefix heuristic for related tiles, not verified geographic deduplication."
    )
    selected: dict[str, dict] = {}
    for _bucket_key, records in sorted(buckets.items()):
        records.sort(key=lambda row: (-len(row["tags"]), len(row["question"]), row["id"]))
        used_images: set[str] = set()
        for row in records:
            if row["image_relative"] in used_images:
                continue
            selected.setdefault(row["id"], row)
            used_images.add(row["image_relative"])
            if len(used_images) == 8:
                break
    raw_cache: dict = {}
    for row in selected.values():
        split = row["split"]
        annotation = DATA / f"Annotations/Annotations_{split}" / row["source_annotation"]
        if str(annotation) not in raw_cache:
            raw_cache[str(annotation)] = json.loads(annotation.read_text(encoding="utf-8"))
        raw = raw_cache[str(annotation)]
        image_path = DATA / row["image_relative"]
        assert image_path.is_file(), image_path
        with Image.open(image_path) as image:
            row["image_size"] = list(image.size)
        row["image_absolute"] = str(image_path)
        if row["kind"] == "referring":
            objects = [item for item in raw["objects"] if item["obj_id"] == row["object_id"]]
            assert len(objects) == 1
            obj = objects[0]
            assert obj["referring_sentence"] == row["question"]
            row["is_unique_in_category"] = obj["is_unique"]
            row["raw_obj_coord"] = obj["obj_coord"]
            row["raw_relative_position"] = obj.get("obj_rel_position")
            row["raw_relative_size"] = obj.get("obj_rel_size")
            row["annotated_same_class_count"] = same_class[split, raw["image"], row["label"]]
        else:
            matches = [item for item in raw["qa_pairs"] if item["ques_id"] == row["question_id"]]
            assert len(matches) == 1 and matches[0]["question"] == row["question"]
        row["source_annotation_absolute"] = str(annotation)
        row["source_annotation_sha256"] = hashlib.sha256(annotation.read_bytes()).hexdigest()
    stats["review_candidate_count"] = len(selected)
    stats["planner_coverage"] = []
    for source in sorted((ROOT / "taskgraph_lab/data").glob("planner_sft*/train.jsonl")):
        datasets: Counter = Counter()
        semantic_targets = 0
        system_semantic_rows = 0
        for line in source.open(encoding="utf-8"):
            row = json.loads(line)
            datasets[row.get("metadata", {}).get("dataset", "unknown")] += 1
            semantic_targets += any(
                "SELECT_SEM(" in msg.get("content", "")
                for msg in row.get("messages", [])
                if msg.get("role") == "assistant"
            )
            system_semantic_rows += any(
                "SELECT_SEM(" in msg.get("content", "")
                for msg in row.get("messages", [])
                if msg.get("role") == "system"
            )
        stats["planner_coverage"].append(
            {
                "file": str(source),
                "datasets": datasets,
                "select_sem_targets": semantic_targets,
                "system_rows_mentioning_select_sem": system_semantic_rows,
            }
        )
    stats["review_image_sizes"] = Counter(
        "x".join(map(str, row["image_size"])) for row in selected.values()
    )
    stats["review_by_split"] = Counter(row["split"] for row in selected.values())
    (OUT / "summary.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (OUT / "review_candidates.jsonl").open("w", encoding="utf-8") as output:
        for row in selected.values():
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "splits": stats["splits"],
                "candidates": len(selected),
                "image_name_overlap": stats["image_name_overlap"],
                "parent_prefix_overlap": stats["parent_prefix_overlap"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
