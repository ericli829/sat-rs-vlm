from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REAL_SAMPLE_IDS = [
    "mme_rs_perception_remote_sensing_count_1099",
    "mme_rs_perception_remote_sensing_count_0026",
    "mme_rs_perception_remote_sensing_count_1057",
    "xlrs_001235",
    "xlrs_000860",
    "xlrs_000259",
    "xlrs_002087",
    "xlrs_000928",
    "xlrs_001998",
    "xlrs_000085",
    "mme_rs_perception_remote_sensing_color_0279",
    "mme_rs_perception_remote_sensing_count_0041",
    "mme_rs_perception_remote_sensing_count_0074",
    "mme_rs_perception_remote_sensing_count_2425",
    "xlrs_000708",
    "xlrs_000771",
    "xlrs_002985",
    "xlrs_002233",
    "xlrs_002028",
    "xlrs_000083",
    "xlrs_000118",
    "xlrs_000211",
    "mme_rs_perception_remote_sensing_position_0642",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def build(source: Path, output: Path, manifest_path: Path) -> dict[str, Any]:
    rows = _read_jsonl(source)
    by_id = {row["sample_id"]: row for row in rows}
    missing = [sample_id for sample_id in REAL_SAMPLE_IDS if sample_id not in by_id]
    if missing:
        raise ValueError(f"benchmark source is missing selected ids: {missing}")
    selected = [by_id[sample_id] for sample_id in REAL_SAMPLE_IDS]
    synthetic = {
        "sample_id": "batch_synthetic_multi_image_change_001",
        "question": (
            "What is the absolute difference between the number of ships in image0 "
            "and the number of ships in image1?"
        ),
        "question_type": "INTEGER",
        "choices": None,
        "inputs": {
            "image0": selected[0]["inputs"]["image0"],
            "image1": selected[14]["inputs"]["image0"],
        },
        "metadata": {
            "dataset": "TaskGraphSynthetic",
            "source_category": "multi-image/change-count",
            "benchmark_only": True,
            "source_sample_ids": [selected[0]["sample_id"], selected[14]["sample_id"]],
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in [*selected, synthetic]),
        encoding="utf-8",
        newline="\n",
    )
    manifest = {
        "version": "taskgraph-batch-benchmark-v1",
        "source": str(source.resolve()),
        "source_sha256": _sha256(source),
        "output": str(output.resolve()),
        "output_sha256": _sha256(output),
        "sample_count": 24,
        "real_sample_count": 23,
        "synthetic_sample_count": 1,
        "sample_ids": [row["sample_id"] for row in [*selected, synthetic]],
        "required_coverage": {
            "simple_count": [selected[0]["sample_id"], selected[14]["sample_id"]],
            "bbox": [selected[3]["sample_id"], selected[4]["sample_id"], selected[17]["sample_id"]],
            "marker": [selected[7]["sample_id"], selected[16]["sample_id"]],
            "relation": [selected[5]["sample_id"], selected[21]["sample_id"]],
            "relational_count": [selected[2]["sample_id"], selected[11]["sample_id"]],
            "rank_or_ordinal": [selected[10]["sample_id"], selected[12]["sample_id"]],
            "multi_image": [synthetic["sample_id"]],
            "route": [selected[8]["sample_id"], selected[18]["sample_id"]],
            "complex_reasoning": [
                selected[9]["sample_id"],
                selected[19]["sample_id"],
                selected[20]["sample_id"],
            ],
            "hard_graph_8_plus": [selected[2]["sample_id"]],
            "source_contract_conflict": [selected[6]["sample_id"]],
            "explicit_alternative": [selected[1]["sample_id"]],
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the fixed 24-sample batch benchmark")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.source, args.output, args.manifest),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
