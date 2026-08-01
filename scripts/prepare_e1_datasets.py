"""生成 E1 balanced 和 detection/counting 专项派生数据，不复制来源数据。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sat_rs_vlm.data.sampling import allocate_quotas, group_by_task, sample_by_task
from sat_rs_vlm.utils.jsonl import read_jsonl, write_jsonl

TRAIN_FOCUS = {
    "detection": 3200,
    "counting": 2400,
    "captioning": 800,
    "vqa": 800,
    "scene_classification": 800,
}
VAL_FOCUS = {
    "detection": 400,
    "counting": 300,
    "captioning": 100,
    "vqa": 100,
    "scene_classification": 100,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare E1 derived datasets.")
    parser.add_argument("--train-input", default="data/processed/qwen3vl_train.jsonl")
    parser.add_argument("--val-input", default="data/processed/qwen3vl_val.jsonl")
    parser.add_argument("--output-dir", default="data/processed/e1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _build(
    source: Path,
    output: Path,
    *,
    total: int | None,
    explicit: dict[str, int] | None,
    seed: int,
    overwrite: bool,
) -> dict[str, Any]:
    if output.exists() and not overwrite:
        raise FileExistsError(f"Derived dataset exists; pass --overwrite: {output}")
    rows = list(read_jsonl(source))
    grouped = group_by_task(rows)
    quotas = allocate_quotas(grouped, total=total, explicit=explicit)
    selected, counts = sample_by_task(rows, quotas, seed=seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, selected)
    return {"source": str(source), "output": str(output), "samples": len(selected), "tasks": counts}


def main() -> int:
    args = parse_args()
    train = Path(args.train_input)
    validation = Path(args.val_input)
    for path in (train, validation):
        if not path.is_file():
            raise SystemExit(f"Input JSONL does not exist: {path}")
    output = Path(args.output_dir)
    reports = [
        _build(
            train,
            output / "qwen3vl_train_balanced_8k.jsonl",
            total=8000,
            explicit=None,
            seed=args.seed,
            overwrite=args.overwrite,
        ),
        _build(
            validation,
            output / "qwen3vl_val_balanced_1k.jsonl",
            total=1000,
            explicit=None,
            seed=args.seed,
            overwrite=args.overwrite,
        ),
        _build(
            train,
            output / "qwen3vl_train_detection_focus_8k.jsonl",
            total=None,
            explicit=TRAIN_FOCUS,
            seed=args.seed,
            overwrite=args.overwrite,
        ),
        _build(
            validation,
            output / "qwen3vl_val_detection_focus_1k.jsonl",
            total=None,
            explicit=VAL_FOCUS,
            seed=args.seed,
            overwrite=args.overwrite,
        ),
    ]
    stats_file = output / "dataset_statistics.json"
    stats_file.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Prepared {len(reports)} E1 datasets under {output}; stats={stats_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
