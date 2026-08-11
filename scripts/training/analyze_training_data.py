"""Analyze Qwen3-VL training composition and exact assistant supervision."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from sat_rs_vlm.data.qwen3vl_collator import Qwen3VLDataCollator
from sat_rs_vlm.data.qwen3vl_dataset import Qwen3VLDataset
from sat_rs_vlm.training.config import (
    TrainingPathOverrides,
    apply_training_overrides,
    load_training_config,
    resolve_path,
)
from sat_rs_vlm.training.data_statistics import analyze_training_data, write_statistics_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--processor-dir", default=None)
    parser.add_argument("--train-file", default=None)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-image-inspection", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_training_config(args.config, allow_unresolved_env=True)
    config = apply_training_overrides(
        config,
        TrainingPathOverrides(
            processor_dir=args.processor_dir,
            train_file=args.train_file,
            image_root=args.image_root,
        ),
    )
    processor_value = config.model.processor_dir or config.model.processor_id
    if not processor_value:
        raise ValueError("Set model.processor_dir/processor_id or --processor-dir")
    processor_source = (
        str(resolve_path(processor_value)) if config.model.processor_dir else processor_value
    )
    train_file = resolve_path(config.data.train_file)
    image_root = resolve_path(config.data.image_root)
    max_samples = args.max_samples or config.statistics.max_samples
    dataset = Qwen3VLDataset(
        train_file,
        max_samples,
        skip_bad_samples=config.data.skip_bad_samples,
    )
    try:
        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        message = 'transformers is required; install with pip install -e ".[model]"'
        raise SystemExit(message) from exc
    processor = transformers.AutoProcessor.from_pretrained(
        processor_source,
        trust_remote_code=config.model.trust_remote_code,
        local_files_only=config.model.local_files_only,
    )
    collator = Qwen3VLDataCollator(processor, config.data.max_seq_length, image_root)
    report = analyze_training_data(
        list(dataset),
        collator,
        image_root=image_root,
        bbox_thresholds=config.statistics.bbox_area_thresholds,
        inspect_images=config.statistics.inspect_images and not args.skip_image_inspection,
    )
    run_name = args.run_name or config.logging.experiment_name
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else resolve_path(config.statistics.output_dir) / run_name
    )
    outputs = write_statistics_report(report, output_dir)
    print(f"Saved statistics JSON: {outputs['summary_json']}")
    print(f"Saved statistics Markdown: {outputs['summary_md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
