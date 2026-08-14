"""Analyze Qwen3-VL training composition and exact assistant supervision."""

from __future__ import annotations

import argparse
import importlib
import time
from pathlib import Path

from sat_rs_vlm.data.qwen3vl_collator import Qwen3VLDataCollator
from sat_rs_vlm.data.qwen3vl_dataset import Qwen3VLDataset
from sat_rs_vlm.training.config import (
    TrainingPathOverrides,
    apply_training_overrides,
    load_training_config,
    resolve_path,
)
from sat_rs_vlm.training.data_statistics import (
    analyze_training_data,
    stratified_sample_by_task,
    task_counts,
    write_statistics_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--processor-dir", default=None)
    parser.add_argument("--train-file", default=None)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--samples-per-task",
        type=int,
        default=None,
        help=(
            "Randomly analyze up to this many samples per task. Recommended for "
            "fast supervised-token diagnostics. Cannot be used with --max-samples."
        ),
    )
    parser.add_argument(
        "--sampling-seed",
        type=int,
        default=None,
        help="Seed for --samples-per-task; defaults to training.seed.",
    )
    parser.add_argument("--skip-image-inspection", action="store_true")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress after this many processed samples; default: 100.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable per-sample processing progress output.",
    )
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
    if args.max_samples is not None and args.samples_per_task is not None:
        raise ValueError("--max-samples and --samples-per-task cannot be used together")
    max_samples = args.max_samples or config.statistics.max_samples
    dataset = Qwen3VLDataset(
        train_file,
        None if args.samples_per_task is not None else max_samples,
        skip_bad_samples=config.data.skip_bad_samples,
    )
    if args.progress_every < 1:
        raise ValueError("--progress-every must be positive")
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
    started = time.perf_counter()
    population = list(dataset)
    population_counts = task_counts(population)
    if args.samples_per_task is not None:
        seed = args.sampling_seed if args.sampling_seed is not None else config.training.seed
        analyzed_samples = stratified_sample_by_task(
            population,
            args.samples_per_task,
            seed=seed,
        )
        analysis_selection = {
            "mode": "stratified_by_task",
            "samples_per_task": args.samples_per_task,
            "seed": seed,
            "population_sample_count": len(population),
            "analyzed_sample_count": len(analyzed_samples),
        }
    else:
        analyzed_samples = population
        analysis_selection = {
            "mode": "prefix_limit" if max_samples is not None else "full_dataset",
            "population_sample_count": len(population),
            "analyzed_sample_count": len(analyzed_samples),
        }
    total_samples = len(analyzed_samples)
    print(
        "Starting training-data statistics: "
        f"samples={total_samples}/{len(population)}, max_seq_length={config.data.max_seq_length}, "
        f"image_inspection={not args.skip_image_inspection}",
        flush=True,
    )

    def show_progress(processed: int, total: int) -> None:
        elapsed = time.perf_counter() - started
        percent = processed / total * 100 if total else 100.0
        rate = processed / elapsed if elapsed > 0 else 0.0
        remaining = (total - processed) / rate if rate > 0 else 0.0
        print(
            "Progress: "
            f"{processed}/{total} ({percent:.1f}%), "
            f"elapsed={elapsed:.1f}s, eta={remaining:.1f}s",
            flush=True,
        )

    report = analyze_training_data(
        analyzed_samples,
        collator,
        image_root=image_root,
        bbox_thresholds=config.statistics.bbox_area_thresholds,
        inspect_images=config.statistics.inspect_images and not args.skip_image_inspection,
        progress_callback=None if args.no_progress else show_progress,
        progress_every=args.progress_every,
        population_task_counts=population_counts,
        analysis_selection=analysis_selection,
        training_sampling_mode=config.data.sampling_mode,
        task_sampling_weights=config.data.task_sampling_weights,
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
    print(f"Statistics completed in {time.perf_counter() - started:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
