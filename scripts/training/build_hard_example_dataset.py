"""Build H1 hard-example plus regular-replay data from Evaluation v1.5 outputs."""

from __future__ import annotations

import argparse

from sat_rs_vlm.training.config import load_training_config, resolve_path
from sat_rs_vlm.training.hard_example_mining import (
    build_hard_example_dataset,
    load_evaluation_ids,
    load_rows,
    resolve_evaluated_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--train-file", default=None)
    parser.add_argument("--evaluation-ids", default=None)
    parser.add_argument("--source-checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def _required(value: str | None, description: str) -> str:
    if not value:
        raise ValueError(f"Missing {description}; set it in hard_adaptation or the CLI")
    return value


def main() -> int:
    args = parse_args()
    config = load_training_config(args.config, allow_unresolved_env=True)
    hard = config.hard_adaptation
    prediction_value = _required(args.predictions or hard.prediction_source, "prediction source")
    train_value = _required(args.train_file or hard.source_train_file, "source training file")
    exclusion_value = _required(
        args.evaluation_ids or hard.evaluation_ids_file,
        "fixed evaluation ID file",
    )
    checkpoint = _required(
        args.source_checkpoint or hard.source_checkpoint or config.lora.initial_adapter_dir,
        "source checkpoint",
    )
    predictions = resolve_evaluated_predictions(resolve_path(prediction_value))
    train_file = resolve_path(train_value)
    exclusions = load_evaluation_ids(resolve_path(exclusion_value))
    output_dir = resolve_path(args.output_dir or hard.output_dir)
    evaluated_rows = load_rows(predictions)
    if not evaluated_rows or any("sample_metrics" not in row for row in evaluated_rows):
        raise ValueError("Hard mining requires Evaluation v1.5 evaluated_predictions.jsonl rows")
    manifest = build_hard_example_dataset(
        load_rows(train_file),
        evaluated_rows,
        exclusions,
        hard,
        seed=args.seed if args.seed is not None else config.training.seed,
        output_dir=output_dir,
        prediction_source=str(predictions),
        source_checkpoint=checkpoint,
    )
    print(f"Saved H1 dataset: {output_dir / 'h1_train.jsonl'}")
    print(f"Saved hard manifest: {output_dir / 'hard_manifest.json'}")
    print(
        f"Samples: hard={manifest['hard_sample_count']}, replay={manifest['regular_replay_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
