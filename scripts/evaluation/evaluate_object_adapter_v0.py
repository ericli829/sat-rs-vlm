"""Evaluate an RS Object Adapter v0 checkpoint on a frozen evaluation tier.

This evaluator deliberately covers only VRSBench detection and counting.  It
loads E1 or E2 through the tier manifest, resolves the requested class from
metadata or the prompt, and never reads a reference label to decide the query
class. E2 remains the default tier.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sat_rs_vlm.configuration.environment import expand_environment  # noqa: E402
from sat_rs_vlm.data.object_adapter_v0 import (  # noqa: E402
    canonical_image_identity,
    extract_answer,
    extract_prompt,
    resolve_counting_class,
    resolve_prompt_class,
)
from sat_rs_vlm.data.task_protocol import parse_count, parse_detection  # noqa: E402
from sat_rs_vlm.models.reliability.checksum import file_sha256  # noqa: E402
from sat_rs_vlm.utils.jsonl import read_jsonl  # noqa: E402


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Object Adapter config must be a mapping: {path}")
    expanded = expand_environment(payload, environ=os.environ, allow_unresolved=False)
    return dict(expanded)


def _tier_rows(
    config: dict[str, Any], tier: str
) -> tuple[list[dict[str, Any]], Path, str | None]:
    data = dict(config.get("data", {}))
    tier = str(tier).strip().upper()
    if tier not in {"E1", "E2"}:
        raise ValueError(f"Object Adapter evaluator supports E1 or E2, got: {tier}")
    manifest_value = data.get("evaluation_tier_manifest")
    if not manifest_value:
        raise ValueError("data.evaluation_tier_manifest is required for tier evaluation")
    manifest_path = _project_path(str(manifest_value))
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Fixed evaluation tier manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = dict(manifest.get("tiers", {}).get(tier, {}))
    if not record:
        raise ValueError(f"Evaluation tier manifest has no {tier} record: {manifest_path}")
    tier_value = Path(str(record.get("path", ""))).expanduser()
    candidates = [
        tier_value,
        PROJECT_ROOT / tier_value,
        manifest_path.parent / tier_value,
    ]
    tier_path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if tier_path is None:
        raise FileNotFoundError(f"{tier} JSONL is missing; checked {candidates}")
    expected_sha = record.get("sha256")
    actual_sha = file_sha256(tier_path)
    if expected_sha and str(expected_sha) != actual_sha:
        raise ValueError(f"{tier} tier SHA256 mismatch: expected={expected_sha}, actual={actual_sha}")
    return list(read_jsonl(tier_path)), tier_path, actual_sha


def _prepare_rows(rows: list[dict[str, Any]], class_vocab: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Prepare only supported rows; class resolution never inspects assistant text."""

    class_to_id = {str(key): int(value) for key, value in class_vocab["class_to_id"].items()}
    prepared: list[dict[str, Any]] = []
    skipped = {"non_vrsbench": 0, "unsupported_task": 0, "class_unresolved": 0, "answer_unparsed": 0}
    for row in rows:
        metadata = row.get("metadata", {})
        dataset = str(metadata.get("dataset", row.get("dataset", ""))) if isinstance(metadata, dict) else ""
        if dataset != "VRSBench":
            skipped["non_vrsbench"] += 1
            continue
        task = str(row.get("task_type", "")).strip().lower()
        if task not in {"detection", "counting"}:
            skipped["unsupported_task"] += 1
            continue
        # Detection queries must be resolved from the prompt only.  Counting
        # keeps the builder's metadata-first rule, but neither path reads the
        # assistant answer to choose a class.
        resolution = (
            resolve_prompt_class(extract_prompt(row), class_vocab)
            if task == "detection"
            else resolve_counting_class(row, class_vocab)
        )
        if resolution.status != "resolved" or resolution.class_name not in class_to_id:
            skipped["class_unresolved"] += 1
            continue
        answer = extract_answer(row)
        item: dict[str, Any] = {
            "id": str(row.get("id", "")),
            "image": canonical_image_identity(row),
            "class_name": resolution.class_name,
            "class_id": class_to_id[resolution.class_name],
            "task_type": task,
            "class_resolution_source": resolution.source,
        }
        if not item["image"]:
            skipped["answer_unparsed"] += 1
            continue
        if task == "detection":
            coordinate_format = str(metadata.get("bbox_target_format", "normalized_0_1")) if isinstance(metadata, dict) else "normalized_0_1"
            parsed = parse_detection(answer, coordinate_format=coordinate_format)
            if parsed is None or not parsed.valid_coordinate_range:
                skipped["answer_unparsed"] += 1
                continue
            item["boxes_xyxy"] = [list(parsed.bbox)]
            item["count"] = None
        else:
            parsed_count = parse_count(answer)
            if parsed_count.value is None:
                skipped["answer_unparsed"] += 1
                continue
            item["boxes_xyxy"] = []
            item["count"] = int(parsed_count.value)
        prepared.append(item)
    prepared.sort(key=lambda item: str(item["id"]))
    return prepared, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/rs_object_adapter_v0_4090.yaml"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument(
        "--evaluation-tier",
        "--tier",
        dest="evaluation_tier",
        default=None,
        choices=("E1", "E2", "e1", "e2"),
        help="Frozen evaluation tier; defaults to evaluation.tier or E2.",
    )
    parser.add_argument(
        "--r1-checkpoint-dir",
        type=Path,
        default=None,
        help="R1 Qwen/PEFT checkpoint; defaults to source_r1_checkpoint in adapter_manifest.json",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = _load_config(_project_path(args.config))
    evaluation_tier = str(
        args.evaluation_tier
        or dict(config.get("evaluation", {})).get("tier", "E2")
    ).strip().upper()
    checkpoint = _project_path(args.checkpoint_dir or args.checkpoint)
    output_dir = _project_path(
        args.output_dir
        or dict(config.get("evaluation", {})).get(
            "output_dir", str(checkpoint / f"evaluation_{evaluation_tier.lower()}")
        )
    )
    try:
        modules = __import__("sat_rs_vlm.training.utils", fromlist=["safe_import_model_dependencies"])
        imported = modules.safe_import_model_dependencies()
        torch = imported["torch"]
        from safetensors.torch import load_file

        from sat_rs_vlm.evaluation.checkpoint_loader import load_finetuned_checkpoint
        from sat_rs_vlm.models.rs_object_adapter import RSObjectAdapter
        from sat_rs_vlm.training.object_adapter_v0 import (
            FrozenVisualFeatureExtractor,
            _box_metrics,
            _cast_features_for_adapter,
            _metrics_from_count_pairs,
            pad_visual_features,
            visual_processor_batch,
        )
        from sat_rs_vlm.training.vision_tuning import load_visual_sidecar, resolve_visual_module

        rows, tier_path, tier_sha = _tier_rows(config, evaluation_tier)
        checkpoint_vocab_path = checkpoint / "class_vocab.json"
        if not checkpoint_vocab_path.is_file():
            raise FileNotFoundError(f"Object Adapter checkpoint is missing class_vocab.json: {checkpoint}")
        class_vocab = json.loads(checkpoint_vocab_path.read_text(encoding="utf-8"))
        eval_rows, skipped = _prepare_rows(rows, class_vocab)
        if args.max_samples is not None:
            eval_rows = eval_rows[: args.max_samples]
        if not eval_rows:
            raise ValueError(
                "No supported, parseable VRSBench detection/counting rows remain in "
                f"{evaluation_tier}"
            )

        adapter_manifest_path = checkpoint / "adapter_manifest.json"
        adapter_manifest = json.loads(adapter_manifest_path.read_text(encoding="utf-8"))
        weight_path = checkpoint / str(adapter_manifest.get("weights", "adapter_model.safetensors"))
        if not weight_path.is_file():
            raise FileNotFoundError(f"Object Adapter weights are missing: {weight_path}")
        expected_weights_sha = adapter_manifest.get("weights_sha256")
        if expected_weights_sha and str(expected_weights_sha) != file_sha256(weight_path):
            raise ValueError("Object Adapter weight checksum mismatch")

        model_cfg = dict(config.get("model", {}))
        loader_config = {
            "local_files_only": bool(model_cfg.get("local_files_only", True)),
            "trust_remote_code": bool(model_cfg.get("trust_remote_code", True)),
            "torch_dtype": str(model_cfg.get("torch_dtype", "bfloat16")),
            "device_map": model_cfg.get("device_map", "auto"),
            "attn_implementation": model_cfg.get("attn_implementation", "sdpa"),
        }
        source_r1_value = (
            args.r1_checkpoint_dir
            or adapter_manifest.get("source_r1_checkpoint")
            or model_cfg.get("checkpoint_dir")
        )
        if not source_r1_value:
            raise ValueError(
                "Object Adapter evaluation needs --r1-checkpoint-dir, "
                "adapter_manifest.source_r1_checkpoint, or model.checkpoint_dir"
            )
        source_r1_checkpoint = _project_path(source_r1_value)
        if not source_r1_checkpoint.is_dir():
            raise FileNotFoundError(
                "Object Adapter evaluation requires the source R1 checkpoint directory: "
                f"{source_r1_checkpoint}. Pass --r1-checkpoint-dir or record source_r1_checkpoint "
                "in adapter_manifest.json."
            )
        source_manifest_path = source_r1_checkpoint / "strategy_manifest.json"
        if not source_manifest_path.is_file():
            raise FileNotFoundError(f"R1 strategy_manifest.json is missing: {source_manifest_path}")
        expected_source_manifest_sha = adapter_manifest.get("source_r1_manifest_sha256")
        if expected_source_manifest_sha and file_sha256(source_manifest_path) != str(expected_source_manifest_sha):
            raise ValueError(
                "Object Adapter source R1 manifest checksum mismatch; use the exact R1 checkpoint "
                "recorded when the adapter was trained."
            )
        model, processor, source_manifest = load_finetuned_checkpoint(
            checkpoint=source_r1_checkpoint,
            eval_model_config=loader_config,
            modules=imported,
        )
        sidecar_name = source_manifest.get("visual_sidecar")
        if not sidecar_name:
            for candidate in ("visual_trainable_weights.safetensors", "h1_visual_weights.safetensors"):
                if (source_r1_checkpoint / candidate).is_file():
                    sidecar_name = candidate
                    break
        if sidecar_name:
            load_visual_sidecar(model, source_r1_checkpoint / str(sidecar_name))
        visual = resolve_visual_module(model)
        for parameter in visual.parameters():
            parameter.requires_grad = False
        visual.eval()
        extractor = FrozenVisualFeatureExtractor(
            visual,
            selected_blocks=tuple(model_cfg.get("selected_blocks", (5, 11, 17, 23))),
            expected_num_blocks=int(model_cfg.get("expected_num_blocks", 24)),
            expected_hidden_size=int(model_cfg.get("expected_hidden_size", 1024)),
        )
        architecture = dict(adapter_manifest.get("architecture", {}))
        adapter = RSObjectAdapter(
            len(class_vocab["classes"]),
            vit_hidden_size=int(architecture.get("vit_hidden_size", 1024)),
            d_model=int(architecture.get("d_model", 256)),
            num_queries=int(architecture.get("num_queries", 64)),
            nhead=int(architecture.get("nhead", dict(config.get("adapter", {})).get("nhead", 8))),
            decoder_layers=int(
                architecture.get("decoder_layers", dict(config.get("adapter", {})).get("decoder_layers", 2))
            ),
            dim_feedforward=int(
                architecture.get("dim_feedforward", dict(config.get("adapter", {})).get("dim_feedforward", 1024))
            ),
            dropout=float(architecture.get("dropout", dict(config.get("adapter", {})).get("dropout", 0.1))),
        )
        device = next(visual.parameters()).device
        adapter.load_state_dict(load_file(str(weight_path), device="cpu"), strict=True)
        adapter.to(device).eval()
        image_root = _project_path(str(dict(config.get("data", {})).get("image_root", ".")))
        batch_size = int(args.batch_size or dict(config.get("evaluation", {})).get("batch_size", 4))
        if batch_size < 1:
            raise ValueError("evaluation batch size must be positive")

        predictions: list[dict[str, Any]] = []
        predicted_counts: list[float] = []
        true_counts: list[int] = []
        proposal_tensors: list[Any] = []
        detection_rows: list[dict[str, Any]] = []
        started = time.perf_counter()
        for start in range(0, len(eval_rows), batch_size):
            group = eval_rows[start : start + batch_size]
            encoded = visual_processor_batch(processor, group, image_root=image_root)
            features, positions = extractor.extract(encoded)
            layer_batch, position_batch, padding_mask = pad_visual_features(features, positions)
            layer_batch = _cast_features_for_adapter(layer_batch, adapter)
            class_ids = torch.as_tensor(
                [int(row["class_id"]) for row in group], dtype=torch.long, device=device
            )
            with torch.no_grad():
                outputs = adapter(
                    layer_batch,
                    position_batch.to(device),
                    class_ids,
                    memory_key_padding_mask=padding_mask.to(device),
                )
            for index, row in enumerate(group):
                logits = outputs["object_logits"][index]
                probabilities = torch.sigmoid(logits)
                predicted_count = float(probabilities.sum().item())
                proposal = torch.cat((logits[:, None], outputs["boxes_cxcywh"][index]), dim=-1)
                proposal_tensors.append(proposal.detach().cpu())
                if row["task_type"] == "counting":
                    predicted_counts.append(predicted_count)
                    true_counts.append(int(row["count"]))
                else:
                    detection_rows.append(row)
                predictions.append(
                    {
                        "id": row["id"],
                        "task_type": row["task_type"],
                        "image": row["image"],
                        "class_name": row["class_name"],
                        "class_id": row["class_id"],
                        "class_resolution_source": row["class_resolution_source"],
                        "predicted_count": predicted_count,
                        "predicted_count_rounded": min(64, max(0, int(predicted_count + 0.5))),
                        "objectness_probability": [float(value) for value in probabilities.cpu().tolist()],
                        "boxes_cxcywh": outputs["boxes_cxcywh"][index].detach().cpu().tolist(),
                        "reference_count": row["count"],
                        "reference_boxes_xyxy": row["boxes_xyxy"],
                    }
                )
        runtime = time.perf_counter() - started
        count_metrics = _metrics_from_count_pairs(predicted_counts, true_counts)
        detection_predictions = [
            proposal for proposal, row in zip(proposal_tensors, eval_rows) if row["task_type"] == "detection"
        ]
        detection_metrics = _box_metrics(detection_predictions, detection_rows)
        metrics = {
            "schema_version": "1.0",
            "experiment": "rs_object_adapter_v0",
            "evaluation_tier": evaluation_tier,
            "evaluation_tier_path": str(tier_path),
            "evaluation_tier_sha256": tier_sha,
            "population_sample_count": len(rows),
            "supported_sample_count": len(eval_rows),
            "unsupported_sample_count": len(rows) - len(eval_rows),
            "supported_ratio": len(eval_rows) / len(rows) if rows else 0.0,
            "unsupported_ratio": (len(rows) - len(eval_rows)) / len(rows) if rows else 0.0,
            "skipped": skipped,
            "counting": count_metrics,
            "detection_proposals": detection_metrics,
            "runtime_seconds": runtime,
            "samples_per_second": len(eval_rows) / runtime if runtime > 0 else None,
            "object_adapter_checkpoint": str(checkpoint),
            "source_r1_checkpoint": str(source_r1_checkpoint),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "predictions.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions),
            encoding="utf-8",
        )
        (output_dir / f"{evaluation_tier.lower()}_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (output_dir / "evaluation_metadata.json").write_text(
            json.dumps(
                {
                    "evaluation_tier": evaluation_tier,
                    "evaluation_tier_sha256": tier_sha,
                    "sample_count": len(eval_rows),
                    "batch_size": batch_size,
                    "source_manifest": source_manifest,
                    "object_adapter_manifest": adapter_manifest,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        extractor.close()
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return 0
    except (FileNotFoundError, ImportError, OSError, ValueError, RuntimeError) as exc:
        print(
            f"Object Adapter v0 {evaluation_tier} evaluation failed: {exc}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
