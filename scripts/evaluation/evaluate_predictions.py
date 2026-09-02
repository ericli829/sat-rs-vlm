"""对已有 predictions.jsonl 执行 v1.5 多任务离线评测。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sat_rs_vlm.evaluation.config import (  # noqa: E402
    EvaluationWorkflowConfig,
    load_evaluation_config,
)
from sat_rs_vlm.evaluation.records import EvaluationError  # noqa: E402
from sat_rs_vlm.evaluation.runner import run_evaluation  # noqa: E402

DEFAULT_CONTRACT = PROJECT_ROOT / "configs" / "eval" / "evaluation_contract_v1.5.yaml"
DEFAULT_SEMANTIC_CONTRACT = (
    PROJECT_ROOT / "configs" / "eval" / "semantic" / "semantic_contract.json"
)
DEFAULT_SEMANTIC_ONTOLOGY = (
    PROJECT_ROOT / "configs" / "eval" / "semantic" / "remote_sensing_ontology.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Evaluation workflow YAML config.")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--semantic", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--semantic-contract", type=Path)
    parser.add_argument("--semantic-ontology", type=Path)
    parser.add_argument(
        "--visual-semantic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run the image-audited LEVIR visual-semantic auxiliary profile.",
    )
    parser.add_argument("--visual-semantic-gold", type=Path)
    parser.add_argument("--visual-semantic-generation-manifest", type=Path)
    parser.add_argument("--visual-semantic-prompt-profile")
    parser.add_argument(
        "--visual-semantic-allow-incomplete-historical-manifest",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--visual-semantic-verify-image-paths",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--latency-semantics",
        choices=("unresolved", "single_sample", "batch_amortized_per_sample"),
        default=None,
    )
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--group-by-task", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--protect-repository",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Refuse output paths inside this repository (disabled for normal integrated use).",
    )
    return parser.parse_args()


def _project_path(value: str | Path) -> Path:
    """配置中的相对路径始终相对于仓库根目录解析。"""

    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    try:
        config = (
            load_evaluation_config(args.config)
            if args.config is not None
            else EvaluationWorkflowConfig()
        )
        settings = config.evaluation
        predictions_value = args.predictions or config.input.predictions
        if predictions_value is None:
            raise EvaluationError(
                "predictions path is required via --predictions or input.predictions"
            )
        output_value = args.output_dir or config.output.output_dir
        tier_sha256 = None
        tier_version = settings.tier_version
        if settings.tier and settings.tiers_manifest:
            tiers_payload = json.loads(
                _project_path(settings.tiers_manifest).read_text(encoding="utf-8")
            )
            tier_record = dict(tiers_payload.get("tiers", {}).get(settings.tier, {}))
            tier_sha256 = tier_record.get("sha256")
            tier_version = tier_version or tiers_payload.get("tier_version")
            if not tier_sha256:
                raise EvaluationError(
                    f"Tier {settings.tier} is missing from {settings.tiers_manifest}"
                )
        outputs = run_evaluation(
            _project_path(predictions_value),
            _project_path(output_value),
            contract_path=_project_path(args.contract or settings.contract or DEFAULT_CONTRACT),
            manifest_path=(
                _project_path(args.manifest or settings.manifest)
                if args.manifest or settings.manifest
                else None
            ),
            strict=settings.strict if args.strict is None else args.strict,
            protected_repository=PROJECT_ROOT if args.protect_repository else None,
            semantic_enabled=settings.semantic if args.semantic is None else args.semantic,
            semantic_contract_path=_project_path(
                args.semantic_contract or settings.semantic_contract or DEFAULT_SEMANTIC_CONTRACT
            ),
            semantic_ontology_path=_project_path(
                args.semantic_ontology or settings.semantic_ontology or DEFAULT_SEMANTIC_ONTOLOGY
            ),
            visual_semantic_enabled=(
                settings.visual_semantic if args.visual_semantic is None else args.visual_semantic
            ),
            visual_semantic_gold_path=(
                _project_path(args.visual_semantic_gold or settings.visual_semantic_gold)
                if args.visual_semantic_gold or settings.visual_semantic_gold
                else None
            ),
            visual_semantic_generation_manifest_path=(
                _project_path(
                    args.visual_semantic_generation_manifest
                    or settings.visual_semantic_generation_manifest
                )
                if args.visual_semantic_generation_manifest
                or settings.visual_semantic_generation_manifest
                else None
            ),
            visual_semantic_prompt_profile=(
                args.visual_semantic_prompt_profile or settings.visual_semantic_prompt_profile
            ),
            visual_semantic_allow_incomplete_historical_manifest=(
                settings.visual_semantic_allow_incomplete_historical_manifest
                if args.visual_semantic_allow_incomplete_historical_manifest is None
                else args.visual_semantic_allow_incomplete_historical_manifest
            ),
            visual_semantic_verify_image_paths=(
                settings.visual_semantic_verify_image_paths
                if args.visual_semantic_verify_image_paths is None
                else args.visual_semantic_verify_image_paths
            ),
            latency_semantics=args.latency_semantics or settings.latency_semantics,
            eval_batch_size=args.eval_batch_size or settings.eval_batch_size,
            group_by_task=(
                settings.group_by_task if args.group_by_task is None else args.group_by_task
            ),
            evaluation_tier=settings.tier,
            evaluation_tier_version=tier_version,
            evaluation_tier_sha256=tier_sha256,
        )
    except (EvaluationError, OSError, ValueError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 2
    for name, path in outputs.items():
        print(f"Saved {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
