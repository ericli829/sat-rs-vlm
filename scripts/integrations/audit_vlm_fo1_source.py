#!/usr/bin/env python3
"""Write an auditable snapshot of the official VLM-FO1 inference contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_REPOSITORY = "https://github.com/om-ai-lab/VLM-FO1"
OFFICIAL_COMMIT = "348c1e8163a8fca5ed621cdfab0c94e3432336bd"
MODEL_NAME = "VLM-FO1-3B-v01"
MODEL_ID = "omlab/VLM-FO1-3B-v01"
UPN_CHECKPOINT_URL = (
    "https://github.com/IDEA-Research/ChatRex/releases/download/upn-large/upn_large.pth"
)


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _weight_inventory(model_dir: Path | None) -> dict[str, Any]:
    if model_dir is None or not model_dir.is_dir():
        return {"present": False, "files": []}
    patterns = ("*.safetensors", "pytorch_model*.bin", "model*.bin", "*.ckpt")
    paths = sorted({path for pattern in patterns for path in model_dir.glob(pattern)})
    return {
        "present": bool(paths),
        "files": [{"name": path.name, "bytes": path.stat().st_size} for path in paths],
    }


def build_audit(model_dir: Path | None) -> dict[str, Any]:
    config_path = model_dir / "config.json" if model_dir else None
    tokenizer_path = model_dir / "tokenizer_config.json" if model_dir else None
    generation_path = model_dir / "generation_config.json" if model_dir else None
    config = _load_json(config_path) if config_path else None
    tokenizer = _load_json(tokenizer_path) if tokenizer_path else None
    generation = _load_json(generation_path) if generation_path else None
    return {
        "schema_version": "vlm-fo1-source-audit-v1",
        "audit_time_utc": datetime.now(timezone.utc).isoformat(),
        "auditor_python": platform.python_version(),
        "official_repository": OFFICIAL_REPOSITORY,
        "official_commit": OFFICIAL_COMMIT,
        "official_commit_source": f"{OFFICIAL_REPOSITORY}/commit/{OFFICIAL_COMMIT}",
        "official_model": {
            "name": MODEL_NAME,
            "model_id": MODEL_ID,
            "base_vlm_family": "Qwen2.5-VL-3B",
            "architecture": (config or {}).get("architectures", ["OmChatQwen25VLForCausalLM"])[0],
            "transformers_version_recorded_by_checkpoint": (config or {}).get(
                "transformers_version", "4.50.1"
            ),
            "local_model_directory": str(model_dir) if model_dir else None,
            "config_sha256": sha256(config_path) if config_path else None,
            "tokenizer_config_sha256": sha256(tokenizer_path) if tokenizer_path else None,
            "generation_config_sha256": sha256(generation_path) if generation_path else None,
            "weight_inventory": _weight_inventory(model_dir),
        },
        "proposal_pipeline": {
            "public_generator": "UPNWrapper from detect_tools/upn",
            "paper_opn_status": (
                "OPN referenced in the paper is not released; official public path uses UPN."
            ),
            "upn_checkpoint_url": UPN_CHECKPOINT_URL,
            "upn_checkpoint_environment_variable": "VLM_FO1_UPN_CHECKPOINT",
            "score_threshold": 0.3,
            "nms_threshold": 0.8,
            "top_k": 100,
            "proposal_output_key": "original_xyxy_boxes",
            "score_output_key": "scores",
            "bbox_format": "pixel xyxy in original image coordinates",
            "upn_transform": {
                "short_edge": 800,
                "max_size": 1333,
                "normalization_mean": [0.485, 0.456, 0.406],
                "normalization_std": [0.229, 0.224, 0.225],
            },
        },
        "fo1_inference": {
            "official_counting_template": (
                "How many {target} are there in this image? "
                "Count each instance of the target object. "
                "Locate them with object indexes and then answer the question "
                "with the number of objects."
            ),
            "bbox_input_field": "messages[0].bbox_list",
            "region_index_output_format": "<ground>label</ground><objects><regionN>...</objects>",
            "region_index_range": "<region0> through <region99>",
            "generation": {
                "max_tokens": 4096,
                "top_p": 0.05,
                "temperature": 0.0,
                "do_sample": False,
                "greedy": True,
            },
            "model_loading": {
                "dtype": "bfloat16",
                "device": "cuda",
                "attn_implementation": "flash_attention_2",
                "slow_tokenizer": True,
            },
        },
        "model_sidecars": {
            "auxiliary_vision_tower": (config or {}).get(
                "mm_vision_tower_aux", "resources/davit-large.pth"
            ),
            "primary_vision_tower": (config or {}).get(
                "mm_vision_tower", "resources/Qwen2.5-VL-3B-Instruct-Vision_Tower"
            ),
            "aux_image_size": (config or {}).get("aux_image_size", 1024),
            "aux_image_aspect_ratio": (config or {}).get("aux_image_aspect_ratio", "dynamic"),
            "uses_region_index_tokens": (config or {}).get("mm_use_region_index_token", True),
            "num_region_tokens": (config or {}).get("mm_num_region_tokens", 100),
            "special_tokens": {
                "image": "<|image_pad|>",
                "region": "<region0>...<region99>",
                "grounding": ["<ground>", "</ground>", "<objects>", "</objects>"],
                "think": ["<think>", "</think>"],
            },
        },
        "environment": {
            "official_requirements": {
                "torch": "2.6.0",
                "torchvision": "0.21.0",
                "transformers": "4.50.1",
                "timm": "1.0.9",
                "accelerate": "1.4.0",
                "ninja": "unversioned in official requirements",
                "upn_ops": "detect_tools/upn/ops editable install",
            },
            "isolated_environment_name": "vlm-fo1",
            "rs_vlm_environment_must_not_install_official_requirements": True,
        },
        "observed_source_differences": [
            (
                "README and scripts expose both Hugging Face and local model paths; "
                "the evaluator requires an explicit local VLM_FO1_MODEL."
            ),
            (
                "The paper mentions OPN, while the public repository documents UPN "
                "as the available object proposal path."
            ),
            (
                "The official script calls filter(min_score=0.3), whose implementation "
                "default nms_value=0.8 is recorded explicitly here."
            ),
        ],
        "source_urls": {
            "upn_inference_script": (
                f"{OFFICIAL_REPOSITORY}/blob/{OFFICIAL_COMMIT}/scripts/inference_with_upn.py"
            ),
            "upn_wrapper": (
                f"{OFFICIAL_REPOSITORY}/blob/{OFFICIAL_COMMIT}/"
                "detect_tools/upn/inference_wrapper.py"
            ),
            "builder": (f"{OFFICIAL_REPOSITORY}/blob/{OFFICIAL_COMMIT}/vlm_fo1/model/builder.py"),
            "task_templates": (
                f"{OFFICIAL_REPOSITORY}/blob/{OFFICIAL_COMMIT}/vlm_fo1/task_templates.py"
            ),
            "requirements": (f"{OFFICIAL_REPOSITORY}/blob/{OFFICIAL_COMMIT}/requirements.txt"),
        },
        "checkpoint_generation_config_observed": generation,
        "checkpoint_tokenizer_config_observed": {
            "eos_token": (tokenizer or {}).get("eos_token"),
            "pad_token": (tokenizer or {}).get("pad_token"),
            "additional_special_tokens_count": len(
                (tokenizer or {}).get("additional_special_tokens", [])
            ),
        },
    }


def render_markdown(audit: dict[str, Any]) -> str:
    model = audit["official_model"]
    proposal = audit["proposal_pipeline"]
    generation = audit["fo1_inference"]["generation"]
    lines = [
        "# VLM-FO1 official source audit",
        "",
        f"- repository: `{audit['official_repository']}`",
        f"- commit: `{audit['official_commit']}`",
        f"- model: `{model['model_id']}` ({model['base_vlm_family']})",
        f"- architecture: `{model['architecture']}`",
        f"- local weight files present: `{model['weight_inventory']['present']}`",
        "",
        "## Public proposal path",
        "",
        f"- generator: `{proposal['public_generator']}`",
        f"- score threshold: `{proposal['score_threshold']}`",
        f"- NMS threshold: `{proposal['nms_threshold']}`",
        f"- top-k: `{proposal['top_k']}`",
        f"- boxes: `{proposal['bbox_format']}`",
        f"- OPN status: {proposal['paper_opn_status']}",
        "",
        "## Official FO1 call",
        "",
        f"- template: `{audit['fo1_inference']['official_counting_template']}`",
        f"- output: `{audit['fo1_inference']['region_index_output_format']}`",
        (
            f"- generation: `max_tokens={generation['max_tokens']}`, "
            f"`top_p={generation['top_p']}`, "
            f"`temperature={generation['temperature']}`, "
            f"`do_sample={generation['do_sample']}`"
        ),
        "- tokenizer: slow tokenizer; region tokens `<region0>` through `<region99>`",
        "",
        "## Vision sidecars",
        "",
        f"- primary tower: `{audit['model_sidecars']['primary_vision_tower']}`",
        f"- auxiliary tower: `{audit['model_sidecars']['auxiliary_vision_tower']}`",
        (
            f"- auxiliary size/aspect: `{audit['model_sidecars']['aux_image_size']}` / "
            f"`{audit['model_sidecars']['aux_image_aspect_ratio']}`"
        ),
        "",
        "## Isolation",
        "",
        (
            "The official requirements are installed only in `vlm-fo1`; the rs-vlm "
            "interpreter communicates through JSONL and never imports the official package."
        ),
        "",
        "## Recorded differences",
        "",
    ]
    lines.extend(f"- {item}" for item in audit["observed_source_differences"])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=PROJECT_ROOT / "reports/integrations/vlm_fo1_source_audit.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=PROJECT_ROOT / "reports/integrations/vlm_fo1_source_audit.md",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_dir = args.model_dir.resolve() if args.model_dir else None
    audit = build_audit(model_dir)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.markdown_output.write_text(render_markdown(audit), encoding="utf-8")
    print(
        json.dumps(
            {"json": str(args.json_output), "markdown": str(args.markdown_output)},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
