#!/usr/bin/env python3
"""Run one minimal VLM-FO1 shared-runtime generation with precomputed boxes.

This is intentionally a GPU smoke.  A CPU-only node may run the static and
loader/path tests, but must not try to materialize the 3B BF16 checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sat_rs_vlm.integrations.vlm_fo1 import extract_count_target_phrase  # noqa: E402
from sat_rs_vlm.integrations.vlm_fo1_loader import (  # noqa: E402
    ensure_official_root,
    validate_model_path,
)


def _boxes(path: Path) -> tuple[list[list[float]], list[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        raw_boxes = payload.get("bbox_list", payload.get("boxes"))
        raw_scores = payload.get("bbox_scores", payload.get("scores"))
    else:
        raw_boxes = payload
        raw_scores = None
    if not isinstance(raw_boxes, list):
        raise ValueError("bbox JSON must be a list or an object containing boxes/bbox_list")
    normalized = []
    for index, box in enumerate(raw_boxes):
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError(f"bbox[{index}] must contain four coordinates")
        normalized.append([float(value) for value in box])
    if raw_scores is None:
        scores = [1.0] * len(normalized)
    elif isinstance(raw_scores, list) and len(raw_scores) == len(normalized):
        scores = [float(value) for value in raw_scores]
    else:
        raise ValueError("scores must have the same length as boxes")
    return normalized, scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path(os.environ.get("VLM_FO1_MODEL", "")))
    parser.add_argument("--fo1-root", type=Path, default=Path(os.environ.get("VLM_FO1_ROOT", "")))
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--bbox-json", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--target-phrase", "--target", dest="target_phrase")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attention-backend", choices=("auto", "sdpa", "flash_attention_2", "eager"), default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        model_path = validate_model_path(args.model)
        official_root = ensure_official_root(args.fo1_root, require_upn=False)
        os.environ["VLM_FO1_ROOT"] = str(official_root)
        if not args.image.is_file():
            raise RuntimeError(f"image does not exist: {args.image.resolve()}")
        if not args.bbox_json.is_file():
            raise RuntimeError(f"bbox JSON does not exist: {args.bbox_json.resolve()}")
        import torch

        if str(args.device).startswith("cuda") and not torch.cuda.is_available():
            print(
                json.dumps(
                    {
                        "status": "skipped",
                        "reason": "CUDA unavailable; refusing to load 3B FO1 on CPU",
                        "model": str(model_path),
                    },
                    ensure_ascii=False,
                )
            )
            return 3
        phrase = args.target_phrase or extract_count_target_phrase(args.question).phrase
        if not phrase:
            raise RuntimeError("question does not yield a supported counting target phrase")
        boxes, scores = _boxes(args.bbox_json)
        # Importing the worker is cheap; loading the checkpoint is deliberately
        # deferred until after all path/device checks above.
        from scripts.integrations.vlm_fo1_worker import (  # noqa: PLC0415
            PipelineConfig,
            build_backend,
            process_request,
        )

        config = PipelineConfig(
            model_path=str(model_path),
            upn_checkpoint="",
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            runtime_mode="shared_rs_vlm",
            proposal_backend="precomputed",
            attention_backend=args.attention_backend,
        )
        backend = build_backend("official", config)
        response = process_request(
            {
                "id": "fo1-shared-smoke",
                "image": str(args.image.resolve()),
                "question": args.question,
                "target_phrase": phrase,
                "bbox_list": boxes,
                "bbox_scores": scores,
            },
            backend,
            config,
        )
        output = {
            "status": response.get("status"),
            "raw_output": response.get("fo1_raw_output", ""),
            "fo1_raw_output": response.get("fo1_raw_output", ""),
            "parsed_region_indexes": response.get("fo1_selected_region_indexes", []),
            "region_count": response.get("fo1_region_count"),
            "count": response.get("fo1_count"),
            "latency_ms": float(response.get("fo1_latency_ms", 0.0)),
            "proposal_count": response.get("proposal_count_used"),
            "error": response.get("error"),
        }
        print(json.dumps(output, ensure_ascii=False))
        return 0 if output["status"] == "ok" else 2
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
