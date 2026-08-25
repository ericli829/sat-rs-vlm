#!/usr/bin/env python3
"""Long-lived JSONL worker for VLM-FO1.

The ``official`` backend retains the original isolated-runtime path.  The
``shared_rs_vlm`` runtime uses the current rs-vlm interpreter and the
compatibility loader, and can consume precomputed proposal boxes without a
UPN CUDA extension.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sat_rs_vlm.integrations.vlm_fo1 import (  # noqa: E402
    FO1_PROMPT_PROFILES,
    build_counting_prompt,
    compact_proposal_evidence,
    extract_count_target_phrase,
    parse_profile_output,
    request_has_reference_leak,
)
from sat_rs_vlm.integrations.vlm_fo1_loader import (  # noqa: E402
    load_fo1_model,
)


def _configure_cache() -> None:
    cache_value = os.environ.get("VLM_FO1_CACHE_DIR", "").strip()
    if not cache_value:
        return
    cache_dir = Path(cache_value).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_dir / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir / "transformers"))


def _configure_official_import_path(*, require_upn: bool = True) -> Path:
    """Add only the official checkout to this worker's import path."""

    value = os.environ.get("VLM_FO1_ROOT", "").strip()
    if not value:
        raise RuntimeError(
            "VLM_FO1_ROOT is required for the official backend; set it to the "
            "official VLM-FO1 checkout"
        )
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f"VLM_FO1_ROOT is not a directory: {root}")
    required = ["vlm_fo1"]
    if require_upn:
        required.extend(("detect_tools", "detect_tools/upn"))
    for relative in required:
        if not (root / relative).is_dir():
            raise RuntimeError(f"VLM_FO1_ROOT is missing required directory: {root / relative}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return root


def _failure(
    request: Mapping[str, Any],
    stage: str,
    error: str,
    *,
    target_phrase: str | None = None,
    target_status: str | None = None,
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "id": request.get("id"),
        "status": "failed",
        "failure_stage": stage,
        "error": error,
    }
    if target_phrase is not None:
        output["target_phrase"] = target_phrase
    if target_status is not None:
        output["target_status"] = target_status
    return output


def _canonicalize_boxes(boxes: Any) -> list[list[float]]:
    """Convert proposal coordinates to JSON-native Python floats."""

    if boxes is None:
        return []
    return [[float(value) for value in box] for box in boxes]


def _canonicalize_scores(scores: Any) -> list[float]:
    """Convert proposal scores to JSON-native Python floats."""

    if scores is None:
        return []
    return [float(value) for value in scores]


def _canonicalize_protocol_response(response: dict[str, Any]) -> dict[str, Any]:
    """Normalize numeric evidence at the worker protocol boundary."""

    for name in (
        "proposal_boxes",
        "selected_region_boxes",
    ):
        if name in response:
            response[name] = _canonicalize_boxes(response[name])
    for name in (
        "proposal_scores",
        "selected_region_scores",
    ):
        if name in response:
            response[name] = _canonicalize_scores(response[name])
    for name in ("upn_latency_ms", "proposal_latency_ms", "fo1_latency_ms"):
        if name in response and response[name] is not None:
            response[name] = float(response[name])
    return response


def ensure_json_serializable(value: Any) -> None:
    """Fail explicitly if a response contains a non-JSON-native value."""

    json.dumps(value, allow_nan=False)


def _request_target(request: Mapping[str, Any]) -> tuple[str | None, str, str | None]:
    result = extract_count_target_phrase(str(request.get("question", "")))
    return result.phrase, result.status, result.reason


def validate_request(
    request: Any, *, prompt_profile: str, proposal_backend: str = "upn"
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not isinstance(request, dict):
        return None, {
            "status": "failed",
            "failure_stage": "protocol_guard",
            "error": "request must be an object",
        }
    if request_has_reference_leak(request):
        return None, _failure(request, "protocol_guard", "reference-answer fields are forbidden")
    required = ("id", "image", "question", "target_phrase")
    missing = [name for name in required if name not in request]
    if missing:
        return None, _failure(request, "protocol_guard", f"missing request fields: {missing}")
    if not all(isinstance(request[name], str) for name in required):
        return None, _failure(
            request, "protocol_guard", "id/image/question/target_phrase must be strings"
        )
    if not str(request["id"]).strip() or not str(request["image"]).strip():
        return None, _failure(request, "protocol_guard", "id and image must be non-empty")
    if prompt_profile not in FO1_PROMPT_PROFILES:
        return None, _failure(
            request, "protocol_guard", f"unsupported prompt profile: {prompt_profile}"
        )
    if proposal_backend not in {"upn", "precomputed"}:
        return None, _failure(
            request, "protocol_guard", f"unsupported proposal backend: {proposal_backend}"
        )
    phrase, status, reason = _request_target(request)
    if status != "supported":
        return None, {
            "id": request["id"],
            "status": "unsupported",
            "target_phrase": None,
            "target_status": status,
            "target_reason": reason,
            "proposal_count_raw": 0,
            "proposal_count_used": 0,
            "proposal_boxes": [],
            "proposal_scores": [],
            "selected_region_indexes": [],
            "selected_region_boxes": [],
            "selected_region_scores": [],
            "fo1_raw_output": "",
            "fo1_region_count": None,
            "fo1_textual_count": None,
            "fo1_count": None,
            "fo1_count_source": None,
            "fo1_count_agrees_with_text": None,
            "zero_proposal_assumed_zero": False,
            "upn_latency_ms": 0.0,
            "proposal_latency_ms": 0.0,
            "fo1_latency_ms": 0.0,
        }
    if str(request["target_phrase"]).strip().lower() != str(phrase).strip().lower():
        return None, _failure(
            request, "protocol_guard", "target_phrase does not match deterministic extraction"
        )
    normalized = dict(request)
    normalized["target_phrase"] = phrase
    if proposal_backend == "precomputed":
        proposal_metadata = request.get("proposal_metadata")
        if isinstance(proposal_metadata, Mapping) and proposal_metadata.get("status") == "failed":
            return None, _failure(
                request,
                "proposal_generation",
                str(proposal_metadata.get("error") or "precomputed proposal generation failed"),
                target_phrase=phrase,
                target_status="supported",
            )
        raw_boxes = request.get("bbox_list")
        if not isinstance(raw_boxes, list):
            return None, _failure(
                request,
                "protocol_guard",
                "precomputed proposal backend requires bbox_list",
            )
        boxes: list[list[float]] = []
        for index, box in enumerate(raw_boxes):
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                return None, _failure(
                    request,
                    "protocol_guard",
                    f"bbox_list[{index}] must contain four numeric coordinates",
                )
            try:
                boxes.append([float(value) for value in box])
            except (TypeError, ValueError) as exc:
                return None, _failure(
                    request,
                    "protocol_guard",
                    f"bbox_list[{index}] contains a non-numeric coordinate: {exc}",
                )
        raw_scores = request.get("bbox_scores")
        if raw_scores is None:
            scores = [1.0] * len(boxes)
        elif isinstance(raw_scores, list) and len(raw_scores) == len(boxes):
            try:
                scores = [float(value) for value in raw_scores]
            except (TypeError, ValueError) as exc:
                return None, _failure(
                    request, "protocol_guard", f"bbox_scores contains a non-numeric value: {exc}"
                )
        else:
            return None, _failure(
                request,
                "protocol_guard",
                "bbox_scores must be a list with the same length as bbox_list",
            )
        normalized["bbox_list"] = boxes
        normalized["bbox_scores"] = scores
    return normalized, None


@dataclass
class PipelineConfig:
    model_path: str
    upn_checkpoint: str
    device: str = "cuda"
    proposal_score_threshold: float = 0.3
    proposal_top_k: int = 100
    nms_threshold: float = 0.8
    max_new_tokens: int = 4096
    temperature: float = 0.0
    top_p: float = 0.05
    prompt_profile: str = "official_fo1"
    count_source: str = "region"
    runtime_mode: str = "official"
    proposal_backend: str = "upn"
    attention_backend: str = "sdpa"


class Backend:
    def infer(self, request: Mapping[str, Any], config: PipelineConfig) -> dict[str, Any]:
        raise NotImplementedError


class MockBackend(Backend):
    """Deterministic synthetic backend used by unit/smoke tests only."""

    def infer(self, request: Mapping[str, Any], config: PipelineConfig) -> dict[str, Any]:
        boxes = [[0.0, 0.0, 10.0, 10.0], [10.0, 10.0, 20.0, 20.0]]
        scores = [0.9, 0.8]
        if config.prompt_profile == "official_fo1":
            raw = "<ground>{}</ground><objects><region0><region1></objects>".format(
                request["target_phrase"]
            )
        elif config.prompt_profile == "integer":
            raw = "2"
        elif config.prompt_profile == "json":
            raw = '{"count":2}'
        else:
            raw = "There are 2."
        parsed = parse_profile_output(
            raw,
            config.prompt_profile,
            proposal_count=len(boxes),
            count_source=config.count_source,
        )
        selected_boxes, selected_scores = compact_proposal_evidence(
            boxes, scores, parsed["selected_region_indexes"]
        )
        return {
            "status": "ok",
            "target_phrase": request["target_phrase"],
            "target_status": "supported",
            "proposal_count_raw": len(boxes),
            "proposal_count_used": len(boxes),
            "proposal_boxes": boxes,
            "proposal_scores": scores,
            "selected_region_indexes": parsed["selected_region_indexes"],
            "selected_region_boxes": selected_boxes,
            "selected_region_scores": selected_scores,
            "fo1_raw_output": raw,
            "fo1_selected_region_indexes": parsed["selected_region_indexes"],
            "fo1_region_count": parsed["region_count"],
            "fo1_textual_count": parsed["textual_count"],
            "fo1_count": parsed["count"],
            "fo1_count_source": parsed["count_source"],
            "fo1_count_agrees_with_text": parsed["count_agrees_with_text"],
            "zero_proposal_assumed_zero": False,
            "upn_latency_ms": 0.0,
            "proposal_latency_ms": 0.0,
            "fo1_latency_ms": 0.0,
        }


class OfficialFO1Backend(Backend):
    """Lazy official implementation; imports never execute in rs-vlm."""

    def __init__(self, config: PipelineConfig) -> None:
        _configure_official_import_path(
            require_upn=not (
                config.runtime_mode == "shared_rs_vlm" and config.proposal_backend == "precomputed"
            )
        )
        try:
            import torch
            from vlm_fo1.mm_utils import prepare_inputs
        except ImportError as exc:  # pragma: no cover - exercised in isolated env
            raise RuntimeError(
                "VLM-FO1 dependencies are unavailable; check VLM_FO1_ROOT and "
                "the selected runtime environment"
            ) from exc
        self._torch = torch
        self._prepare_inputs = prepare_inputs
        self._proposal_backend = config.proposal_backend
        self._runtime_mode = config.runtime_mode
        self._model_path = config.model_path
        # The upstream helper enables Qwen vision-token handling by checking
        # the model-name string.  The distributed FO1 directory is named
        # ``VLM-FO1-3B-v01`` (without ``qwen``), so preserve the local path for
        # loading while supplying an explicit Qwen alias to ``prepare_inputs``.
        self._prepare_model_name = str(config.model_path)
        if "qwen" not in self._prepare_model_name.lower():
            self._prepare_model_name = f"qwen2.5-vl::{self._prepare_model_name}"
        self._upn = None
        # The public helpers print prompts/tokens to stdout.  stdout is the
        # machine-readable JSONL channel, so suppress library chatter while
        # constructing the long-lived backend as well as during inference.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            if config.runtime_mode == "shared_rs_vlm":
                if config.proposal_backend == "precomputed":
                    self._upn = None
                else:
                    try:
                        from detect_tools.upn import UPNWrapper
                    except ImportError as exc:
                        raise RuntimeError(
                            "shared_rs_vlm with proposal_backend=upn requires the "
                            "official UPN Python package; use precomputed proposals "
                            "when the CUDA extension is unavailable"
                        ) from exc
                    self._upn = UPNWrapper(config.upn_checkpoint)
                bundle = load_fo1_model(
                    config.model_path,
                    config.device,
                    attention_backend=config.attention_backend,
                )
                self._model_path = str(bundle.model_path)
                self._tokenizer = bundle.tokenizer
                self._model = bundle.model
                self._processors = bundle.image_processors
            else:
                try:
                    from detect_tools.upn import UPNWrapper
                    from vlm_fo1.model.builder import load_pretrained_model
                except ImportError as exc:  # pragma: no cover - isolated env only
                    raise RuntimeError(
                        "official FO1 dependencies are unavailable; run with VLM_FO1_PYTHON "
                        "from the isolated vlm-fo1 environment"
                    ) from exc
                self._upn = UPNWrapper(config.upn_checkpoint)
                self._tokenizer, self._model, self._processors = load_pretrained_model(
                    config.model_path, device=config.device
                )

    def infer(self, request: Mapping[str, Any], config: PipelineConfig) -> dict[str, Any]:
        from PIL import Image

        try:
            image = Image.open(str(request["image"])).convert("RGB")
        except Exception as exc:
            return _failure(
                request,
                "image_load",
                str(exc),
                target_phrase=request["target_phrase"],
                target_status="supported",
            )
        try:
            proposal_metadata = (
                dict(request.get("proposal_metadata", {}))
                if isinstance(request.get("proposal_metadata"), Mapping)
                else {}
            )
            proposal_provider = request.get("proposal_provider") or proposal_metadata.get("provider")
            proposal_model = request.get("proposal_model") or proposal_metadata.get("model_id")
            if self._proposal_backend == "precomputed":
                raw_boxes = request.get("bbox_list") or []
                boxes = _canonicalize_boxes(raw_boxes)
                raw_scores = request.get("bbox_scores")
                if raw_scores is None:
                    raw_scores = [1.0] * len(boxes)
                scores = _canonicalize_scores(raw_scores)
                proposal_count_raw = len(boxes)
                upn_latency_ms = 0.0
                proposal_latency_ms = float(request.get("proposal_latency_ms", 0.0))
            else:
                if self._upn is None:
                    raise RuntimeError("UPN backend was not initialized")
                start = time.perf_counter()
                with (
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    raw_proposals = self._upn.inference(image)
                    filtered = self._upn.filter(
                        raw_proposals,
                        min_score=config.proposal_score_threshold,
                        nms_value=config.nms_threshold,
                    )
                upn_latency_ms = (time.perf_counter() - start) * 1000.0
                proposal_latency_ms = upn_latency_ms
                filtered_boxes = filtered.get("original_xyxy_boxes") or []
                filtered_scores = filtered.get("scores") or []
                boxes = _canonicalize_boxes(filtered_boxes[0]) if filtered_boxes else []
                scores = _canonicalize_scores(filtered_scores[0]) if filtered_scores else []
                proposal_count_raw = len(boxes)
        except Exception as exc:
            return _failure(
                request,
                "proposal_generation",
                str(exc),
                target_phrase=request["target_phrase"],
                target_status="supported",
            )
        try:
            boxes = _canonicalize_boxes(boxes[: config.proposal_top_k])
            scores = _canonicalize_scores(scores[: config.proposal_top_k])
            proposal_count_used = len(boxes)
            if not boxes:
                if config.prompt_profile == "official_fo1":
                    zero_raw = "<ground>{}</ground><objects></objects>".format(
                        request["target_phrase"]
                    )
                    zero_source = "region" if config.count_source == "auto" else config.count_source
                elif config.prompt_profile == "integer":
                    zero_raw = "0"
                    zero_source = "text"
                elif config.prompt_profile == "json":
                    zero_raw = '{"count":0}'
                    zero_source = "text"
                else:
                    zero_raw = "There are 0."
                    zero_source = "text"
                return {
                    "id": request["id"],
                    "status": "ok",
                    "target_phrase": request["target_phrase"],
                    "target_status": "supported",
                    "proposal_count_raw": proposal_count_raw,
                    "proposal_count_used": 0,
                    "proposal_boxes": [],
                    "proposal_scores": [],
                    "proposal_provider": proposal_provider,
                    "proposal_model": proposal_model,
                    "proposal_metadata": proposal_metadata,
                    "selected_region_indexes": [],
                    "selected_region_boxes": [],
                    "selected_region_scores": [],
                    "fo1_raw_output": zero_raw,
                    "fo1_selected_region_indexes": [],
                    "fo1_region_count": 0,
                    "fo1_textual_count": None,
                    "fo1_count": 0,
                    "fo1_count_source": zero_source,
                    "zero_proposal_assumed_zero": True,
                    "fo1_count_agrees_with_text": None,
                    "upn_latency_ms": upn_latency_ms,
                    "proposal_latency_ms": proposal_latency_ms,
                    "fo1_latency_ms": 0.0,
                }
            prompt = build_counting_prompt(
                str(request["question"]), str(request["target_phrase"]), config.prompt_profile
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": str(request["image"])}},
                        {"type": "text", "text": prompt},
                    ],
                    "bbox_list": boxes,
                }
            ]
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                generation_kwargs = self._prepare_inputs(
                    self._prepare_model_name,
                    self._model,
                    self._processors,
                    self._tokenizer,
                    messages,
                    device=config.device,
                    max_tokens=config.max_new_tokens,
                    top_p=config.top_p,
                    temperature=config.temperature,
                    do_sample=False,
                )
            # ``prepare_inputs`` installs a TextStreamer which writes tokens
            # to stdout.  Streaming is not part of the sidecar contract.
            generation_kwargs.pop("streamer", None)
            start = time.perf_counter()
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
                self._torch.inference_mode(),
            ):
                output_ids = self._model.generate(**generation_kwargs)
            fo1_latency_ms = (time.perf_counter() - start) * 1000.0
            input_ids = generation_kwargs.get("inputs")
            if input_ids is None:
                input_ids = generation_kwargs.get("input_ids")
            prompt_len = int(input_ids.shape[1]) if input_ids is not None else 0
            raw_output = self._tokenizer.decode(output_ids[0, prompt_len:]).strip()
            parsed = parse_profile_output(
                raw_output,
                config.prompt_profile,
                proposal_count=proposal_count_used,
                count_source=config.count_source,
            )
            if not parsed["parse_ok"]:
                return _failure(
                    request,
                    "fo1_parse" if config.prompt_profile == "official_fo1" else "count_parse",
                    str(parsed.get("parse_error") or "FO1 count output did not parse"),
                    target_phrase=request["target_phrase"],
                    target_status="supported",
                ) | {
                    "proposal_count_raw": proposal_count_raw,
                    "proposal_count_used": proposal_count_used,
                    "proposal_boxes": boxes,
                    "proposal_scores": scores,
                    "proposal_provider": proposal_provider,
                    "proposal_model": proposal_model,
                    "proposal_metadata": proposal_metadata,
                    "selected_region_indexes": parsed["selected_region_indexes"],
                    "fo1_selected_region_indexes": parsed["selected_region_indexes"],
                    "fo1_region_count": parsed["region_count"],
                    "fo1_textual_count": parsed["textual_count"],
                    "fo1_count_source": parsed["count_source"],
                    "fo1_count_agrees_with_text": parsed["count_agrees_with_text"],
                    "zero_proposal_assumed_zero": False,
                    "fo1_raw_output": raw_output,
                    "upn_latency_ms": upn_latency_ms,
                    "proposal_latency_ms": proposal_latency_ms,
                    "fo1_latency_ms": fo1_latency_ms,
                }
            selected_boxes, selected_scores = compact_proposal_evidence(
                boxes, scores, parsed["selected_region_indexes"]
            )
            selected_boxes = _canonicalize_boxes(selected_boxes)
            selected_scores = _canonicalize_scores(selected_scores)
            return {
                "id": request["id"],
                "status": "ok",
                "target_phrase": request["target_phrase"],
                "target_status": "supported",
                "proposal_count_raw": proposal_count_raw,
                "proposal_count_used": proposal_count_used,
                "proposal_boxes": boxes,
                "proposal_scores": scores,
                "proposal_provider": proposal_provider,
                "proposal_model": proposal_model,
                "proposal_metadata": proposal_metadata,
                "selected_region_indexes": parsed["selected_region_indexes"],
                "selected_region_boxes": selected_boxes,
                "selected_region_scores": selected_scores,
                "fo1_raw_output": raw_output,
                "fo1_selected_region_indexes": parsed["selected_region_indexes"],
                "fo1_region_count": parsed["region_count"],
                "fo1_count": parsed["count"],
                "fo1_textual_count": parsed["textual_count"],
                "fo1_count_source": parsed["count_source"],
                "fo1_count_agrees_with_text": parsed["count_agrees_with_text"],
                "zero_proposal_assumed_zero": False,
                "upn_latency_ms": upn_latency_ms,
                "proposal_latency_ms": proposal_latency_ms,
                "fo1_latency_ms": fo1_latency_ms,
            }
        except Exception as exc:
            return _failure(
                request,
                "fo1_inference",
                str(exc),
                target_phrase=request["target_phrase"],
                target_status="supported",
            )


def build_backend(name: str, config: PipelineConfig) -> Backend:
    if name == "mock":
        return MockBackend()
    if name == "official":
        return OfficialFO1Backend(config)
    raise ValueError(f"unsupported backend: {name}")


def process_request(
    request: Any,
    backend: Backend,
    config: PipelineConfig,
) -> dict[str, Any]:
    normalized, error = validate_request(
        request,
        prompt_profile=config.prompt_profile,
        proposal_backend=config.proposal_backend,
    )
    if error is not None:
        return error
    assert normalized is not None
    response = backend.infer(normalized, config)
    response.setdefault("id", normalized["id"])
    response.setdefault("target_phrase", normalized["target_phrase"])
    response.setdefault("target_status", "supported")
    return _canonicalize_protocol_response(response)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("official", "mock"), default="official")
    parser.add_argument(
        "--runtime-mode",
        choices=("official", "shared_rs_vlm"),
        default=os.environ.get("VLM_FO1_RUNTIME_MODE", "official"),
    )
    parser.add_argument(
        "--proposal-backend",
        choices=("upn", "precomputed"),
        default=os.environ.get("VLM_FO1_PROPOSAL_BACKEND", "upn"),
    )
    parser.add_argument(
        "--attention-backend",
        choices=("auto", "sdpa", "flash_attention_2", "eager"),
        default=os.environ.get("VLM_FO1_ATTENTION_BACKEND", "sdpa"),
    )
    parser.add_argument("--model", default=os.environ.get("VLM_FO1_MODEL", ""))
    parser.add_argument("--upn-checkpoint", default=os.environ.get("VLM_FO1_UPN_CHECKPOINT", ""))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--proposal-score-threshold", type=float, default=0.3)
    parser.add_argument("--proposal-top-k", type=int, default=100)
    parser.add_argument("--nms-threshold", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.05)
    parser.add_argument("--prompt-profile", choices=FO1_PROMPT_PROFILES, default="official_fo1")
    parser.add_argument("--count-source", choices=("region", "text", "auto"), default="region")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        _configure_cache()
    except OSError as exc:
        print(
            json.dumps({"status": "failed", "failure_stage": "cache_init", "error": str(exc)}),
            flush=True,
        )
        return 2
    config = PipelineConfig(
        model_path=args.model,
        upn_checkpoint=args.upn_checkpoint,
        device=args.device,
        proposal_score_threshold=args.proposal_score_threshold,
        proposal_top_k=args.proposal_top_k,
        nms_threshold=args.nms_threshold,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        prompt_profile=args.prompt_profile,
        count_source=args.count_source,
        runtime_mode=args.runtime_mode,
        proposal_backend=args.proposal_backend,
        attention_backend=args.attention_backend,
    )
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            backend = build_backend(args.backend, config)
    except Exception as exc:
        print(
            json.dumps({"status": "failed", "failure_stage": "backend_init", "error": str(exc)}),
            flush=True,
        )
        return 2
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            response = process_request(request, backend, config)
        except json.JSONDecodeError as exc:
            response = {
                "status": "failed",
                "failure_stage": "protocol_guard",
                "error": f"invalid JSON: {exc}",
            }
        except Exception as exc:  # keep the long-lived protocol alive for the next id
            response = {"status": "failed", "failure_stage": "worker", "error": str(exc)}
        try:
            ensure_json_serializable(response)
        except (TypeError, ValueError) as exc:
            response = {
                "id": response.get("id") if isinstance(response, dict) else None,
                "status": "failed",
                "failure_stage": "protocol_serialization",
                "error": str(exc),
            }
        print(json.dumps(response, ensure_ascii=False, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
