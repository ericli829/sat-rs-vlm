"""Proposal-provider wrapper that reranks detector candidates with a retriever."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sat_rs_vlm.integrations.retrievers.protocol import RetrieverProvider

from .protocol import ProposalError, ProposalProvider, ProposalResult


def _min_max_normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high - low <= 1e-12:
        return [1.0] * len(values)
    return [(value - low) / (high - low) for value in values]


class CLIPRerankedProposalProvider:
    """Rerank detector proposals with a configured region retriever."""

    provider_name = "clip_rerank"

    def __init__(
        self,
        base_provider: ProposalProvider,
        retriever: RetrieverProvider,
        config: Mapping[str, Any],
        *,
        base_provider_name: str,
        retriever_name: str,
    ) -> None:
        self.base_provider = base_provider
        self.retriever = retriever
        self.base_provider_name = base_provider_name
        self.retriever_name = retriever_name
        self.config = dict(config)
        self.detector_weight = float(self.config.get("detector_weight", 0.35))
        self.retriever_weight = float(self.config.get("retriever_weight", 0.65))
        weight_total = self.detector_weight + self.retriever_weight
        if not math.isfinite(weight_total) or weight_total <= 0.0:
            raise ProposalError("clip_rerank weights must have a positive finite total")
        if self.detector_weight < 0.0 or self.retriever_weight < 0.0:
            raise ProposalError("clip_rerank weights must be non-negative")
        self.detector_weight /= weight_total
        self.retriever_weight /= weight_total
        top_k_value = self.config.get("candidate_top_k")
        self.candidate_top_k = int(top_k_value) if top_k_value is not None else None
        if self.candidate_top_k is not None and self.candidate_top_k < 1:
            raise ProposalError("clip_rerank candidate_top_k must be positive")
        self.fail_open = bool(self.config.get("fail_open", True))
        self.model_id = (
            f"clip-rerank:{getattr(base_provider, 'model_id', base_provider_name)}:"
            f"{getattr(retriever, 'model_id', retriever_name)}"
        )
        self.model_identity = {
            "provider": self.provider_name,
            "base_provider": self.base_provider_name,
            "base_model_id": str(getattr(base_provider, "model_id", base_provider_name)),
            "retriever": self.retriever_name,
            "retriever_model_id": str(getattr(retriever, "model_id", retriever_name)),
            "detector_weight": self.detector_weight,
            "retriever_weight": self.retriever_weight,
            "candidate_top_k": self.candidate_top_k,
        }

    @staticmethod
    def _base_metadata(result: ProposalResult) -> dict[str, Any]:
        metadata = dict(result.metadata)
        metadata.setdefault("clip_rerank", {})
        return metadata

    def _fallback(
        self,
        result: ProposalResult,
        *,
        started: float,
        reason: str,
    ) -> ProposalResult:
        metadata = self._base_metadata(result)
        metadata["clip_rerank"] = {
            "status": "fallback",
            "reason": reason,
            "fail_open": self.fail_open,
            "base_provider": self.base_provider_name,
            "retriever_provider": self.retriever_name,
            "candidate_count": len(result.boxes_xyxy),
            "detector_weight": self.detector_weight,
            "retriever_weight": self.retriever_weight,
            "candidate_top_k": self.candidate_top_k,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
        }
        return ProposalResult(
            boxes_xyxy=[list(box) for box in result.boxes_xyxy],
            scores=list(result.scores),
            latency_ms=result.latency_ms,
            provider=result.provider,
            model_id=result.model_id,
            metadata=metadata,
        )

    def predict(self, image_path: Path, target_phrase: str) -> ProposalResult:
        started = time.perf_counter()
        result = self.base_provider.predict(image_path, target_phrase)
        candidate_count = len(result.boxes_xyxy)
        if candidate_count == 0:
            metadata = self._base_metadata(result)
            metadata["clip_rerank"] = {
                "status": "no_candidates",
                "base_provider": self.base_provider_name,
                "retriever_provider": self.retriever_name,
                "candidate_count": 0,
                "detector_weight": self.detector_weight,
                "retriever_weight": self.retriever_weight,
                "candidate_top_k": self.candidate_top_k,
                "latency_ms": (time.perf_counter() - started) * 1000.0,
            }
            return ProposalResult(
                boxes_xyxy=[],
                scores=[],
                latency_ms=result.latency_ms,
                provider=result.provider,
                model_id=result.model_id,
                metadata=metadata,
            )

        try:
            scored = self.retriever.score_regions(
                Path(image_path), target_phrase, result.boxes_xyxy
            )
            if len(scored.scores) != candidate_count:
                raise ProposalError(
                    "clip_rerank retriever returned an unexpected score count: "
                    f"{len(scored.scores)} != {candidate_count}"
                )
            detector_scores = [float(score) for score in result.scores]
            retriever_scores = [float(score) for score in scored.scores]
            detector_normalized = _min_max_normalize(detector_scores)
            retriever_normalized = _min_max_normalize(retriever_scores)
            fused_scores = [
                self.detector_weight * detector_score
                + self.retriever_weight * retriever_score
                for detector_score, retriever_score in zip(
                    detector_normalized, retriever_normalized, strict=True
                )
            ]
            order = sorted(
                range(candidate_count),
                key=lambda index: (-fused_scores[index], index),
            )
            retained = (
                order[: self.candidate_top_k]
                if self.candidate_top_k is not None
                else order
            )
            metadata = self._base_metadata(result)
            metadata["clip_rerank"] = {
                "status": "applied",
                "base_provider": self.base_provider_name,
                "base_model_id": str(getattr(self.base_provider, "model_id", "unknown")),
                "retriever_provider": scored.provider,
                "retriever_model_id": scored.model_id,
                "candidate_count": candidate_count,
                "detector_weight": self.detector_weight,
                "retriever_weight": self.retriever_weight,
                "candidate_top_k": self.candidate_top_k,
                "original_order": list(range(candidate_count)),
                "ranked_order": order,
                "retained_indices": retained,
                "filtered_indices": [index for index in order if index not in retained],
                "detector_scores": detector_scores,
                "retriever_scores": retriever_scores,
                "detector_normalized_scores": detector_normalized,
                "retriever_normalized_scores": retriever_normalized,
                "fused_scores": fused_scores,
                "retriever_metadata": dict(scored.metadata),
                "latency_ms": (time.perf_counter() - started) * 1000.0,
            }
            return ProposalResult(
                boxes_xyxy=[list(result.boxes_xyxy[index]) for index in retained],
                scores=[fused_scores[index] for index in retained],
                latency_ms=(time.perf_counter() - started) * 1000.0,
                provider=self.provider_name,
                model_id=self.model_id,
                metadata=metadata,
            )
        except Exception as exc:
            if not self.fail_open:
                raise ProposalError(f"clip_rerank failed: {exc}") from exc
            return self._fallback(
                result,
                started=started,
                reason=f"{type(exc).__name__}: {exc}",
            )

    def close(self) -> None:
        try:
            self.base_provider.close()
        finally:
            self.retriever.close()
