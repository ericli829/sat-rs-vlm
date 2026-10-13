"""LOCATE proposal adapter using CountingSystem's existing tiler and fusion."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sat_rs_vlm.integrations.detectors.protocol import ProposalResult
from sat_rs_vlm.integrations.detectors.strategies import resolve_detection_strategy
from sat_rs_vlm.taskgraph.schema import TargetSpec

from .bootstrap import ensure_counting_system_importable
from .detector_bridge import CountingProposalDetectorBridge

ensure_counting_system_importable()
from counting_system.executor import CountExecutor, CountParams  # noqa: E402
from counting_system.runtime import ImageRef  # noqa: E402
from counting_system.target import build_target  # noqa: E402


class StrategyProposalProvider:
    """Named strategies reuse the same tile planner/fusion as COUNT; legacy delegates."""

    provider_name = "strategy_proposals"

    def __init__(self, legacy_provider: Any, strategies: Mapping[str, Any]) -> None:
        self.base_provider = legacy_provider
        raw = legacy_provider
        while getattr(raw, "base_provider", None) is not None:
            raw = raw.base_provider
        self._raw_provider = raw
        self.strategies = dict(strategies)
        self.model_id = getattr(raw, "model_id", "DINO")
        self._executor = CountExecutor(
            {
                "detection_strategies": self.strategies,
                "gate": {"enabled": False},
                "count": {
                    "score_threshold": 0.2,
                    "nms_iou": 0.4,
                    "cross_scale_iou": 0.5,
                    "keep_raw_proposals": False,
                },
            },
            detector=CountingProposalDetectorBridge(raw),
        )

    def predict(self, image_path: Path, target_phrase: str) -> ProposalResult:
        category = target_phrase.split(".", 1)[0].strip().casefold()
        strategy = resolve_detection_strategy(category, self.strategies)
        if strategy["name"] == "legacy":
            predictor = getattr(
                self.base_provider, "predict_detector_only", self.base_provider.predict
            )
            return predictor(image_path, target_phrase)
        if not callable(getattr(self._raw_provider, "predict_raw", None)):
            raise RuntimeError(
                "named DINO strategies require an untruncated raw proposal interface"
            )
        spec = build_target(category, prompt=TargetSpec(category=category).detection_phrase())
        spec.category = category
        result = self._executor.run(
            {"image": ImageRef(str(image_path))}, CountParams(target=spec, entire=True)
        )
        return ProposalResult(
            [list(det.bbox_xyxy_global) for det in result.detections],
            [float(det.score) for det in result.detections],
            float(result.provenance["latency_sec"]) * 1000,
            self.provider_name,
            self.model_id,
            {
                **result.provenance,
                "coordinate_mode": "absolute_original_pixel_xyxy",
                "candidate_ids": [det.provenance.get("candidate_id") for det in result.detections],
            },
        )

    predict_detector_only = predict

    def close(self) -> None:
        self._executor.close()
        self.base_provider.close()
