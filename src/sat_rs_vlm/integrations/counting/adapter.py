"""TaskGraph DetectionProvider backed by 04_counting_system_plan COUNT pipeline."""

from __future__ import annotations

import time
from typing import Any

from sat_rs_vlm.taskgraph.providers import DetectionRequest, DetectionSet
from sat_rs_vlm.taskgraph.runtime_types import ImageRef, Region

from .bootstrap import ensure_counting_system_importable
from .bridge import to_counting_scope, to_counting_target, to_taskgraph_entity_set

ensure_counting_system_importable()

from counting_system.executor import CountExecutor, CountParams  # noqa: E402


class CountingSystemDetectionAdapter:
    """Run tiled COUNT (core-ownership + NMS + fusion) then expose EntitySet.

    TaskGraph COUNT(image/Region) calls ``detect`` then ``len(entities)``.
    COUNT(EntitySet|SelectResult) never reaches this adapter.
    LOCATE object targets reuse the same detector path via ``task_hint``.
    """

    provider_name = "counting_system"

    def __init__(
        self,
        executor: CountExecutor | None = None,
        *,
        config: dict[str, Any] | None = None,
        detector: Any = None,
    ) -> None:
        self._owns_executor = executor is None
        self._executor = executor or CountExecutor(config, detector=detector)
        inner = getattr(self._executor.detector, "impl_name", None) or getattr(
            self._executor.detector, "name", None
        )
        self.provider_name = f"counting_system:{inner}" if inner else "counting_system"
        self.detect_requests: list[DetectionRequest] = []

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> CountingSystemDetectionAdapter:
        cfg = dict(config or {})
        backend = str(cfg.pop("backend", "fake"))
        extra: dict[str, Any] = {
            "detector": {"backend": backend},
            "gate": {"enabled": bool(cfg.get("gate", False))},
        }
        if "score_threshold" in cfg:
            extra.setdefault("count", {})["score_threshold"] = float(cfg["score_threshold"])
        if backend == "fake":
            from counting_system.detector.fake import FakeDetector

            return cls(CountExecutor(extra, detector=FakeDetector()))
        return cls(CountExecutor(extra))

    def detect(self, request: DetectionRequest) -> DetectionSet:
        self.detect_requests.append(request)
        started = time.perf_counter()
        scope = to_counting_scope(request.scope)
        hint = str(request.task_hint or "COUNT").upper()
        entire = isinstance(request.scope, ImageRef) and hint in {
            "COUNT",
            "COUNTING",
            "LOCATE",
        }
        if isinstance(request.scope, Region):
            entire = False
        result = self._executor.run(
            {"image": scope},
            CountParams(target=to_counting_target(request.target), entire=entire),
        )
        image = request.scope if isinstance(request.scope, ImageRef) else request.scope.image
        detections = to_taskgraph_entity_set(
            result.detections,
            image,
            extra_provenance={
                "fusion": result.provenance.get("fusion"),
                "tiles_run": result.provenance.get("tiles_run"),
                "detector_calls": result.provenance.get("detector_calls"),
                "entire": result.provenance.get("entire"),
                "task_hint": request.task_hint,
            },
        )
        return DetectionSet(
            detections=detections,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            provider=self.provider_name,
            metadata={
                "counting_mode": result.provenance.get("mode"),
                "detector_calls": result.provenance.get("detector_calls"),
                "tiles_run": result.provenance.get("tiles_run"),
                "fusion": result.provenance.get("fusion"),
                "inner_provider": result.provenance.get("provider"),
            },
        )

    def close(self) -> None:
        if self._owns_executor:
            self._executor.close()
