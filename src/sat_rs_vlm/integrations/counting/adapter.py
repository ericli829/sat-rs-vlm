"""TaskGraph CountingProvider backed by 04_counting_system_plan tiling/fusion."""

from __future__ import annotations

import time
from typing import Any

from sat_rs_vlm.taskgraph.providers import CountingRequest, CountingResult
from sat_rs_vlm.taskgraph.runtime_types import ImageRef, Region

from .bootstrap import ensure_counting_system_importable
from .bridge import to_counting_scope, to_counting_target, to_taskgraph_entity_set

ensure_counting_system_importable()

from counting_system.executor import CountExecutor, CountParams  # noqa: E402

_EXECUTOR_CONFIG_KEYS = ("scale", "count", "gate", "detector", "retriever")
_TILED_KINDS = {"tiled"}
_FORBIDDEN_MAIN_ENV_KINDS = {"auto", "lae_mmdet", "grounding_dino"}
_SIDECAR_KINDS = {"lae_dino_lae1m", "lae_dino_dior", "lae_dino_dota"}


class CountingSystemProvider:
    """COUNT(image|Region) backend. LOCATE must not use this provider."""

    provider_name = "counting_system"

    def __init__(
        self,
        executor: CountExecutor | None = None,
        *,
        config: dict[str, Any] | None = None,
        detector: Any = None,
        owns_executor: bool | None = None,
    ) -> None:
        if executor is None:
            if detector is None:
                detector = self._build_detector("fake", {})
            self._executor = CountExecutor(config, detector=detector)
            self._owns_executor = True if owns_executor is None else owns_executor
        else:
            self._executor = executor
            self._owns_executor = True if owns_executor is None else owns_executor
        inner = getattr(self._executor.detector, "impl_name", None) or getattr(
            self._executor.detector, "name", None
        )
        self.provider_name = f"counting_system:{inner}" if inner else "counting_system"
        self.count_requests: list[CountingRequest] = []
        self.closed = False

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> CountingSystemProvider:
        cfg = dict(config or {})
        detector_section = dict(cfg.get("detector") or {})
        if (
            "backend" in cfg
            and "kind" not in detector_section
            and "backend" not in detector_section
        ):
            detector_section["kind"] = cfg["backend"]
        detector_kind = str(
            detector_section.get("kind") or detector_section.get("backend") or "fake"
        ).strip().lower()
        executor_config = {
            key: value
            for key, value in cfg.items()
            if key in _EXECUTOR_CONFIG_KEYS and value is not None
        }
        executor_config["detector"] = detector_section
        gate = dict(executor_config.get("gate") or {})
        if "enabled" not in gate:
            gate["enabled"] = False
        executor_config["gate"] = gate
        detector = cls._build_detector(detector_kind, detector_section)
        return cls(CountExecutor(executor_config, detector=detector), owns_executor=True)

    @staticmethod
    def _build_detector(kind: str, detector_section: dict[str, Any]) -> Any:
        if kind in _TILED_KINDS or kind.startswith("tiled("):
            raise ValueError(
                "Counting provider unavailable: detector.kind='tiled' would double-tile. "
                "counting_system owns Global/Native/Fine tiling; use "
                "detector.kind='lae_dino_lae1m'."
            )
        if kind in _FORBIDDEN_MAIN_ENV_KINDS:
            raise ValueError(
                "Counting provider unavailable: production counting must use "
                "detector.kind='lae_dino_lae1m' (isolated sidecar) or 'fake'. "
                "Do not import MMDetection into the main environment."
            )
        if kind == "fake":
            from counting_system.detector.fake import FakeDetector

            return FakeDetector()
        if kind in _SIDECAR_KINDS:
            from sat_rs_vlm.integrations.detectors.registry import create_proposal_provider

            from .detector_bridge import CountingProposalDetectorBridge

            proposal_cfg = {
                key: value
                for key, value in detector_section.items()
                if key not in {"kind", "backend"}
            }
            proposal_cfg.setdefault("score_threshold", 0.0)
            try:
                proposal = create_proposal_provider(kind, proposal_cfg)
            except Exception as exc:
                raise RuntimeError(
                    f"Counting provider unavailable: LAE sidecar failure: {exc}"
                ) from exc
            return CountingProposalDetectorBridge(proposal)
        raise ValueError(
            f"Counting provider unavailable: unsupported detector kind {kind!r}"
        )

    def count(self, request: CountingRequest) -> CountingResult:
        self.count_requests.append(request)
        started = time.perf_counter()
        entire = bool(request.entire)
        entire_source = "CountingRequest.entire"
        if isinstance(request.scope, Region):
            entire = False
            entire_source = "region_frozen_semantic"
        result = self._executor.run(
            {"image": to_counting_scope(request.scope)},
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
                "entire": entire,
                "requested_entire": request.entire,
                "entire_source": entire_source,
            },
        )
        return CountingResult(
            count=int(result.count),
            detections=detections,
            provider=self.provider_name,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            metadata={
                "counting_mode": result.provenance.get("mode"),
                "detector_calls": result.provenance.get("detector_calls"),
                "tiles_run": result.provenance.get("tiles_run"),
                "fusion": result.provenance.get("fusion"),
                "inner_provider": result.provenance.get("provider"),
                "entire": entire,
                "requested_entire": request.entire,
                "entire_source": entire_source,
            },
        )

    def close(self) -> None:
        if self.closed:
            return
        if self._owns_executor:
            detector = getattr(self._executor, "detector", None)
            closer = getattr(detector, "close", None)
            if closer is not None:
                closer()
            self._executor.close()
        self.closed = True
