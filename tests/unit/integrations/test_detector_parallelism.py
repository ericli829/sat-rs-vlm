from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from PIL import Image

from sat_rs_vlm.integrations.detectors.parallel import (
    resolve_parallel_workers,
)
from sat_rs_vlm.integrations.detectors.protocol import ProposalResult
from sat_rs_vlm.integrations.detectors.tiled import TiledProposalProvider


def test_auto_parallel_workers_uses_free_vram_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sat_rs_vlm.integrations.detectors.parallel.available_cuda_memory_gb",
        lambda: 18.0,
    )
    assert (
        resolve_parallel_workers(
            "auto",
            max_workers=3,
            worker_vram_gb=4.0,
            vram_reserve_gb=6.0,
        )
        == 3
    )

    monkeypatch.setattr(
        "sat_rs_vlm.integrations.detectors.parallel.available_cuda_memory_gb",
        lambda: 9.0,
    )
    assert (
        resolve_parallel_workers(
            "auto",
            max_workers=3,
            worker_vram_gb=4.0,
            vram_reserve_gb=6.0,
        )
        == 1
    )


def test_tiled_provider_parallelizes_base_requests_and_preserves_order(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "parallel.png"
    Image.new("RGB", (30, 10), "white").save(image_path)

    class ParallelProvider:
        provider_name = "fixture"
        model_id = "fixture-v1"

        def __init__(self) -> None:
            self._lock = threading.Lock()
            self.active = 0
            self.max_active = 0
            self.calls: list[str] = []

        def predict(self, tile_path: Path, target_phrase: str) -> ProposalResult:
            with self._lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                self.calls.append(tile_path.name)
            time.sleep(0.03)
            with self._lock:
                self.active -= 1
            return ProposalResult(
                boxes_xyxy=[],
                scores=[],
                latency_ms=30.0,
                provider=self.provider_name,
                model_id=self.model_id,
                metadata={"query": target_phrase},
            )

        def close(self) -> None:
            return None

    base = ParallelProvider()
    provider = TiledProposalProvider(
        base,
        {
            "tile_size": 10,
            "overlap_ratio": 0.0,
            "global_nms_iou": 0.4,
            "parallel_workers": 3,
        },
        base_provider_name="fixture",
    )

    result = provider.predict(image_path, "aircraft")

    assert result.boxes_xyxy == []
    assert len(base.calls) == 3
    assert base.max_active >= 2
    assert result.metadata["tile_count"] == 3
    assert result.metadata["parallel_workers"] == 3
    assert [item["tile_id"] for item in result.metadata["tiles"]] == [0, 1, 2]


def test_tiled_provider_reuses_first_auto_worker_count(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "cached-workers.png"
    Image.new("RGB", (10, 10), "white").save(image_path)
    worker_counts = iter((3, 1))

    class Provider:
        provider_name = "fixture"
        model_id = "fixture-v1"

        def predict(self, _tile_path: Path, _target_phrase: str) -> ProposalResult:
            return ProposalResult([], [], 1.0, self.provider_name, self.model_id)

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        "sat_rs_vlm.integrations.detectors.tiled.resolve_parallel_workers",
        lambda *_args, **_kwargs: next(worker_counts),
    )
    provider = TiledProposalProvider(
        Provider(),
        {"tile_size": 10, "parallel_workers": "auto"},
        base_provider_name="fixture",
    )
    try:
        assert provider.predict(image_path, "aircraft").metadata["parallel_workers"] == 3
        assert provider.predict(image_path, "ships").metadata["parallel_workers"] == 3
    finally:
        provider.close()


def test_tiled_provider_caches_same_image_and_query(tmp_path: Path) -> None:
    image_path = tmp_path / "cached-query.png"
    Image.new("RGB", (10, 10), "white").save(image_path)

    class Provider:
        provider_name = "fixture"
        model_id = "fixture-v1"

        def __init__(self) -> None:
            self.calls = 0

        def predict(self, _tile_path: Path, _target_phrase: str) -> ProposalResult:
            self.calls += 1
            return ProposalResult([], [], 1.0, self.provider_name, self.model_id)

        def close(self) -> None:
            return None

    base = Provider()
    provider = TiledProposalProvider(
        base,
        {"tile_size": 10, "parallel_workers": 1, "proposal_cache_size": 2},
        base_provider_name="fixture",
    )
    try:
        first = provider.predict(image_path, "aircraft")
        second = provider.predict(image_path, " AIRCRAFT ")
        assert base.calls == 1
        assert first.metadata["proposal_cache_hit"] is False
        assert second.metadata["proposal_cache_hit"] is True
        assert second.boxes_xyxy == first.boxes_xyxy
    finally:
        provider.close()
