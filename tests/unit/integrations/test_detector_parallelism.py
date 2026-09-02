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
