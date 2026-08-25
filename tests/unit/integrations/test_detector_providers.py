from __future__ import annotations

import sys
from pathlib import Path

import pytest

from scripts.integrations import precompute_vlm_fo1_proposals as precompute
from scripts.integrations.lae_dino_worker import _extract_predictions
from scripts.integrations.vlm_fo1_worker import validate_request
from sat_rs_vlm.integrations.detectors.cache import ProposalCache
from sat_rs_vlm.integrations.detectors.grounding_dino import _nms
from sat_rs_vlm.integrations.detectors.lae_dino_sidecar import _LAESidecarClient
from sat_rs_vlm.integrations.detectors.protocol import (
    ProposalError,
    ProposalResult,
    canonicalize_proposals,
    proposal_cache_key,
)
from sat_rs_vlm.integrations.detectors.registry import create_proposal_provider
from sat_rs_vlm.integrations.vlm_fo1 import request_has_reference_leak


def test_provider_registry_is_lazy_and_exposes_mock() -> None:
    provider = create_proposal_provider("mock", {})
    try:
        assert provider.provider_name == "mock"
    finally:
        provider.close()
    with pytest.raises(ProposalError, match="unsupported proposal provider"):
        create_proposal_provider("unknown", {})


def test_reference_guard_covers_provider_metadata() -> None:
    assert request_has_reference_leak({"proposal_metadata": {"reference": "secret"}})


def test_precomputed_proposal_failure_is_not_silently_counted_as_zero() -> None:
    normalized, response = validate_request(
        {
            "id": "failed",
            "image": "image.png",
            "question": "How many airplanes are visible?",
            "target_phrase": "airplanes",
            "bbox_list": [],
            "bbox_scores": [],
            "proposal_metadata": {"status": "failed", "error": "detector crashed"},
        },
        prompt_profile="official_fo1",
        proposal_backend="precomputed",
    )
    assert normalized is None
    assert response is not None
    assert response["failure_stage"] == "proposal_generation"


def test_canonicalize_normalized_invalid_and_deterministic_order() -> None:
    boxes, scores, stats = canonicalize_proposals(
        [[0.8, 0.8, 0.2, 0.2], [0.0, 0.0, 1.2, 1.0], [1.0, 1.0, 1.0, 1.0], [float("nan")] * 4],
        [0.5, 0.9, 0.99, 0.8],
        image_width=100,
        image_height=50,
        coordinate_mode="normalized",
        top_k=2,
    )
    assert boxes == [[0.0, 0.0, 100.0, 50.0], [20.0, 10.0, 80.0, 40.0]]
    assert scores == [0.9, 0.5]
    assert stats == {"invalid_count": 2, "reordered_count": 2, "clamped_count": 1}


def test_canonicalize_rejects_mismatched_lengths() -> None:
    assert canonicalize_proposals([], [], image_width=10, image_height=10) == (
        [],
        [],
        {"invalid_count": 0, "reordered_count": 0, "clamped_count": 0},
    )
    with pytest.raises(ProposalError, match="length mismatch"):
        canonicalize_proposals([[0, 0, 1, 1]], [], image_width=10, image_height=10)


def test_optional_nms_is_deterministic_and_empty_safe() -> None:
    boxes, scores = _nms(
        [[0, 0, 10, 10], [1, 1, 9, 9], [20, 20, 30, 30]],
        [0.8, 0.7, 0.6],
        0.5,
    )
    assert boxes == [[0, 0, 10, 10], [20, 20, 30, 30]]
    assert scores == [0.8, 0.6]
    assert _nms([], [], 0.5) == ([], [])


def test_cache_round_trip_and_key_includes_query_and_parameters(tmp_path: Path) -> None:
    image = tmp_path / "image.bin"
    image.write_bytes(b"image")
    key = proposal_cache_key(
        provider="mock",
        model_identity={"revision": "v1"},
        image=image,
        target_phrase="airplanes",
        parameters={"top_k": 10},
    )
    assert key != proposal_cache_key(
        provider="mock",
        model_identity={"revision": "v1"},
        image=image,
        target_phrase="ships",
        parameters={"top_k": 10},
    )
    cache = ProposalCache(tmp_path / "cache")
    result = ProposalResult(
        boxes_xyxy=[[1, 2, 3, 4]],
        scores=[0.75],
        latency_ms=1,
        provider="mock",
        model_id="mock-v1",
    )
    assert cache.get(key) is None
    cache.put(key, result)
    loaded = cache.get(key)
    assert loaded is not None
    assert loaded.to_dict() == result.to_dict()
    assert cache.misses == 1
    assert cache.hits == 1


def test_precompute_uses_target_phrase_and_never_sends_reference(monkeypatch, tmp_path: Path) -> None:
    image = tmp_path / "image.bin"
    image.write_bytes(b"image")
    rows = [
        {
            "id": "sample-1",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image.name},
                        {"type": "text", "text": "How many airplanes are visible?"},
                    ],
                },
                {"role": "assistant", "content": "999 airplanes; secret reference answer"},
            ],
        }
    ]
    provider = create_proposal_provider("mock", {})
    monkeypatch.setattr(precompute, "create_proposal_provider", lambda *_args, **_kwargs: provider)
    output, counts = precompute.precompute_rows(
        rows,
        provider_name="mock",
        provider_config={},
        image_root=tmp_path,
    )
    assert counts == {"ok": 1, "unsupported": 0, "failed": 0}
    assert provider.calls == [(image.resolve(), "airplanes")]
    assert output[0]["target_phrase"] == "airplanes"
    assert output[0]["proposal_metadata"]["target_phrase"] == "airplanes"
    # The original row remains intact for evaluator-side metrics, but its
    # assistant/reference content was not present in the provider request.
    assert "secret reference answer" in output[0]["messages"][1]["content"]


def test_precompute_row_failure_does_not_stop_following_rows(monkeypatch, tmp_path: Path) -> None:
    image = tmp_path / "image.bin"
    image.write_bytes(b"image")

    class FlakyProvider:
        provider_name = "mock"
        calls = 0

        def predict(self, _image_path: Path, _target_phrase: str) -> ProposalResult:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("one sample failed")
            return ProposalResult([[0, 0, 1, 1]], [0.5], 1.0, "mock", "flaky")

        def close(self) -> None:
            return None

    provider = FlakyProvider()
    monkeypatch.setattr(precompute, "create_proposal_provider", lambda *_args, **_kwargs: provider)
    rows = [
        {"id": "one", "question": "How many airplanes are visible?", "image": image.name},
        {"id": "two", "question": "How many airplanes are visible?", "image": image.name},
    ]
    output, counts = precompute.precompute_rows(
        rows, provider_name="mock", provider_config={}, image_root=tmp_path
    )
    assert counts == {"ok": 1, "unsupported": 0, "failed": 1}
    assert output[0]["proposal_metadata"]["status"] == "failed"
    assert output[1]["proposal_metadata"]["status"] == "ok"


def _sidecar_fixture(tmp_path: Path, body: str) -> tuple[_LAESidecarClient, Path]:
    source_root = tmp_path / "source"
    source_root.mkdir()
    config = tmp_path / "config.py"
    config.write_text("# explicit test config\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    image = tmp_path / "image.bin"
    image.write_bytes(b"image")
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import json, sys\n"
        "for line in sys.stdin:\n"
        "    json.loads(line)\n"
        f"    sys.stdout.write({body!r} + '\\n'); sys.stdout.flush()\n",
        encoding="utf-8",
    )
    client = _LAESidecarClient(
        {
            "source_root": str(source_root),
            "config_path": str(config),
            "checkpoint": str(checkpoint),
            "worker_python": sys.executable,
            "worker_script": str(worker),
            "device": "cpu",
        }
    )
    return client, image


def test_lae_sidecar_rejects_malformed_json(tmp_path: Path) -> None:
    client, image = _sidecar_fixture(tmp_path, "'not-json'")
    try:
        with pytest.raises(ProposalError, match="invalid JSON"):
            client.request(image, "airplanes")
    finally:
        client.close()


def test_lae_sidecar_reports_process_crash(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    config = tmp_path / "config.py"
    config.write_text("# explicit test config\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    image = tmp_path / "image.bin"
    image.write_bytes(b"image")
    worker = tmp_path / "crash.py"
    worker.write_text("raise SystemExit(7)\n", encoding="utf-8")
    client = _LAESidecarClient(
        {
            "source_root": str(source_root),
            "config_path": str(config),
            "checkpoint": str(checkpoint),
            "worker_python": sys.executable,
            "worker_script": str(worker),
        }
    )
    try:
        with pytest.raises(ProposalError, match="exited"):
            client.request(image, "airplanes")
    finally:
        client.close()


def test_lae_worker_extracts_mmdetection2_boxes_without_segmentation_payload() -> None:
    boxes, scores = _extract_predictions(
        (
            [[[1.0, 2.0, 10.0, 20.0, 0.8]]],
            [[[0, 1], [1, 0]]],
        )
    )
    assert boxes == [[1.0, 2.0, 10.0, 20.0]]
    assert scores == [0.8]
