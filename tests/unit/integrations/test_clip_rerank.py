from __future__ import annotations

from pathlib import Path

import pytest

from sat_rs_vlm.integrations.detectors.clip_rerank import (
    CLIPRerankedProposalProvider,
)
from sat_rs_vlm.integrations.detectors.protocol import ProposalError, ProposalResult
from sat_rs_vlm.integrations.detectors.registry import create_proposal_provider
from sat_rs_vlm.integrations.retrievers.protocol import RetrievalResult


class _FakeProposalProvider:
    provider_name = "fake_detector"
    model_id = "fake-detector-v1"

    def __init__(self, result: ProposalResult) -> None:
        self.result = result
        self.closed = False

    def predict(self, image_path: Path, target_phrase: str) -> ProposalResult:
        return self.result

    def close(self) -> None:
        self.closed = True


class _FakeRetriever:
    provider_name = "fake_retriever"
    model_id = "fake-retriever-v1"

    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.closed = False
        self.calls: list[tuple[Path, str, int]] = []

    def score_regions(
        self, image_path: Path, query: str, regions_xyxy: list[list[float]]
    ) -> RetrievalResult:
        self.calls.append((image_path, query, len(regions_xyxy)))
        return RetrievalResult(
            scores=self.scores,
            latency_ms=1.0,
            provider=self.provider_name,
            model_id=self.model_id,
            metadata={"fixture": True},
        )

    def close(self) -> None:
        self.closed = True


def _base_result(count: int = 3) -> ProposalResult:
    return ProposalResult(
        boxes_xyxy=[[float(index), 0.0, float(index + 1), 1.0] for index in range(count)],
        scores=[0.2, 0.8, 0.5][:count],
        latency_ms=2.0,
        provider="fake_detector",
        model_id="fake-detector-v1",
        metadata={"base": True},
    )


def test_clip_rerank_fuses_scores_and_records_original_indices() -> None:
    base = _FakeProposalProvider(_base_result())
    retriever = _FakeRetriever([0.1, 0.2, 0.9])
    provider = CLIPRerankedProposalProvider(
        base,
        retriever,
        {"detector_weight": 0.35, "retriever_weight": 0.65},
        base_provider_name="fake_detector",
        retriever_name="fake_retriever",
    )

    result = provider.predict(Path("image.png"), "building")
    trace = result.metadata["clip_rerank"]

    assert [box[0] for box in result.boxes_xyxy] == [2.0, 1.0, 0.0]
    assert trace["status"] == "applied"
    assert trace["ranked_order"] == [2, 1, 0]
    assert trace["retained_indices"] == [2, 1, 0]
    assert trace["detector_scores"] == [0.2, 0.8, 0.5]
    assert retriever.calls == [(Path("image.png"), "building", 3)]
    provider.close()
    assert base.closed and retriever.closed


def test_clip_rerank_accepts_distinct_detector_and_referring_queries() -> None:
    provider = CLIPRerankedProposalProvider(
        _FakeProposalProvider(_base_result()),
        retriever := _FakeRetriever([0.1, 0.2, 0.9]),
        {},
        base_provider_name="fake_detector",
        retriever_name="fake_retriever",
    )

    result = provider.predict_with_rerank_query(
        Path("image.png"), "building", "white cylindrical building"
    )

    assert retriever.calls == [(Path("image.png"), "white cylindrical building", 3)]
    assert result.metadata["clip_rerank"]["detector_query"] == "building"
    assert result.metadata["clip_rerank"]["clip_query"] == "white cylindrical building"
    provider.close()


def test_clip_rerank_top_k_is_explicit_and_auditable() -> None:
    provider = CLIPRerankedProposalProvider(
        _FakeProposalProvider(_base_result()),
        _FakeRetriever([0.1, 0.2, 0.9]),
        {"candidate_top_k": 1},
        base_provider_name="fake_detector",
        retriever_name="fake_retriever",
    )

    result = provider.predict(Path("image.png"), "building")
    trace = result.metadata["clip_rerank"]

    assert [box[0] for box in result.boxes_xyxy] == [2.0]
    assert trace["retained_indices"] == [2]
    assert trace["filtered_indices"] == [1, 0]


def test_clip_rerank_pool_cap_limits_retriever_call_and_maps_indices() -> None:
    base = _FakeProposalProvider(_base_result())
    retriever = _FakeRetriever([0.9, 0.1])
    provider = CLIPRerankedProposalProvider(
        base,
        retriever,
        {"rerank_pool_k": 2},
        base_provider_name="fake_detector",
        retriever_name="fake_retriever",
    )

    result = provider.predict_with_rerank_query(
        Path("image.png"), "building", "white building", top_k=1
    )
    trace = result.metadata["clip_rerank"]

    # detector scores [0.2, 0.8, 0.5] -> top-2 pool = global indices [1, 2]
    assert retriever.calls == [(Path("image.png"), "white building", 2)]
    assert trace["rerank_pool_k"] == 2
    assert trace["rerank_pool_size"] == 2
    assert trace["pool_order"] == [1, 2]
    assert trace["raw_candidate_count"] == 3
    assert [box[0] for box in result.boxes_xyxy] == [1.0]
    # retained index is reported in the original candidate space
    assert trace["retained_indices"] == [1]
    provider.close()


def test_clip_rerank_empty_candidates_does_not_call_retriever() -> None:
    base = _FakeProposalProvider(_base_result(0))
    retriever = _FakeRetriever([])
    provider = CLIPRerankedProposalProvider(
        base,
        retriever,
        {},
        base_provider_name="fake_detector",
        retriever_name="fake_retriever",
    )

    result = provider.predict(Path("image.png"), "building")

    assert result.boxes_xyxy == []
    assert result.metadata["clip_rerank"]["status"] == "no_candidates"
    assert retriever.calls == []


def test_clip_rerank_fail_open_preserves_detector_candidates() -> None:
    base = _FakeProposalProvider(_base_result())
    retriever = _FakeRetriever([0.1])
    provider = CLIPRerankedProposalProvider(
        base,
        retriever,
        {},
        base_provider_name="fake_detector",
        retriever_name="fake_retriever",
    )

    result = provider.predict(Path("image.png"), "building")

    assert result.provider == "fake_detector"
    assert result.scores == [0.2, 0.8, 0.5]
    assert result.metadata["clip_rerank"]["status"] == "fallback"
    assert "unexpected score count" in result.metadata["clip_rerank"]["reason"]


def test_clip_rerank_can_fail_closed() -> None:
    provider = CLIPRerankedProposalProvider(
        _FakeProposalProvider(_base_result()),
        _FakeRetriever([0.1]),
        {"fail_open": False},
        base_provider_name="fake_detector",
        retriever_name="fake_retriever",
    )

    with pytest.raises(ProposalError, match="unexpected score count"):
        provider.predict(Path("image.png"), "building")


def test_clip_rerank_registry_composes_mock_providers() -> None:
    provider = create_proposal_provider(
        "clip_rerank",
        {
            "base_provider": "mock",
            "base_config": {},
            "retriever": {"provider": "mock", "config": {}},
            "candidate_top_k": 1,
        },
    )

    try:
        result = provider.predict(Path("image.png"), "building")
    finally:
        provider.close()

    assert provider.provider_name == "clip_rerank"
    assert len(result.boxes_xyxy) == 1
    assert result.metadata["clip_rerank"]["status"] == "applied"
