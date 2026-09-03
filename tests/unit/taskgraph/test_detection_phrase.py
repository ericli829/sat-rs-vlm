"""Validate the open-set detection synonym phrase and existing phrase behavior."""

from __future__ import annotations

from pathlib import Path

from sat_rs_vlm.taskgraph.providers import DetectionRequest, ProposalDetectionAdapter
from sat_rs_vlm.taskgraph.runtime_types import ImageRef
from sat_rs_vlm.taskgraph.schema import TargetSpec


def _target(category: str, **attributes: object) -> TargetSpec:
    return TargetSpec(category=category, attributes=dict(attributes))


def test_detection_phrase_groups_synonyms_with_dot_separators() -> None:
    assert _target("building").detection_phrase() == (
        "building . house . home . residential"
    )
    assert _target("car").detection_phrase() == "car . vehicle . truck . sedan"


def test_detection_phrase_keeps_unknown_category_unchanged() -> None:
    assert _target("airport").detection_phrase() == "airport"
    assert _target("custom thing").detection_phrase() == "custom thing"


def test_detection_phrase_preserves_attribute_ordering() -> None:
    target = _target("house", size="large")
    # detection_phrase intentionally ignores attributes (detector query is
    # category-only); the plain phrase still carries them.
    assert target.detection_phrase() == "house . building . home . residential"
    assert target.phrase() == "large house"


def test_case_insensitive_synonym_lookup() -> None:
    assert _target("BUILDING").detection_phrase() == (
        "BUILDING . house . home . residential"
    )
    assert _target("Vehicle").detection_phrase() == (
        "Vehicle . car . truck . bus"
    )


def test_proposal_detection_adapter_sends_synonym_phrase(tmp_path: Path) -> None:
    """LOCATE detection sends the synonym phrase to the LAE sidecar."""

    class RecordingProvider:
        provider_name = "mock_lae"

        def __init__(self) -> None:
            self.phrases: list[str] = []

        def predict(self, _image_path: Path, target_phrase: str):
            from sat_rs_vlm.integrations.detectors.protocol import ProposalResult

            self.phrases.append(target_phrase)
            return ProposalResult([], [], 0.0, self.provider_name, "mock")

        def close(self) -> None:  # pragma: no cover
            pass

    provider = RecordingProvider()
    adapter = ProposalDetectionAdapter(provider)  # type: ignore[arg-type]
    image_path = tmp_path / "source.png"
    from PIL import Image

    Image.new("RGB", (64, 64), "white").save(image_path)
    image = ImageRef(str(image_path), width=64, height=64)
    result = adapter.detect(
        DetectionRequest(
            image,
            _target("building"),
            "LOCATE",
            apply_locate_policy=False,
            use_clip_rerank=False,
        )
    )
    assert provider.phrases == ["building . house . home . residential"]
    assert result.provider == "mock_lae"
