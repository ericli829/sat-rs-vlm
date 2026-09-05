from pathlib import Path

import pytest

from sat_rs_vlm.taskgraph.capabilities import TargetCapability, TargetCapabilityClassifier

ONTOLOGY = Path("configs/eval/semantic/remote_sensing_ontology.json")


@pytest.mark.parametrize("category", ["car", "ship", "building", "bridge"])
def test_object_targets_have_explicit_detector_capability(category: str) -> None:
    classifier = TargetCapabilityClassifier.from_ontology_path(ONTOLOGY)

    decision = classifier.classify(category)

    assert decision.capability is TargetCapability.DETECTOR
    assert decision.effective_capability is TargetCapability.DETECTOR
    assert decision.used_fallback is False
    assert decision.source == "ontology.capabilities"


@pytest.mark.parametrize("category", ["harbor", "farmland", "industrial area"])
def test_semantic_region_targets_have_explicit_retriever_capability(category: str) -> None:
    classifier = TargetCapabilityClassifier.from_ontology_path(ONTOLOGY)

    decision = classifier.classify(category)

    assert decision.capability is TargetCapability.RETRIEVER
    assert decision.effective_capability is TargetCapability.RETRIEVER
    assert decision.used_fallback is False


def test_unknown_target_records_explicit_fallback() -> None:
    classifier = TargetCapabilityClassifier.from_ontology_path(ONTOLOGY)

    decision = classifier.classify("unmapped target")

    assert decision.capability is TargetCapability.UNRESOLVED
    assert decision.effective_capability is TargetCapability.DETECTOR
    assert decision.source == "explicit_unresolved_policy"
