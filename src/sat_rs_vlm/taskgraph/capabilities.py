"""Deterministic target-capability classification for TaskGraph LOCATE."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from sat_rs_vlm.semantics.ontology import load_ontology

from .schema import TargetSpec


class TargetCapability(str, Enum):
    DETECTOR = "DETECTOR"
    RETRIEVER = "RETRIEVER"
    UNRESOLVED = "UNRESOLVED"


class TargetCapabilityError(ValueError):
    """A target cannot be routed under the configured unresolved policy."""


@dataclass(frozen=True)
class TargetCapabilityDecision:
    requested_category: str
    canonical_category: str | None
    capability: TargetCapability
    effective_capability: TargetCapability
    source: str
    reason: str
    unresolved_policy: str

    @property
    def used_fallback(self) -> bool:
        return self.capability is TargetCapability.UNRESOLVED

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_category": self.requested_category,
            "canonical_category": self.canonical_category,
            "capability": self.capability.value,
            "effective_capability": self.effective_capability.value,
            "source": self.source,
            "reason": self.reason,
            "unresolved_policy": self.unresolved_policy,
            "used_fallback": self.used_fallback,
        }


_DEFAULT_DETECTOR_CATEGORIES = frozenset(
    {"aircraft", "bridge", "building", "ship", "storage_tank", "vehicle"}
)
_DEFAULT_RETRIEVER_CATEGORIES = frozenset(
    {
        "airport",
        "farmland",
        "forest",
        "harbor",
        "industrial_area",
        "parking_lot",
        "residential_area",
        "river",
        "road",
        "runway",
        "water",
    }
)


def _normalized(value: object) -> str:
    return " ".join(
        str(value)
        .casefold()
        .replace("_", " ")
        .replace("-", " ")
        .split()
    )


def _normalized_set(values: Iterable[object]) -> set[str]:
    return {normalized for value in values if (normalized := _normalized(value))}


class TargetCapabilityClassifier:
    """Classify object instances and semantic regions without an extra model.

    Ontology capability metadata is authoritative. The built-in taxonomy is
    only used when no ontology is supplied, which keeps fixture runtimes
    dependency-light while preserving a deterministic production contract.
    """

    def __init__(
        self,
        ontology: Mapping[str, Any] | None = None,
        *,
        unresolved_policy: str = "detector_fallback",
        detector_overrides: Iterable[object] = (),
        retriever_overrides: Iterable[object] = (),
        legacy_region_overrides: Iterable[object] = (),
    ) -> None:
        policy = _normalized(unresolved_policy).replace(" ", "_")
        if policy not in {"error", "detector_fallback", "retriever_fallback"}:
            raise ValueError(
                "unresolved_policy must be error, detector_fallback, or retriever_fallback"
            )
        self.unresolved_policy = policy
        payload = dict(ontology or {})
        objects = payload.get("objects", {})
        self._aliases: dict[str, str] = {}
        if isinstance(objects, Mapping):
            for canonical, aliases in objects.items():
                canonical_name = str(canonical)
                self._aliases[_normalized(canonical_name)] = canonical_name
                if isinstance(aliases, Iterable) and not isinstance(aliases, (str, bytes)):
                    for alias in aliases:
                        self._aliases[_normalized(alias)] = canonical_name

        capability_metadata = payload.get("capabilities", {})
        detector_values: Iterable[object] = _DEFAULT_DETECTOR_CATEGORIES
        retriever_values: Iterable[object] = _DEFAULT_RETRIEVER_CATEGORIES
        source = "default_taxonomy"
        if isinstance(capability_metadata, Mapping):
            declared_detector = capability_metadata.get("detector")
            declared_retriever = capability_metadata.get(
                "region_retriever", capability_metadata.get("retriever")
            )
            if isinstance(declared_detector, Iterable) and not isinstance(
                declared_detector, (str, bytes)
            ):
                detector_values = declared_detector
                source = "ontology.capabilities"
            if isinstance(declared_retriever, Iterable) and not isinstance(
                declared_retriever, (str, bytes)
            ):
                retriever_values = declared_retriever
                source = "ontology.capabilities"

        detector_categories = _normalized_set(detector_values)
        retriever_categories = _normalized_set(retriever_values)
        detector_categories.update(_normalized_set(detector_overrides))
        retriever_categories.update(_normalized_set(retriever_overrides))
        retriever_categories.update(_normalized_set(legacy_region_overrides))
        overlap = detector_categories.intersection(retriever_categories)
        if overlap:
            raise ValueError(
                "target capability categories cannot be both detector and retriever: "
                + ", ".join(sorted(overlap))
            )
        self.detector_categories = frozenset(detector_categories)
        self.retriever_categories = frozenset(retriever_categories)
        self._detector_aliases = tuple(
            sorted(
                (
                    (alias, canonical)
                    for alias, canonical in self._aliases.items()
                    if _normalized(canonical) in self.detector_categories
                ),
                key=lambda item: (-len(item[0].split()), -len(item[0]), item[0]),
            )
        )
        self.metadata_source = source

    @classmethod
    def from_ontology_path(
        cls,
        path: str | Path,
        *,
        unresolved_policy: str = "detector_fallback",
        detector_overrides: Iterable[object] = (),
        retriever_overrides: Iterable[object] = (),
        legacy_region_overrides: Iterable[object] = (),
    ) -> TargetCapabilityClassifier:
        ontology = load_ontology(Path(path).expanduser().resolve())
        return cls(
            ontology,
            unresolved_policy=unresolved_policy,
            detector_overrides=detector_overrides,
            retriever_overrides=retriever_overrides,
            legacy_region_overrides=legacy_region_overrides,
        )

    def classify(self, target: TargetSpec | str) -> TargetCapabilityDecision:
        requested = target.category if isinstance(target, TargetSpec) else str(target)
        normalized = _normalized(requested)
        canonical = self._aliases.get(normalized)
        if canonical is None:
            tokens = normalized.split()
            for alias, alias_canonical in self._detector_aliases:
                alias_tokens = alias.split()
                if len(alias_tokens) > len(tokens):
                    continue
                if any(
                    tokens[index : index + len(alias_tokens)] == alias_tokens
                    for index in range(len(tokens) - len(alias_tokens) + 1)
                ):
                    canonical = alias_canonical
                    break
        canonical_key = _normalized(canonical) if canonical is not None else normalized
        if canonical_key in self.detector_categories:
            return TargetCapabilityDecision(
                requested,
                canonical,
                TargetCapability.DETECTOR,
                TargetCapability.DETECTOR,
                self.metadata_source,
                "ontology taxonomy marks the target as an object instance",
                self.unresolved_policy,
            )
        if canonical_key in self.retriever_categories:
            return TargetCapabilityDecision(
                requested,
                canonical,
                TargetCapability.RETRIEVER,
                TargetCapability.RETRIEVER,
                self.metadata_source,
                "ontology taxonomy marks the target as a semantic region",
                self.unresolved_policy,
            )
        if self.unresolved_policy == "error":
            raise TargetCapabilityError(
                f"target capability is unresolved for category {requested!r}"
            )
        fallback = (
            TargetCapability.RETRIEVER
            if self.unresolved_policy == "retriever_fallback"
            else TargetCapability.DETECTOR
        )
        return TargetCapabilityDecision(
            requested,
            canonical,
            TargetCapability.UNRESOLVED,
            fallback,
            "explicit_unresolved_policy",
            "ontology has no capability metadata for the target",
            self.unresolved_policy,
        )
