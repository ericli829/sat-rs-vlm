"""Auditable GeoRSCLIP verification shared by TaskGraph detection and counting."""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from PIL import Image

from sat_rs_vlm.integrations.detectors.dino_georsclip import (
    DinoGeoRSCLIPTwoStage,
    resolve_target,
)
from sat_rs_vlm.integrations.detectors.protocol import ProposalError, ProposalResult
from sat_rs_vlm.taskgraph.providers import (
    CountingRequest,
    CountingResult,
    DetectionRequest,
    DetectionSet,
)
from sat_rs_vlm.taskgraph.runtime_types import Entity, EntitySet, ImageRef, Region
from sat_rs_vlm.taskgraph.schema import TargetSpec


class VerificationStatus(str, Enum):
    MATCH = "MATCH"
    REJECT = "REJECT"


@dataclass(frozen=True)
class VerificationResult:
    detections: EntitySet
    decisions: tuple[dict[str, Any], ...]
    metadata: dict[str, Any]

    @property
    def matched(self) -> tuple[Entity, ...]:
        return tuple(
            e
            for e in self.detections.entities
            if e.provenance.get("verification_status") == "MATCH"
        )


class UnavailablePromptScorer:
    """Record a CLIP setup failure while keeping the DINO output path usable."""

    def __init__(self, reason: str, model_id: str = "GeoRSCLIP") -> None:
        self.initialization_error = reason
        self.model_id = model_id
        self.region_crop_policy = "square_letterbox_gray"

    def score_prompt_groups(self, *args: Any) -> Any:
        raise RuntimeError(self.initialization_error)

    def close(self) -> None:
        pass


class UnavailableCandidateProvider:
    provider_name = "unavailable_dino_fallback"

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def detect(self, request: DetectionRequest) -> DetectionSet:
        return DetectionSet(
            EntitySet((), {"candidate_pool_complete": False}),
            0.0,
            self.provider_name,
            {"candidate_pool_complete": False, "detector_error": self.reason},
        )

    def count(self, request: CountingRequest) -> CountingResult:
        return CountingResult(
            0,
            EntitySet((), {"candidate_pool_complete": False}),
            self.provider_name,
            0.0,
            {"candidate_pool_complete": False, "detector_error": self.reason},
        )

    def close(self) -> None:
        pass


class GeoRSCLIPCandidateVerifier:
    """Reuse the established prompt ensemble and 80/20 scorer without detecting again.

    Positive margins match; all other scored margins reject.
    Intrinsic attributes supplied upstream are scored and joined with AND.
    Scoring failures keep available DINO candidates, with explicit diagnostics.
    Similarity margins never replace the original detector score.
    """

    provider_name = "georsclip_candidate_verification"

    def __init__(
        self,
        retriever: Any,
        prompt_catalog: Mapping[str, Any],
        config: Mapping[str, Any] | None = None,
    ) -> None:
        self.retriever = retriever
        self.config = dict(config or {})
        self.scorer = DinoGeoRSCLIPTwoStage(
            None,
            retriever,
            prompt_catalog,
            coarse_margin_threshold=float(self.config.get("coarse_margin_threshold", 0.0)),
            qualified_margin_threshold=float(self.config.get("attribute_margin_threshold", 0.0)),
            unrestricted_weight=0.8,
            contrast_weight=0.2,
            coarse_score_weight=0.5,
            attribute_score_weight=0.5,
        )
        supported = self.config.get(
            "sample_checked_attributes",
            self.config.get(
                "supported_attributes",
                {
                    "color": ["red", "white", "blue", "orange"],
                },
            ),
        )
        if not isinstance(supported, Mapping):
            raise ValueError("candidate_verification.supported_attributes must be a mapping")
        if any(
            not isinstance(values, Sequence) or isinstance(values, str)
            for values in supported.values()
        ):
            raise ValueError("each supported attribute must define a sequence of modifier values")
        self.supported = {
            str(k): {str(v).strip().casefold() for v in values} for k, values in supported.items()
        }
        self.closed = False

    def normalize_target(self, target: TargetSpec) -> TargetSpec:
        """Local adapter for direct-count phrases, preserving unhandled modifiers."""
        try:
            category, _group, modifier = resolve_target(target.category, self.scorer.groups)
        except ProposalError:
            return target
        attributes = dict(target.attributes)
        vocabularies = {
            "color": {
                "red",
                "white",
                "blue",
                "orange",
                "black",
                "gray",
                "grey",
                "green",
                "yellow",
                "brown",
                "purple",
                "pink",
            },
            "size": {"large", "big", "small", "tall", "long", "larger", "smaller"},
            "shape": {"round", "square", "rectangular", "circular", "triangular"},
            "state": {"moving", "parked", "open", "closed", "abandoned", "unplanted"},
        }
        unknown = []
        for word in (modifier or "").split():
            kind = next((key for key, values in vocabularies.items() if word in values), None)
            if kind is None:
                unknown.append(word)
            elif kind not in attributes:
                attributes[kind] = word
            elif str(attributes[kind]).casefold() != word:
                # Conflicting conjunctions remain unsupported, rather than losing a constraint.
                attributes[kind] = f"{attributes[kind]} {word}"
        if unknown:
            attributes["state"] = " ".join([str(attributes.get("state", "")), *unknown]).strip()
        return TargetSpec(category=category, attributes=attributes)

    def normalize_direct_target(self, target: TargetSpec, question: str) -> TargetSpec:
        normalized = self.normalize_target(target)
        parsed = self.normalize_target(TargetSpec(category=question))
        attributes = dict(normalized.attributes)
        if parsed.category == normalized.category:
            attributes = {**parsed.attributes, **attributes}
        # Direct counting supports leading intrinsic modifiers. Preserve an
        # relative/part clause as an explicit modifier supplied to CLIP.
        clause = re.search(r"\b(?:with|having|that|which|whose)\b.+", question, re.IGNORECASE)
        if clause:
            attributes["has_part"] = clause.group(0).strip()
        return TargetSpec(category=normalized.category, attributes=attributes)

    @staticmethod
    def _verdict(margin: float, threshold: float) -> VerificationStatus:
        if margin > threshold:
            return VerificationStatus.MATCH
        return VerificationStatus.REJECT

    def verify_candidates(
        self,
        image: ImageRef,
        scope: ImageRef | Region,
        target_spec: TargetSpec,
        candidates: EntitySet,
    ) -> VerificationResult:
        started = time.perf_counter()
        target = self.normalize_target(target_spec)
        entities = list(candidates.entities)
        decisions: list[dict[str, Any]] = []
        for index, entity in enumerate(entities):
            decisions.append(
                {
                    "candidate_id": entity.provenance.get(
                        "candidate_id",
                        entity.region.provenance.get("candidate_id", f"candidate_{index + 1:04d}"),
                    ),
                    "bbox_xyxy_global": list(entity.region.bbox_xyxy_global),
                    "dino_category": entity.label,
                    "dino_score": entity.score,
                    "clip_model_id": getattr(self.retriever, "model_id", "unknown"),
                    "status": "MATCH",
                    "clip_verified": False,
                    "reason": "not_scored",
                    "attributes": {},
                    "coarse_margin": None,
                    "mixed_attribute_score": None,
                    "combined_score": None,
                }
            )
        errors: dict[str, str] = {}
        if getattr(self.retriever, "initialization_error", None):
            errors["model_init"] = str(self.retriever.initialization_error)
        if self.config.get("prompt_catalog_error"):
            errors["prompt_catalog"] = str(self.config["prompt_catalog_error"])
        reports: dict[str, Any] = {}
        valid: list[int] = []
        id_counts = Counter(str(decision["candidate_id"]) for decision in decisions)
        try:
            with Image.open(image.path) as source:
                width, height = source.size
            bounds = (
                scope.bbox_xyxy_global
                if isinstance(scope, Region)
                else (0.0, 0.0, float(width), float(height))
            )
            for index, entity in enumerate(entities):
                box = entity.region.bbox_xyxy_global
                if id_counts[str(decisions[index]["candidate_id"])] > 1:
                    decisions[index].update(status="REJECT", reason="duplicate_candidate_id")
                    continue
                if (
                    entity.region.image.uri_or_key != image.uri_or_key
                    or entity.score is not None
                    and not math.isfinite(float(entity.score))
                    or not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)
                    or not (
                        bounds[0] <= box[0]
                        and bounds[1] <= box[1]
                        and box[2] <= bounds[2]
                        and box[3] <= bounds[3]
                    )
                ):
                    decisions[index].update(status="REJECT", reason="invalid_global_bbox_or_scope")
                else:
                    valid.append(index)
            category, group, _modifier = resolve_target(target.category, self.scorer.groups)
        except Exception as exc:
            errors["input_or_category"] = f"{type(exc).__name__}: {exc}"
            for decision in decisions:
                if decision["status"] != "REJECT":
                    decision.update(
                        status="MATCH", reason="DINO_fallback: " + errors["input_or_category"]
                    )
            valid = []

        supported = {key: str(value).strip() for key, value in target.attributes.items()}
        unchecked = {
            key: value
            for key, value in target.attributes.items()
            if str(value).strip().casefold() not in self.supported.get(key, set())
        }
        if valid:
            proposals = ProposalResult(
                boxes_xyxy=[list(entities[i].region.bbox_xyxy_global) for i in valid],
                scores=[float(entities[i].score or 0.0) for i in valid],
                latency_ms=0.0,
                provider="existing_fused_candidates",
                model_id="upstream",
                metadata=dict(candidates.provenance),
            )
            # Each call uses the same image embedding cache; all candidates are batched.
            for key, modifier in (supported or {"__category__": None}).items():
                try:
                    reports[key] = self.scorer.run(
                        image.path,
                        target_category=category,
                        group_name=group,
                        modifier=modifier,
                        proposal_result=proposals,
                    )
                except Exception as exc:
                    errors[key] = f"{type(exc).__name__}: {exc}"
            for position, index in enumerate(valid):
                decision = decisions[index]
                if not reports:
                    decision.update(status="MATCH", reason="DINO_fallback: CLIP scoring failed")
                    continue
                row = next(iter(reports.values()))["candidates"][position]
                coarse = float(row["coarse_margin"])
                coarse_status = self._verdict(coarse, self.scorer.coarse_margin_threshold)
                decision["coarse_margin"] = coarse
                conditions = []
                for key, report in reports.items():
                    if key == "__category__":
                        continue
                    attr_row = report["candidates"][position]
                    value = attr_row["mixed_attribute_score"]
                    verdict = (
                        self._verdict(float(value), self.scorer.qualified_margin_threshold)
                        if value is not None
                        else VerificationStatus.REJECT
                    )
                    conditions.append(verdict)
                    decision["attributes"][key] = {
                        "value": supported[key],
                        "status": verdict.value,
                        "prompts": report["attribute_filter"]["prompts"],
                        "qualified_score": attr_row["qualified_score"],
                        "unrestricted_score": attr_row["unrestricted_score"],
                        "contrast_score": attr_row["contrast_score"],
                        "margin": value,
                    }
                margins = [
                    item["margin"]
                    for item in decision["attributes"].values()
                    if item["margin"] is not None
                ]
                mixed = sum(margins) / len(margins) if margins else None
                decision["mixed_attribute_score"] = mixed
                decision["combined_score"] = (
                    0.5 * coarse + 0.5 * mixed if mixed is not None else coarse
                )
                if coarse_status is VerificationStatus.REJECT:
                    status, reason = "REJECT", "category_margin_below_threshold"
                elif VerificationStatus.REJECT in conditions:
                    status, reason = "REJECT", "required_attribute_margin_below_threshold"
                elif errors:
                    status, reason = "MATCH", "DINO_fallback: some CLIP conditions failed"
                else:
                    status, reason = "MATCH", "all_required_margins_positive"
                decision.update(status=status, reason=reason, clip_verified=not errors)

        counts = {
            status.value: sum(d["status"] == status.value for d in decisions)
            for status in VerificationStatus
        }
        input_complete = (
            candidates.provenance.get("candidate_pool_complete") is not False
            and not candidates.provenance.get("candidate_truncated")
            and candidates.provenance.get("instance_resolution_complete") is not False
            and not candidates.provenance.get("unresolved_instance_regions")
        )
        complete = input_complete and not errors
        if any(
            e.provenance.get("instance_status", "CONFIRMED_INSTANCE") != "CONFIRMED_INSTANCE"
            for e, d in zip(entities, decisions, strict=True)
            if d["status"] != "REJECT"
        ):
            complete = False
        metadata = {
            "enabled": True,
            "method": "fixed_category_ensemble_80_20_50_50",
            "target_spec": target.model_dump(mode="json"),
            "model_id": getattr(self.retriever, "model_id", "unknown"),
            "candidate_count": len(entities),
            "counts": counts,
            "attributes_without_prior_sample_check": unchecked,
            "errors": errors,
            "candidate_pool_complete": complete,
            "input_candidate_pool_complete": input_complete,
            "decision_policy": "binary_always_output",
            "count_output_available": True,
            "clip_verified": not errors,
            "coarse_margin_threshold": self.scorer.coarse_margin_threshold,
            "attribute_margin_threshold": self.scorer.qualified_margin_threshold,
            "attribute_formula": "0.8*(qualified-unrestricted)+0.2*(qualified-contrast)",
            "ranking_formula": "0.5*coarse_margin+0.5*mean(required_attribute_margins)",
            "multiple_attributes": "AND",
            "scores_are_probabilities": False,
            "crop_policy": getattr(self.retriever, "region_crop_policy", "unknown"),
            "category_prompts": (
                next(iter(reports.values()))["coarse_filter"]["prompts"] if reports else {}
            ),
            "decisions": decisions,
            "latency_ms": (time.perf_counter() - started) * 1000,
        }
        kept = []
        for entity, decision in zip(entities, decisions, strict=True):
            if decision["status"] == "REJECT":
                continue
            provenance = {
                **entity.provenance,
                "candidate_id": decision["candidate_id"],
                "verification_status": decision["status"],
                "attribute_verdict": decision["status"],
                "clip_verification": decision,
            }
            kept.append(
                replace(
                    entity,
                    provenance=provenance,
                    region=replace(
                        entity.region,
                        provenance={
                            **entity.region.provenance,
                            "candidate_id": decision["candidate_id"],
                            "verification_status": decision["status"],
                        },
                    ),
                )
            )
        kept.sort(
            key=lambda e: -(
                e.provenance["clip_verification"]["combined_score"]
                if e.provenance["clip_verification"]["combined_score"] is not None
                else -math.inf
            )
        )
        provenance = {
            **candidates.provenance,
            "clip_verification": metadata,
            "allow_partial_output": True,
            "candidate_pool_complete": complete,
            "resolution_status": (
                "EMPTY" if not kept else "PRIMARY_RESOLVED" if len(kept) == 1 else "MULTIPLE_VALID"
            ),
        }
        return VerificationResult(EntitySet(tuple(kept), provenance), tuple(decisions), metadata)

    def close(self) -> None:
        if not self.closed:
            self.retriever.close()
            self.closed = True


def _provider_pool(
    detections: EntitySet, metadata: Mapping[str, Any], *, count_complete: bool = True
) -> EntitySet:
    provenance = dict(detections.provenance)
    provenance["candidate_pool_complete"] = (
        count_complete
        and provenance.get("candidate_pool_complete") is not False
        and metadata.get("candidate_pool_complete") is not False
    )
    provenance["candidate_truncated"] = bool(
        provenance.get("candidate_truncated") or metadata.get("candidate_truncated")
    )
    provenance["instance_resolution_complete"] = (
        provenance.get("instance_resolution_complete") is not False
        and metadata.get("instance_resolution_complete") is not False
    )
    if metadata.get("unresolved_instance_regions"):
        provenance["unresolved_instance_regions"] = metadata["unresolved_instance_regions"]
    return EntitySet(detections.entities, provenance)


class VerifiedDetectionProvider:
    def __init__(self, provider: Any, verifier: GeoRSCLIPCandidateVerifier) -> None:
        self._provider, self._verifier = provider, verifier
        self.provider_name = f"{provider.provider_name}+georsclip_verification"

    def normalize_target(self, target: TargetSpec) -> TargetSpec:
        return self._verifier.normalize_target(target)

    def detect(self, request: DetectionRequest) -> DetectionSet:
        started = time.perf_counter()
        target = self.normalize_target(request.target)
        try:
            raw = self._provider.detect(
                replace(request, target=target, apply_locate_policy=False, use_clip_rerank=False)
            )
        except Exception as exc:
            raw = DetectionSet(
                getattr(exc, "detections", EntitySet((), {"candidate_pool_complete": False})),
                0.0,
                self._provider.provider_name,
                {
                    "candidate_pool_complete": False,
                    "detector_error": f"{type(exc).__name__}: {exc}",
                },
            )
        image = request.scope.image if isinstance(request.scope, Region) else request.scope
        verified = self._verifier.verify_candidates(
            image, request.scope, target, _provider_pool(raw.detections, raw.metadata)
        )
        return DetectionSet(
            verified.detections,
            (time.perf_counter() - started) * 1000,
            self.provider_name,
            {
                **raw.metadata,
                "clip_verification": verified.metadata,
                "candidate_pool_complete": verified.metadata["candidate_pool_complete"],
            },
        )

    def close(self) -> None:
        self._provider.close()
        self._verifier.close()


class VerifiedCountingProvider:
    def __init__(self, provider: Any, verifier: GeoRSCLIPCandidateVerifier) -> None:
        self._provider, self._verifier = provider, verifier
        self.provider_name = f"{provider.provider_name}+georsclip_verification"

    def normalize_target(self, target: TargetSpec) -> TargetSpec:
        return self._verifier.normalize_target(target)

    def normalize_direct_target(self, target: TargetSpec, question: str) -> TargetSpec:
        return self._verifier.normalize_direct_target(target, question)

    def count(self, request: CountingRequest) -> CountingResult:
        started = time.perf_counter()
        target = self.normalize_target(request.target)
        try:
            raw = self._provider.count(replace(request, target=target))
        except Exception as exc:
            raw = CountingResult(
                0,
                getattr(exc, "detections", EntitySet((), {"candidate_pool_complete": False})),
                self._provider.provider_name,
                0.0,
                {
                    "candidate_pool_complete": False,
                    "detector_error": f"{type(exc).__name__}: {exc}",
                },
            )
        image = request.scope.image if isinstance(request.scope, Region) else request.scope
        count_semantics_complete = (
            raw.count is not None
            and raw.count == len(raw.detections.entities)
            and raw.metadata.get("count_status") != "UNRESOLVED"
        )
        verified = self._verifier.verify_candidates(
            image,
            request.scope,
            target,
            _provider_pool(raw.detections, raw.metadata, count_complete=count_semantics_complete),
        )
        complete = verified.metadata["candidate_pool_complete"]
        return CountingResult(
            len(verified.matched),
            verified.detections,
            self.provider_name,
            (time.perf_counter() - started) * 1000,
            {
                **raw.metadata,
                "count_before_verification": raw.count,
                "input_count_semantics_complete": count_semantics_complete,
                "clip_verification": verified.metadata,
                "candidate_pool_complete": complete,
                "count_status": "FILTERED" if complete else "AVAILABLE_CANDIDATES_FALLBACK",
                "count_output_available": True,
            },
        )

    def close(self) -> None:
        self._provider.close()
        self._verifier.close()
