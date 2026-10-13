"""Two-stage DINO proposal filtering with GeoRSCLIP prompt ensembles."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from sat_rs_vlm.integrations.retrievers.protocol import RetrieverProvider

from .protocol import ProposalError, ProposalProvider

PROMPT_TEMPLATES = (
    "satellite imagery of a {term}.",
    "aerial photograph of a {term} viewed from above.",
    "an overhead remote sensing image showing a {term}.",
)
UNRESTRICTED_WEIGHT = 0.80
CONTRAST_WEIGHT = 0.20
COARSE_SCORE_WEIGHT = 0.50
ATTRIBUTE_SCORE_WEIGHT = 0.50


def _normalize(value: str) -> str:
    value = str(value).casefold().replace("_", " ").replace("-", " ")
    return re.sub(r"[^a-z0-9 ]+", " ", value).strip()


def _extract_counted_phrase(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value).casefold()).strip()
    stop = (
        r"are|is|were|was|can|could|do|does|did|have|has|had|in|within|inside|"
        r"on|at|from|to|along|near|around|among|between|there|located|present|"
        r"visible|shown|depicted|parked|driving|moving|docked|standing|with|of"
    )
    patterns = (
        rf"\bhow many\s+(.+?)(?=\s+(?:{stop})\b|[?.!,;]|$)",
        rf"\bnumber of\s+(.+?)(?=\s+(?:{stop})\b|[?.!,;]|$)",
        rf"\bcount(?:ing)?(?:\s+the|\s+of)?\s+(.+?)(?=\s+(?:{stop})\b|[?.!,;]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return text.strip(" ?.!;,:")


def resolve_target(
    target_or_question: str,
    groups: Mapping[str, Any],
    *,
    modifier_override: str | None = None,
) -> tuple[str, str, str | None]:
    """Resolve an object alias and optional leading modifier from user text."""

    target_phrase = _extract_counted_phrase(target_or_question)
    normalized_phrase = _normalize(target_phrase)
    matches: list[tuple[int, str, str, re.Match[str]]] = []
    for group_name, raw_group in groups.items():
        if not isinstance(raw_group, Mapping):
            continue
        aliases = raw_group.get("aliases", ())
        if not isinstance(aliases, Sequence) or isinstance(aliases, str):
            continue
        for raw_alias in aliases:
            alias = _normalize(str(raw_alias))
            if not alias:
                continue
            words = r"\s+".join(re.escape(word) for word in alias.split())
            pattern = re.compile(rf"(?<![a-z0-9]){words}(?:s|es)?(?![a-z0-9])")
            match = pattern.search(normalized_phrase)
            if match:
                matches.append((len(alias), str(group_name), alias, match))
    if not matches:
        supported = sorted(
            {
                str(alias)
                for raw_group in groups.values()
                if isinstance(raw_group, Mapping)
                for alias in raw_group.get("aliases", ())
            }
        )
        raise ProposalError(
            f"could not map target {target_phrase!r} to a configured CLIP group; "
            f"supported target examples: {', '.join(supported[:18])}"
        )
    _length, group_name, target_category, match = max(matches, key=lambda item: (item[0], item[1]))
    inferred_modifier = normalized_phrase[: match.start()].strip()
    inferred_modifier = re.sub(r"^(?:a|an|the|some|many)\s+", "", inferred_modifier)
    modifier = str(modifier_override).strip() if modifier_override else inferred_modifier
    return target_category, group_name, modifier or None


def prompts_for_terms(terms: Sequence[str], templates: Sequence[str]) -> list[str]:
    return [str(template).format(term=str(term)) for term in terms for template in templates]


class DinoGeoRSCLIPTwoStage:
    """Detect broad candidates, CLIP-screen their class, then rank an attribute."""

    def __init__(
        self,
        detector: ProposalProvider | None,
        retriever: RetrieverProvider,
        prompt_catalog: Mapping[str, Any],
        *,
        coarse_margin_threshold: float = 0.0,
        qualified_margin_threshold: float = 0.0,
        unrestricted_weight: float = UNRESTRICTED_WEIGHT,
        contrast_weight: float = CONTRAST_WEIGHT,
        coarse_score_weight: float = COARSE_SCORE_WEIGHT,
        attribute_score_weight: float = ATTRIBUTE_SCORE_WEIGHT,
    ) -> None:
        self.detector = detector
        self.retriever = retriever
        self.prompt_catalog = dict(prompt_catalog)
        self.groups = self.prompt_catalog.get("groups", {})
        self.templates = self.prompt_catalog.get("templates", PROMPT_TEMPLATES)
        self.coarse_margin_threshold = float(coarse_margin_threshold)
        self.qualified_margin_threshold = float(qualified_margin_threshold)
        self.unrestricted_weight = float(unrestricted_weight)
        self.contrast_weight = float(contrast_weight)
        self.coarse_score_weight = float(coarse_score_weight)
        self.attribute_score_weight = float(attribute_score_weight)
        if not math.isfinite(self.coarse_margin_threshold):
            raise ProposalError("coarse margin threshold must be finite")
        if not math.isfinite(self.qualified_margin_threshold):
            raise ProposalError("qualified margin threshold must be finite")
        if not all(
            math.isfinite(weight) and weight >= 0.0
            for weight in (self.unrestricted_weight, self.contrast_weight)
        ):
            raise ProposalError("attribute mixture weights must be finite and non-negative")
        total = self.unrestricted_weight + self.contrast_weight
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ProposalError("attribute mixture weights must add up to 1")
        if not all(
            math.isfinite(weight) and weight >= 0.0
            for weight in (self.coarse_score_weight, self.attribute_score_weight)
        ):
            raise ProposalError("combined score weights must be finite and non-negative")
        combined_total = self.coarse_score_weight + self.attribute_score_weight
        if not math.isclose(combined_total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ProposalError("combined score weights must add up to 1")
        if not isinstance(self.groups, Mapping) or not isinstance(self.templates, Sequence):
            raise ProposalError("prompt catalog must define groups and templates")
        if not callable(getattr(retriever, "score_prompt_groups", None)):
            raise ProposalError(
                "the configured retriever must support prompt-group scoring; "
                "use the GeoRSCLIP/OpenCLIP provider"
            )

    def run(
        self,
        image_path: str | Path,
        *,
        target_category: str,
        group_name: str | None = None,
        modifier: str | None = None,
        contrast_modifier: str | None = None,
        unrestricted_query: str | None = None,
        qualified_query: str | None = None,
        contrast_query: str | None = None,
        proposal_result: Any = None,
    ) -> dict[str, Any]:
        resolved_image = Path(image_path).expanduser().resolve()
        if not resolved_image.is_file():
            raise ProposalError(f"input image does not exist: {resolved_image}")
        category = str(target_category).strip()
        if not category:
            raise ProposalError("target_category is required")
        if group_name is None:
            category, group_name, inferred_modifier = resolve_target(
                category, self.groups, modifier_override=modifier
            )
            modifier = modifier or inferred_modifier
        modifier = str(modifier or "").strip()
        raw_group = self.groups.get(group_name)
        if not isinstance(raw_group, Mapping):
            raise ProposalError(f"unknown prompt group {group_name!r}")
        positive_terms = [str(term).strip() for term in raw_group.get("positive_terms", ())]
        negative_terms = [str(term).strip() for term in raw_group.get("negative_terms", ())]
        if not positive_terms or not negative_terms:
            raise ProposalError(f"prompt group {group_name!r} needs positive and negative terms")

        detector_phrase = " . ".join(positive_terms)
        base_query = unrestricted_query or category
        limited_query = qualified_query or f"{modifier} {category}".strip()
        alternative_modifier = contrast_modifier or f"non-{modifier}"
        alternative_query = contrast_query or f"{alternative_modifier} {category}"
        coarse_prompts = {
            "positive": prompts_for_terms(positive_terms, self.templates),
            "negative": prompts_for_terms(negative_terms, self.templates),
        }
        attribute_prompts = {
            "qualified": [limited_query],
            "unrestricted": [base_query],
            "contrast": [alternative_query],
        }
        if proposal_result is None:
            if self.detector is None:
                raise ProposalError("a detector or an existing proposal_result is required")
            proposal_result = self.detector.predict(resolved_image, detector_phrase)
        candidate_rows = [
            {
                "id": index + 1,
                "bbox_xyxy": [float(value) for value in box],
                "dino_score": float(score),
                "coarse_positive_score": None,
                "coarse_negative_score": None,
                "coarse_margin": None,
                "coarse_selected": False,
                "coarse_rank": None,
                "qualified_score": None,
                "unrestricted_score": None,
                "contrast_score": None,
                "delta_qualified_minus_unrestricted": None,
                "delta_qualified_minus_contrast": None,
                "mixed_attribute_score": None,
                "combined_score": None,
                "combined_selected": False,
                "combined_rank": None,
                "qualified_selected": False,
                "qualified_rank": None,
            }
            for index, (box, score) in enumerate(
                zip(proposal_result.boxes_xyxy, proposal_result.scores, strict=True)
            )
        ]
        if not candidate_rows:
            return self._result(
                resolved_image,
                category,
                group_name,
                modifier,
                detector_phrase,
                positive_terms,
                negative_terms,
                [],
                proposal_result,
                None,
                None,
                [],
                [],
                base_query,
                limited_query,
                alternative_query,
                coarse_prompts=coarse_prompts,
                attribute_prompts=attribute_prompts,
            )

        regions = [row["bbox_xyxy"] for row in candidate_rows]
        coarse_result = self.retriever.score_prompt_groups(resolved_image, coarse_prompts, regions)
        positive_scores = coarse_result.scores_by_group["positive"]
        negative_scores = coarse_result.scores_by_group["negative"]
        for row, positive, negative in zip(
            candidate_rows, positive_scores, negative_scores, strict=True
        ):
            margin = float(positive - negative)
            row.update(
                {
                    "coarse_positive_score": float(positive),
                    "coarse_negative_score": float(negative),
                    "coarse_margin": margin,
                    "coarse_selected": margin > self.coarse_margin_threshold,
                }
            )

        coarse_ranked = sorted(
            candidate_rows,
            key=lambda row: (
                -float(row["coarse_margin"]),
                -float(row["dino_score"]),
                int(row["id"]),
            ),
        )
        for rank, row in enumerate(coarse_ranked, start=1):
            row["coarse_rank"] = rank
        unqualified_rows = [row for row in coarse_ranked if row["coarse_selected"]]

        attribute_result = None
        qualified_rows: list[dict[str, Any]] = []
        if unqualified_rows and modifier:
            coarse_regions = [row["bbox_xyxy"] for row in unqualified_rows]
            attribute_result = self.retriever.score_prompt_groups(
                resolved_image, attribute_prompts, coarse_regions
            )
            for index, row in enumerate(unqualified_rows):
                limited_score = float(attribute_result.scores_by_group["qualified"][index])
                unrestricted_score = float(attribute_result.scores_by_group["unrestricted"][index])
                contrast_score = float(attribute_result.scores_by_group["contrast"][index])
                unrestricted_delta = limited_score - unrestricted_score
                contrast_delta = limited_score - contrast_score
                mixed_score = (
                    self.unrestricted_weight * unrestricted_delta
                    + self.contrast_weight * contrast_delta
                )
                combined_score = (
                    self.coarse_score_weight * float(row["coarse_margin"])
                    + self.attribute_score_weight * mixed_score
                )
                selected = (
                    bool(row["coarse_selected"]) and mixed_score > self.qualified_margin_threshold
                )
                row.update(
                    {
                        "qualified_score": limited_score,
                        "unrestricted_score": unrestricted_score,
                        "contrast_score": contrast_score,
                        "delta_qualified_minus_unrestricted": unrestricted_delta,
                        "delta_qualified_minus_contrast": contrast_delta,
                        "mixed_attribute_score": mixed_score,
                        "combined_score": combined_score,
                        "combined_selected": selected,
                        "qualified_selected": selected,
                    }
                )
            qualified_ranked = sorted(
                unqualified_rows,
                key=lambda row: (
                    -float(row["combined_score"]),
                    -float(row["dino_score"]),
                    int(row["id"]),
                ),
            )
            for rank, row in enumerate(qualified_ranked, start=1):
                row["combined_rank"] = rank
            qualified_rows = [row for row in qualified_ranked if row["combined_selected"]]
            for rank, row in enumerate(qualified_rows, start=1):
                row["qualified_rank"] = rank
        elif not modifier:
            qualified_rows = list(unqualified_rows)
            for rank, row in enumerate(qualified_rows, start=1):
                row.update(
                    {
                        "combined_score": row["coarse_margin"],
                        "combined_selected": True,
                        "combined_rank": rank,
                        "qualified_selected": True,
                        "qualified_rank": rank,
                    }
                )

        return self._result(
            resolved_image,
            category,
            group_name,
            modifier,
            detector_phrase,
            positive_terms,
            negative_terms,
            candidate_rows,
            proposal_result,
            coarse_result,
            attribute_result,
            unqualified_rows,
            qualified_rows,
            base_query,
            limited_query,
            alternative_query,
            coarse_prompts=coarse_prompts,
            attribute_prompts=attribute_prompts,
        )

    def _result(
        self,
        image_path: Path,
        target_category: str,
        group_name: str,
        modifier: str,
        detector_phrase: str,
        positive_terms: list[str],
        negative_terms: list[str],
        candidates: list[dict[str, Any]],
        proposal_result: Any,
        coarse_result: Any,
        attribute_result: Any,
        unqualified_rows: list[dict[str, Any]],
        qualified_rows: list[dict[str, Any]],
        unrestricted_query: str | None,
        qualified_query: str | None,
        contrast_query: str | None,
        *,
        coarse_prompts: Mapping[str, Sequence[str]] | None = None,
        attribute_prompts: Mapping[str, Sequence[str]] | None = None,
    ) -> dict[str, Any]:
        return {
            "image": str(image_path),
            "target_category": target_category,
            "prompt_group": group_name,
            "modifier": modifier,
            "detector": {
                "provider": proposal_result.provider,
                "model_id": proposal_result.model_id,
                "latency_ms": proposal_result.latency_ms,
                "prompt": detector_phrase,
                "metadata": dict(proposal_result.metadata),
            },
            "georsclip": {
                "provider": coarse_result.provider if coarse_result else None,
                "model_id": coarse_result.model_id if coarse_result else None,
                "coarse_latency_ms": coarse_result.latency_ms if coarse_result else 0.0,
                "attribute_latency_ms": (attribute_result.latency_ms if attribute_result else 0.0),
            },
            "coarse_filter": {
                "positive_terms": positive_terms,
                "negative_terms": negative_terms,
                "prompts": dict(coarse_prompts or {}),
                "margin_threshold_strictly_greater_than": self.coarse_margin_threshold,
                "dino_candidate_count": len(candidates),
                "unqualified_detection_count": len(unqualified_rows),
            },
            "attribute_filter": {
                "required": bool(modifier),
                "qualified_query": qualified_query,
                "unrestricted_query": unrestricted_query,
                "contrast_query": contrast_query,
                "prompts": dict(attribute_prompts or {}),
                "formula": ("0.80 * (qualified - unrestricted) + 0.20 * (qualified - contrast)"),
                "unrestricted_weight": self.unrestricted_weight,
                "contrast_weight": self.contrast_weight,
                "margin_threshold_strictly_greater_than": (self.qualified_margin_threshold),
                "qualified_detection_count": len(qualified_rows),
            },
            "combined_ranking": {
                "formula": (
                    f"{self.coarse_score_weight:.2f} * coarse_margin + "
                    f"{self.attribute_score_weight:.2f} * mixed_attribute_score"
                ),
                "coarse_score_weight": self.coarse_score_weight,
                "attribute_score_weight": self.attribute_score_weight,
                "selection_rule": (
                    "coarse_margin > coarse threshold and mixed_attribute_score > "
                    "qualified threshold"
                ),
            },
            "score_definition": (
                "cosine similarity of normalized GeoRSCLIP image and normalized text "
                "prototype embeddings; scores and margins are not probabilities"
            ),
            "candidates": candidates,
            "unqualified_results": unqualified_rows,
            "qualified_results": qualified_rows,
        }
