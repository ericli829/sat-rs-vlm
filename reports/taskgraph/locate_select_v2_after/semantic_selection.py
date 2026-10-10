"""Bounded, serial finite decisions over an existing candidate pool."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .input_composer import InputComposer
from .providers import FiniteDecisionRequest, SemanticVLMProvider
from .runtime_types import EntitySet, Region, RegionSet, RuntimeObject, SelectStatus


@dataclass(frozen=True)
class SemanticSelectionConfig:
    batch_size: int = 8
    max_candidates: int = 64
    max_pixels: int = 409600

    def __post_init__(self) -> None:
        if not 1 <= self.batch_size <= 8 or self.max_candidates < 1 or self.max_pixels < 131072:
            raise ValueError(
                "semantic_select requires batch_size=1..8, max_candidates>0, max_pixels>=131072"
            )

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> SemanticSelectionConfig:
        return cls(**(raw or {}))


@dataclass(frozen=True)
class CandidateSelection:
    indices: tuple[int, ...]
    status: SelectStatus
    metadata: dict[str, Any]


class BoundedSemanticSelector:
    def __init__(
        self, provider: SemanticVLMProvider, config: SemanticSelectionConfig | None = None
    ) -> None:
        self.provider = provider
        self.config = config or SemanticSelectionConfig()

    def select(
        self,
        candidates: EntitySet | Region | RegionSet,
        *,
        query: str,
        cardinality: str,
        composer: InputComposer,
        reference: RuntimeObject | None = None,
        scope: RuntimeObject | None = None,
        purpose: str = "select_semantic",
    ) -> CandidateSelection:
        items = (
            candidates.entities
            if isinstance(candidates, EntitySet)
            else candidates.regions
            if isinstance(candidates, RegionSet)
            else (candidates,)
        )
        all_ids = [str(item.provenance["candidate_id"]) for item in items]
        selected: set[int] = set()
        uncertain: set[int] = set()
        ambiguous = False
        batches: list[dict[str, Any]] = []
        provider_pixels = getattr(getattr(self.provider, "model_config", None), "max_pixels", None)
        pixel_budget = (
            min(self.config.max_pixels, 2 * provider_pixels)
            if provider_pixels
            else self.config.max_pixels
        )
        for start in range(0, min(len(items), self.config.max_candidates), self.config.batch_size):
            end = min(start + self.config.batch_size, len(items), self.config.max_candidates)
            subset = (
                EntitySet(tuple(items[start:end]))
                if isinstance(candidates, EntitySet)
                else RegionSet(tuple(items[start:end]))
            )
            contract = (
                "Choose exactly one marked candidate only when it uniquely satisfies "
                "the complete query. Choose NONE if none matches, AMBIGUOUS if several match, "
                "UNRESOLVED if evidence is insufficient."
                if cardinality == "SINGLE"
                else "Evaluate every marked candidate against the complete query. "
                "For the candidate being verified, use MATCH, NO_MATCH, or UNCERTAIN. "
                "Empty sets are valid; uncertainty is not a negative."
            )
            prompt = (
                f"Task: SELECT_SEM_{cardinality}\nSelect ONLY from the existing marked candidates. "
                "Do not detect new objects, return new boxes, or replace candidates. "
                "Background landmarks "
                "in the query are context, not additional detection tasks.\n"
                f"Complete query (verbatim):\n{query}\n\n{contract}"
            )
            try:
                model_input = composer.compose_selection(
                    subset,
                    reference=reference,
                    scope=scope,
                    question=prompt,
                    pixel_budget=pixel_budget,
                )
            except Exception as exc:
                batches.append(
                    {"candidate_ids": all_ids[start:end], "error": f"{type(exc).__name__}: {exc}"}
                )
                uncertain.update(range(start, end))
                continue
            mapping = model_input.metadata.get("candidate_mapping", {})
            expected_ids = all_ids[start:end]
            record: dict[str, Any] = {
                "candidate_ids": expected_ids,
                "mapping": mapping,
                "visual_metadata": model_input.metadata,
                "decisions": [],
            }
            batches.append(record)
            labels = tuple(mapping)
            if (
                len(labels) != end - start
                or [mapping[label].get("candidate_id") for label in labels] != expected_ids
            ):
                record["error"] = "candidate_mapping_mismatch"
                uncertain.update(range(start, end))
                continue
            choices = (
                (*labels, "NONE", "AMBIGUOUS", "UNRESOLVED")
                if cardinality == "SINGLE"
                else ("MATCH", "NO_MATCH", "UNCERTAIN")
            )
            checks = (None,) if cardinality == "SINGLE" else labels
            for label in checks:
                instruction = (
                    "Reason about all marked candidates and the full query. "
                    "A finite decision follows."
                    if label is None
                    else f"Verify ONLY candidate {label} against the complete query. "
                    "Reason from the supplied context; a three-state finite decision follows."
                )
                suffix = (
                    "\n\nSELECT_SEM_SINGLE decision:"
                    if label is None
                    else f"\n\nCandidate {label} semantic verdict:"
                )
                try:
                    result = self.provider.reason_and_decide(
                        FiniteDecisionRequest(
                            model_input=model_input,
                            decision_mode="SINGLE",
                            candidate_ids=tuple(choices),
                            candidate_texts=tuple(choices),
                            single_decision_suffix=suffix,
                            multi_verify_template="{choice_id}: {option_text}",
                            purpose=purpose,
                            reasoning_instruction=instruction,
                        )
                    )
                    scores = result.scores
                    if (
                        not result.cache_reused
                        or result.decision_mode != "SINGLE"
                        or len(result.selected_ids) != 1
                        or set(scores) != set(choices)
                        or not all(math.isfinite(float(v)) for v in scores.values())
                        or result.selected_ids[0] not in choices
                    ):
                        raise ValueError("invalid_finite_semantic_output")
                    decision = result.selected_ids[0]
                    # A tied terminal score is evidence insufficiency, not arbitrary argmax.
                    ranked = sorted((float(v) for v in scores.values()), reverse=True)
                    if float(scores[decision]) < ranked[0] - 1e-8:
                        raise ValueError("selected_id_score_mismatch")
                    if ranked[0] - ranked[1] <= 1e-8:
                        decision = "UNRESOLVED" if label is None else "UNCERTAIN"
                    record["decisions"].append(
                        {
                            "label": label,
                            "decision": decision,
                            "scores": scores,
                            "provider": result.provider,
                            "cache_reused": result.cache_reused,
                            "reasoning_text": result.reasoning_text,
                            "latency_ms": result.latency_ms,
                            "purpose": purpose,
                            "suffix": suffix,
                        }
                    )
                    if label is None:
                        if decision in labels:
                            selected.add(start + labels.index(decision))
                        elif decision == "AMBIGUOUS":
                            ambiguous = True
                        elif decision == "UNRESOLVED":
                            uncertain.update(range(start, end))
                    else:
                        index = start + labels.index(label)
                        if decision == "MATCH":
                            selected.add(index)
                        elif decision == "UNCERTAIN":
                            uncertain.add(index)
                except Exception as exc:
                    record["decisions"].append(
                        {"label": label, "error": f"{type(exc).__name__}: {exc}"}
                    )
                    uncertain.update(
                        range(start, end) if label is None else (start + labels.index(label),)
                    )
        unseen = tuple(range(min(len(items), self.config.max_candidates), len(items)))
        uncertain.update(unseen)
        if uncertain:
            status = SelectStatus.UNRESOLVED
        elif ambiguous or (cardinality == "SINGLE" and len(selected) > 1):
            status = SelectStatus.AMBIGUOUS
        else:
            status = SelectStatus.OK if selected else SelectStatus.EMPTY
        return CandidateSelection(
            tuple(sorted(selected)),
            status,
            {
                "query": query,
                "selection_type": cardinality,
                "purpose": purpose,
                "semantic_selection_ancestor": purpose == "select_semantic",
                "all_candidate_ids": all_ids,
                "selected_candidate_ids": [all_ids[i] for i in sorted(selected)],
                "uncertain_candidate_ids": [all_ids[i] for i in sorted(uncertain)],
                "unseen_candidate_ids": [all_ids[i] for i in unseen],
                "batches": batches,
                "candidate_pool_complete": not uncertain and not ambiguous,
                "budget": {
                    "batch_size": self.config.batch_size,
                    "max_candidates": self.config.max_candidates,
                    "max_pixels": self.config.max_pixels,
                    "effective_max_pixels": pixel_budget,
                },
                "inference_serial": True,
            },
        )
