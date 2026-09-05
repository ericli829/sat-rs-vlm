"""Greedy token filtering for canonical TaskGraph Planner DSL generation."""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from taskgraph_lab.taskgraph.dsl.constraint import CanonicalDSLPrefixGrammar, PrefixAnalysis


class ConstraintRecoveryState(StrEnum):
    NORMAL = "normal"
    FINISH_CURRENT_NODE = "finish_current_node"
    FORCED_FINAL = "forced_final"
    DONE = "done"
    ABORTED = "aborted"


@dataclass
class ConstraintStats:
    examined_candidates: int = 0
    rejected_candidates: int = 0
    repeat_guard_rejections: int = 0
    grammar_dead_ends: int = 0
    failure_reason: str | None = None
    recovery_state: ConstraintRecoveryState = ConstraintRecoveryState.NORMAL
    recovery_transitions: tuple[str, ...] = (ConstraintRecoveryState.NORMAL.value,)
    finish_node_tokens: int = 0
    forced_final_tokens: int = 0
    repeat_guard_triggered: bool = False
    candidate_limit_hits: int = 0
    handler_calls: int = 0
    handler_total_ms: float = 0.0
    handler_max_ms: float = 0.0


class GreedyDSLLogitsProcessor:
    """Select the highest-logit token that preserves a valid DSL prefix.

    This is exact for greedy decoding: candidates are inspected in descending
    logit order, and the first grammar-valid candidate is forced.  It avoids a
    full-vocabulary Python scan in the common case where the model's top token
    is already legal.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        prompt_width: int,
        image_refs_by_row: Iterable[Iterable[str]],
        initial_top_k: int = 64,
        max_candidate_checks: int = 512,
        max_nodes: int | None = 24,
        repeat_guard_repetitions: int | None = 4,
        max_finish_node_tokens: int = 32,
    ) -> None:
        if prompt_width < 1:
            raise ValueError("prompt_width must be positive")
        if initial_top_k < 1:
            raise ValueError("initial_top_k must be positive")
        if max_candidate_checks < initial_top_k:
            raise ValueError("max_candidate_checks must be >= initial_top_k")
        if max_finish_node_tokens < 1:
            raise ValueError("max_finish_node_tokens must be positive")
        eos = tokenizer.eos_token_id
        if eos is None:
            raise ValueError("tokenizer must define eos_token_id for constrained decoding")
        self.tokenizer = tokenizer
        self.prompt_width = int(prompt_width)
        self.initial_top_k = int(initial_top_k)
        self.max_candidate_checks = int(max_candidate_checks)
        self.max_finish_node_tokens = int(max_finish_node_tokens)
        self.eos_token_ids = {
            int(value) for value in (eos if isinstance(eos, (list, tuple)) else [eos])
        }
        self.special_token_ids = {int(value) for value in tokenizer.all_special_ids}
        self.grammars = [
            CanonicalDSLPrefixGrammar(
                refs,
                max_nodes=max_nodes,
                repeat_guard_repetitions=repeat_guard_repetitions,
            )
            for refs in image_refs_by_row
        ]
        self.stats = [ConstraintStats() for _ in self.grammars]

    @staticmethod
    def _transition(stats: ConstraintStats, state: ConstraintRecoveryState) -> None:
        if stats.recovery_state is state:
            return
        stats.recovery_state = state
        stats.recovery_transitions = (*stats.recovery_transitions, state.value)

    def _decode(self, token_ids: Any) -> str:
        values = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
        return str(
            self.tokenizer.decode(
                values,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )

    def _analysis_with_candidate(
        self,
        row: int,
        generated_ids: Any,
        candidate_id: int,
        current_text: str,
    ) -> PrefixAnalysis | None:
        if candidate_id in self.special_token_ids:
            return None
        candidate_ids = [*generated_ids.tolist(), int(candidate_id)]
        candidate_text = self._decode(candidate_ids)
        if candidate_text == current_text:
            return None
        return self.grammars[row].analyze(candidate_text)

    def _choose(self, row: int, generated_ids: Any, row_scores: Any) -> int:
        current_text = self._decode(generated_ids)
        current = self.grammars[row].analyze(current_text)
        stats = self.stats[row]
        if current.valid_prefix and current.complete:
            stats.failure_reason = None
            self._transition(stats, ConstraintRecoveryState.DONE)
            return min(self.eos_token_ids)
        if not current.valid_prefix:
            stats.grammar_dead_ends += 1
            stats.failure_reason = "constraint_abort"
            self._transition(stats, ConstraintRecoveryState.ABORTED)
            return min(self.eos_token_ids)

        if current.force_final:
            stats.repeat_guard_triggered = True
            self._transition(stats, ConstraintRecoveryState.FORCED_FINAL)

        vocabulary = int(row_scores.shape[-1])
        checked: set[int] = set()
        candidate_limit = min(vocabulary, self.max_candidate_checks)
        top_k = min(self.initial_top_k, candidate_limit)
        while True:
            candidates = row_scores.topk(top_k).indices.tolist()
            for raw_candidate in candidates:
                candidate_id = int(raw_candidate)
                if candidate_id in checked:
                    continue
                checked.add(candidate_id)
                stats.examined_candidates += 1
                if candidate_id in self.eos_token_ids:
                    stats.rejected_candidates += 1
                    continue
                analysis = self._analysis_with_candidate(
                    row,
                    generated_ids,
                    candidate_id,
                    current_text,
                )
                if analysis is not None and analysis.valid_prefix:
                    stats.failure_reason = None
                    if analysis.force_final and not current.force_final:
                        stats.repeat_guard_triggered = True
                        self._transition(stats, ConstraintRecoveryState.FINISH_CURRENT_NODE)
                        stats.finish_node_tokens += 1
                        if stats.finish_node_tokens > self.max_finish_node_tokens:
                            stats.failure_reason = "constraint_abort"
                            self._transition(stats, ConstraintRecoveryState.ABORTED)
                            return min(self.eos_token_ids)
                        if analysis.current_node_complete:
                            self._transition(stats, ConstraintRecoveryState.FORCED_FINAL)
                    elif stats.recovery_state is ConstraintRecoveryState.FORCED_FINAL:
                        stats.forced_final_tokens += 1
                    return candidate_id
                stats.rejected_candidates += 1
                if analysis is not None and analysis.reason in {
                    "repeat_guard",
                    "forced_final_required",
                    "max_nodes",
                }:
                    stats.repeat_guard_rejections += 1
            if top_k >= candidate_limit:
                stats.grammar_dead_ends += 1
                stats.candidate_limit_hits += 1
                stats.failure_reason = "constraint_abort"
                self._transition(stats, ConstraintRecoveryState.ABORTED)
                return min(self.eos_token_ids)
            top_k = min(candidate_limit, top_k * 4)

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        if int(input_ids.shape[0]) != len(self.grammars):
            raise ValueError(
                "constrained decoder row count changed; beam search is unsupported and "
                "num_beams must remain 1"
            )
        for row in range(int(input_ids.shape[0])):
            generated_ids = input_ids[row, self.prompt_width :]
            started = time.perf_counter()
            chosen = self._choose(row, generated_ids, scores[row])
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            stats = self.stats[row]
            stats.handler_calls += 1
            stats.handler_total_ms += elapsed_ms
            stats.handler_max_ms = max(stats.handler_max_ms, elapsed_ms)
            chosen_score = scores[row, chosen].clone()
            scores[row].fill_(float("-inf"))
            scores[row, chosen] = chosen_score
        return scores

    def diagnostics(
        self,
        row: int,
        continuation_ids: Any,
        *,
        max_new_tokens: int,
        pad_token_id: int,
    ) -> dict[str, Any]:
        ids = continuation_ids.tolist()
        semantic_ids = [
            int(value)
            for value in ids
            if int(value) != int(pad_token_id) and int(value) not in self.eos_token_ids
        ]
        text = self._decode(semantic_ids)
        analysis = self.grammars[row].analyze(text)
        stats = self.stats[row]
        if analysis.valid_prefix and analysis.complete:
            termination_reason = (
                "repeat_guard_forced_final"
                if stats.repeat_guard_triggered
                else "final"
            )
            failure = None
        elif stats.failure_reason is not None:
            termination_reason = stats.failure_reason
            failure = stats.failure_reason
        elif len(semantic_ids) >= max_new_tokens:
            termination_reason = "max_tokens"
            failure = "max_tokens"
        else:
            termination_reason = "error"
            failure = analysis.reason or "incomplete_program"
        return {
            "termination_reason": termination_reason,
            "constraint_failure": failure,
            "constraint_examined_candidates": stats.examined_candidates,
            "constraint_rejected_candidates": stats.rejected_candidates,
            "repeat_guard_rejections": stats.repeat_guard_rejections,
            "grammar_dead_end_count": stats.grammar_dead_ends,
            "constraint_recovery_state": stats.recovery_state.value,
            "constraint_recovery_transitions": list(stats.recovery_transitions),
            "constraint_finish_node_tokens": stats.finish_node_tokens,
            "constraint_forced_final_tokens": stats.forced_final_tokens,
            "constraint_candidate_limit_hits": stats.candidate_limit_hits,
            "constraint_handler_calls": stats.handler_calls,
            "constraint_handler_total_ms": stats.handler_total_ms,
            "constraint_handler_mean_ms": (
                stats.handler_total_ms / stats.handler_calls if stats.handler_calls else 0.0
            ),
            "constraint_handler_max_ms": stats.handler_max_ms,
        }
