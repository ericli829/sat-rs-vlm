from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from sat_rs_vlm.models.hf_vlm_engine import HuggingFaceVLMEngine
from sat_rs_vlm.taskgraph.choice_config import ChoiceSystemConfig

CHOICE_CONFIG = ChoiceSystemConfig()


@dataclass
class FakeCache:
    length: int
    clone_counter: list[int] = field(default_factory=lambda: [0])

    def get_seq_length(self) -> int:
        return self.length

    def __deepcopy__(self, memo: dict[int, object]) -> FakeCache:
        del memo
        self.clone_counter[0] += 1
        return FakeCache(self.length, self.clone_counter)


class CacheTokenizer:
    def __init__(
        self,
        *,
        multi_token_labels: bool = False,
        multi_token_binary: bool = False,
    ) -> None:
        self.multi_token_labels = multi_token_labels
        self.multi_token_binary = multi_token_binary

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        if "Candidate option A" in text:
            return [31]
        if "Candidate option B" in text:
            return [32]
        if "Candidate option C" in text:
            return [33]
        mapping = {
            "\n\nFinal choice:": [20],
            "\n\nFinal choice: ": [20],
            " A": [1, 3] if self.multi_token_labels else [1],
            " B": [2, 4] if self.multi_token_labels else [2],
            " C": [5],
            "A": [1, 3] if self.multi_token_labels else [1],
            "B": [2, 4] if self.multi_token_labels else [2],
            "C": [5],
            " YES": [10, 12] if self.multi_token_binary else [10],
            " NO": [11, 13] if self.multi_token_binary else [11],
        }
        return mapping[text]


class CacheBatch(dict[str, torch.Tensor]):
    def to(self, device: object) -> CacheBatch:
        for key, value in self.items():
            self[key] = value.to(device)
        return self


class CacheProcessor:
    def __init__(
        self,
        *,
        multi_token_labels: bool = False,
        multi_token_binary: bool = False,
    ) -> None:
        self.tokenizer = CacheTokenizer(
            multi_token_labels=multi_token_labels,
            multi_token_binary=multi_token_binary,
        )

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> CacheBatch:
        assert messages
        assert kwargs["return_tensors"] == "pt"
        return CacheBatch(
            input_ids=torch.tensor([[40, 41]]),
            attention_mask=torch.ones((1, 2), dtype=torch.long),
            pixel_values=torch.ones((1, 1)),
        )

    def batch_decode(self, token_ids: torch.Tensor, **kwargs: Any) -> list[str]:
        assert token_ids.shape[-1] in {1, 2}
        assert kwargs["skip_special_tokens"] is True
        return ["A looks possible, B is weaker, therefore C may be correct."]


class CacheModel:
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.model = SimpleNamespace(rope_deltas=torch.tensor([[0]]))
        self.generate_calls = 0
        self.visual_prefill_calls = 0
        self.continuation_calls = 0
        self.cache_clone_counter = [0]

    def generate(self, **kwargs: Any) -> SimpleNamespace:
        self.generate_calls += 1
        if kwargs.get("pixel_values") is not None:
            self.visual_prefill_calls += 1
        input_ids = kwargs["input_ids"]
        sequences = torch.cat([input_ids, torch.tensor([[50, 51]])], dim=-1)
        return SimpleNamespace(
            sequences=sequences,
            past_key_values=FakeCache(3, self.cache_clone_counter),
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        *,
        next_sequence_length: int,
        past_key_values: FakeCache,
        attention_mask: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        return {
            "input_ids": input_ids[:, -next_sequence_length:],
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": True,
        }

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        self.continuation_calls += 1
        assert "pixel_values" not in kwargs
        input_ids = kwargs["input_ids"]
        last_token = int(input_ids[0, -1])
        logits = torch.full((1, input_ids.shape[-1], 64), -10.0)
        if last_token == 20:
            logits[0, -1, 1] = 1.0
            logits[0, -1, 2] = 5.0
        elif last_token == 1:
            logits[0, -1, 3] = -3.0
        elif last_token == 2:
            logits[0, -1, 4] = 6.0
        elif last_token == 31:
            logits[0, -1, 10] = 5.0
            logits[0, -1, 11] = 1.0
        elif last_token == 32:
            logits[0, -1, 10] = 1.0
            logits[0, -1, 11] = 5.0
        elif last_token == 33:
            logits[0, -1, 10] = 4.0
            logits[0, -1, 11] = 1.0
        elif last_token == 10:
            logits[0, -1, 12] = 4.0
        elif last_token == 11:
            logits[0, -1, 13] = 4.0
        cache = kwargs["past_key_values"]
        return SimpleNamespace(
            logits=logits,
            past_key_values=FakeCache(
                cache.length + input_ids.shape[-1],
                cache.clone_counter,
            ),
        )


def _engine(
    *,
    multi_token_labels: bool = False,
    multi_token_binary: bool = False,
) -> HuggingFaceVLMEngine:
    engine = object.__new__(HuggingFaceVLMEngine)
    engine._torch = torch
    engine._processor = CacheProcessor(
        multi_token_labels=multi_token_labels,
        multi_token_binary=multi_token_binary,
    )
    engine._model = CacheModel()
    engine._image_module = SimpleNamespace()
    engine.model_id = "fake-qwen"
    engine.device = "cpu"
    engine.dtype = "float32"
    engine.max_new_tokens = 8
    engine._model_class_name = "CacheModel"
    engine._model_identity = f"fake-qwen:CacheModel:{id(engine._model)}"
    engine._active_sessions = {}
    engine._open_image = lambda path: object()
    return engine


def test_reasoning_session_is_reused_without_second_visual_prefill_and_released() -> None:
    engine = _engine()
    reasoning = engine.reason_with_cache("reason", ["image.png"])

    assert engine.active_session_count == 1
    result = engine.score_choice_from_cache(
        reasoning.session,
        single_choice_suffix=CHOICE_CONFIG.single_choice_suffix,
        multi_verify_template=CHOICE_CONFIG.multi_verify_template,
        choice_ids=("A", "B"),
        option_texts=("A first", "B second"),
        answer_type="CHOICE_SINGLE",
    )

    assert result.selected_ids == ("B",)
    assert result.method == "kv_cached_single_token_logits"
    assert result.cache_reused is True
    assert result.metadata["initial_prefill_tokens"] == 2
    assert result.metadata["reasoning_tokens"] == 2
    assert engine._model.generate_calls == 1
    assert engine._model.visual_prefill_calls == 1
    assert engine._model.cache_clone_counter[0] == 0
    assert result.metadata["reasoning_cache_mode"] == "consume_in_place"
    assert result.latency_ms["choice_total_ms"] >= 0.0
    assert result.latency_ms["cache_clone_ms"] >= 0.0
    assert result.latency_ms["suffix_tokenize_ms"] >= 0.0
    assert result.latency_ms["choice_total_ms"] >= result.latency_ms["cache_clone_ms"]
    reasoning.session.close()
    assert engine.active_session_count == 0
    with pytest.raises(RuntimeError, match="closed"):
        engine.score_choice_from_cache(
            reasoning.session,
            single_choice_suffix=CHOICE_CONFIG.single_choice_suffix,
            multi_verify_template=CHOICE_CONFIG.multi_verify_template,
            choice_ids=("A", "B"),
            option_texts=("A", "B"),
            answer_type="CHOICE_SINGLE",
        )


def test_cache_cannot_cross_model_identity() -> None:
    first = _engine()
    second = _engine()
    reasoning = first.reason_with_cache("reason", [])
    try:
        with pytest.raises(ValueError, match="different model"):
            second.score_choice_from_cache(
                reasoning.session,
                single_choice_suffix=CHOICE_CONFIG.single_choice_suffix,
                multi_verify_template=CHOICE_CONFIG.multi_verify_template,
                choice_ids=("A", "B"),
                option_texts=("A", "B"),
                answer_type="CHOICE_SINGLE",
            )
    finally:
        reasoning.session.close()


def test_terminal_eos_is_removed_before_cached_suffix() -> None:
    engine = _engine()
    engine._model.generation_config = SimpleNamespace(eos_token_id=51)
    reasoning = engine.reason_with_cache("reason", [])
    try:
        assert reasoning.session.reasoning_tokens == 1
        assert reasoning.session._sequence_ids.shape[-1] == 3
        result = engine.score_choice_from_cache(
            reasoning.session,
            single_choice_suffix=CHOICE_CONFIG.single_choice_suffix,
            multi_verify_template=CHOICE_CONFIG.multi_verify_template,
            choice_ids=("A", "B"),
            option_texts=("A", "B"),
            answer_type="CHOICE_SINGLE",
        )
        assert result.selected_ids == ("B",)
    finally:
        reasoning.session.close()


def test_multi_token_choice_label_uses_full_continuation_probability() -> None:
    engine = _engine(multi_token_labels=True)
    result = engine.reason_and_choose(
        "reason",
        [],
        choice_ids=("A", "B"),
        option_texts=("A first", "B second"),
        answer_type="CHOICE_SINGLE",
        single_choice_suffix=CHOICE_CONFIG.single_choice_suffix,
        multi_verify_template=CHOICE_CONFIG.multi_verify_template,
    )

    assert result.selected_ids == ("B",)
    assert result.method == "kv_cached_multi_token_continuation_logprob"
    assert result.metadata["choice_scored_tokens"] == 4
    assert result.metadata["session_released"] is True
    assert engine.active_session_count == 0


def test_multi_choice_independently_scores_yes_no_from_one_reasoning_prefix() -> None:
    engine = _engine()
    result = engine.reason_and_choose(
        "reason",
        ["image.png"],
        choice_ids=("A", "B", "C"),
        option_texts=("alpha", "beta", "gamma"),
        answer_type="CHOICE_MULTI",
        single_choice_suffix=CHOICE_CONFIG.single_choice_suffix,
        multi_verify_template=CHOICE_CONFIG.multi_verify_template,
        multi_select_threshold=0.0,
    )

    assert result.selected_ids == ("A", "C")
    assert result.scores == {"A": 4.0, "B": -4.0, "C": 3.0}
    assert result.method == "kv_cached_binary_verification"
    assert engine._model.generate_calls == 1
    assert engine._model.visual_prefill_calls == 1
    assert result.metadata["choice_suffix_tokens"] == 3
    assert result.metadata["reasoning_cache_mode"] == "fork_per_option"
    assert engine._model.cache_clone_counter[0] == 3
    assert result.latency_ms["cache_clone_ms"] >= 0.0
    assert result.latency_ms["choice_total_ms"] >= result.latency_ms["cache_clone_ms"]
    assert engine.active_session_count == 0


def test_repeated_high_level_calls_do_not_accumulate_sessions() -> None:
    engine = _engine()
    for _ in range(10):
        result = engine.reason_and_choose(
            "reason",
            [],
            choice_ids=("A", "B"),
            option_texts=("alpha", "beta"),
            answer_type="CHOICE_SINGLE",
            single_choice_suffix=CHOICE_CONFIG.single_choice_suffix,
            multi_verify_template=CHOICE_CONFIG.multi_verify_template,
        )
        assert result.cache_reused is True
        assert engine.active_session_count == 0


def test_choice_label_boundary_handles_suffix_with_or_without_whitespace() -> None:
    without_space = _engine().reason_and_choose(
        "reason",
        [],
        choice_ids=("A", "B"),
        option_texts=("alpha", "beta"),
        answer_type="CHOICE_SINGLE",
        single_choice_suffix="\n\nFinal choice:",
        multi_verify_template=CHOICE_CONFIG.multi_verify_template,
    )
    with_space = _engine().reason_and_choose(
        "reason",
        [],
        choice_ids=("A", "B"),
        option_texts=("alpha", "beta"),
        answer_type="CHOICE_SINGLE",
        single_choice_suffix="\n\nFinal choice: ",
        multi_verify_template=CHOICE_CONFIG.multi_verify_template,
    )

    assert without_space.selected_ids == with_space.selected_ids == ("B",)
    assert without_space.method == with_space.method == "kv_cached_single_token_logits"


def test_multi_choice_yes_no_uses_multi_token_continuation_when_required() -> None:
    engine = _engine(multi_token_binary=True)
    result = engine.reason_and_choose(
        "reason",
        [],
        choice_ids=("A", "B"),
        option_texts=("alpha", "beta"),
        answer_type="CHOICE_MULTI",
        single_choice_suffix=CHOICE_CONFIG.single_choice_suffix,
        multi_verify_template=CHOICE_CONFIG.multi_verify_template,
    )

    assert result.selected_ids == ("A",)
    assert result.metadata["continuation_methods"] == ["multi_token_continuation_logprob"]
    assert result.metadata["choice_scored_tokens"] == 8
    assert engine.active_session_count == 0


def test_cached_choice_outputs_only_json_serializable_metadata() -> None:
    engine = _engine()
    result = engine.reason_and_choose(
        "reason",
        [],
        choice_ids=("A", "B"),
        option_texts=("alpha", "beta"),
        answer_type="CHOICE_SINGLE",
        single_choice_suffix=CHOICE_CONFIG.single_choice_suffix,
        multi_verify_template=CHOICE_CONFIG.multi_verify_template,
    )

    json.dumps({"latency_ms": result.latency_ms, "metadata": result.metadata})
    assert engine.active_session_count == 0
