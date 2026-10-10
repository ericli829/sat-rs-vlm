from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from sat_rs_vlm.models.hf_vlm_engine import HuggingFaceVLMEngine
from sat_rs_vlm.taskgraph.choice_config import ChoiceSystemConfig

CHOICE_CONFIG = ChoiceSystemConfig()

torch = pytest.importorskip(
    "torch", reason="KV-cache engine tests require the optional model extra"
)


@dataclass
class FakeCache:
    length: int
    clone_counter: list[int] = field(default_factory=lambda: [0])

    def get_seq_length(self) -> int:
        return self.length

    def crop(self, max_length: int) -> None:
        self.length = min(self.length, max_length)

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
        prefix = "\n\nFinal choice: "
        if text.startswith(prefix) and text != prefix:
            return [20] + self.encode(text[len(prefix):], add_special_tokens=False)
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


REGION_LABELS = (
    "top_left", "top", "top_right", "left", "center", "right",
    "bottom_left", "bottom", "bottom_right",
)
REGION_SUFFIX = "\n\nQuestion: Where is the change?\nAnswer: "
REGION_CONTEXT_IDS = {
    "top_left": (30,), "top": (31,), "top_right": (32, 52),
    "left": (33,), "center": (34,), "right": (35,),
    "bottom_left": (36, 56, 57), "bottom": (37,), "bottom_right": (38, 58),
}


class ContextualRegionTokenizer:
    """Emulate a tokenizer that merges the suffix's trailing space into a label token."""

    eos_token_id = 63

    def __init__(self) -> None:
        self.encoded_texts: list[str] = []

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        self.encoded_texts.append(text)
        if text == REGION_SUFFIX:
            # The space is a standalone token when the suffix is encoded alone.
            return [20, 21]
        if text.startswith(REGION_SUFFIX):
            label = text[len(REGION_SUFFIX):]
            if label in REGION_CONTEXT_IDS:
                # Joint encoding absorbs that space into the first label token.
                return [20, *REGION_CONTEXT_IDS[label]]
        if text in REGION_LABELS:
            # Deliberately different from the whitespace-prefixed contextual IDs.
            return [10 + REGION_LABELS.index(text)]
        return [60]

    def decode(self, token_ids: list[int], **kwargs: Any) -> str:
        del kwargs
        return "".join(self.convert_ids_to_tokens(token) for token in token_ids)

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return {
            20: "Answer:", 21: "Ġ", 30: "Ġtop_left", 31: "Ġtop",
            32: "Ġtop", 33: "Ġleft", 34: "Ġcenter", 35: "Ġright",
            36: "Ġbottom", 37: "Ġbottom", 38: "Ġbottom",
            52: "_right", 56: "_left", 57: "_tail", 58: "_right",
            63: "<eos>",
        }.get(int(token_id), str(token_id))


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
        self.prompt_prefill_calls = 0
        self.cache_clone_counter = [0]
        self.region_choice_token_id: int | None = None
        self.region_token_successors = dict(zip(
            (tokens[0] for tokens in REGION_CONTEXT_IDS.values()),
            (tokens[1] if len(tokens) > 1 else 63 for tokens in REGION_CONTEXT_IDS.values()),
            strict=True,
        ))
        self.region_mode = False

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
        position_ids = kwargs.pop("position_ids")
        del kwargs
        assert position_ids.shape == (3, input_ids.shape[0], next_sequence_length)
        return {
            "input_ids": input_ids[:, -next_sequence_length:],
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "use_cache": True,
        }

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        if kwargs.get("past_key_values") is None:
            self.prompt_prefill_calls += 1
        else:
            self.continuation_calls += 1
            assert "pixel_values" not in kwargs
        input_ids = kwargs["input_ids"]
        last_token = int(input_ids[0, -1])
        logits = torch.full((1, input_ids.shape[-1], 64), -10.0)
        if self.region_mode and last_token == 20:
            logits[0, -1, int(self.region_choice_token_id)] = 10.0
        elif self.region_mode:
            logits[0, -1, self.region_token_successors.get(last_token, 63)] = 10.0
        elif last_token == 20:
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
        cache = kwargs.get("past_key_values")
        if cache is None:
            cache = FakeCache(0, self.cache_clone_counter)
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


def test_original_collator_tensors_bypass_reprocessing_and_are_moved_before_cached_generation(monkeypatch) -> None:
    import sat_rs_vlm.models.hf_vlm_engine as module

    engine = _engine()
    engine._model.get_input_embeddings = lambda: SimpleNamespace(weight=torch.empty(1))
    engine._processor.tokenizer.eos_token_id = 63
    encoded = dict(CacheProcessor().apply_chat_template([{}], return_tensors="pt"))
    moved = []
    original_move = module.move_to_device
    def track_move(value, device, torch_module):
        moved.append((value, device))
        return original_move(value, device, torch_module)
    monkeypatch.setattr(module, "move_to_device", track_move)
    def forbidden_template(*args, **kwargs):
        raise AssertionError("Original collator inputs must not be rebuilt")
    monkeypatch.setattr(engine._processor, "apply_chat_template", forbidden_template)
    generation = {}
    original_generate = engine._model.generate
    def track_generate(**kwargs):
        generation.update(kwargs)
        return original_generate(**kwargs)
    monkeypatch.setattr(engine._model, "generate", track_generate)
    reasoning = engine.reason_with_cache("original prompt", [], max_new_tokens=128,
                                         encoded_inputs=encoded)
    assert moved[0][0] is encoded
    for key, value in encoded.items():
        assert torch.equal(generation[key], value)
    assert generation["max_new_tokens"] == 128
    assert generation["do_sample"] is False
    result = engine.decode_from_cache(reasoning.session, suffix="\n\nFinal choice: ",
                                      allowed_outputs=("A", "B"))
    assert result.text == "B"
    assert engine._model.visual_prefill_calls == 1
    assert engine._model.cache_clone_counter[0] == 0
    assert engine.active_session_count == 0


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


def test_cached_suffix_uses_query_length_qwen_mrope_positions() -> None:
    engine = _engine()
    reasoning = engine.reason_with_cache("reason", [])
    try:
        reasoning.session._rope_deltas = torch.tensor([[-2]])
        attention_mask = torch.ones((1, 7), dtype=torch.long)

        position_ids = engine._cached_position_ids(reasoning.session, attention_mask, 2)

        assert position_ids.shape == (3, 1, 2)
        assert position_ids[:, 0, :].tolist() == [[3, 4], [3, 4], [3, 4]]
    finally:
        reasoning.session.close()


def test_peft_wrapped_qwen_rope_state_is_captured_and_restored() -> None:
    engine = _engine()
    holder = SimpleNamespace(rope_deltas=torch.tensor([[-7]]))
    engine._model.model = SimpleNamespace()
    engine._model.base_model = SimpleNamespace(model=SimpleNamespace(model=holder))

    reasoning = engine.reason_with_cache("reason", [])
    try:
        assert reasoning.session._rope_deltas.tolist() == [[-7]]

        holder.rope_deltas = torch.tensor([[99]])
        engine._restore_rope_state(reasoning.session)

        assert holder.rope_deltas.tolist() == [[-7]]
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


def test_constrained_decode_retains_visual_cache_without_reencoding_or_cloning() -> None:
    engine = _engine(multi_token_labels=True)
    engine._processor.tokenizer.eos_token_id = 63
    reasoning = engine.reason_with_cache("describe the change", ["before.png", "after.png"])
    # Deliberately omit a direction from the free description. The label stage
    # still uses the same multimodal session, not a description-only prefill.
    reasoning.session.reasoning_text = "A building appeared."
    result = engine.decode_from_cache(
        reasoning.session, suffix="\n\nFinal choice: ", allowed_outputs=("A", "B")
    )
    assert result.text == "B"
    assert result.generated_tokens == (2, 4)
    assert result.cache_reused is True
    assert result.metadata["visual_reprefill"] is False
    assert result.metadata["reasoning_cache_mode"] == "consume_in_place"
    assert engine._model.generate_calls == 1
    assert engine._model.visual_prefill_calls == 1
    assert engine._model.continuation_calls == 3
    assert engine._model.cache_clone_counter[0] == 0
    assert result.latency_ms["cache_clone_ms"] >= 0
    assert reasoning.session.closed
    assert engine.active_session_count == 0


def test_constrained_decode_can_extend_a_label_that_is_a_prefix_of_another() -> None:
    engine = _engine()
    tokenizer = engine._processor.tokenizer
    tokenizer.eos_token_id = 63
    encode = tokenizer.encode
    tokenizer.encode = lambda text, **kw: [1, 3] if text == "AB" else encode(text, **kw)
    reasoning = engine.reason_with_cache("reason", ["image.png"])
    result = engine.decode_from_cache(
        reasoning.session, suffix="\n\nFinal choice: ", allowed_outputs=("A", "AB")
    )
    assert result.text == "AB"
    assert result.generated_tokens == (1, 3)
    assert reasoning.session.closed


@pytest.mark.parametrize("target_label", REGION_LABELS)
def test_region_labels_use_joint_context_tokenization_and_keep_raw_eos_scores(target_label: str) -> None:
    engine = _engine()
    tokenizer = ContextualRegionTokenizer()
    engine._processor.tokenizer = tokenizer
    engine._processor.tokenizer.eos_token_id = 63
    engine._model.region_mode = True
    engine._model.region_choice_token_id = REGION_CONTEXT_IDS[target_label][0]
    reasoning = engine.reason_with_cache("describe", ["before.png", "after.png"])

    result = engine.decode_from_cache(
        reasoning.session,
        suffix=REGION_SUFFIX,
        allowed_outputs=REGION_LABELS,
        record_probabilities=True,
        score_candidates=True,
    )

    assert result.text == target_label
    assert result.generated_tokens == REGION_CONTEXT_IDS[target_label]
    diagnostics = result.metadata["probability_diagnostics"]
    assert diagnostics["candidate_tokenization_method"].startswith("tokenize_complete_suffix_plus_output")
    assert [token["token_id"] for token in diagnostics["shared_context_prefix_tokens"]] == [20]
    assert [token["token_id"] for token in diagnostics["separately_tokenized_suffix_tokens"]] == [20, 21]
    assert set(diagnostics["candidate_tokenizations"]) == set(REGION_LABELS)
    assert set(diagnostics["candidate_scores"]) == set(REGION_LABELS)
    for label in REGION_LABELS:
        tokenization = diagnostics["candidate_tokenizations"][label]
        assert [token["token_id"] for token in tokenization["full_context_tokens"]] == [
            20, *REGION_CONTEXT_IDS[label]
        ]
        assert [token["token_id"] for token in tokenization["continuation_tokens_after_shared_context"]] == list(
            REGION_CONTEXT_IDS[label]
        )
        score = diagnostics["candidate_scores"][label]
        assert score["token_ids"] == list(REGION_CONTEXT_IDS[label])
        assert score["label_log_probability"] == pytest.approx(sum(score["token_log_probabilities"]))
        assert score["terminated_log_probability"] == pytest.approx(
            score["label_log_probability"] + score["eos_log_probability"]
        )
    assert sum(
        score["normalized_terminated_probability"]
        for score in diagnostics["candidate_scores"].values()
    ) == pytest.approx(1)
    assert engine._model.cache_clone_counter[0] == 0
    assert engine._model.visual_prefill_calls == 1
    assert reasoning.session.closed


def test_prompt_only_prefill_starts_direct_region_decode_without_free_generation() -> None:
    engine = _engine()
    engine._processor.tokenizer = ContextualRegionTokenizer()
    engine._processor.tokenizer.eos_token_id = 63
    engine._model.region_mode = True
    engine._model.region_choice_token_id = REGION_CONTEXT_IDS["center"][0]

    prefill = engine.prefill_with_cache("classify the description directly", [])
    assert prefill.reasoning_text == ""
    assert prefill.session.reasoning_tokens == 0
    assert torch.equal(prefill.session._rope_deltas, torch.zeros((1, 1), dtype=torch.long))
    assert engine._model.generate_calls == 0
    assert engine._model.prompt_prefill_calls == 1
    result = engine.decode_from_cache(prefill.session, suffix=REGION_SUFFIX, allowed_outputs=REGION_LABELS)
    assert result.text == "center"
    assert result.generated_tokens == REGION_CONTEXT_IDS["center"]
    assert engine.active_session_count == 0


def test_constrained_decode_masks_illegal_high_logits_and_premature_eos(monkeypatch) -> None:
    engine = _engine(multi_token_labels=True)
    engine._processor.tokenizer.eos_token_id = 63
    original_forward = CacheModel.__call__
    def high_illegal_logits(model, **kwargs):
        result = original_forward(model, **kwargs)
        if int(kwargs["input_ids"][0, -1]) == 20:
            result.logits[0, -1, 60] = 1000.0
            result.logits[0, -1, 63] = 2000.0
        return result
    monkeypatch.setattr(CacheModel, "__call__", high_illegal_logits)
    reasoning = engine.reason_with_cache("reason", [])
    result = engine.decode_from_cache(
        reasoning.session, suffix="\n\nFinal choice: ", allowed_outputs=("A", "B"), record_probabilities=True
    )
    assert result.text == "B"
    assert reasoning.session.closed
    first = result.metadata["probability_diagnostics"]["steps"][0]
    assert first["permitted_probability_mass"] == 0
    assert first["permitted_log_probability_mass"] < -1000
    assert sum(item["normalized_probability"] for item in first["alternatives"]) == pytest.approx(1)


def test_constrained_decode_releases_session_when_cache_continuation_fails(monkeypatch) -> None:
    engine = _engine()
    engine._processor.tokenizer.eos_token_id = 63
    reasoning = engine.reason_with_cache("reason", [])
    def failing_forward(*args, **kwargs):
        raise RuntimeError("continuation failed")
    monkeypatch.setattr(CacheModel, "__call__", failing_forward)
    with pytest.raises(RuntimeError, match="continuation failed"):
        engine.decode_from_cache(reasoning.session, suffix="\n\nFinal choice: ", allowed_outputs=("A", "B"))
    assert reasoning.session.closed
    assert engine.active_session_count == 0


def test_probability_diagnostics_are_normalized_and_do_not_change_greedy_decode():
    engine = _engine(multi_token_labels=True)
    engine._processor.tokenizer.eos_token_id = 63
    reasoning = engine.reason_with_cache("reason", ["before.png", "after.png"])
    result = engine.decode_from_cache(reasoning.session, suffix="\n\nFinal choice: ",
                                     allowed_outputs=("A", "B"),
                                     record_probabilities=True, score_candidates=True)
    diagnostic = result.metadata["probability_diagnostics"]
    assert result.text == "B"
    assert diagnostic["candidate_cache_mode"] == "crop_to_suffix_prefix_between_candidates"
    for step in diagnostic["steps"]:
        assert sum(a["normalized_probability"] for a in step["alternatives"]) == pytest.approx(1)
        assert sum(a["raw_probability"] for a in step["alternatives"]) == pytest.approx(step["permitted_probability_mass"])
        best = max(step["alternatives"], key=lambda a: a["normalized_probability"])
        assert step["selected_token_id"] == best["token_id"]
    scores = diagnostic["candidate_scores"]
    for field in ["normalized_terminated_probability", "normalized_label_probability", "normalized_mean_token_score"]:
        assert sum(score[field] for score in scores.values()) == pytest.approx(1)
    assert scores["B"]["token_ids"] == [2, 4]
    assert scores["B"]["label_log_probability"] == pytest.approx(sum(scores["B"]["token_log_probabilities"]))
    assert scores["B"]["terminated_log_probability"] == pytest.approx(scores["B"]["label_log_probability"] + scores["B"]["eos_log_probability"])
    assert engine._model.visual_prefill_calls == 1
    assert engine._model.cache_clone_counter[0] == 0
    assert engine.active_session_count == 0
    json.dumps(diagnostic, allow_nan=False)


def test_probability_trace_does_not_duplicate_shared_prefix_mass():
    engine = _engine()
    tokenizer = engine._processor.tokenizer
    tokenizer.eos_token_id = 63
    encode = tokenizer.encode
    tokenizer.encode = lambda text, **kw: [1, 3] if text == "AB" else encode(text, **kw)
    reasoning = engine.reason_with_cache("reason", [])
    result = engine.decode_from_cache(reasoning.session, suffix="\n\nFinal choice: ",
                                     allowed_outputs=("A", "AB"), record_probabilities=True,
                                     score_candidates=True)
    first = result.metadata["probability_diagnostics"]["steps"][0]
    assert len(first["alternatives"]) == 1
    assert first["alternatives"][0]["labels"] == ["A", "AB"]
    assert first["alternatives"][0]["normalized_probability"] == 1
    assert result.text == "AB"
