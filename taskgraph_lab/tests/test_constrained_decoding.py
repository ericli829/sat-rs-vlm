from __future__ import annotations

import time

import torch

from taskgraph_lab.evaluation.constrained_decoding import GreedyDSLLogitsProcessor


class _CharacterTokenizer:
    eos_token_id = 0
    all_special_ids = [0]

    def __init__(self, vocabulary: str) -> None:
        self.tokens = ["", *vocabulary]
        self.ids = {token: index for index, token in enumerate(self.tokens)}

    def decode(self, values: list[int], **_: object) -> str:
        return "".join(self.tokens[value] for value in values)


def _processor(tokenizer: _CharacterTokenizer) -> GreedyDSLLogitsProcessor:
    return GreedyDSLLogitsProcessor(
        tokenizer,
        prompt_width=1,
        image_refs_by_row=[["image0"]],
        initial_top_k=2,
    )


def test_highest_invalid_token_is_rejected_for_next_valid_token() -> None:
    tokenizer = _CharacterTokenizer("XI")
    processor = _processor(tokenizer)
    input_ids = torch.tensor([[tokenizer.ids["X"]]])  # one-token dummy prompt
    scores = torch.tensor([[0.0, 10.0, 9.0]])

    constrained = processor(input_ids, scores)

    assert constrained[0, tokenizer.ids["I"]].item() == 9.0
    assert torch.isneginf(constrained[0, tokenizer.ids["X"]])
    assert processor.stats[0].rejected_candidates == 1


def test_complete_final_forces_eos_and_prevents_trailing_node() -> None:
    program = 'n1=COUNT_IMAGE($image0,T("ship"),true)\nFINAL($n1,INTEGER)'
    vocabulary = "".join(dict.fromkeys(program + "X"))
    tokenizer = _CharacterTokenizer(vocabulary)
    processor = _processor(tokenizer)
    generated = [tokenizer.ids[character] for character in program]
    input_ids = torch.tensor([[tokenizer.ids["X"], *generated]])
    scores = torch.zeros((1, len(tokenizer.tokens)))
    scores[0, tokenizer.ids["X"]] = 10.0

    constrained = processor(input_ids, scores)

    assert torch.isfinite(constrained[0, tokenizer.eos_token_id])
    assert torch.isneginf(constrained[0, tokenizer.ids["X"]])


def test_candidate_checks_are_bounded_without_full_vocabulary_scan() -> None:
    tokenizer = _CharacterTokenizer("I" + "".join(chr(0x400 + index) for index in range(5000)))
    processor = GreedyDSLLogitsProcessor(
        tokenizer,
        prompt_width=1,
        image_refs_by_row=[["image0"]],
        initial_top_k=4,
        max_candidate_checks=8,
    )
    input_ids = torch.tensor([[tokenizer.ids["I"]]])
    scores = torch.arange(len(tokenizer.tokens), dtype=torch.float32).unsqueeze(0)

    started = time.perf_counter()
    constrained = processor(input_ids, scores)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    assert torch.isfinite(constrained[0, tokenizer.eos_token_id])
    assert processor.stats[0].examined_candidates == 8
    assert processor.stats[0].failure_reason == "constraint_abort"
    assert processor.stats[0].handler_max_ms < 100.0
    assert elapsed_ms < 100.0


def test_repeat_completion_transitions_to_forced_final() -> None:
    prefix = (
        'n1=LOCATE($image0,T("ship"))\n'
        "n2=SELECT_REL($n1,$n1,NEAR)\n"
        "n3=SELECT_REL($n2,$n1,NEAR)\n"
        "n4=SELECT_REL($n3,$n1,NEAR)\n"
        "n5=SELECT_REL($n4,$n1,NEAR"
    )
    vocabulary = "".join(dict.fromkeys(prefix + ")\nFINAL($n5,CHOICE_SINGLE)X"))
    tokenizer = _CharacterTokenizer(vocabulary)
    processor = GreedyDSLLogitsProcessor(
        tokenizer,
        prompt_width=1,
        image_refs_by_row=[["image0"]],
        initial_top_k=2,
        max_candidate_checks=8,
        repeat_guard_repetitions=4,
    )
    generated = [tokenizer.ids[character] for character in prefix]
    for character in ")\nFINAL($n5,CHOICE_SINGLE)":
        input_ids = torch.tensor([[tokenizer.ids["X"], *generated]])
        scores = torch.zeros((1, len(tokenizer.tokens)))
        scores[0, tokenizer.ids[character]] = 10.0
        constrained = processor(input_ids, scores)
        assert torch.isfinite(constrained[0, tokenizer.ids[character]])
        generated.append(tokenizer.ids[character])

    diagnostics = processor.diagnostics(
        0,
        torch.tensor(generated),
        max_new_tokens=512,
        pad_token_id=-1,
    )
    assert diagnostics["termination_reason"] == "repeat_guard_forced_final"
    assert diagnostics["constraint_recovery_transitions"] == [
        "normal",
        "finish_current_node",
        "forced_final",
    ]


def test_impossible_node_closure_aborts_at_candidate_cap() -> None:
    tokenizer = _CharacterTokenizer("IXYZ")
    processor = GreedyDSLLogitsProcessor(
        tokenizer,
        prompt_width=1,
        image_refs_by_row=[["image0"]],
        initial_top_k=2,
        max_candidate_checks=2,
        max_finish_node_tokens=1,
    )
    input_ids = torch.tensor([[tokenizer.ids["X"], tokenizer.ids["I"]]])
    scores = torch.tensor([[0.0, 0.0, 1.0, 3.0, 2.0]])
    constrained = processor(input_ids, scores)
    assert torch.isfinite(constrained[0, tokenizer.eos_token_id])
    assert processor.stats[0].failure_reason == "constraint_abort"
