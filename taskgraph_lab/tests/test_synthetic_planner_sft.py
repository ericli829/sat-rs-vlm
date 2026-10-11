from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from sat_rs_vlm.taskgraph.providers import PlannerRequest, Qwen3VLPlannerProvider
from sat_rs_vlm.taskgraph.runtime_types import ImageRef
from taskgraph_lab.tools.mix_planner_synthetic import build_mixture
from taskgraph_lab.tools.verify_synthetic_planner_sft import (
    inspect_batch,
    validate_message_contract,
    verify_sft,
)
from taskgraph_lab.training.planner_dataset import PlannerSFTDataset, file_sha256

torch = pytest.importorskip("torch")


class ReversibleFakeProcessor:
    """One Unicode code point per token, with an explicit assistant boundary."""

    def __init__(self, padding_side: str = "right") -> None:
        self.tokenizer = SimpleNamespace(padding_side=padding_side, pad_token_id=0)

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        text = "".join(f"<{m['role']}>{m['content']}<end>\n" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def __call__(
        self,
        *,
        text: list[str],
        padding: bool = True,
        truncation: bool = True,
        max_length: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        values = [[ord(character) + 1 for character in item] for item in text]
        if truncation and max_length is not None:
            values = [item[:max_length] for item in values]
        width = max(len(item) for item in values)
        ids = []
        masks = []
        for item in values:
            pads = [0] * (width - len(item)) if padding else []
            if self.tokenizer.padding_side == "left":
                ids.append(pads + item)
                masks.append([0] * len(pads) + [1] * len(item))
            else:
                ids.append(item + pads)
                masks.append([1] * len(item) + [0] * len(pads))
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks)}


def _row(sample_id: str, *, synthetic: bool, family: str = "category") -> dict[str, Any]:
    question = f"Locate target {sample_id}."
    request = PlannerRequest(question, "FREE_FORM", (), {"image0": ImageRef("symbolic.png")})
    messages = Qwen3VLPlannerProvider._messages(request, "Return canonical DSL.")
    messages.append(
        {
            "role": "assistant",
            "content": 'INTENT(OTHER)\nn1=LOCATE($image0,T("building"))\nFINAL($n1,BBOX)',
        }
    )
    return {
        "id": sample_id,
        "messages": messages,
        "task_type": "planner",
        "metadata": {
            "source": "synthetic_programmatic_plan" if synthetic else "existing_teacher",
            "family": family,
            "category": "building",
            "original_question": question,
        },
    }


def _write(path: Path, rows: list[dict[str, Any]], *, newline: str = "\n") -> Path:
    path.write_bytes((newline.join(json.dumps(row) for row in rows) + newline).encode())
    return path


def _sources(tmp_path: Path) -> dict[str, Path]:
    return {
        "base_train": _write(
            tmp_path / "old_train.jsonl",
            [_row(f"old-{index}", synthetic=False) for index in range(4)],
        ),
        "base_test": _write(
            tmp_path / "fixed_test.jsonl", [_row("test-only", synthetic=False)], newline="\r\n"
        ),
        "synthetic_train": _write(
            tmp_path / "new_train.jsonl",
            [
                _row(f"new-{index}", synthetic=True, family=f"family-{index % 2}")
                for index in range(6)
            ],
        ),
        "synthetic_validation": _write(
            tmp_path / "new_validation.jsonl", [_row("validation-only", synthetic=True)]
        ),
    }


def test_mixture_preserves_old_rows_and_fixed_test_bytes(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    hashes = {key: file_sha256(path) for key, path in sources.items()}
    first = build_mixture(**sources, output_dir=tmp_path / "first", synthetic_fraction=0.5, seed=9)
    second = build_mixture(
        **sources, output_dir=tmp_path / "second", synthetic_fraction=0.5, seed=9
    )
    assert first["base_training_count"] == 4
    assert first["synthetic_selected_count"] == 4
    assert first["actual_synthetic_fraction"] == 0.5
    assert first["selected_synthetic_ids"] == second["selected_synthetic_ids"]
    assert (tmp_path / "first/test.jsonl").read_bytes() == sources["base_test"].read_bytes()
    mixed = [json.loads(line) for line in (tmp_path / "first/train.jsonl").read_text().splitlines()]
    originals = [json.loads(line) for line in sources["base_train"].read_text().splitlines()]
    assert [row for row in mixed if row["id"].startswith("old-")] == sorted(
        originals, key=lambda row: next(i for i, mixed_row in enumerate(mixed) if row == mixed_row)
    )
    assert {key: file_sha256(path) for key, path in sources.items()} == hashes
    assert first["distributions"]["train"]["final_types"] == {"BBOX": 8}
    assert first["splits"]["test"]["sha256"] == hashes["base_test"]


def test_mixture_rejects_insufficient_distinct_capacity_and_existing_output(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    target = tmp_path / "too-large"
    with pytest.raises(ValueError, match="only 6 available"):
        build_mixture(**sources, output_dir=target, synthetic_fraction=0.9)
    assert not target.exists()
    target.mkdir()
    with pytest.raises(FileExistsError, match="new output"):
        build_mixture(**sources, output_dir=target, synthetic_fraction=0.5)


def test_mixture_rejects_synthetic_validation_overlap(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    validation = _row("different-id", synthetic=True)
    validation["metadata"]["original_question"] = "Locate target new-0."
    _write(sources["synthetic_validation"], [validation])
    with pytest.raises(ValueError, match="questions overlap"):
        build_mixture(**sources, output_dir=tmp_path / "rejected", synthetic_fraction=0.5)


def test_mixture_rejects_fake_answer_audits_and_nonbbox_synthetic_labels(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    bad = _row("new-only", synthetic=True)
    bad["answer_audit"] = {"valid": True}
    _write(sources["synthetic_train"], [bad])
    with pytest.raises(ValueError, match="cannot claim answer audit"):
        build_mixture(**sources, output_dir=tmp_path / "audit", synthetic_fraction=0.2)
    bad.pop("answer_audit")
    bad["messages"][2]["content"] = bad["messages"][2]["content"].replace(
        "FINAL($n1,BBOX)", 'FINAL_QUESTION($n1,CHOICE_SINGLE,"Which building?")'
    )
    _write(sources["synthetic_train"], [bad])
    with pytest.raises(ValueError, match="synthetic final must be BBOX"):
        build_mixture(**sources, output_dir=tmp_path / "wrong-label", synthetic_fraction=0.2)


def test_mixture_preserves_historical_graphs_rejected_by_current_typechecker(
    tmp_path: Path,
) -> None:
    sources = _sources(tmp_path)
    original = [json.loads(line) for line in sources["base_train"].read_text().splitlines()]
    original[0]["messages"][2]["content"] = (
        'INTENT(ATTRIBUTE_QUERY)\nn1=LOCATE($image0,T("building"))\n'
        'n2=SELECT_REL($n1,$n1,NEAR)\nn3=ATTRIBUTE($n2,"color")\n'
        "FINAL($n3,CHOICE_SINGLE)"
    )
    _write(sources["base_train"], original)
    manifest = build_mixture(**sources, output_dir=tmp_path / "historical", synthetic_fraction=0.5)
    audit = manifest["base_current_dsl_validation"]
    assert audit["invalid_count"] == 1
    assert audit["failure_codes_rows"]["attribute_requires_singleton"] == 1
    mixed = [
        json.loads(line) for line in (tmp_path / "historical/train.jsonl").read_text().splitlines()
    ]
    assert original[0] in mixed


@pytest.mark.parametrize("side", ["left", "right"])
def test_reversible_processor_checks_complete_assistant_only_supervision(side: str) -> None:
    rows = [_row("short", synthetic=True), _row("a-longer-question", synthetic=True)]
    stats = inspect_batch(rows, ReversibleFakeProcessor(side), max_seq_length=1024)
    assert stats["assistant_tokens"][0] == stats["assistant_tokens"][1]
    assert stats["full_sequence_tokens"][0] < stats["full_sequence_tokens"][1]


def test_verifier_loads_real_dataset_jsonl_without_visual_inputs(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "sft.jsonl", [_row("one", synthetic=True), _row("two", synthetic=True)]
    )
    report = verify_sft([path], processor=ReversibleFakeProcessor(), max_seq_length=1024)
    assert report["valid"]
    assert report["count"] == 2
    assert report["assistant_only_mask_exact"]
    assert report["truncated_count"] == 0
    assert report["model_weights_loaded"] is False


def test_verifier_rejects_truncation_and_output_budget(tmp_path: Path) -> None:
    row = _row("one", synthetic=True)
    with pytest.raises(ValueError, match="exceeds max_seq_length|No assistant tokens"):
        inspect_batch([row], ReversibleFakeProcessor(), max_seq_length=180)
    path = _write(tmp_path / "budget.jsonl", [row])
    with pytest.raises(ValueError, match="exceed max_output_tokens"):
        verify_sft(
            [path], processor=ReversibleFakeProcessor(), max_seq_length=1024, max_output_tokens=10
        )


def test_message_contract_rejects_nonproduction_payload_and_nonbbox() -> None:
    row = _row("one", synthetic=True)
    validate_message_contract(row)
    payload = json.loads(row["messages"][1]["content"])
    payload["unexpected"] = True
    row["messages"][1]["content"] = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    with pytest.raises(ValueError, match="production Planner serialization"):
        validate_message_contract(row)


def test_synthetic_plan_does_not_bypass_existing_accepted_answer_gate(tmp_path: Path) -> None:
    path = _write(tmp_path / "synthetic.jsonl", [_row("one", synthetic=True)])
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Planner prompt")
    with pytest.raises(ValueError, match="missing sample_id"):
        PlannerSFTDataset(path, system_prompt=prompt)
