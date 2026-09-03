from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import scripts.evaluate_rs_vlm as evaluate_script
from scripts.evaluate_rs_vlm import (
    build_generation_kwargs,
    generate_prediction,
    iter_evaluation_batches,
    resolve_evaluation_outputs,
    summarize,
    validate_local_adapter,
)

from sat_rs_vlm.evaluation.inference import count_decoded_output_tokens


class FakeTensor:
    """提供 shape 和设备移动能力的最小 tensor 替身。"""

    def __init__(self, shape: tuple[int, ...], device: str = "cpu") -> None:
        self.shape = shape
        self.device = device

    def to(self, device: object) -> FakeTensor:
        self.device = str(device)
        return self


class FakeOutputIds:
    """模拟 generate 返回的二维 token tensor。"""

    def __getitem__(self, key: object) -> list[list[int]]:
        assert key == (slice(None), slice(3, None))
        return [[9, 10]]


def test_greedy_generation_omits_temperature() -> None:
    kwargs = build_generation_kwargs(
        {"max_new_tokens": 64, "do_sample": False, "temperature": 0.0, "num_beams": 1}
    )

    assert kwargs == {"max_new_tokens": 64, "do_sample": False, "num_beams": 1}


def test_sampling_generation_includes_sampling_parameters() -> None:
    kwargs = build_generation_kwargs(
        {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20}
    )

    assert kwargs["temperature"] == 0.7
    assert kwargs["top_p"] == 0.8
    assert kwargs["top_k"] == 20


def test_count_decoded_output_tokens_reports_method_input() -> None:
    tokenizer = SimpleNamespace(encode=lambda text, **_: text.split())

    assert count_decoded_output_tokens(
        SimpleNamespace(tokenizer=tokenizer), ["one two", "three"]
    ) == [2, 1]


def test_validate_local_adapter_requires_config_and_weights(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    with pytest.raises(FileNotFoundError, match="adapter_config.json"):
        validate_local_adapter(str(adapter), local_files_only=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="weights"):
        validate_local_adapter(str(adapter), local_files_only=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")

    validate_local_adapter(str(adapter), local_files_only=True)


def test_generate_prediction_moves_batch_to_model_input_device() -> None:
    input_ids = FakeTensor((1, 3))
    pixel_values = FakeTensor((4, 8))

    class Model:
        def get_input_embeddings(self) -> Any:
            return SimpleNamespace(weight=FakeTensor((1,), "cuda:0"))

        def generate(self, **kwargs: Any) -> FakeOutputIds:
            assert kwargs["input_ids"].device == "cuda:0"
            assert kwargs["pixel_values"].device == "cuda:0"
            assert "temperature" not in kwargs
            return FakeOutputIds()

    class Processor:
        def batch_decode(self, token_ids: Any, **kwargs: Any) -> list[str]:
            assert token_ids == [[9, 10]]
            assert kwargs["clean_up_tokenization_spaces"] is False
            return ["answer"]

    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        device=lambda value: value,
        is_tensor=lambda value: isinstance(value, FakeTensor),
        inference_mode=lambda: nullcontext(),
    )
    collator = lambda batch: {  # noqa: E731
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }

    prediction = generate_prediction(
        Model(),
        Processor(),
        collator,  # type: ignore[arg-type]
        {"id": "sample"},
        {"do_sample": False},
        torch,
    )

    assert prediction == "answer"


def test_summary_reports_empty_prediction_rate() -> None:
    summary = summarize(
        [
            {"task_type": "vqa", "prediction": "", "reference": "yes"},
            {"task_type": "vqa", "prediction": "yes", "reference": "yes"},
        ]
    )

    assert summary["overall"]["empty_predictions"] == 1
    assert summary["overall"]["empty_prediction_rate"] == 0.5
    assert summary["by_task"]["vqa"]["empty_prediction_rate"] == 0.5


def test_evaluation_batches_group_by_task_and_preserve_original_indexes() -> None:
    dataset = [
        {"id": "caption-1", "task_type": "captioning"},
        {"id": "vqa-1", "task_type": "vqa"},
        {"id": "caption-2", "task_type": "captioning"},
    ]

    batches = list(iter_evaluation_batches(dataset, 2, group_by_task=True))

    assert batches == [
        ("captioning", [(0, dataset[0]), (2, dataset[2])]),
        ("vqa", [(1, dataset[1])]),
    ]


def test_evaluation_batches_reject_non_positive_batch_size() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        list(iter_evaluation_batches([], 0, group_by_task=True))


def test_evaluation_outputs_keep_legacy_files_and_isolate_v15(tmp_path: Path) -> None:
    summary, predictions, evaluation_dir = resolve_evaluation_outputs(
        tmp_path / "eval.yaml",
        {
            "output": {
                "summary_file": "unused-summary.json",
                "predictions_file": "unused-predictions.jsonl",
            }
        },
        checkpoint=None,
        output_dir=tmp_path / "run",
    )

    assert summary == tmp_path / "run" / "summary.json"
    assert predictions == tmp_path / "run" / "predictions.jsonl"
    assert evaluation_dir == tmp_path / "run" / "evaluation_v1_5"


def test_evaluate_writes_system_telemetry_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    eval_file = tmp_path / "eval.jsonl"
    eval_file.write_text("{}\n", encoding="utf-8")
    output_dir = tmp_path / "run"
    sample = {
        "id": "sample-1",
        "task_type": "vqa",
        "messages": [{"role": "assistant", "content": "yes"}],
        "metadata": {"dataset": "fixture"},
    }

    class Parameter:
        dtype = "torch.float32"

        @staticmethod
        def numel() -> int:
            return 4

        @staticmethod
        def element_size() -> int:
            return 4

    class Model:
        def eval(self) -> None:
            return None

        @staticmethod
        def parameters():
            return iter((Parameter(),))

    class Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    torch = SimpleNamespace(cuda=Cuda(), __version__="test", version=SimpleNamespace(cuda=None))

    monkeypatch.setattr(evaluate_script, "Qwen3VLDataset", lambda *_: [sample])
    monkeypatch.setattr(evaluate_script, "Qwen3VLDataCollator", lambda *_args, **_kwargs: object())
    prediction_calls: list[None] = []

    def fake_timed_predictions(*_args, **_kwargs):
        prediction_calls.append(None)
        return ["yes"], 12.5

    monkeypatch.setattr(evaluate_script, "timed_predictions", fake_timed_predictions)

    evaluation_arguments: dict[str, Any] = {}

    def fake_run_evaluation(*_args, **kwargs):
        evaluation_arguments.update(kwargs)
        destination = Path(_args[1])
        destination.mkdir(parents=True)
        metrics = destination / "metrics.json"
        metrics.write_text("{}\n", encoding="utf-8")
        return {"metrics": metrics}

    monkeypatch.setattr(evaluate_script, "run_evaluation", fake_run_evaluation)
    config = {
        "model": {"base_model": "fixture/model", "torch_dtype": "float32"},
        "data": {
            "eval_file": str(eval_file),
            "image_root": str(tmp_path),
            "eval_batch_size": 1,
        },
        "generation": {"max_new_tokens": 8, "do_sample": False},
        "evaluation": {"semantic": False, "warmup_runs": 1, "repeat_runs": 2},
        "output": {"summary_file": "unused", "predictions_file": "unused"},
    }

    result = evaluate_script.evaluate(
        tmp_path / "config.yaml",
        output_dir=output_dir,
        loaded_model=Model(),
        loaded_processor=SimpleNamespace(
            tokenizer=SimpleNamespace(encode=lambda text, **_: text.split())
        ),
        loaded_modules={"torch": torch},
        config_override=config,
    )

    metadata = json.loads((output_dir / "evaluation_metadata.json").read_text(encoding="utf-8"))
    telemetry = json.loads((output_dir / "telemetry_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "system_manifest.json").read_text(encoding="utf-8"))
    prediction = json.loads((output_dir / "predictions.jsonl").read_text(encoding="utf-8"))

    assert metadata["peak_cpu_rss_mb"] > 0
    assert metadata["failed_samples"] == 0
    assert telemetry["prediction_loop"]["success"] is True
    assert telemetry["single_sample_full_system_e2e_available"] is False
    assert manifest["system"]["total_parameter_count"] == 4
    assert manifest["benchmark"]["warmup_runs"] == 1
    assert manifest["benchmark"]["repeat_runs"] == 2
    assert telemetry["tokens"]["output_token_count"] == 1
    assert len(prediction_calls) == 3
    assert evaluation_arguments["resource_benchmark"]["resources"]["peak_cpu_rss_mb"] > 0
    assert prediction["latency_semantics"] == "batch_amortized_model_path"
    assert result["system_manifest"] == str(output_dir / "system_manifest.json")
