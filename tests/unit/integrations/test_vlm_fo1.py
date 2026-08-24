from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
from scripts.integrations.vlm_fo1_worker import (
    MockBackend,
    PipelineConfig,
    build_backend,
    process_request,
    validate_request,
)

from sat_rs_vlm.evaluation.counting_protocol import PROTOCOL_NAME
from sat_rs_vlm.evaluation.ensemble import (
    EnsembleComparisonError,
    majority_vote_counting,
    median_vote_counting,
    pairwise_counting_comparison,
)
from sat_rs_vlm.integrations.vlm_fo1 import (
    build_counting_prompt,
    extract_count_target_phrase,
    is_official_fo1_model_path,
    parse_profile_output,
    parse_region_indexes,
)
from sat_rs_vlm.integrations.vlm_fo1_loader import (
    ensure_official_root,
    load_fo1_model,
    patch_config_compatibility,
    resolve_attention_backend,
    validate_model_path,
)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("How many small vehicles are visible?", "small vehicles"),
        ("How many large storage tanks are visible?", "large storage tanks"),
        ("How many planes are visible?", "planes"),
        ("How many tennis courts can be seen?", "tennis courts"),
        ("How many ships are there?", "ships"),
        ("What is the total number of planes visible?", "planes"),
        ("What is the number of storage tanks in the image?", "storage tanks"),
        ("Number of tennis courts visible in the image?", "tennis courts"),
        ("How many unique airplanes are visible?", "airplanes"),
    ],
)
def test_target_phrase_extraction_is_open_vocabulary(question: str, expected: str) -> None:
    result = extract_count_target_phrase(question)
    assert result.supported
    assert result.phrase == expected


@pytest.mark.parametrize(
    "question",
    [
        "How many unique object categories are present?",
        "How many lanes are on the highway?",
        "How many objects are visible?",
        "Are there more ships than harbors?",
    ],
)
def test_unsupported_target_is_explicit(question: str) -> None:
    result = extract_count_target_phrase(question)
    assert result.status == "unsupported"
    assert result.phrase is None


def test_comparative_target_is_not_instance_counting() -> None:
    result = extract_count_target_phrase("How many more ships than boats are visible?")
    assert result.status == "unsupported"


def test_prompt_profiles_are_switchable() -> None:
    question = "How many small vehicles are visible?"
    assert build_counting_prompt(question, "small vehicles", "plain") == question
    assert "integer only" in build_counting_prompt(question, "small vehicles", "integer")
    assert '"count"' in build_counting_prompt(question, "small vehicles", "json")
    assert build_counting_prompt(question, "small vehicles", "official_fo1").startswith(
        "How many small vehicles are there in this image?"
    )


def test_profile_specific_count_parsing_keeps_region_and_text_evidence() -> None:
    assert parse_profile_output("There are 7.", "plain")["count"] == 7
    assert parse_profile_output("7", "integer")["count"] == 7
    assert parse_profile_output('{"count":7}', "json")["count"] == 7
    official = parse_profile_output(
        "<ground>airplanes</ground><objects><region1><region3><region8></objects> 3",
        "official_fo1",
        proposal_count=10,
    )
    assert official["region_count"] == 3
    assert official["textual_count"] == 3
    assert official["count"] == 3
    disagreement = parse_profile_output(
        "<ground>airplanes</ground><objects><region1><region3><region8></objects> 4",
        "official_fo1",
        proposal_count=10,
    )
    assert disagreement["region_count"] == 3
    assert disagreement["textual_count"] == 4
    assert disagreement["count"] == 3
    assert disagreement["count_agrees_with_text"] is False
    assert parse_profile_output("There are 7.", "official_fo1")["parse_ok"] is False
    assert parse_profile_output(
        "There are 7.", "official_fo1", count_source="text"
    )["count"] == 7
    assert parse_profile_output(
        "There are 7.", "official_fo1", count_source="auto"
    )["count_source"] == "text"


def test_region_index_parser_preserves_evidence_and_zero_count() -> None:
    parsed = parse_region_indexes(
        "<ground>ships</ground><objects><region2><region2><region0></objects>",
        proposal_count=3,
    )
    assert parsed["parse_ok"] is True
    assert parsed["selected_region_indexes"] == [2, 0]
    zero = parse_region_indexes("<ground>ships</ground><objects></objects>", proposal_count=3)
    assert zero["parse_ok"] is True
    assert zero["selected_region_indexes"] == []


def test_region_index_parser_rejects_out_of_range() -> None:
    parsed = parse_region_indexes(
        "<ground>ships</ground><objects><region4></objects>", proposal_count=2
    )
    assert parsed["parse_ok"] is False
    assert parsed["invalid_region_indexes"] == [4]


def test_worker_json_protocol_and_failure_handling() -> None:
    config = PipelineConfig(model_path="mock", upn_checkpoint="mock")
    backend = MockBackend()
    request = {
        "id": "sample-1",
        "image": "missing.png",
        "question": "How many small vehicles are visible?",
        "target_phrase": "small vehicles",
    }
    response = process_request(request, backend, config)
    assert response["status"] == "ok"
    assert response["fo1_count"] == 2
    assert response["selected_region_indexes"] == [0, 1]
    bad = process_request({**request, "reference": "2"}, backend, config)
    assert bad["status"] == "failed"
    assert bad["failure_stage"] == "protocol_guard"
    upper_bad = process_request({**request, "Reference": "2"}, backend, config)
    assert upper_bad["failure_stage"] == "protocol_guard"
    nested_bad = process_request(
        {**request, "metadata": {"nested": [{"ground_truth": 2}]}}, backend, config
    )
    assert nested_bad["failure_stage"] == "protocol_guard"
    mismatch = process_request({**request, "target_phrase": "ships"}, backend, config)
    assert mismatch["failure_stage"] == "protocol_guard"


def test_precomputed_proposals_are_validated_without_upn() -> None:
    config = PipelineConfig(
        model_path="mock",
        upn_checkpoint="",
        runtime_mode="shared_rs_vlm",
        proposal_backend="precomputed",
    )
    request = {
        "id": "precomputed-1",
        "image": "missing.png",
        "question": "How many ships are visible?",
        "target_phrase": "ships",
        "bbox_list": [[0, 0, 10, 10]],
    }
    response = process_request(request, MockBackend(), config)
    assert response["status"] == "ok"
    normalized, error = validate_request(
        {key: value for key, value in request.items() if key != "bbox_list"},
        prompt_profile="official_fo1",
        proposal_backend="precomputed",
    )
    assert normalized is None
    assert error is not None
    assert "bbox_list" in error["error"]


def test_mock_does_not_require_official_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLM_FO1_ROOT", raising=False)
    response = process_request(
        {
            "id": "mock-no-root",
            "image": "missing.png",
            "question": "How many ships are visible?",
            "target_phrase": "ships",
        },
        MockBackend(),
        PipelineConfig(model_path="mock", upn_checkpoint="mock"),
    )
    assert response["status"] == "ok"


def test_official_backend_missing_root_fails_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLM_FO1_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="VLM_FO1_ROOT"):
        build_backend("official", PipelineConfig(model_path="mock", upn_checkpoint="mock"))


def test_official_model_path_helper() -> None:
    assert is_official_fo1_model_path("VLM-FO1-3B-v01")
    assert not is_official_fo1_model_path("some-other-model")


def test_shared_loader_validates_local_model_before_hf(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    with pytest.raises(RuntimeError, match="does not exist"):
        validate_model_path(missing)
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    with pytest.raises(RuntimeError, match="missing config.json"):
        validate_model_path(incomplete)


def test_shared_attention_auto_falls_back_without_flash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    assert resolve_attention_backend("auto") == "sdpa"


def test_shared_attention_explicit_flash_is_clear_without_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match="flash_attn is unavailable"):
        resolve_attention_backend("flash_attention_2")


def test_shared_root_does_not_require_upn_source(tmp_path: Path) -> None:
    (tmp_path / "vlm_fo1").mkdir()
    assert ensure_official_root(tmp_path, require_upn=False) == tmp_path.resolve()


def test_shared_config_patch_uses_tokenizer_pad_token_id() -> None:
    config = types.SimpleNamespace(eos_token_id=2)
    tokenizer = types.SimpleNamespace(pad_token_id=151643, eos_token_id=2)

    patches = patch_config_compatibility(config, tokenizer)

    assert config.pad_token_id == 151643
    assert patches == {
        "pad_token_id": {"source": "tokenizer.pad_token_id", "value": 151643}
    }


def test_shared_config_patch_falls_back_to_tokenizer_eos_token_id() -> None:
    config = types.SimpleNamespace(eos_token_id=2)
    tokenizer = types.SimpleNamespace(pad_token_id=None, eos_token_id=151644)

    patches = patch_config_compatibility(config, tokenizer)

    assert config.pad_token_id == 151644
    assert patches["pad_token_id"] == {
        "source": "tokenizer.eos_token_id",
        "value": 151644,
    }


def test_shared_config_patch_fails_without_any_pad_or_eos_id() -> None:
    config = types.SimpleNamespace(eos_token_id=None)
    tokenizer = types.SimpleNamespace(pad_token_id=None, eos_token_id=None)

    with pytest.raises(RuntimeError, match="no fallback is available"):
        patch_config_compatibility(config, tokenizer)


def test_shared_loader_passes_patched_config_to_from_pretrained(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text("{}", encoding="utf-8")
    calls: list[str] = []
    text_config = types.SimpleNamespace(_attn_implementation_internal=None)
    vision_config = types.SimpleNamespace(_attn_implementation_internal=None)
    config = types.SimpleNamespace(
        eos_token_id=151644,
        text_config=text_config,
        vision_config=vision_config,
    )
    tokenizer = types.SimpleNamespace(pad_token_id=151643, eos_token_id=151644)
    captured: dict[str, object] = {}

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> object:
            calls.append("config")
            assert kwargs["local_files_only"] is True
            return config

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> object:
            calls.append("tokenizer")
            assert kwargs["local_files_only"] is True
            return tokenizer

    class FakeModel:
        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> "FakeModel":
            calls.append("model")
            captured.update(kwargs)
            return cls()

        def get_vision_tower(self) -> None:
            return None

        def get_vision_tower_aux(self) -> None:
            return None

        def eval(self) -> "FakeModel":
            return self

        def to(self, **kwargs: object) -> "FakeModel":
            return self

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_torch.float32 = object()
    fake_torch.bfloat16 = object()
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoConfig = FakeAutoConfig
    fake_transformers.AutoTokenizer = FakeAutoTokenizer
    fake_vlm = types.ModuleType("vlm_fo1")
    fake_vlm.__path__ = []
    fake_vlm_model = types.ModuleType("vlm_fo1.model")
    fake_vlm_model.OmChatQwen25VLForCausalLM = FakeModel

    monkeypatch.delenv("VLM_FO1_ROOT", raising=False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "vlm_fo1", fake_vlm)
    monkeypatch.setitem(sys.modules, "vlm_fo1.model", fake_vlm_model)

    bundle = load_fo1_model(model_path, "cpu", attention_backend="sdpa")

    assert calls == ["config", "tokenizer", "model"]
    assert captured["config"] is config
    assert config.pad_token_id == 151643
    assert config._attn_implementation_internal == "sdpa"
    assert text_config._attn_implementation_internal == "sdpa"
    assert vision_config._attn_implementation_internal == "sdpa"
    assert bundle.config_compatibility_patches == {
        "pad_token_id": {"source": "tokenizer.pad_token_id", "value": 151643},
        "attention_backend": {
            "source": "loader.attention_backend",
            "value": "sdpa",
            "targets": ["config", "config.text_config", "config.vision_config"],
        },
    }


def test_worker_early_exit_returns_one_failed_row_per_request(tmp_path: Path) -> None:
    from scripts.evaluation.evaluate_vlm_fo1 import _run_worker

    script = tmp_path / "exit_after_one.py"
    script.write_text(
        "import json, sys\n"
        "line = sys.stdin.readline()\n"
        "if line:\n"
        "    request = json.loads(line)\n"
        "    print(json.dumps({'id': request['id'], 'status': 'ok'}), flush=True)\n",
        encoding="utf-8",
    )
    requests = [
        {"id": "one", "image": "a", "question": "q", "target_phrase": "q"},
        {"id": "two", "image": "b", "question": "q", "target_phrase": "q"},
        {"id": "three", "image": "c", "question": "q", "target_phrase": "q"},
    ]
    settings = {
        "worker_python": Path(__import__("sys").executable),
        "worker_script": script,
        "backend": "mock",
        "model": Path("mock"),
        "upn_checkpoint": Path("mock"),
        "device": "cpu",
        "proposal_score_threshold": 0.3,
        "proposal_top_k": 100,
        "nms_threshold": 0.8,
        "max_new_tokens": 8,
        "temperature": 0.0,
        "top_p": 0.05,
        "prompt_profile": "plain",
        "count_source": "text",
    }
    responses = _run_worker(requests, settings)
    assert [response["id"] for response in responses] == ["one", "two", "three"]
    assert responses[0]["status"] == "ok"
    assert all(response["status"] == "failed" for response in responses[1:])


def test_validate_request_marks_unsupported_without_backend_call() -> None:
    request = {
        "id": "sample-unsupported",
        "image": "image.png",
        "question": "How many lanes are on the highway?",
        "target_phrase": "",
    }
    normalized, response = validate_request(request, prompt_profile="official_fo1")
    assert normalized is None
    assert response is not None
    assert response["status"] == "unsupported"
    assert response["target_status"] == "unsupported"


def _row(sample_id: str, prediction: str, reference: str = "2") -> dict[str, str]:
    return {
        "id": sample_id,
        "task_type": "counting",
        "prediction": prediction,
        "reference": reference,
    }


def test_pairwise_alignment_and_oracle() -> None:
    result = pairwise_counting_comparison(
        [_row("a", "2"), _row("b", "1")],
        [_row("a", "1"), _row("b", "2")],
    )
    assert result["pairwise_prediction_agreement"] == 0.0
    assert result["oracle_accuracy"] == 1.0
    assert result["correctness_overlap"]["a_only_correct"] == 1
    assert result["correctness_overlap"]["b_only_correct"] == 1


def test_duplicate_id_rejected() -> None:
    with pytest.raises(EnsembleComparisonError, match="duplicate"):
        pairwise_counting_comparison(
            [_row("a", "1"), _row("a", "2")], [_row("a", "1"), _row("b", "2")]
        )


def test_majority_vote_has_no_router_search() -> None:
    result = majority_vote_counting([[_row("a", "2")], [_row("a", "2")], [_row("a", "1")]])
    assert result["accuracy"] == 1.0
    assert result["threshold_search"]["performed"] is False


def test_median_vote_preserves_missing_predictions() -> None:
    result = median_vote_counting([[_row("a", "1")], [_row("a", "")], [_row("a", "3")]])
    assert result["rows"][0]["prediction"] == 2
    assert result["threshold_search"]["performed"] is False


def test_mock_evaluator_writes_standard_outputs(tmp_path: Path) -> None:
    from scripts.evaluation.evaluate_vlm_fo1 import evaluate

    source = tmp_path / "tier.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "sample-1",
                "task_type": "counting",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": "image.png"},
                            {"type": "text", "text": "How many ships are visible?"},
                        ],
                    },
                    {"role": "assistant", "content": '{"count":2}'},
                ],
                "metadata": {"dataset": "VRSBench"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    audit = tmp_path / "audit.json"
    audit.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "out"
    outputs = evaluate(
        {
            "scope": "e_count_v2",
            "input": source,
            "output_dir": output_dir,
            "image_root": None,
            "max_samples": 4,
            "backend": "mock",
            "worker_python": Path(__import__("sys").executable),
            "worker_script": Path("scripts/integrations/vlm_fo1_worker.py"),
            "model": Path("mock"),
            "upn_checkpoint": Path("mock"),
            "device": "cpu",
            "proposal_score_threshold": 0.3,
            "proposal_top_k": 100,
            "nms_threshold": 0.8,
            "max_new_tokens": 4096,
            "temperature": 0.0,
            "top_p": 0.05,
            "prompt_profile": "official_fo1",
            "audit": audit,
        }
    )
    assert set(outputs) == {"predictions", "metrics", "summary", "provenance", "diagnostics"}
    metrics = json.loads(outputs["metrics"].read_text(encoding="utf-8"))
    assert metrics["metrics_protocol"] == PROTOCOL_NAME
    assert metrics["n"] == 1
    diagnostics = json.loads(outputs["diagnostics"].read_text(encoding="utf-8"))
    assert diagnostics["prompt_profile"] == "official_fo1"
    assert diagnostics["count_source"] == "region"
    assert diagnostics["region_count_available_count"] == 1
    assert diagnostics["textual_count_available_count"] == 0
    assert diagnostics["failure_stage_histogram"] == {}
    prediction = json.loads(outputs["predictions"].read_text(encoding="utf-8"))
    assert prediction["prediction"] == "2"
    assert {
        "id",
        "task_type",
        "question",
        "reference",
        "prediction",
        "proposal_boxes",
        "proposal_scores",
        "selected_region_indexes",
        "selected_region_boxes",
    } <= prediction.keys()


def test_full_quantity_scope_keeps_parseable_quantity_population() -> None:
    from scripts.evaluation.evaluate_vlm_fo1 import _select_rows

    def row(sample_id: str, reference: str, task_type: str = "counting") -> dict[str, object]:
        return {
            "id": sample_id,
            "task_type": task_type,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "image.png"},
                        {"type": "text", "text": "How many ships are visible?"},
                    ],
                },
                {"role": "assistant", "content": reference},
            ],
            "metadata": {"dataset": "VRSBench", "qa_type": "object quantity"},
        }

    selected = _select_rows(
        [row("numeric", "2"), row("non_numeric", "Multiple"), row("other", "2", "vqa")],
        "full_vrsbench_quantity",
    )
    assert [item["id"] for item in selected] == ["numeric"]
