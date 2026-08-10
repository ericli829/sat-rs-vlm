from __future__ import annotations

import json
from pathlib import Path

import pytest

from sat_rs_vlm.quantization.sensitivity import (
    SensitivityResult,
    build_sensitivity_groups,
    build_sensitivity_report,
    calculate_sensitivity,
    classify_component,
    quantize_named_linear_modules,
    write_sensitivity_report,
)


def test_component_classifier_uses_multiple_qwen_naming_candidates() -> None:
    assert classify_component("model.visual.blocks.0.mlp") == "vision_encoder"
    assert classify_component("model.visual.merger.mlp") == "multimodal_projector"
    assert classify_component("model.mm_projector.linear") == "multimodal_projector"
    assert classify_component("model.language_model.layers.0.self_attn") == "language_model"


def test_group_scan_and_selected_dynamic_quantization() -> None:
    torch = pytest.importorskip("torch")

    class ToyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.visual = torch.nn.Sequential(torch.nn.Linear(4, 4))
            self.projector = torch.nn.Linear(4, 4)
            self.language_model = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.Linear(4, 2),
            )

    model = ToyModel()
    groups = build_sensitivity_groups(
        model,
        torch,
        method="component_wise",
        skip_modules=("visual",),
    )
    names = {group.name for group in groups}
    assert names == {"language_model", "multimodal_projector"}

    target = next(group for group in groups if group.name == "multimodal_projector")
    quantized = quantize_named_linear_modules(model, torch, target.module_names)
    modules = dict(quantized.named_modules())
    assert "quantized.dynamic" in modules["projector"].__class__.__module__
    assert isinstance(modules["language_model.0"], torch.nn.Linear)
    assert isinstance(model.projector, torch.nn.Linear)


def test_sensitivity_uses_v15_primary_metrics_not_keyword_hit() -> None:
    baseline = {
        "by_task": {
            "vqa": {
                "metrics": {
                    "normalized_accuracy": {"value": 0.8},
                    "keyword_hit": {"value": 1.0},
                }
            }
        }
    }
    quantized = {
        "by_task": {
            "vqa": {
                "metrics": {
                    "normalized_accuracy": {"value": 0.6},
                    "keyword_hit": {"value": 0.0},
                }
            }
        }
    }

    score, deltas = calculate_sensitivity(baseline, quantized)

    assert score == pytest.approx(0.2)
    assert len(deltas) == 1
    assert next(iter(deltas)).endswith("normalized_accuracy")


def test_sensitivity_report_round_trip(tmp_path: Path) -> None:
    result = SensitivityResult(
        name="language_model",
        kind="component",
        module_names=("language_model.0",),
        parameter_count=20,
        sensitivity_score=0.05,
        metric_deltas={"metric": {"delta": -0.05, "higher_is_better": True}},
        evaluation_dir="results/language_model",
    )
    report = build_sensitivity_report(
        model_source="merged-model",
        method="component_wise",
        baseline_evaluation_dir="results/baseline",
        results=[result],
    )
    outputs = write_sensitivity_report(report, tmp_path)

    written = json.loads(outputs["json"].read_text(encoding="utf-8"))
    assert written["metric_contract"] == "evaluation-v1.5"
    assert written["results"][0]["name"] == "language_model"
    assert outputs["markdown"].is_file()
