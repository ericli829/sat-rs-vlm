from __future__ import annotations

from pathlib import Path

import pytest

from sat_rs_vlm.application.inference_service import InferenceService
from sat_rs_vlm.models.mock_model import MockVLMEngine


@pytest.fixture
def inference_service() -> InferenceService:
    return InferenceService(engine=MockVLMEngine())


@pytest.fixture
def fake_plugin_root(tmp_path: Path) -> Path:
    """Create a minimal external plugin pack shared by unit and integration tests."""
    root = tmp_path / "plugin-pack"
    plugin = root / "plugins" / "fake_strategy"
    (plugin / "configs").mkdir(parents=True)
    (plugin / "docs").mkdir()
    (plugin / "tests").mkdir()
    (plugin / "plugin.yaml").write_text(
        """schema_version: "1"
plugin:
  name: fake_strategy
  display_name: Fake Strategy
  version: 0.1.0
  description: Test plugin.
  api_version: "1"
  status: experimental
entrypoint:
  module: strategy.py
  class: FakePlugin
compatibility:
  python: ">=3.10"
  platforms: [windows, linux, darwin]
  requires_cuda: false
  supports_cpu: true
dependencies:
  requirements_file: requirements.txt
paths:
  default_train_config: configs/train.yaml
  default_smoke_config: configs/smoke.yaml
  checkpoints_dir: checkpoints
  reports_dir: reports
  logs_dir: logs
  docs_dir: docs
capabilities:
  adapter_based: false
  quantized_base: false
outputs:
  manifest_file: strategy_manifest.json
  train_report_file: train_report.json
  evaluation_report_file: evaluation_report.json
""",
        encoding="utf-8",
    )
    (plugin / "strategy.py").write_text(
        """from pathlib import Path
from typing import Any, Mapping
from sat_rs_vlm.plugins import ExternalFineTuningPlugin, PluginContext

class FakePlugin(ExternalFineTuningPlugin):
    name = "fake_strategy"
    version = "0.1.0"
    def validate(self, context: PluginContext, config: Mapping[str, Any]) -> None:
        return None
    def prepare_model(self, context: PluginContext, model: Any, processor: Any,
                      config: Mapping[str, Any]) -> Any:
        return model
    def build_training_arguments(self, context: PluginContext,
                                 config: Mapping[str, Any]) -> Mapping[str, Any]:
        return {}
    def save_artifacts(self, context: PluginContext, model: Any, processor: Any,
                       output_dir: Path) -> None:
        return None
""",
        encoding="utf-8",
    )
    (plugin / "requirements.txt").write_text("", encoding="utf-8")
    (plugin / "configs" / "train.yaml").write_text("{}\n", encoding="utf-8")
    (plugin / "configs" / "smoke.yaml").write_text("{}\n", encoding="utf-8")
    (plugin / "docs" / "README.md").write_text("# Fake\n", encoding="utf-8")
    (plugin / "tests" / "test_plugin.py").write_text("def test_ok(): pass\n", encoding="utf-8")
    return root
