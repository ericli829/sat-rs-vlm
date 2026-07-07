import importlib

import pytest

from sat_rs_vlm.infrastructure.config import ModelConfig
from sat_rs_vlm.models.hf_vlm_engine import MODEL_EXTRA_MESSAGE
from sat_rs_vlm.models.mock_model import MockVLMEngine
from sat_rs_vlm.models.model_factory import create_vlm_engine


def test_create_mock_backend() -> None:
    engine = create_vlm_engine(ModelConfig(backend="mock"))
    assert isinstance(engine, MockVLMEngine)


def test_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported model backend"):
        create_vlm_engine(ModelConfig(backend="unknown"))


def test_huggingface_backend_missing_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    import sat_rs_vlm.models.hf_vlm_engine as hf_vlm_engine

    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None) -> object:
        if name == "torch":
            raise ModuleNotFoundError("No module named 'torch'")
        return real_import_module(name, package)

    monkeypatch.setattr(hf_vlm_engine.importlib, "import_module", fake_import_module)
    with pytest.raises(ImportError, match=r"pip install -e"):
        create_vlm_engine(ModelConfig(backend="huggingface", model_id="dummy/model"))
    assert MODEL_EXTRA_MESSAGE
