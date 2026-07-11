from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest

import sat_rs_vlm.models.hf_vlm_engine as hf_vlm_engine
from sat_rs_vlm.models.hf_vlm_engine import TORCH_LOAD_MESSAGE, HuggingFaceVLMEngine


class FakeBatch(dict[str, Any]):
    """模拟 transformers BatchFeature，并记录输入被移动到的设备。"""

    moved_to: object | None = None

    def to(self, device: object) -> FakeBatch:
        self.moved_to = device
        return self


class FakeProcessor:
    """记录 chat template 和解码参数的 processor 替身。"""

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] | None = None
        self.template_kwargs: dict[str, Any] = {}
        self.decoded_ids: Any = None
        self.decode_kwargs: dict[str, Any] = {}
        self.batch = FakeBatch(input_ids=[[1, 2, 3]], pixel_values="pixels")

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> FakeBatch:
        self.messages = messages
        self.template_kwargs = kwargs
        return self.batch

    def batch_decode(self, token_ids: Any, **kwargs: Any) -> list[str]:
        self.decoded_ids = token_ids
        self.decode_kwargs = kwargs
        return ["  遥感图像回答。  "]


class FakeProcessorLoader:
    """模拟 AutoProcessor.from_pretrained。"""

    processor = FakeProcessor()
    model_id: str | None = None
    kwargs: dict[str, Any] = {}

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: Any) -> FakeProcessor:
        cls.processor = FakeProcessor()
        cls.model_id = model_id
        cls.kwargs = kwargs
        return cls.processor


class FakeModel:
    """模拟已加载的视觉语言模型。"""

    device = "cuda:0"

    def __init__(self) -> None:
        self.eval_called = False
        self.to_calls: list[object] = []
        self.generate_kwargs: dict[str, Any] = {}

    def eval(self) -> None:
        self.eval_called = True

    def to(self, device: object) -> FakeModel:
        self.to_calls.append(device)
        self.device = str(device)
        return self

    def generate(self, **kwargs: Any) -> list[list[int]]:
        self.generate_kwargs = kwargs
        return [[1, 2, 3, 9, 10]]


class FakeQwenModelLoader:
    """模拟 Qwen3VLForConditionalGeneration.from_pretrained。"""

    model = FakeModel()
    model_id: str | None = None
    kwargs: dict[str, Any] = {}

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: Any) -> FakeModel:
        cls.model = FakeModel()
        cls.model_id = model_id
        cls.kwargs = kwargs
        return cls.model


class FakeAutoModelLoader(FakeQwenModelLoader):
    """模拟 AutoModelForImageTextToText 回退类。"""


def fake_torch() -> SimpleNamespace:
    """返回满足引擎初始化和 inference_mode 的最小 torch 替身。"""

    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        float16="float16",
        bfloat16="bfloat16",
        inference_mode=lambda: nullcontext(),
    )


def install_fake_modules(
    monkeypatch: pytest.MonkeyPatch,
    transformers: object,
) -> None:
    """让动态导入返回测试替身，避免下载或加载真实模型。"""

    modules = {
        "torch": fake_torch(),
        "transformers": transformers,
        "PIL.Image": SimpleNamespace(),
    }

    def fake_import_module(name: str, package: str | None = None) -> object:
        del package
        return modules[name]

    monkeypatch.setattr(hf_vlm_engine.importlib, "import_module", fake_import_module)


def test_engine_prefers_qwen3vl_and_uses_auto_device_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transformers = SimpleNamespace(
        AutoProcessor=FakeProcessorLoader,
        Qwen3VLForConditionalGeneration=FakeQwenModelLoader,
        AutoModelForImageTextToText=FakeAutoModelLoader,
    )
    install_fake_modules(monkeypatch, transformers)

    engine = HuggingFaceVLMEngine("local/qwen3-vl", device="auto", dtype="auto")

    assert FakeQwenModelLoader.model_id == "local/qwen3-vl"
    assert FakeQwenModelLoader.kwargs["device_map"] == "auto"
    assert FakeQwenModelLoader.kwargs["dtype"] == "auto"
    assert FakeQwenModelLoader.model.to_calls == []
    assert engine.device == "cuda:0"


def test_engine_falls_back_to_image_text_auto_model(monkeypatch: pytest.MonkeyPatch) -> None:
    transformers = SimpleNamespace(
        AutoProcessor=FakeProcessorLoader,
        AutoModelForImageTextToText=FakeAutoModelLoader,
    )
    install_fake_modules(monkeypatch, transformers)

    HuggingFaceVLMEngine("local/generic-vlm", device="cpu", dtype="float16")

    assert FakeAutoModelLoader.model_id == "local/generic-vlm"
    assert FakeAutoModelLoader.kwargs["dtype"] == "float16"
    assert "device_map" not in FakeAutoModelLoader.kwargs
    assert FakeAutoModelLoader.model.to_calls == ["cpu"]


def test_generate_uses_multimodal_chat_template_and_decodes_only_new_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transformers = SimpleNamespace(
        AutoProcessor=FakeProcessorLoader,
        Qwen3VLForConditionalGeneration=FakeQwenModelLoader,
    )
    install_fake_modules(monkeypatch, transformers)
    engine = HuggingFaceVLMEngine("local/qwen3-vl")
    before = object()
    after = object()

    answer = engine._generate("请描述变化。", [before, after])

    processor = FakeProcessorLoader.processor
    assert answer == "遥感图像回答。"
    assert processor.messages == [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": before},
                {"type": "image", "image": after},
                {"type": "text", "text": "请描述变化。"},
            ],
        }
    ]
    assert processor.template_kwargs == {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    assert processor.batch.moved_to == "cuda:0"
    assert processor.decoded_ids == [[9, 10]]
    assert processor.decode_kwargs == {
        "skip_special_tokens": True,
        "clean_up_tokenization_spaces": False,
    }


def test_engine_reports_pytorch_dll_load_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_import_module(name: str, package: str | None = None) -> object:
        del package
        if name == "torch":
            raise OSError("Error loading c10.dll")
        raise AssertionError(f"Unexpected import: {name}")

    monkeypatch.setattr(hf_vlm_engine.importlib, "import_module", fake_import_module)

    with pytest.raises(RuntimeError, match="PyTorch is installed but cannot be loaded"):
        HuggingFaceVLMEngine("local/qwen3-vl")
    assert "pytorch.org/get-started/locally" in TORCH_LOAD_MESSAGE
