"""HuggingFace 多模态模型引擎。

算法/流程：
    1. 仅在用户选择 huggingface 后端时动态导入 torch、transformers、PIL。
    2. 从配置读取 model_id/device/dtype 等参数，不在代码中硬编码模型 ID。
    3. 优先用 Qwen3VLForConditionalGeneration，并兼容通用多模态 AutoModel。
    4. 将单图或双图组织为多模态 messages，通过 chat template 插入视觉 token。
    5. 调用 generate 后裁掉输入 token，只解码模型新生成的回答。
    6. 将生成文本收敛为统一 InferenceResult，无法解析结构化检测框时至少返回 answer。

注意：
    不同 VLM 的 processor 调用细节可能不同。本实现覆盖遵循 HuggingFace 多模态聊天
    模板协议的模型，并优先适配 Qwen3-VL；其他协议应增加独立输入适配器。
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from sat_rs_vlm.domain.entities import RemoteSensingInput
from sat_rs_vlm.domain.result import InferenceResult
from sat_rs_vlm.domain.tasks import TaskType

MODEL_EXTRA_MESSAGE = 'HuggingFace model dependencies are missing. Run: pip install -e ".[model]"'
TORCH_LOAD_MESSAGE = (
    "PyTorch is installed but cannot be loaded. On Windows this usually means the wheel or "
    "one of its DLL dependencies is incompatible. Reinstall a matching PyTorch build from "
    "https://pytorch.org/get-started/locally/ and verify with: "
    'python -c "import torch; print(torch.__version__, torch.version.cuda)"'
)
MODEL_CLASS_NAMES = (
    "Qwen3VLForConditionalGeneration",
    "AutoModelForImageTextToText",
    "AutoModelForVision2Seq",
)


class HuggingFaceVLMEngine:
    """基于 transformers 的真实 VLM 推理引擎。

    参数：
        model_id：HuggingFace 模型 ID 或本地模型目录。
        device：运行设备；auto 使用 accelerate 自动分配模型。
        dtype：torch dtype 名称；auto 使用模型推荐精度。
        max_new_tokens：生成最大新 token 数。
        trust_remote_code：是否信任远程模型代码。
        local_files_only：是否只使用本地缓存文件。
    """

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 256,
        trust_remote_code: bool = True,
        local_files_only: bool = False,
    ) -> None:
        """初始化 HuggingFace 模型和处理器。

        异常：
            ValueError：model_id 为空或 dtype 不支持。
            ImportError：缺少 `[model]` 依赖或 transformers 不支持多模态模型时抛出。
            RuntimeError：PyTorch 已安装但 DLL/二进制无法加载时抛出。
        """

        if not model_id:
            raise ValueError(
                "HuggingFace backend requires model_id. Set model.model_id in config or pass "
                "--model-id <id>."
            )

        try:
            self._torch = importlib.import_module("torch")
        except ModuleNotFoundError as exc:
            raise ImportError(MODEL_EXTRA_MESSAGE) from exc
        except OSError as exc:
            raise RuntimeError(f"{TORCH_LOAD_MESSAGE}\nOriginal error: {exc}") from exc

        try:
            transformers = importlib.import_module("transformers")
            self._image_module = importlib.import_module("PIL.Image")
        except ModuleNotFoundError as exc:
            raise ImportError(MODEL_EXTRA_MESSAGE) from exc

        self.model_id = model_id
        self.device = self._resolve_device(device)
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens

        processor_cls = transformers.AutoProcessor
        model_cls = self._resolve_model_class(transformers)
        self._model_class_name = str(getattr(model_cls, "__name__", type(model_cls).__name__))
        processor_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only,
        }
        model_kwargs = dict(processor_kwargs)
        torch_dtype = self._resolve_dtype(dtype)
        model_kwargs["dtype"] = torch_dtype if torch_dtype is not None else "auto"
        if device == "auto":
            model_kwargs["device_map"] = "auto"

        self._processor = processor_cls.from_pretrained(model_id, **processor_kwargs)
        self._model = model_cls.from_pretrained(model_id, **model_kwargs)
        if device != "auto" and hasattr(self._model, "to"):
            self._model = self._model.to(self.device)
        if hasattr(self._model, "eval"):
            self._model.eval()
        model_device = getattr(self._model, "device", None)
        if model_device is not None:
            self.device = str(model_device)

    @staticmethod
    def _resolve_model_class(transformers: Any) -> Any:
        """选择支持视觉语言条件生成的模型类。

        参数：
            transformers：动态导入的 transformers 模块。

        返回值：
            Any：Qwen3-VL 专用类或兼容的多模态 AutoModel 类。

        异常：
            ImportError：当前 transformers 没有任何兼容多模态模型类时抛出。
        """

        for class_name in MODEL_CLASS_NAMES:
            model_cls = getattr(transformers, class_name, None)
            if model_cls is not None:
                return model_cls
        raise ImportError(
            "Transformers does not provide a compatible vision-language model class. "
            "Expected one of: "
            + ", ".join(MODEL_CLASS_NAMES)
            + '. Upgrade with: pip install -e ".[model]"'
        )

    def infer(self, input_data: RemoteSensingInput) -> InferenceResult:
        """执行真实模型生成式推理。

        参数：
            input_data：遥感推理输入。若 second_image_path 存在，会同时读取双图。

        返回值：
            InferenceResult：answer 为模型生成文本，raw_output 保存后端和 prompt 信息。
        """

        images = [self._open_image(input_data.image_path)]
        if input_data.second_image_path:
            images.append(self._open_image(input_data.second_image_path))

        prompt = self._build_prompt(input_data)
        generated = self._generate(prompt=prompt, images=images)
        return InferenceResult(
            task_type=input_data.task_type,
            answer=generated,
            confidence=None,
            raw_output={
                "engine": "huggingface",
                "model_id": self.model_id,
                "model_class": self._model_class_name,
                "device": self.device,
                "prompt": prompt,
            },
        )

    def _resolve_device(self, device: str) -> str:
        """解析运行设备。

        参数：
            device：配置设备字符串。

        返回值：
            str：cuda 或 cpu 等 torch 设备字符串。
        """

        if device == "auto":
            return "cuda" if bool(self._torch.cuda.is_available()) else "cpu"
        return device

    def _resolve_dtype(self, dtype: str) -> Any | None:
        """解析 torch dtype。

        参数：
            dtype：auto 或 torch dtype 属性名。

        返回值：
            Any | None：torch dtype 对象；auto 返回 None。
        """

        if dtype == "auto":
            return None
        if not hasattr(self._torch, dtype):
            raise ValueError(f"Unsupported torch dtype: {dtype}")
        return getattr(self._torch, dtype)

    def _open_image(self, image_path: str) -> Any:
        """读取图像并转为 RGB。

        参数：
            image_path：图像文件路径。

        返回值：
            PIL.Image.Image：与源文件句柄解耦的 RGB 图像对象。

        异常：
            FileNotFoundError：图像不存在时抛出。
        """

        path = Path(image_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Image file does not exist: {image_path}")
        with self._image_module.open(path) as image:
            return image.convert("RGB")

    def _build_prompt(self, input_data: RemoteSensingInput) -> str:
        """构造面向 VLM 的任务提示词。

        参数：
            input_data：包含原始 prompt、任务类型和单双图路径的输入对象。

        返回值：
            str：包含任务提示和用户指令的组合 prompt，不向模型暴露本地文件路径。
        """

        task_hint = {
            TaskType.DETECTION: "Return object locations if the model can infer them.",
            TaskType.COUNTING: "Return the estimated count.",
            TaskType.CHANGE_DETECTION: "Compare the before and after images and summarize changes.",
            TaskType.CAPTIONING: "Describe the remote-sensing image.",
            TaskType.SCENE_CLASSIFICATION: "Classify the remote-sensing scene.",
            TaskType.SEGMENTATION: "Describe likely semantic regions.",
            TaskType.VQA: "Answer the remote-sensing question.",
            TaskType.UNKNOWN: "Interpret the remote-sensing image.",
        }[input_data.task_type]
        if input_data.second_image_path:
            return (
                f"{task_hint}\nThe first image is before and the second is after.\n"
                f"{input_data.prompt}"
            )
        return f"{task_hint}\n{input_data.prompt}"

    @staticmethod
    def _build_messages(prompt: str, images: list[Any]) -> list[dict[str, Any]]:
        """构造 HuggingFace 多模态聊天消息。

        参数：
            prompt：已经加入任务提示的文本。
            images：按语义顺序排列的 PIL 图像；变化检测中先 before 后 after。

        返回值：
            list[dict[str, Any]]：可传给 processor.apply_chat_template 的 messages。
        """

        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def _generate(self, prompt: str, images: list[Any]) -> str:
        """调用 transformers generate 完成文本生成。

        参数：
            prompt：组合后的模型输入文本。
            images：PIL 图像列表，单图或双图。

        返回值：
            str：仅包含模型新增 token 的解码文本。
        """

        apply_chat_template = getattr(self._processor, "apply_chat_template", None)
        if apply_chat_template is None:
            raise RuntimeError(
                "The selected processor does not support multimodal chat templates. "
                "Use a Qwen3-VL compatible processor or add a model-specific input adapter."
            )
        messages = self._build_messages(prompt, images)
        encoded = apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        if hasattr(encoded, "to"):
            model_device = getattr(self._model, "device", self.device)
            encoded = encoded.to(model_device)
        with self._torch.inference_mode():
            output_ids = self._model.generate(**encoded, max_new_tokens=self.max_new_tokens)
        if hasattr(self._processor, "batch_decode"):
            generated_ids = [
                output[len(input_row) :]
                for input_row, output in zip(input_ids, output_ids, strict=True)
            ]
            decoded = self._processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return str(decoded[0]).strip() if decoded else ""
        return str(output_ids)
