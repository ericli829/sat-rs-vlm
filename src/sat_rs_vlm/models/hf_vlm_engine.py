"""HuggingFace 多模态模型引擎。

算法/流程：
    1. 仅在用户选择 huggingface 后端时动态导入 torch、transformers、PIL。
    2. 从配置读取 model_id/device/dtype 等参数，不在代码中硬编码模型 ID。
    3. 用 AutoProcessor 和 AutoModelForVision2Seq/AutoModelForCausalLM 加载模型。
    4. 将图像和 prompt 编码后调用 generate。
    5. 将生成文本收敛为统一 InferenceResult，无法解析结构化检测框时至少返回 answer。

注意：
    不同 VLM 的 processor 调用细节可能不同，本实现是保守通用入口，后续可按
    具体模型在该模块内扩展适配器。
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from sat_rs_vlm.domain.entities import RemoteSensingInput
from sat_rs_vlm.domain.result import InferenceResult
from sat_rs_vlm.domain.tasks import TaskType

MODEL_EXTRA_MESSAGE = 'HuggingFace model dependencies are missing. Run: pip install -e ".[model]"'


class HuggingFaceVLMEngine:
    """基于 transformers 的真实 VLM 推理引擎。

    参数：
        model_id：HuggingFace 模型 ID 或本地模型目录。
        device：运行设备，auto 时优先 CUDA，否则 CPU。
        dtype：torch dtype 名称，auto 时交由 transformers 默认处理。
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
            ImportError：缺少 `[model]` 可选依赖时抛出，并提示安装命令。
        """

        if not model_id:
            raise ValueError(
                "HuggingFace backend requires model_id. Set model.model_id in config or pass "
                "--model-id <id>."
            )

        try:
            self._torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
            self._image_module = importlib.import_module("PIL.Image")
        except ModuleNotFoundError as exc:
            raise ImportError(MODEL_EXTRA_MESSAGE) from exc

        self.model_id = model_id
        self.device = self._resolve_device(device)
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens

        processor_cls = transformers.AutoProcessor
        model_cls = getattr(transformers, "AutoModelForVision2Seq", None)
        if model_cls is None:
            model_cls = transformers.AutoModelForCausalLM
        load_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only,
        }
        torch_dtype = self._resolve_dtype(dtype)
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype

        self._processor = processor_cls.from_pretrained(model_id, **load_kwargs)
        self._model = model_cls.from_pretrained(model_id, **load_kwargs)
        if hasattr(self._model, "to"):
            self._model = self._model.to(self.device)
        if hasattr(self._model, "eval"):
            self._model.eval()

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
            PIL.Image.Image：RGB 图像对象。

        异常：
            FileNotFoundError：图像不存在时抛出。
        """

        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file does not exist: {image_path}")
        return self._image_module.open(path).convert("RGB")

    def _build_prompt(self, input_data: RemoteSensingInput) -> str:
        """构造面向 VLM 的任务提示词。

        参数：
            input_data：包含原始 prompt、任务类型和单双图路径的输入对象。

        返回值：
            str：包含任务提示、图像路径和用户指令的组合 prompt。
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
                f"{task_hint}\nBefore image: {input_data.image_path}\n"
                f"After image: {input_data.second_image_path}\nUser prompt: {input_data.prompt}"
            )
        return f"{task_hint}\nImage: {input_data.image_path}\nUser prompt: {input_data.prompt}"

    def _generate(self, prompt: str, images: list[Any]) -> str:
        """调用 transformers generate 完成文本生成。

        参数：
            prompt：组合后的模型输入文本。
            images：PIL 图像列表，单图或双图。

        返回值：
            str：解码后的模型生成文本。
        """

        encoded = self._processor(text=prompt, images=images, return_tensors="pt")
        if hasattr(encoded, "to"):
            encoded = encoded.to(self.device)
        with self._torch.inference_mode():
            output_ids = self._model.generate(**encoded, max_new_tokens=self.max_new_tokens)
        if hasattr(self._processor, "batch_decode"):
            decoded = self._processor.batch_decode(output_ids, skip_special_tokens=True)
            return str(decoded[0]).strip() if decoded else ""
        return str(output_ids)
