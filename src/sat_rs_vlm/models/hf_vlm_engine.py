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

import copy
import importlib
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
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


@dataclass
class CachedGenerationSession:
    """Opaque, model-bound KV state retained between reasoning and constrained choice."""

    session_id: str
    model_identity: str
    model_id: str
    reasoning_text: str
    initial_prefill_tokens: int
    reasoning_tokens: int
    latency_ms: dict[str, float | None]
    metadata: dict[str, Any] = field(default_factory=dict)
    closed: bool = False
    _past_key_values: Any = field(default=None, repr=False)
    _sequence_ids: Any = field(default=None, repr=False)
    _attention_mask: Any = field(default=None, repr=False)
    _rope_deltas: Any = field(default=None, repr=False)
    _release_callback: Callable[[str], None] | None = field(default=None, repr=False)

    def close(self) -> None:
        if self.closed:
            return
        self._past_key_values = None
        self._sequence_ids = None
        self._attention_mask = None
        self._rope_deltas = None
        self.closed = True
        if self._release_callback is not None:
            self._release_callback(self.session_id)
            self._release_callback = None

    release = close

    def __enter__(self) -> CachedGenerationSession:
        if self.closed:
            raise RuntimeError("cached generation session is closed")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@dataclass(frozen=True)
class CachedReasoningResult:
    reasoning_text: str
    session: CachedGenerationSession
    latency_ms: dict[str, float | None]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CachedChoiceEngineResult:
    selected_ids: tuple[str, ...]
    scores: dict[str, float]
    answer_type: str
    reasoning_text: str
    method: str
    cache_reused: bool
    latency_ms: dict[str, float | None]
    metadata: dict[str, Any] = field(default_factory=dict)


class HuggingFaceVLMEngine:
    """基于 transformers 的真实 VLM 推理引擎。

    参数：
        model_id：HuggingFace 模型 ID 或本地模型目录。
        adapter_path：可选的本地 PEFT/LoRA adapter 目录。
        device：运行设备；auto 使用 accelerate 自动分配模型。
        dtype：torch dtype 名称；auto 使用模型推荐精度。
        max_new_tokens：生成最大新 token 数。
        trust_remote_code：是否信任远程模型代码。
        local_files_only：是否只使用本地缓存文件。
    """

    def __init__(
        self,
        model_id: str,
        adapter_path: str | None = None,
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
        self.adapter_path = adapter_path
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

        if adapter_path:
            try:
                peft = importlib.import_module("peft")
            except ModuleNotFoundError as exc:
                raise ImportError(MODEL_EXTRA_MESSAGE) from exc
            from sat_rs_vlm.models.qwen3vl_loader import load_qwen3vl

            self._model, self._processor = load_qwen3vl(
                modules={"torch": self._torch, "transformers": transformers, "peft": peft},
                base_model=model_id,
                processor_source=model_id,
                model_kwargs=model_kwargs,
                processor_kwargs=processor_kwargs,
                adapter_path=adapter_path,
            )
        else:
            self._processor = processor_cls.from_pretrained(model_id, **processor_kwargs)
            self._model = model_cls.from_pretrained(model_id, **model_kwargs)
        if device != "auto" and hasattr(self._model, "to"):
            self._model = self._model.to(self.device)
        if hasattr(self._model, "eval"):
            self._model.eval()
        model_device = getattr(self._model, "device", None)
        if model_device is not None:
            self.device = str(model_device)
        adapter_identity = self.adapter_path or "none"
        self._model_identity = (
            f"{self.model_id}:{self._model_class_name}:{adapter_identity}:{id(self._model)}"
        )
        self._active_sessions: dict[str, CachedGenerationSession] = {}

    @property
    def model_identity(self) -> str:
        return self._model_identity

    @property
    def active_session_count(self) -> int:
        return len(self._active_sessions)

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

    def generate_text(
        self,
        prompt: str,
        image_paths: list[Any],
        *,
        allowed_outputs: tuple[str, ...] | None = None,
    ) -> str:
        """Generate from arbitrary multi-image or text-only input.

        This public boundary lets higher-level runtimes reuse the existing Qwen
        processor/model/generation stack without constructing another transformers
        loader.  An empty image list is intentionally supported for authoritative
        structured results such as detector counts.
        """

        images = [self._open_image(path) for path in image_paths]
        return self._generate(prompt=prompt, images=images, allowed_outputs=allowed_outputs)

    def _release_session(self, session_id: str) -> None:
        self._active_sessions.pop(session_id, None)

    def _rope_state_holder(self) -> Any | None:
        pending = [self._model]
        visited: set[int] = set()
        while pending:
            candidate = pending.pop(0)
            identity = id(candidate)
            if identity in visited:
                continue
            visited.add(identity)
            if hasattr(candidate, "rope_deltas"):
                return candidate
            for attribute in ("model", "base_model"):
                nested = getattr(candidate, attribute, None)
                if nested is not None and id(nested) not in visited:
                    pending.append(nested)
        return None

    def _restore_rope_state(self, session: CachedGenerationSession) -> None:
        if session._rope_deltas is None:
            return
        holder = self._rope_state_holder()
        if holder is not None:
            holder.rope_deltas = session._rope_deltas

    def _cached_position_ids(
        self,
        session: CachedGenerationSession,
        attention_mask: Any,
        next_sequence_length: int,
    ) -> Any:
        """Build query-length Qwen M-RoPE positions for an external cache continuation."""

        rope_deltas = session._rope_deltas
        if rope_deltas is None:
            return None
        batch_size = int(attention_mask.shape[0])
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids = position_ids.masked_fill(attention_mask == 0, 0)
        position_ids = position_ids[..., -next_sequence_length:]
        position_ids = position_ids.view(1, batch_size, -1).repeat(3, 1, 1)
        delta_batch = int(rope_deltas.shape[0])
        if delta_batch < 1 or batch_size % delta_batch != 0:
            raise RuntimeError("cached Qwen rope_deltas batch does not match the continuation")
        deltas = rope_deltas.repeat_interleave(batch_size // delta_batch, dim=0)
        return position_ids + deltas.to(device=position_ids.device)

    def _peak_vram_mb(self) -> float | None:
        cuda = getattr(self._torch, "cuda", None)
        if cuda is None or not bool(cuda.is_available()):
            return None
        try:
            return float(cuda.max_memory_allocated()) / (1024.0 * 1024.0)
        except (AttributeError, RuntimeError):
            return None

    def reason_with_cache(
        self,
        prompt: str,
        image_paths: list[Any],
        *,
        max_new_tokens: int | None = None,
    ) -> CachedReasoningResult:
        """Generate free reasoning once and retain its actual Transformers KV cache."""

        images = [self._open_image(path) for path in image_paths]
        apply_chat_template = getattr(self._processor, "apply_chat_template", None)
        if apply_chat_template is None:
            raise RuntimeError("The selected processor does not support multimodal chat templates.")
        encoded = apply_chat_template(
            self._build_messages(prompt, images),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        if hasattr(encoded, "to"):
            model_device = getattr(self._model, "device", self.device)
            encoded = encoded.to(model_device)
        input_ids = encoded["input_ids"]
        initial_prefill_tokens = int(input_ids.shape[-1])
        started = time.perf_counter()
        with self._torch.inference_mode():
            generated = self._model.generate(
                **encoded,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=False,
                use_cache=True,
                return_dict_in_generate=True,
            )
        reasoning_generate_ms = (time.perf_counter() - started) * 1000.0
        cache = getattr(generated, "past_key_values", None)
        sequences = getattr(generated, "sequences", None)
        if cache is None or sequences is None:
            raise RuntimeError(
                "cached reasoning requires generate() to return sequences and past_key_values"
            )
        terminal_ids: set[int] = set()
        generation_config = getattr(self._model, "generation_config", None)
        eos_token_id = getattr(generation_config, "eos_token_id", None)
        if isinstance(eos_token_id, int):
            terminal_ids.add(eos_token_id)
        elif eos_token_id is not None:
            terminal_ids.update(int(value) for value in eos_token_id)
        tokenizer = getattr(self._processor, "tokenizer", None)
        tokenizer_eos = getattr(tokenizer, "eos_token_id", None)
        if isinstance(tokenizer_eos, int):
            terminal_ids.add(tokenizer_eos)
        if (
            int(sequences.shape[-1]) > initial_prefill_tokens
            and int(sequences[0, -1].item()) in terminal_ids
        ):
            # Transformers' returned cache excludes the just-selected final token. Drop a
            # terminal EOS so the constrained suffix remains in the same assistant turn.
            sequences = sequences[:, :-1]
        reasoning_ids = sequences[:, initial_prefill_tokens:]
        reasoning_tokens = int(reasoning_ids.shape[-1])
        decoded = self._processor.batch_decode(
            reasoning_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        reasoning_text = str(decoded[0]).strip() if decoded else ""
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            attention_mask = self._torch.ones_like(input_ids)
        if reasoning_tokens:
            attention_mask = self._torch.cat(
                [
                    attention_mask,
                    attention_mask.new_ones((attention_mask.shape[0], reasoning_tokens)),
                ],
                dim=-1,
            )
        rope_holder = self._rope_state_holder()
        rope_deltas = getattr(rope_holder, "rope_deltas", None)
        clone_rope = getattr(rope_deltas, "clone", None)
        if callable(clone_rope):
            rope_deltas = clone_rope()
        session_id = uuid.uuid4().hex
        latency: dict[str, float | None] = {
            "vision_prefill_ms": None,
            "text_prefill_ms": None,
            "total_prefill_ms": None,
            "reasoning_decode_ms": reasoning_generate_ms,
            "reasoning_total_ms": reasoning_generate_ms,
        }
        session = CachedGenerationSession(
            session_id=session_id,
            model_identity=self._model_identity,
            model_id=self.model_id,
            reasoning_text=reasoning_text,
            initial_prefill_tokens=initial_prefill_tokens,
            reasoning_tokens=reasoning_tokens,
            latency_ms=latency,
            metadata={
                "device": self.device,
                "dtype": self.dtype,
                "reasoning_generate_includes_prefill": True,
                "peak_vram_mb": self._peak_vram_mb(),
            },
            _past_key_values=cache,
            _sequence_ids=sequences,
            _attention_mask=attention_mask,
            _rope_deltas=rope_deltas,
            _release_callback=self._release_session,
        )
        self._active_sessions[session_id] = session
        return CachedReasoningResult(
            reasoning_text,
            session,
            dict(latency),
            {
                "initial_prefill_tokens": initial_prefill_tokens,
                "reasoning_tokens": reasoning_tokens,
                **session.metadata,
            },
        )

    def _validate_session(self, session: CachedGenerationSession) -> None:
        if session.closed:
            raise RuntimeError("cached generation session is closed")
        if session.model_identity != self._model_identity:
            raise ValueError("cached generation session belongs to a different model")
        if session.session_id not in self._active_sessions:
            raise RuntimeError("cached generation session is not active")

    def _token_ids(self, text: str) -> list[int]:
        tokenizer = getattr(self._processor, "tokenizer", self._processor)
        encode = getattr(tokenizer, "encode", None)
        if encode is None:
            raise RuntimeError("choice scoring requires a processor tokenizer.encode API")
        values = encode(text, add_special_tokens=False)
        if hasattr(values, "tolist"):
            values = values.tolist()
        ids = [int(value) for value in values]
        if not ids:
            raise ValueError(f"choice continuation tokenized to an empty sequence: {text!r}")
        return ids

    @staticmethod
    def _continuation(prefix: str, label: str) -> str:
        return label if prefix[-1:].isspace() else " " + label

    def _prepare_cached_prefix(
        self,
        session: CachedGenerationSession,
        suffix: str,
        *,
        fork_cache: bool,
    ) -> tuple[Any, Any, Any, int, int, float, float, float]:
        tokenize_started = time.perf_counter()
        suffix_ids = self._token_ids(suffix)
        suffix_tokenize_ms = (time.perf_counter() - tokenize_started) * 1000.0
        sequence_ids = session._sequence_ids
        if sequence_ids is None:
            raise RuntimeError("cached generation session has no sequence state")
        suffix_tensor = self._torch.tensor(
            [suffix_ids], dtype=sequence_ids.dtype, device=sequence_ids.device
        )
        full_ids = self._torch.cat([sequence_ids, suffix_tensor], dim=-1)
        attention_mask = self._torch.cat(
            [
                session._attention_mask,
                session._attention_mask.new_ones(
                    (session._attention_mask.shape[0], len(suffix_ids))
                ),
            ],
            dim=-1,
        )
        clone_started = time.perf_counter()
        cache = copy.deepcopy(session._past_key_values) if fork_cache else session._past_key_values
        cache_clone_ms = (time.perf_counter() - clone_started) * 1000.0
        past_length = int(cache.get_seq_length())
        next_sequence_length = int(full_ids.shape[-1]) - past_length
        if next_sequence_length < 1:
            raise RuntimeError("cached prefix has no unprocessed continuation tokens")
        self._restore_rope_state(session)
        position_ids = self._cached_position_ids(
            session,
            attention_mask,
            next_sequence_length,
        )
        prepared = self._model.prepare_inputs_for_generation(
            full_ids,
            next_sequence_length=next_sequence_length,
            past_key_values=cache,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            is_first_iteration=False,
        )
        started = time.perf_counter()
        with self._torch.inference_mode():
            outputs = self._model(**prepared, return_dict=True)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return (
            outputs.logits[:, -1, :],
            outputs.past_key_values,
            attention_mask,
            int(full_ids.shape[-1]),
            len(suffix_ids),
            suffix_tokenize_ms,
            cache_clone_ms,
            elapsed_ms,
        )

    def _score_continuations(
        self,
        session: CachedGenerationSession,
        *,
        prefix: str,
        continuations: tuple[str, ...],
        fork_reasoning_cache: bool,
    ) -> tuple[dict[str, float], str, int, int, float, float, float, float]:
        (
            base_logits,
            base_cache,
            base_attention_mask,
            _,
            suffix_tokens,
            suffix_tokenize_ms,
            cache_clone_ms,
            suffix_ms,
        ) = self._prepare_cached_prefix(session, prefix, fork_cache=fork_reasoning_cache)
        continuation_tokenize_started = time.perf_counter()
        tokenizations = {
            label: self._token_ids(self._continuation(prefix, label)) for label in continuations
        }
        suffix_tokenize_ms += (time.perf_counter() - continuation_tokenize_started) * 1000.0
        scored_tokens = sum(len(values) for values in tokenizations.values())
        scoring_ms = 0.0
        if all(len(values) == 1 for values in tokenizations.values()):
            scoring_started = time.perf_counter()
            scores = {
                label: float(base_logits[0, values[0]].float().item())
                for label, values in tokenizations.items()
            }
            scoring_ms = (time.perf_counter() - scoring_started) * 1000.0
            method = "single_token_logits"
        else:
            scores = {}
            length_normalize = len({len(values) for values in tokenizations.values()}) > 1
            for label, values in tokenizations.items():
                candidate_clone_started = time.perf_counter()
                candidate_cache = copy.deepcopy(base_cache)
                cache_clone_ms += (time.perf_counter() - candidate_clone_started) * 1000.0
                candidate_scoring_started = time.perf_counter()
                log_probs = self._torch.log_softmax(base_logits.float(), dim=-1)
                score = float(log_probs[0, values[0]].item())
                candidate_attention = base_attention_mask
                for index in range(1, len(values)):
                    previous = self._torch.tensor(
                        [[values[index - 1]]],
                        dtype=session._sequence_ids.dtype,
                        device=session._sequence_ids.device,
                    )
                    candidate_attention = self._torch.cat(
                        [
                            candidate_attention,
                            candidate_attention.new_ones((candidate_attention.shape[0], 1)),
                        ],
                        dim=-1,
                    )
                    self._restore_rope_state(session)
                    position_ids = self._cached_position_ids(session, candidate_attention, 1)
                    with self._torch.inference_mode():
                        outputs = self._model(
                            input_ids=previous,
                            attention_mask=candidate_attention,
                            past_key_values=candidate_cache,
                            position_ids=position_ids,
                            use_cache=True,
                            return_dict=True,
                        )
                    candidate_cache = outputs.past_key_values
                    log_probs = self._torch.log_softmax(outputs.logits[:, -1, :].float(), dim=-1)
                    score += float(log_probs[0, values[index]].item())
                scores[label] = score / len(values) if length_normalize else score
                scoring_ms += (time.perf_counter() - candidate_scoring_started) * 1000.0
            method = "multi_token_continuation_logprob"
        return (
            scores,
            method,
            suffix_tokens,
            scored_tokens,
            suffix_tokenize_ms,
            cache_clone_ms,
            suffix_ms,
            scoring_ms,
        )

    def score_choice_from_cache(
        self,
        session: CachedGenerationSession,
        *,
        single_choice_suffix: str,
        multi_verify_template: str,
        choice_ids: tuple[str, ...],
        option_texts: tuple[str, ...],
        answer_type: str,
        multi_select_threshold: float = 0.0,
    ) -> CachedChoiceEngineResult:
        """Score legal choices from a model-bound reasoning cache without visual replay."""

        choice_started = time.perf_counter()
        self._validate_session(session)
        if answer_type not in {"CHOICE_SINGLE", "CHOICE_MULTI"}:
            raise ValueError("answer_type must be CHOICE_SINGLE or CHOICE_MULTI")
        if not choice_ids or len(choice_ids) != len(option_texts):
            raise ValueError("choice ids and option texts must be non-empty and aligned")
        if len(choice_ids) != len(set(choice_ids)):
            raise ValueError("choice ids must be unique")
        suffix_tokens = 0
        scored_tokens = 0
        suffix_tokenize_ms = 0.0
        cache_clone_ms = 0.0
        suffix_ms = 0.0
        scoring_ms = 0.0
        methods: set[str] = set()
        selected_ids: tuple[str, ...]
        if answer_type == "CHOICE_SINGLE":
            (
                scores,
                method,
                current_suffix_tokens,
                current_scored_tokens,
                current_tokenize_ms,
                current_clone_ms,
                current_suffix_ms,
                current_scoring_ms,
            ) = self._score_continuations(
                session,
                prefix=single_choice_suffix,
                continuations=choice_ids,
                fork_reasoning_cache=False,
            )
            methods.add(method)
            suffix_tokens += current_suffix_tokens
            scored_tokens += current_scored_tokens
            suffix_tokenize_ms += current_tokenize_ms
            cache_clone_ms += current_clone_ms
            suffix_ms += current_suffix_ms
            scoring_ms += current_scoring_ms
            selected_ids = (max(choice_ids, key=lambda item: scores[item]),)
            result_method = f"kv_cached_{method}"
        else:
            scores = {}
            for choice_id, option_text in zip(choice_ids, option_texts, strict=True):
                verification_suffix = multi_verify_template.format(
                    choice_id=choice_id,
                    option_text=option_text,
                )
                (
                    binary_scores,
                    method,
                    current_suffix_tokens,
                    current_scored_tokens,
                    current_tokenize_ms,
                    current_clone_ms,
                    current_suffix_ms,
                    current_scoring_ms,
                ) = self._score_continuations(
                    session,
                    prefix=verification_suffix,
                    continuations=("YES", "NO"),
                    fork_reasoning_cache=True,
                )
                methods.add(method)
                scores[choice_id] = binary_scores["YES"] - binary_scores["NO"]
                suffix_tokens += current_suffix_tokens
                scored_tokens += current_scored_tokens
                suffix_tokenize_ms += current_tokenize_ms
                cache_clone_ms += current_clone_ms
                suffix_ms += current_suffix_ms
                scoring_ms += current_scoring_ms
            selected_ids = tuple(
                choice_id for choice_id in choice_ids if scores[choice_id] > multi_select_threshold
            )
            result_method = "kv_cached_binary_verification"
        peak_vram_mb = self._peak_vram_mb()
        choice_total_ms = (time.perf_counter() - choice_started) * 1000.0
        reasoning_total_ms = float(
            session.latency_ms.get("reasoning_total_ms")
            or session.latency_ms.get("reasoning_decode_ms")
            or 0.0
        )
        total_ms = reasoning_total_ms + choice_total_ms
        latency = {
            **session.latency_ms,
            "cache_clone_ms": cache_clone_ms,
            "suffix_tokenize_ms": suffix_tokenize_ms,
            "choice_suffix_prefill_ms": suffix_ms,
            "choice_scoring_ms": scoring_ms,
            "choice_total_ms": choice_total_ms,
            "total_ms": total_ms,
        }
        return CachedChoiceEngineResult(
            selected_ids=selected_ids,
            scores=scores,
            answer_type=answer_type,
            reasoning_text=session.reasoning_text,
            method=result_method,
            cache_reused=True,
            latency_ms=latency,
            metadata={
                "initial_prefill_tokens": session.initial_prefill_tokens,
                "reasoning_tokens": session.reasoning_tokens,
                "choice_suffix_tokens": suffix_tokens,
                "choice_scored_tokens": scored_tokens,
                "continuation_methods": sorted(methods),
                "reasoning_cache_mode": (
                    "consume_in_place" if answer_type == "CHOICE_SINGLE" else "fork_per_option"
                ),
                "multi_select_threshold": multi_select_threshold,
                "model_id": self.model_id,
                "model_identity": self._model_identity,
                "device": self.device,
                "dtype": self.dtype,
                "peak_vram_mb": peak_vram_mb,
            },
        )

    def reason_and_choose(
        self,
        prompt: str,
        image_paths: list[Any],
        *,
        choice_ids: tuple[str, ...],
        option_texts: tuple[str, ...],
        answer_type: str,
        single_choice_suffix: str,
        multi_verify_template: str,
        multi_select_threshold: float = 0.0,
        reasoning_max_new_tokens: int | None = None,
    ) -> CachedChoiceEngineResult:
        reasoning = self.reason_with_cache(
            prompt,
            image_paths,
            max_new_tokens=reasoning_max_new_tokens,
        )
        try:
            result = self.score_choice_from_cache(
                reasoning.session,
                single_choice_suffix=single_choice_suffix,
                multi_verify_template=multi_verify_template,
                choice_ids=choice_ids,
                option_texts=option_texts,
                answer_type=answer_type,
                multi_select_threshold=multi_select_threshold,
            )
            return replace(
                result,
                metadata={**result.metadata, "session_released": True},
            )
        finally:
            reasoning.session.close()

    def close(self) -> None:
        for session in list(self._active_sessions.values()):
            session.close()

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

    def _open_image(self, image_path: Any) -> Any:
        """读取图像并转为 RGB。

        参数：
            image_path：图像文件路径。

        返回值：
            PIL.Image.Image：与源文件句柄解耦的 RGB 图像对象。

        异常：
            FileNotFoundError：图像不存在时抛出。
        """

        if not isinstance(image_path, (str, Path)):
            convert = getattr(image_path, "convert", None)
            if not callable(convert):
                raise TypeError("image input must be a path or PIL-compatible image")
            converted = convert("RGB")
            copy = getattr(converted, "copy", None)
            return copy() if callable(copy) else converted
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

    def _selection_prefix_constraint(
        self,
        *,
        allowed_outputs: tuple[str, ...],
        prompt_token_count: int,
    ) -> Callable[[int, Any], list[int]]:
        """Return a tokenizer-aware finite-state mask for compatibility fallback only."""

        tokenizer = getattr(self._processor, "tokenizer", self._processor)
        encode = getattr(tokenizer, "encode", None)
        if encode is None:
            raise RuntimeError("selection constrained decoding requires processor.tokenizer.encode")
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is None:
            eos_token_id = getattr(
                getattr(self._model, "generation_config", None), "eos_token_id", None
            )
        if eos_token_id is None:
            raise RuntimeError("selection constrained decoding requires an EOS token id")
        sequences = tuple(
            tuple(int(token) for token in encode(value, add_special_tokens=False))
            for value in allowed_outputs
        )
        if not sequences or any(not sequence for sequence in sequences):
            raise ValueError("allowed selection outputs must tokenize to non-empty sequences")

        def allowed_next_tokens(_: int, input_ids: Any) -> list[int]:
            values = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
            generated = tuple(int(token) for token in values[prompt_token_count:])
            matching = [
                sequence for sequence in sequences if sequence[: len(generated)] == generated
            ]
            next_tokens = {
                sequence[len(generated)] for sequence in matching if len(sequence) > len(generated)
            }
            if any(len(sequence) == len(generated) for sequence in matching):
                next_tokens.add(int(eos_token_id))
            return sorted(next_tokens) or [int(eos_token_id)]

        return allowed_next_tokens

    def _generate(
        self,
        prompt: str,
        images: list[Any],
        *,
        allowed_outputs: tuple[str, ...] | None = None,
    ) -> str:
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
        generation_kwargs: dict[str, Any] = {"max_new_tokens": self.max_new_tokens}
        if allowed_outputs:
            generation_kwargs["prefix_allowed_tokens_fn"] = self._selection_prefix_constraint(
                allowed_outputs=allowed_outputs,
                prompt_token_count=len(input_ids[0]),
            )
            generation_kwargs["max_new_tokens"] = min(
                self.max_new_tokens,
                max(len(value) for value in allowed_outputs) + 2,
            )
        with self._torch.inference_mode():
            output_ids = self._model.generate(**encoded, **generation_kwargs)
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
