"""Replaceable capability contracts and adapters for production providers."""

from __future__ import annotations

import json
import math
import re
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
from pathlib import Path
from typing import Any, Protocol, cast

from PIL import Image

from sat_rs_vlm.infrastructure.config import ModelConfig
from sat_rs_vlm.integrations.detectors.protocol import ProposalProvider
from sat_rs_vlm.integrations.locators.protocol import LocatorProvider
from sat_rs_vlm.integrations.retrievers.protocol import RetrieverProvider
from sat_rs_vlm.models.hf_vlm_engine import HuggingFaceVLMEngine

from .runtime_types import (
    BBox,
    ChoiceScoreResult,
    Entity,
    EntitySet,
    ImageRef,
    Region,
    RuntimeObject,
)
from .schema import TargetSpec, TaskGraph


def _bbox_contains(
    outer: Sequence[float], inner: Sequence[float], *, tolerance: float = 1e-6
) -> bool:
    return (
        float(inner[0]) >= float(outer[0]) - tolerance
        and float(inner[1]) >= float(outer[1]) - tolerance
        and float(inner[2]) <= float(outer[2]) + tolerance
        and float(inner[3]) <= float(outer[3]) + tolerance
    )


def _bbox_intersection(
    left: Sequence[float], right: Sequence[float]
) -> tuple[float, float, float, float] | None:
    box = (
        max(float(left[0]), float(right[0])),
        max(float(left[1]), float(right[1])),
        min(float(left[2]), float(right[2])),
        min(float(left[3]), float(right[3])),
    )
    return box if box[0] < box[2] and box[1] < box[3] else None


@dataclass(frozen=True)
class DetectionRequest:
    scope: ImageRef | Region
    target: TargetSpec
    task_hint: str | None = None


@dataclass(frozen=True)
class DetectionSet:
    detections: EntitySet
    latency_ms: float
    provider: str
    metadata: dict[str, Any] = field(default_factory=dict)


class DetectionProvider(Protocol):
    provider_name: str

    def detect(self, request: DetectionRequest) -> DetectionSet: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class CountingRequest:
    scope: ImageRef | Region
    target: TargetSpec
    entire: bool


@dataclass(frozen=True)
class CountingResult:
    count: int
    detections: EntitySet
    provider: str
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


class CountingProvider(Protocol):
    provider_name: str

    def count(self, request: CountingRequest) -> CountingResult: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class RegionRetrievalRequest:
    image: ImageRef | Region
    query: str
    search_scope: Region | None = None
    max_candidates: int | None = None

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("region retrieval query must not be empty")
        if self.max_candidates is not None and self.max_candidates < 1:
            raise ValueError("max_candidates must be positive")
        if self.search_scope is None:
            return
        image = self.image if isinstance(self.image, ImageRef) else self.image.image
        if self.search_scope.image.uri_or_key != image.uri_or_key:
            raise ValueError("search_scope must reference the same image")
        if isinstance(self.image, Region) and not _bbox_contains(
            self.image.bbox_xyxy_global, self.search_scope.bbox_xyxy_global
        ):
            raise ValueError("nested search_scope must be contained by input Region")

    def effective_scope(self) -> Region | None:
        if self.search_scope is not None:
            return self.search_scope
        return self.image if isinstance(self.image, Region) else None


@dataclass(frozen=True)
class RegionCandidate:
    region: Region
    relevance_score: float
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        score = float(self.relevance_score)
        if not math.isfinite(score):
            raise ValueError("region candidate relevance_score must be finite")
        object.__setattr__(self, "relevance_score", score)


@dataclass(frozen=True)
class RegionCandidates:
    candidates: tuple[RegionCandidate, ...]
    provider: str
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        latency = float(self.latency_ms)
        if not math.isfinite(latency) or latency < 0.0:
            raise ValueError("region candidate latency_ms must be finite and non-negative")
        if not str(self.provider).strip():
            raise ValueError("region candidate provider must not be empty")
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "latency_ms", latency)


class RegionRetrieverProvider(Protocol):
    provider_name: str

    def retrieve(self, request: RegionRetrievalRequest) -> RegionCandidates: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class ModelSource:
    role: str
    value: RuntimeObject


@dataclass(frozen=True)
class ModelInput:
    visual_inputs: tuple[Any, ...]
    structured_context: str
    question: str
    options: tuple[str, ...] = ()
    visual_roles: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VLMRequest:
    model_input: ModelInput
    output_contract: str = "text"


@dataclass(frozen=True)
class VLMResult:
    text: str
    provider: str
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class CachedChoiceUnavailableError(RuntimeError):
    """The configured backend explicitly lacks cached choice capability."""


@dataclass(frozen=True)
class ChoiceScoringRequest:
    model_input: ModelInput
    answer_type: str
    choice_ids: tuple[str, ...]
    option_texts: tuple[str, ...]
    single_choice_suffix: str
    multi_verify_template: str
    multi_select_threshold: float = 0.0
    purpose: str = "final_choice"

    def __post_init__(self) -> None:
        if self.answer_type not in {"CHOICE_SINGLE", "CHOICE_MULTI"}:
            raise ValueError("choice scoring answer_type is invalid")
        if not self.choice_ids or len(self.choice_ids) != len(self.option_texts):
            raise ValueError("choice ids and option texts must be non-empty and aligned")
        if not self.single_choice_suffix:
            raise ValueError("single choice suffix must not be empty")
        if (
            "{choice_id}" not in self.multi_verify_template
            or "{option_text}" not in self.multi_verify_template
        ):
            raise ValueError("multi verify template must include choice_id and option_text")


@dataclass(frozen=True)
class FiniteDecisionRequest:
    """Generic cached reasoning-to-finite-decision request.

    Benchmark choice is one caller of this primitive. Intermediate semantic
    alignment uses canonical values directly and never parses the free
    reasoning text.
    """

    model_input: ModelInput
    decision_mode: str
    candidate_ids: tuple[str, ...]
    candidate_texts: tuple[str, ...]
    single_decision_suffix: str
    multi_verify_template: str
    select_threshold: float = 0.0
    purpose: str = "semantic_decision"
    reasoning_instruction: str | None = None

    def __post_init__(self) -> None:
        if self.decision_mode not in {"SINGLE", "MULTI", "BINARY"}:
            raise ValueError("finite decision mode must be SINGLE, MULTI, or BINARY")
        if not self.candidate_ids or len(self.candidate_ids) != len(self.candidate_texts):
            raise ValueError("finite decision candidates must be non-empty and aligned")
        if len(self.candidate_ids) != len(set(self.candidate_ids)):
            raise ValueError("finite decision candidate ids must be unique")
        if self.decision_mode == "BINARY" and len(self.candidate_ids) != 2:
            raise ValueError("binary finite decision requires exactly two candidates")
        if not self.single_decision_suffix:
            raise ValueError("single decision suffix must not be empty")
        if (
            "{choice_id}" not in self.multi_verify_template
            or "{option_text}" not in self.multi_verify_template
        ):
            raise ValueError("multi verify template must include choice_id and option_text")


@dataclass(frozen=True)
class FiniteDecisionResult:
    selected_ids: tuple[str, ...]
    scores: dict[str, float]
    decision_mode: str
    reasoning_text: str | None
    provider: str
    model_id: str
    method: str
    cache_reused: bool
    latency_ms: dict[str, float | None] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.decision_mode not in {"SINGLE", "MULTI", "BINARY"}:
            raise ValueError("finite decision result mode is invalid")
        if self.decision_mode in {"SINGLE", "BINARY"} and len(self.selected_ids) != 1:
            raise ValueError("single and binary decisions require exactly one selected id")
        if len(self.selected_ids) != len(set(self.selected_ids)):
            raise ValueError("selected finite decision ids must be unique")
        if any(candidate not in self.scores for candidate in self.selected_ids):
            raise ValueError("every selected finite decision id must have a score")


class SemanticVLMProvider(Protocol):
    provider_name: str

    def infer(self, request: VLMRequest) -> VLMResult: ...

    def reason_and_decide(self, request: FiniteDecisionRequest) -> FiniteDecisionResult: ...

    def reason_and_choose(self, request: ChoiceScoringRequest) -> ChoiceScoreResult: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class PlannerRequest:
    question: str
    question_type: str
    choices: tuple[str, ...]
    inputs: Mapping[str, ImageRef]
    sample_id: str | None = None


class PlannerProvider(Protocol):
    provider_name: str

    def plan(self, request: PlannerRequest) -> TaskGraph: ...


class PlannerFailedError(RuntimeError):
    """A Planner exhausted its bounded generation and validation attempts."""

    error_type = "planner_failed"
    stage = "planner"


class Qwen3VLPlannerProvider:
    """Text-only Qwen3-VL 4B Planner with the validated lab DSL boundary."""

    provider_name = "qwen3vl_lora"

    def __init__(self, config: Mapping[str, Any], *, role: str = "planner_4b") -> None:
        self.config = dict(config)
        self.role = role
        self.model_id = str(
            self.config.get("model_id")
            or self.config.get("base_model")
            or self.config.get("model_dir")
            or ""
        ).strip()
        self.adapter_path = str(
            self.config.get("adapter_path")
            or self.config.get("adapter")
            or self.config.get("lora_path")
            or ""
        ).strip()
        self.processor_id = str(
            self.config.get("processor_id") or self.config.get("processor_path") or self.model_id
        ).strip()
        self.device = str(self.config.get("device", "auto"))
        self.dtype = str(self.config.get("dtype", self.config.get("torch_dtype", "auto")))
        self.local_files_only = bool(self.config.get("local_files_only", True))
        self.max_new_tokens = int(self.config.get("max_new_tokens", 512))
        self.max_prompt_tokens = int(self.config.get("max_prompt_tokens", 2048))
        self.max_attempts = int(self.config.get("max_attempts", 2))
        self.constraint_top_k = int(self.config.get("constraint_top_k", 64))
        self.constraint_max_candidate_checks = int(
            self.config.get("constraint_max_candidate_checks", 256)
        )
        self.constraint_max_nodes = int(self.config.get("constraint_max_nodes", 24))
        self.repeat_guard_repetitions = int(self.config.get("repeat_guard_repetitions", 4))
        self.max_finish_node_tokens = int(self.config.get("max_finish_node_tokens", 32))
        self._model: Any | None = None
        self._processor: Any | None = None
        self._tokenizer: Any | None = None
        self._torch: Any | None = None
        self._load_info: dict[str, Any] = {}
        self.last_metadata: dict[str, Any] = {}
        self._validate_config()

    @staticmethod
    def _local_path(value: str, *, code: str, label: str) -> Path:
        path = Path(value).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"{code}: {label} directory does not exist: {path}")
        return path

    @staticmethod
    def _model_stem(value: str | Path) -> str:
        text = str(value).replace("\\", "/").rstrip("/").casefold()
        return text.rsplit("/", 1)[-1]

    def _validate_adapter_base(self, adapter_config: Mapping[str, Any]) -> None:
        declared = adapter_config.get("base_model_name_or_path")
        if not declared:
            return
        configured = self._model_stem(self.model_id)
        adapter_base = self._model_stem(str(declared))
        if configured == adapter_base or configured in adapter_base or adapter_base in configured:
            return
        raise ValueError(
            "PLANNER_ADAPTER_BASE_MISMATCH: LoRA adapter targets "
            f"{declared!r}, configured base is {self.model_id!r}"
        )

    def _validate_config(self) -> None:
        if not self.model_id:
            raise ValueError("MISSING_LOCAL_PLANNER_MODEL: planner model_id is required")
        if not self.adapter_path:
            raise ValueError("MISSING_LOCAL_PLANNER_ADAPTER: planner adapter_path is required")
        if not self.local_files_only:
            raise ValueError("Qwen3VL Planner requires local_files_only=true")
        if self.max_new_tokens < 1 or self.max_prompt_tokens < 1:
            raise ValueError("Planner token limits must be positive")
        if not 1 <= self.max_attempts <= 3:
            raise ValueError("Planner max_attempts must be between 1 and 3")
        if self.constraint_top_k < 1 or (
            self.constraint_max_candidate_checks < self.constraint_top_k
        ):
            raise ValueError("Planner constraint candidate limits are invalid")
        if self.constraint_max_nodes < 1 or self.repeat_guard_repetitions < 1:
            raise ValueError("Planner constraint node/repeat limits are invalid")
        self.base_path = self._local_path(
            self.model_id,
            code="MISSING_LOCAL_PLANNER_MODEL",
            label="Planner base model",
        )
        self.adapter_dir = self._local_path(
            self.adapter_path,
            code="MISSING_LOCAL_PLANNER_ADAPTER",
            label="Planner LoRA adapter",
        )
        if self.processor_id:
            self.processor_path = self._local_path(
                self.processor_id,
                code="MISSING_LOCAL_PLANNER_PROCESSOR",
                label="Planner processor",
            )
        else:
            self.processor_path = self.base_path
        adapter_config_path = self.adapter_dir / "adapter_config.json"
        try:
            adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"INVALID_LOCAL_PLANNER_ADAPTER: cannot read {adapter_config_path}: {exc}"
            ) from exc
        if not isinstance(adapter_config, Mapping):
            raise ValueError("INVALID_LOCAL_PLANNER_ADAPTER: adapter_config.json must be an object")
        self.adapter_config = dict(adapter_config)
        self._validate_adapter_base(self.adapter_config)

    @property
    def load_info(self) -> dict[str, Any]:
        return dict(self._load_info)

    def _load(self) -> tuple[Any, Any, Any, Any]:
        if self._model is not None and self._processor is not None:
            return self._model, self._processor, self._tokenizer, self._torch
        try:
            import peft
            import torch
            import transformers
        except (ImportError, OSError) as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "PLANNER_DEPENDENCY_MISSING: install the model extras for Qwen3-VL Planner"
            ) from exc
        from sat_rs_vlm.models.qwen3vl_loader import load_qwen3vl

        torch_dtype = None if self.dtype == "auto" else getattr(torch, self.dtype, None)
        if self.dtype != "auto" and torch_dtype is None:
            raise ValueError(f"Unsupported Planner torch dtype: {self.dtype}")
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": bool(self.config.get("trust_remote_code", True)),
            "local_files_only": True,
            "dtype": torch_dtype if torch_dtype is not None else "auto",
        }
        if self.device == "auto":
            model_kwargs["device_map"] = "auto"
        if self.config.get("attn_implementation"):
            model_kwargs["attn_implementation"] = self.config["attn_implementation"]
        model, processor = load_qwen3vl(
            modules={"torch": torch, "transformers": transformers, "peft": peft},
            base_model=str(self.base_path),
            processor_source=str(self.processor_path),
            model_kwargs=model_kwargs,
            processor_kwargs={
                "trust_remote_code": model_kwargs["trust_remote_code"],
                "local_files_only": True,
            },
            adapter_path=str(self.adapter_dir),
        )
        tokenizer = getattr(processor, "tokenizer", processor)
        if hasattr(tokenizer, "padding_side"):
            tokenizer.padding_side = "left"
        model_device = getattr(model, "device", None)
        if model_device is None:
            try:
                model_device = next(model.parameters()).device
            except (AttributeError, StopIteration):
                model_device = self.device
        self._model, self._processor, self._tokenizer, self._torch = (
            model,
            processor,
            tokenizer,
            torch,
        )
        self._load_info = {
            "base_model_path": str(self.base_path),
            "adapter_path": str(self.adapter_dir),
            "adapter_config": dict(self.adapter_config),
            "model_class": type(model).__name__,
            "dtype": self.dtype,
            "device": str(model_device),
            "role": self.role,
        }
        return model, processor, tokenizer, torch

    @staticmethod
    def _question_type(value: str, choices: Sequence[str]) -> str:
        normalized = str(value).upper()
        if choices:
            return (
                "MULTIPLE_CHOICE_MULTI"
                if normalized.endswith("_MULTI") or normalized == "MULTI"
                else "MULTIPLE_CHOICE_SINGLE"
            )
        return normalized if normalized in {"FREE_FORM", "BOOLEAN", "INTEGER"} else "FREE_FORM"

    @staticmethod
    def _messages(
        request: PlannerRequest,
        system_prompt: str,
        *,
        previous_prediction: str | None = None,
        diagnostic: str | None = None,
    ) -> list[dict[str, str]]:
        inputs = {
            str(key).removeprefix("$"): {
                "type": "image",
                "uri_or_key": value.uri_or_key,
            }
            for key, value in request.inputs.items()
        }
        payload = {
            "question": request.question,
            "question_type": Qwen3VLPlannerProvider._question_type(
                request.question_type, request.choices
            ),
            "choices": list(request.choices) if request.choices else None,
            "inputs": inputs,
        }
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            },
        ]
        if previous_prediction is not None:
            messages.append({"role": "assistant", "content": previous_prediction.strip()})
        if diagnostic is not None:
            messages.append({"role": "user", "content": diagnostic})
        return messages

    def _system_prompt(self) -> str:
        configured = self.config.get("system_prompt_path") or self.config.get("prompt_path")
        path = (
            Path(str(configured)).expanduser()
            if configured
            else Path(__file__).resolve().parents[3]
            / "taskgraph_lab"
            / "prompts"
            / "planner_student_system_prompt.txt"
        )
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[3] / path
        if not path.is_file():
            raise FileNotFoundError(f"PLANNER_SYSTEM_PROMPT_MISSING: {path}")
        prompt = path.read_text(encoding="utf-8").strip()
        if not prompt:
            raise ValueError(f"PLANNER_SYSTEM_PROMPT_EMPTY: {path}")
        return prompt

    def _input_device(self, model: Any, torch: Any) -> Any:
        if self.device != "auto":
            return torch.device(self.device)
        model_device = getattr(model, "device", None)
        if model_device is not None and str(model_device) != "meta":
            return model_device
        try:
            return next(model.parameters()).device
        except (AttributeError, StopIteration):
            return torch.device("cpu")

    def _generate(
        self,
        request: PlannerRequest,
        messages: list[dict[str, str]],
    ) -> tuple[str, dict[str, Any]]:
        model, processor, tokenizer, torch = self._load()
        apply_chat_template = getattr(processor, "apply_chat_template", None)
        if not callable(apply_chat_template):
            raise RuntimeError("PLANNER_PROCESSOR_INVALID: processor lacks apply_chat_template")
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        try:
            prompt = str(apply_chat_template(messages, **template_kwargs))
        except TypeError:
            template_kwargs.pop("enable_thinking")
            prompt = str(apply_chat_template(messages, **template_kwargs))
        encoded = processor(
            text=[prompt],
            images=None,
            videos=None,
            padding=True,
            truncation=True,
            max_length=self.max_prompt_tokens,
            return_tensors="pt",
        )
        input_device = self._input_device(model, torch)
        if hasattr(encoded, "to"):
            encoded = encoded.to(input_device)
        else:
            encoded = {
                key: value.to(input_device) if hasattr(value, "to") else value
                for key, value in dict(encoded).items()
            }
        input_ids = encoded["input_ids"]
        prompt_width = int(input_ids.shape[1])
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            raise RuntimeError("PLANNER_TOKENIZER_INVALID: tokenizer lacks pad/eos token")
        from taskgraph_lab.evaluation.constrained_decoding import GreedyDSLLogitsProcessor

        constraint = GreedyDSLLogitsProcessor(
            tokenizer,
            prompt_width=prompt_width,
            image_refs_by_row=[tuple(f"${str(key).removeprefix('$')}" for key in request.inputs)],
            initial_top_k=self.constraint_top_k,
            max_candidate_checks=self.constraint_max_candidate_checks,
            max_nodes=self.constraint_max_nodes,
            repeat_guard_repetitions=self.repeat_guard_repetitions,
            max_finish_node_tokens=self.max_finish_node_tokens,
        )
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                num_beams=1,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=int(pad_token_id),
                use_cache=True,
                logits_processor=[constraint],
            )
        continuation = generated[0, prompt_width:]
        if hasattr(processor, "decode"):
            text = str(
                processor.decode(
                    continuation,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            ).strip()
        else:
            text = str(
                tokenizer.decode(
                    continuation.tolist(),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            ).strip()
        metadata = constraint.diagnostics(
            0,
            continuation,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=int(pad_token_id),
        )
        metadata.update(
            {
                "latency_ms": (time.perf_counter() - started) * 1000.0,
                "prompt_tokens": int(encoded.get("attention_mask", input_ids).sum().item()),
                "generated_tokens": int((continuation != int(pad_token_id)).sum().item()),
                "constrained": True,
                "vision_inputs": 0,
            }
        )
        return text, metadata

    @staticmethod
    def _to_production_graph(text: str, request: PlannerRequest) -> TaskGraph:
        from taskgraph_lab.taskgraph.canonicalize import canonicalize_target
        from taskgraph_lab.taskgraph.dsl import parse_taskgraph_dsl

        target = parse_taskgraph_dsl(text)
        canonical = canonicalize_target(target)
        final = dict(canonical["final"])
        final.setdefault("question", "")
        payload = {
            "version": "taskgraph-v1.1",
            "question": request.question,
            "question_type": Qwen3VLPlannerProvider._question_type(
                request.question_type, request.choices
            ),
            "choices": list(request.choices) if request.choices else None,
            "inputs": {
                str(key).removeprefix("$"): {
                    "type": "image",
                    "uri_or_key": value.uri_or_key,
                }
                for key, value in request.inputs.items()
            },
            **canonical,
            "final": final,
        }
        return TaskGraph.model_validate(payload)

    def plan(self, request: PlannerRequest) -> TaskGraph:
        if not request.question.strip():
            raise ValueError("Planner question must not be empty")
        system_prompt = self._system_prompt()
        attempts: list[dict[str, Any]] = []
        messages = self._messages(request, system_prompt)
        for attempt_number in range(1, self.max_attempts + 1):
            prediction, generation_metadata = self._generate(request, messages)
            try:
                graph = self._to_production_graph(prediction, request)
            except Exception as exc:
                attempts.append(
                    {
                        "attempt": attempt_number,
                        "prediction": prediction,
                        "termination_reason": "planner_failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        **generation_metadata,
                    }
                )
                if attempt_number >= self.max_attempts:
                    self.last_metadata = {
                        "role": self.role,
                        "provider": self.provider_name,
                        "status": "planner_failed",
                        "attempts": attempts,
                        "load": self.load_info,
                    }
                    raise PlannerFailedError(
                        "planner_failed for sample "
                        f"{request.sample_id or request.question!r}: {exc}"
                    ) from exc
                messages = self._messages(
                    request,
                    system_prompt,
                    previous_prediction=prediction,
                    diagnostic=(
                        "Previous plan is invalid.\n\nError:\n"
                        f"{type(exc).__name__}: {exc}\n\n"
                        "Regenerate the complete valid TaskGraph DSL only."
                    ),
                )
                continue
            attempts.append(
                {
                    "attempt": attempt_number,
                    "termination_reason": "final",
                    "prediction": prediction,
                    "planner_output": prediction,
                    **generation_metadata,
                }
            )
            self.last_metadata = {
                "role": self.role,
                "provider": self.provider_name,
                "status": "executed",
                "planner_output": prediction,
                "attempts": attempts,
                "load": self.load_info,
                "vision_inputs": 0,
            }
            return graph
        raise AssertionError("Planner generated no attempts")

    def close(self) -> None:
        close = getattr(self._model, "close", None)
        if callable(close):
            close()
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._torch = None


@dataclass(frozen=True)
class EvidenceSufficiencyRequest:
    question: str
    region: Region | None = None
    task_hint: str | None = None
    evidence: tuple[RuntimeObject, ...] = ()
    sample_id: str | None = None
    evidence_version: str = "v1"

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("evidence sufficiency question must not be empty")
        if self.region is None and not self.evidence:
            raise ValueError("evidence sufficiency requires region or evidence")
        if not self.evidence_version.strip():
            raise ValueError("evidence_version must not be empty")

    @property
    def sources(self) -> tuple[RuntimeObject, ...]:
        region = (self.region,) if self.region is not None else ()
        return region + self.evidence


class EvidenceSufficiencyStatus(str, Enum):
    SUFFICIENT = "SUFFICIENT"
    NEED_MORE_EVIDENCE = "NEED_MORE_EVIDENCE"
    UNRESOLVED = "UNRESOLVED"
    ERROR = "ERROR"


@dataclass(frozen=True)
class EvidenceSufficiencyResult:
    status: EvidenceSufficiencyStatus | str
    score: float | None = None
    reason_code: str | None = None
    provider: str = "unknown"
    model_id: str = "unknown"
    method: str = "unknown"
    cache_reused: bool = False
    latency_ms: dict[str, float | None] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", EvidenceSufficiencyStatus(self.status))
        if self.score is not None and not 0.0 <= self.score <= 1.0:
            raise ValueError("evidence sufficiency score must be between 0 and 1")


class EvidenceSufficiencyProvider(Protocol):
    provider_name: str

    def assess(self, request: EvidenceSufficiencyRequest) -> EvidenceSufficiencyResult: ...


class ProposalDetectionAdapter:
    """Adapt the existing LAE/other ProposalProvider without changing it."""

    def __init__(self, provider: ProposalProvider) -> None:
        self._provider = provider
        self.provider_name = provider.provider_name

    @staticmethod
    def _image_scope(scope: ImageRef | Region) -> tuple[ImageRef, tuple[float, float]]:
        if isinstance(scope, ImageRef):
            return scope, (0.0, 0.0)
        return scope.image, (scope.bbox_xyxy_global[0], scope.bbox_xyxy_global[1])

    def detect(self, request: DetectionRequest) -> DetectionSet:
        image_ref, offset = self._image_scope(request.scope)
        source_path = image_ref.path.resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"detection image does not exist: {source_path}")
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="taskgraph_detection_") as temp_dir:
            detector_path = source_path
            if isinstance(request.scope, Region):
                with Image.open(source_path) as source:
                    crop = source.convert("RGB").crop(request.scope.bbox_xyxy_global)
                    detector_path = Path(temp_dir) / "scope.png"
                    crop.save(detector_path)
            proposal_query = request.target.category.strip()
            result = self._provider.predict(detector_path, proposal_query)
        result_metadata = dict(result.metadata)
        result_metadata.update(
            {
                "proposal_query": proposal_query,
                "original_target_spec": {
                    "category": request.target.category,
                    "attributes": dict(request.target.attributes),
                },
            }
        )
        entities = []
        for index, (box, score) in enumerate(zip(result.boxes_xyxy, result.scores, strict=True)):
            global_box = (
                float(box[0]) + offset[0],
                float(box[1]) + offset[1],
                float(box[2]) + offset[0],
                float(box[3]) + offset[1],
            )
            entities.append(
                Entity(
                    region=Region(
                        image=image_ref,
                        bbox_xyxy_global=global_box,
                        provenance={
                            "provider": result.provider,
                            "coordinate_mode": "absolute_original_pixel_xyxy",
                            "candidate_id": f"candidate_{index + 1:04d}",
                        },
                    ),
                    label=request.target.category,
                    score=float(score),
                    provenance={
                        "provider": result.provider,
                        "model_id": result.model_id,
                        "candidate_id": f"candidate_{index + 1:04d}",
                        "scale_tile_metadata": result_metadata,
                    },
                )
            )
        return DetectionSet(
            detections=EntitySet(
                tuple(entities),
                provenance={
                    "provider": result.provider,
                    "model_id": result.model_id,
                    "proposal_metadata": result_metadata,
                },
            ),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            provider=result.provider,
            metadata=result_metadata,
        )

    def close(self) -> None:
        self._provider.close()


class FakeDetectionProvider:
    provider_name = "fake_lae"

    def __init__(self, boxes: Sequence[Sequence[float]] | None = None) -> None:
        self.boxes = [cast(BBox, tuple(float(item) for item in box)) for box in (boxes or [])]
        self.calls: list[DetectionRequest] = []

    def detect(self, request: DetectionRequest) -> DetectionSet:
        self.calls.append(request)
        image = request.scope if isinstance(request.scope, ImageRef) else request.scope.image
        entities = []
        for index, box in enumerate(self.boxes):
            global_box = box
            if isinstance(request.scope, Region):
                clipped = _bbox_intersection(box, request.scope.bbox_xyxy_global)
                if clipped is None:
                    continue
                global_box = clipped
            entities.append(
                Entity(
                    Region(
                        image,
                        global_box,
                        {
                            "provider": self.provider_name,
                            "search_scope": (
                                list(request.scope.bbox_xyxy_global)
                                if isinstance(request.scope, Region)
                                else None
                            ),
                        },
                    ),
                    request.target.category,
                    max(0.01, 0.99 - index * 0.01),
                    {"provider": self.provider_name},
                )
            )
        return DetectionSet(
            EntitySet(tuple(entities), {"provider": self.provider_name}),
            0.0,
            self.provider_name,
            {"deterministic": True},
        )

    def close(self) -> None:
        return None


class FakeCountingProvider:
    """Deterministic COUNT fixture; independent from FakeDetectionProvider."""

    provider_name = "fake_counting"

    def __init__(self, boxes: Sequence[Sequence[float]] | None = None) -> None:
        self.boxes = [cast(BBox, tuple(float(item) for item in box)) for box in (boxes or [])]
        self.calls: list[CountingRequest] = []
        self.closed = False

    def count(self, request: CountingRequest) -> CountingResult:
        self.calls.append(request)
        image = request.scope if isinstance(request.scope, ImageRef) else request.scope.image
        entire = bool(request.entire)
        entire_source = "CountingRequest.entire"
        if isinstance(request.scope, Region):
            entire = False
            entire_source = "region_frozen_semantic"
        entities = []
        for index, box in enumerate(self.boxes):
            global_box = box
            if isinstance(request.scope, Region):
                clipped = _bbox_intersection(box, request.scope.bbox_xyxy_global)
                if clipped is None:
                    continue
                global_box = clipped
            entities.append(
                Entity(
                    Region(
                        image,
                        global_box,
                        {
                            "provider": self.provider_name,
                            "entire": entire,
                            "entire_source": entire_source,
                        },
                    ),
                    request.target.category,
                    max(0.01, 0.99 - index * 0.01),
                    {"provider": self.provider_name},
                )
            )
        detections = EntitySet(
            tuple(entities),
            {
                "provider": self.provider_name,
                "entire": entire,
                "entire_source": entire_source,
                "requested_entire": request.entire,
            },
        )
        return CountingResult(
            count=len(detections.entities),
            detections=detections,
            provider=self.provider_name,
            latency_ms=0.0,
            metadata={
                "deterministic": True,
                "entire": entire,
                "entire_source": entire_source,
                "requested_entire": request.entire,
            },
        )

    def close(self) -> None:
        self.closed = True


class LocatorRegionRetrieverAdapter:
    """Expose the existing UHR Locator as the generic candidate capability."""

    def __init__(self, locator: LocatorProvider) -> None:
        self._locator = locator
        self.provider_name = locator.provider_name

    def retrieve(self, request: RegionRetrievalRequest) -> RegionCandidates:
        image = request.image if isinstance(request.image, ImageRef) else request.image.image
        source_path = image.path.resolve()
        scope = request.effective_scope()
        offset = (0.0, 0.0)
        if scope is None:
            result = self._locator.locate(source_path, request.query)
        else:
            offset = scope.bbox_xyxy_global[:2]
            with tempfile.TemporaryDirectory(prefix="taskgraph_retrieval_") as temp_dir:
                crop_path = Path(temp_dir) / "scope.png"
                with Image.open(source_path) as source:
                    source.convert("RGB").crop(scope.bbox_xyxy_global).save(crop_path)
                result = self._locator.locate(crop_path, request.query)
        candidates_list = []
        for index, (box, score) in enumerate(zip(result.regions_xyxy, result.scores, strict=True)):
            global_box = (
                float(box[0]) + offset[0],
                float(box[1]) + offset[1],
                float(box[2]) + offset[0],
                float(box[3]) + offset[1],
            )
            if scope is not None:
                clipped = _bbox_intersection(global_box, scope.bbox_xyxy_global)
                if clipped is None:
                    continue
                global_box = clipped
            details = (
                dict(result.region_details[index]) if index < len(result.region_details) else {}
            )
            candidates_list.append(
                RegionCandidate(
                    Region(
                        image,
                        global_box,
                        {
                            "locator": self.provider_name,
                            "coordinate_mode": "absolute_original_pixel_xyxy",
                            "search_scope": (
                                list(scope.bbox_xyxy_global) if scope is not None else None
                            ),
                        },
                    ),
                    float(score),
                    {
                        "locator": self.provider_name,
                        "local_bbox_xyxy": list(box),
                        "global_bbox_xyxy": list(global_box),
                        "details": details,
                    },
                )
            )
            if (
                request.max_candidates is not None
                and len(candidates_list) >= request.max_candidates
            ):
                break
        candidates = tuple(candidates_list)
        search_plan = getattr(result, "search_plan", None)
        return RegionCandidates(
            candidates,
            self.provider_name,
            result.latency_ms.get("total", 0.0),
            {
                "locator": self.provider_name,
                "provider_provenance": dict(getattr(result, "provider_provenance", {})),
                "latency_ms": dict(result.latency_ms),
                "search_plan": (search_plan.to_dict() if hasattr(search_plan, "to_dict") else None),
                "depth_reached": getattr(result, "depth_reached", None),
            },
        )

    def close(self) -> None:
        self._locator.close()


class ScoredGridRegionRetrieverAdapter:
    """Adapt score-only RetrieverProvider by supplying explicit grid candidates."""

    def __init__(
        self,
        provider: RetrieverProvider,
        *,
        grid_size: int = 3,
        default_max_candidates: int = 5,
        candidate_window_ratio: float | None = None,
    ) -> None:
        if grid_size < 1:
            raise ValueError("retriever grid_size must be positive")
        if default_max_candidates < 1:
            raise ValueError("retriever default_max_candidates must be positive")
        if candidate_window_ratio is not None and not 0.0 < candidate_window_ratio <= 1.0:
            raise ValueError("retriever candidate_window_ratio must be in (0, 1]")
        self._provider = provider
        self.provider_name = provider.provider_name
        self.grid_size = grid_size
        self.default_max_candidates = default_max_candidates
        self.candidate_window_ratio = candidate_window_ratio

    def retrieve(self, request: RegionRetrievalRequest) -> RegionCandidates:
        image = request.image if isinstance(request.image, ImageRef) else request.image.image
        with Image.open(image.path) as source:
            width, height = source.size
        effective_scope = request.effective_scope()
        scope = (
            effective_scope.bbox_xyxy_global
            if effective_scope is not None
            else (0.0, 0.0, float(width), float(height))
        )
        scope_width = scope[2] - scope[0]
        scope_height = scope[3] - scope[1]
        window_ratio = self.candidate_window_ratio or 1.0 / self.grid_size
        cell_width = scope_width * window_ratio
        cell_height = scope_height * window_ratio
        if self.grid_size == 1:
            x_starts = [scope[0] + (scope_width - cell_width) / 2.0]
            y_starts = [scope[1] + (scope_height - cell_height) / 2.0]
        else:
            x_stride = (scope_width - cell_width) / (self.grid_size - 1)
            y_stride = (scope_height - cell_height) / (self.grid_size - 1)
            x_starts = [scope[0] + x * x_stride for x in range(self.grid_size)]
            y_starts = [scope[1] + y * y_stride for y in range(self.grid_size)]
        boxes = [
            (x_start, y_start, x_start + cell_width, y_start + cell_height)
            for y_start in y_starts
            for x_start in x_starts
        ]
        scored = self._provider.score_regions(image.path, request.query, boxes)
        order = sorted(range(len(boxes)), key=lambda index: (-scored.scores[index], index))
        order = order[: request.max_candidates or self.default_max_candidates]
        candidates = tuple(
            RegionCandidate(
                Region(
                    image,
                    boxes[index],
                    {
                        "retriever": self.provider_name,
                        "coordinate_mode": "absolute_original_pixel_xyxy",
                        "search_scope": list(scope),
                    },
                ),
                scored.scores[index],
                {
                    "provider": self.provider_name,
                    "model_id": scored.model_id,
                    "bbox_xyxy_global": list(boxes[index]),
                    "search_scope": list(scope),
                    "tile": {
                        "level": 1,
                        "index": index,
                        "row": index // self.grid_size,
                        "column": index % self.grid_size,
                        "grid_size": self.grid_size,
                    },
                    "candidate_geometry": {
                        "layout": "uniform_sliding_grid",
                        "window_ratio": window_ratio,
                        "overlapping": window_ratio > 1.0 / self.grid_size,
                    },
                    "provider_metadata": dict(getattr(scored, "metadata", {})),
                },
            )
            for index in order
        )
        return RegionCandidates(
            candidates,
            self.provider_name,
            scored.latency_ms,
            {"provider_metadata": dict(getattr(scored, "metadata", {}))},
        )

    def close(self) -> None:
        self._provider.close()


class FakeRegionRetriever:
    provider_name = "fake_region_retriever"

    def __init__(self, candidates: Sequence[tuple[Sequence[float], float]] | None = None) -> None:
        self._candidates = list(candidates or [])

    def retrieve(self, request: RegionRetrievalRequest) -> RegionCandidates:
        image = request.image if isinstance(request.image, ImageRef) else request.image.image
        scope = request.effective_scope()
        candidates = []
        for box, score in self._candidates:
            global_box = cast(BBox, tuple(float(value) for value in box))
            if scope is not None:
                clipped = _bbox_intersection(global_box, scope.bbox_xyxy_global)
                if clipped is None:
                    continue
                global_box = clipped
            candidates.append(
                RegionCandidate(
                    Region(
                        image,
                        global_box,
                        {
                            "fake": True,
                            "search_scope": (
                                list(scope.bbox_xyxy_global) if scope is not None else None
                            ),
                        },
                    ),
                    float(score),
                    {"provider": self.provider_name},
                )
            )
            if request.max_candidates is not None and len(candidates) >= request.max_candidates:
                break
        return RegionCandidates(
            tuple(candidates),
            self.provider_name,
        )

    def close(self) -> None:
        return None


class LazyQwenSemanticProvider:
    """Lazy semantic adapter reusing the repository HuggingFace VLM engine."""

    provider_name = "qwen3_vl"

    def __init__(self, model_config: ModelConfig, *, role: str = "semantic") -> None:
        self.model_config = model_config
        self.role = role
        self._engine: HuggingFaceVLMEngine | None = None

    def _load(self) -> HuggingFaceVLMEngine:
        if self._engine is None:
            self._engine = HuggingFaceVLMEngine(
                model_id=self.model_config.model_id,
                adapter_path=self.model_config.adapter_path,
                device=self.model_config.device,
                dtype=self.model_config.dtype,
                max_new_tokens=self.model_config.max_new_tokens,
                trust_remote_code=self.model_config.trust_remote_code,
                local_files_only=self.model_config.local_files_only,
            )
        return self._engine

    @property
    def engine_identity(self) -> str | None:
        return self._engine.model_identity if self._engine is not None else None

    @staticmethod
    def _prompt(model_input: ModelInput, *, reasoning_instruction: str | None = None) -> str:
        prompt_parts = []
        if model_input.visual_inputs:
            roles = model_input.visual_roles or tuple(
                f"VISUAL_{index}" for index in range(1, len(model_input.visual_inputs) + 1)
            )
            manifest = "\n".join(
                f"[image_{index}] role: {role}" for index, role in enumerate(roles, start=1)
            )
            prompt_parts.append("Visual inputs:\n" + manifest)
        if model_input.structured_context:
            prompt_parts.append("Structured results:\n" + model_input.structured_context)
        prompt_parts.append("Question:\n" + model_input.question)
        if model_input.options:
            prompt_parts.append("Options:\n" + "\n".join(model_input.options))
        if reasoning_instruction:
            prompt_parts.append(reasoning_instruction)
        return "\n\n".join(prompt_parts)

    def infer(self, request: VLMRequest) -> VLMResult:
        model_input = request.model_input
        prompt = self._prompt(model_input)
        engine = self._load()
        allowed_outputs: tuple[str, ...] | None = None
        if (
            request.output_contract == "selection"
            and self.model_config.selection_constrained_decoding
        ):
            candidate_mapping = model_input.metadata.get("candidate_mapping")
            if isinstance(candidate_mapping, Mapping):
                labels = tuple(str(label) for label in candidate_mapping)
                # SELECT v1 intentionally keeps the candidate canvas small.
                # 2^8 produces only 256 valid complete strings and is practical
                # for a trie-like token mask; larger sets fall back to parsing.
                if 0 < len(labels) <= 8 and all(
                    len(label) == 1 and label.isalpha() for label in labels
                ):
                    allowed_outputs = ("NONE",) + tuple(
                        ",".join(group)
                        for size in range(1, len(labels) + 1)
                        for group in combinations(labels, size)
                    )
        if not model_input.visual_inputs:
            # Qwen is a VLM, but structured authoritative choice must remain text-only.
            generated = engine.generate_text(
                prompt=prompt, image_paths=[], allowed_outputs=allowed_outputs
            )
        else:
            generated = engine.generate_text(
                prompt=prompt,
                image_paths=list(model_input.visual_inputs),
                allowed_outputs=allowed_outputs,
            )
        return VLMResult(
            generated,
            f"{self.provider_name}:{self.role}",
            metadata={
                "model_id": str(getattr(engine, "model_id", "unknown")),
                "output_contract": request.output_contract,
                "constrained_decoding": allowed_outputs is not None,
                "allowed_output_count": len(allowed_outputs or ()),
            },
        )

    @staticmethod
    def _choice_instruction(purpose: str) -> str:
        if purpose == "route_choice":
            return (
                "Analyze the route-planning problem using the marked start, goal, obstacles, "
                "spatial layout, and supplied route options. Compare feasible routes carefully. "
                "The final option will be selected in a separate constrained step."
            )
        if purpose == "select_relation":
            return (
                "Analyze which candidate object or objects satisfy the requested relation. "
                "Use candidate labels only as visual references during reasoning. A separate "
                "constrained step will determine the final selection."
            )
        return (
            "Analyze the visual evidence, question, and all candidate options carefully. "
            "Reason through the problem before making the final decision. The final option "
            "will be selected in a separate constrained step."
        )

    def reason_and_decide(self, request: FiniteDecisionRequest) -> FiniteDecisionResult:
        engine = self._load()
        scorer = getattr(engine, "reason_and_choose", None)
        if not callable(scorer):
            raise CachedChoiceUnavailableError(
                "Qwen engine does not expose cached reasoning-to-decision scoring"
            )
        answer_type = "CHOICE_MULTI" if request.decision_mode == "MULTI" else "CHOICE_SINGLE"
        result = scorer(
            self._prompt(
                request.model_input,
                reasoning_instruction=request.reasoning_instruction,
            ),
            list(request.model_input.visual_inputs),
            choice_ids=request.candidate_ids,
            option_texts=request.candidate_texts,
            answer_type=answer_type,
            single_choice_suffix=request.single_decision_suffix,
            multi_verify_template=request.multi_verify_template,
            multi_select_threshold=request.select_threshold,
            reasoning_max_new_tokens=self.model_config.max_new_tokens,
        )
        return FiniteDecisionResult(
            selected_ids=result.selected_ids,
            scores=result.scores,
            decision_mode=request.decision_mode,
            reasoning_text=result.reasoning_text,
            provider=f"{self.provider_name}:{self.role}",
            model_id=engine.model_id,
            method=result.method,
            cache_reused=result.cache_reused,
            latency_ms=result.latency_ms,
            metadata={**result.metadata, "purpose": request.purpose},
        )

    def reason_and_choose(self, request: ChoiceScoringRequest) -> ChoiceScoreResult:
        decided = self.reason_and_decide(
            FiniteDecisionRequest(
                model_input=request.model_input,
                decision_mode=("MULTI" if request.answer_type == "CHOICE_MULTI" else "SINGLE"),
                candidate_ids=request.choice_ids,
                candidate_texts=request.option_texts,
                single_decision_suffix=request.single_choice_suffix,
                multi_verify_template=request.multi_verify_template,
                select_threshold=request.multi_select_threshold,
                purpose=request.purpose,
                reasoning_instruction=self._choice_instruction(request.purpose),
            )
        )
        return ChoiceScoreResult(
            selected_ids=decided.selected_ids,
            scores=decided.scores,
            answer_type=request.answer_type,
            reasoning_text=decided.reasoning_text,
            provider=decided.provider,
            model_id=decided.model_id,
            method=decided.method,
            cache_reused=decided.cache_reused,
            latency_ms=decided.latency_ms,
            metadata=decided.metadata,
        )

    def close(self) -> None:
        if self._engine is not None:
            self._engine.close()
        self._engine = None


class FakeSemanticVLMProvider:
    provider_name = "fake_vlm"

    def __init__(
        self,
        responses: Mapping[str, str] | None = None,
        default: str = "A",
        *,
        choice_scores: Mapping[str, Mapping[str, float]] | None = None,
    ) -> None:
        self.responses = dict(responses or {})
        self.default = default
        self.choice_scores = {
            purpose: {choice_id: float(score) for choice_id, score in scores.items()}
            for purpose, scores in (choice_scores or {}).items()
        }
        self.calls: list[VLMRequest] = []
        self.semantic_calls: list[FiniteDecisionRequest] = []
        self.choice_calls: list[ChoiceScoringRequest] = []

    def infer(self, request: VLMRequest) -> VLMResult:
        self.calls.append(request)
        response = self.responses.get(request.output_contract)
        if response is None and request.output_contract in {
            "choice",
            "choice_single",
            "choice_multi",
        }:
            response = self.responses.get("choice")
        if response is None and request.output_contract in {
            "choice",
            "choice_single",
            "choice_multi",
        }:
            match = re.search(r"^value:\s*(.+)$", request.model_input.structured_context, re.M)
            if match:
                value = match.group(1).strip().casefold()
                for index, option in enumerate(request.model_input.options):
                    normalized = re.sub(r"^\s*[\(\[]?[A-Z][\)\].:]?\s*", "", option).strip()
                    if normalized.casefold() == value:
                        response = chr(ord("A") + index)
                        break
        response = response or self.default
        return VLMResult(response, self.provider_name, 1.0, {"deterministic": True})

    def _fixture_selected_ids(self, request: ChoiceScoringRequest) -> tuple[str, ...]:
        response = self.responses.get(request.purpose)
        if response is None and request.purpose == "select_relation":
            response = self.responses.get("selection")
        if response is None:
            response = self.responses.get(request.answer_type.casefold())
        if response is None:
            response = self.responses.get("choice")
        response = (response or self.default).strip()
        try:
            import json

            payload = json.loads(response)
        except (ValueError, TypeError):
            payload = None
        values: list[str] = []
        if isinstance(payload, dict):
            raw = payload.get("selected_ids", payload.get("choice_ids"))
            if isinstance(raw, list):
                values = [str(item).strip().upper() for item in raw]
        if not values and response.upper() in request.choice_ids:
            values = [response.upper()]
        if not values and request.purpose == "select_relation":
            raw_items = [item.strip().upper() for item in response.split(",")]
            if raw_items and all(item in request.choice_ids for item in raw_items):
                values = raw_items
            elif raw_items and all(item.isdigit() for item in raw_items):
                indices = [int(item) for item in raw_items]
                if 0 not in indices and all(
                    1 <= item <= len(request.choice_ids) for item in indices
                ):
                    indices = [item - 1 for item in indices]
                if all(0 <= item < len(request.choice_ids) for item in indices):
                    values = [request.choice_ids[item] for item in indices]
        return tuple(dict.fromkeys(item for item in values if item in request.choice_ids))

    def _finite_fixture_selected_ids(self, request: FiniteDecisionRequest) -> tuple[str, ...]:
        response = self.responses.get(request.purpose)
        if response is None and request.purpose.startswith("semantic_"):
            response = self.responses.get(request.purpose.removeprefix("semantic_"))
        if response is None:
            response = self.responses.get(request.decision_mode.casefold())
        response = (response or self.default).strip()
        try:
            import json

            payload = json.loads(response)
        except (ValueError, TypeError):
            payload = None
        values: list[str] = []
        if isinstance(payload, dict):
            raw = payload.get("selected_ids", payload.get("candidate_ids"))
            if isinstance(raw, list):
                values = [str(item).strip() for item in raw]
        if not values:
            canonical = {item.casefold(): item for item in request.candidate_ids}
            normalized = response.casefold()
            if normalized in {"true", "1"}:
                normalized = "yes"
            elif normalized in {"false", "0"}:
                normalized = "no"
            if normalized in canonical:
                values = [canonical[normalized]]
        if not values and request.decision_mode == "MULTI":
            raw_items = [item.strip() for item in response.split(",")]
            if raw_items and all(item in request.candidate_ids for item in raw_items):
                values = raw_items
        return tuple(dict.fromkeys(item for item in values if item in request.candidate_ids))

    def reason_and_decide(self, request: FiniteDecisionRequest) -> FiniteDecisionResult:
        self.semantic_calls.append(request)
        fixture_scores = self.choice_scores.get(request.purpose)
        if fixture_scores is None:
            fixture_scores = self.choice_scores.get(request.decision_mode.casefold())
        selected_fixture = self._finite_fixture_selected_ids(request)
        scores = {
            candidate_id: (
                float(fixture_scores[candidate_id])
                if fixture_scores is not None and candidate_id in fixture_scores
                else (1.0 if candidate_id in selected_fixture else -1.0)
            )
            for candidate_id in request.candidate_ids
        }
        selected_ids: tuple[str, ...]
        if request.decision_mode in {"SINGLE", "BINARY"}:
            selected_ids = (max(request.candidate_ids, key=lambda item: scores[item]),)
            method = "fake_kv_cached_logits"
            cache_mode = "consume_in_place"
        else:
            selected_ids = tuple(
                candidate_id
                for candidate_id in request.candidate_ids
                if scores[candidate_id] > request.select_threshold
            )
            method = "fake_kv_cached_binary_verification"
            cache_mode = "fork_per_option"
        reasoning = self.responses.get(
            f"{request.purpose}_reasoning",
            self.responses.get("reasoning", "Fake free reasoning is never parsed."),
        )
        return FiniteDecisionResult(
            selected_ids=selected_ids,
            scores=scores,
            decision_mode=request.decision_mode,
            reasoning_text=reasoning,
            provider=self.provider_name,
            model_id="fake-model",
            method=method,
            cache_reused=True,
            latency_ms={
                "vision_prefill_ms": 0.0,
                "text_prefill_ms": 0.0,
                "total_prefill_ms": 0.0,
                "reasoning_decode_ms": 0.0,
                "reasoning_total_ms": 0.0,
                "cache_clone_ms": 0.0,
                "suffix_tokenize_ms": 0.0,
                "choice_suffix_prefill_ms": 0.0,
                "choice_scoring_ms": 0.0,
                "choice_total_ms": 0.0,
                "total_ms": 0.0,
            },
            metadata={
                "initial_prefill_tokens": 1,
                "reasoning_tokens": 1,
                "choice_suffix_tokens": 1,
                "choice_scored_tokens": len(request.candidate_ids),
                "visual_prefill_count": 1 if request.model_input.visual_inputs else 0,
                "reasoning_pass_count": 1,
                "session_released": True,
                "reasoning_cache_mode": cache_mode,
                "peak_vram_mb": None,
                "purpose": request.purpose,
            },
        )

    def reason_and_choose(self, request: ChoiceScoringRequest) -> ChoiceScoreResult:
        self.choice_calls.append(request)
        fixture_scores = self.choice_scores.get(request.purpose)
        if fixture_scores is None:
            fixture_scores = self.choice_scores.get(request.answer_type.casefold())
        selected_fixture = self._fixture_selected_ids(request)
        scores = {
            choice_id: (
                float(fixture_scores[choice_id])
                if fixture_scores is not None and choice_id in fixture_scores
                else (1.0 if choice_id in selected_fixture else -1.0)
            )
            for choice_id in request.choice_ids
        }
        selected_ids: tuple[str, ...]
        if request.answer_type == "CHOICE_SINGLE":
            selected_ids = (max(request.choice_ids, key=lambda item: scores[item]),)
            method = "fake_kv_cached_logits"
        else:
            selected_ids = tuple(
                choice_id
                for choice_id in request.choice_ids
                if scores[choice_id] > request.multi_select_threshold
            )
            method = "fake_kv_cached_binary_verification"
        reasoning = self.responses.get(
            f"{request.purpose}_reasoning",
            self.responses.get("reasoning", "Fake free reasoning; letters A/B/C are not parsed."),
        )
        return ChoiceScoreResult(
            selected_ids=selected_ids,
            scores=scores,
            answer_type=request.answer_type,
            reasoning_text=reasoning,
            provider=self.provider_name,
            model_id="fake-model",
            method=method,
            cache_reused=True,
            latency_ms={
                "vision_prefill_ms": 0.0,
                "text_prefill_ms": 0.0,
                "total_prefill_ms": 0.0,
                "reasoning_decode_ms": 0.0,
                "reasoning_total_ms": 0.0,
                "cache_clone_ms": 0.0,
                "suffix_tokenize_ms": 0.0,
                "choice_suffix_prefill_ms": 0.0,
                "choice_scoring_ms": 0.0,
                "choice_total_ms": 0.0,
                "total_ms": 0.0,
            },
            metadata={
                "initial_prefill_tokens": 1,
                "reasoning_tokens": 1,
                "choice_suffix_tokens": 1,
                "choice_scored_tokens": len(request.choice_ids),
                "visual_prefill_count": 1 if request.model_input.visual_inputs else 0,
                "reasoning_pass_count": 1,
                "session_released": True,
                "reasoning_cache_mode": (
                    "consume_in_place"
                    if request.answer_type == "CHOICE_SINGLE"
                    else "fork_per_option"
                ),
                "peak_vram_mb": None,
                "purpose": request.purpose,
            },
        )

    def close(self) -> None:
        return None


class FixturePlannerProvider:
    provider_name = "fixture_planner"

    def __init__(self, fixtures: Mapping[str, TaskGraph | Mapping[str, Any]]) -> None:
        self.fixtures = {
            key: value if isinstance(value, TaskGraph) else TaskGraph.model_validate(value)
            for key, value in fixtures.items()
        }

    def plan(self, request: PlannerRequest) -> TaskGraph:
        key = request.sample_id or request.question
        try:
            graph = self.fixtures[key]
        except KeyError as exc:
            raise KeyError(f"no fixture TaskGraph for {key!r}") from exc
        if tuple(graph.choices or ()) != request.choices:
            raise ValueError("fixture graph choices differ from original dataset options")
        return graph


class FakeEvidenceSufficiencyProvider:
    provider_name = "fake_evidence_sufficiency"

    def __init__(self, status: str = "SUFFICIENT", score: float = 1.0) -> None:
        self.status = EvidenceSufficiencyStatus(status)
        self.score = score
        self.calls: list[EvidenceSufficiencyRequest] = []

    def assess(self, request: EvidenceSufficiencyRequest) -> EvidenceSufficiencyResult:
        self.calls.append(request)
        return EvidenceSufficiencyResult(
            self.status,
            self.score,
            reason_code="fake_fixture",
            provider=self.provider_name,
            model_id="fake",
            method="fake_structured_status",
        )


def parse_selection_indices(text: str, count: int) -> tuple[int, ...]:
    """Parse a candidate selection without mistaking prose numbers for ids.

    Constrained decoding normally yields canonical ``A,C`` or ``NONE``.  This
    parser deliberately remains tolerant for model/back-end compatibility, but
    numeric ids are accepted only when the entire reply is a numeric list or
    they are explicitly introduced as candidate/option identifiers.
    """

    normalized = text.strip()
    if normalized.casefold() in {"none", "no", "empty", "null"}:
        return ()

    letters = re.findall(r"(?<![A-Z0-9])([A-Z])(?![A-Z0-9])", normalized.upper())
    if letters:
        values = [ord(item) - ord("A") for item in letters]
        if any(item < 0 or item >= count for item in values):
            raise ValueError(f"selection provider returned out-of-range candidate ids: {letters}")
        return tuple(dict.fromkeys(values))
    numeric_list = re.fullmatch(r"\s*\[?\s*\d+(?:\s*[,，]\s*\d+)*\s*\]?\s*", normalized)
    explicit_numeric = re.findall(
        r"(?:candidate|candidates|option|options|候选|选项)\s*#?\s*(\d+)",
        normalized,
        flags=re.IGNORECASE,
    )
    values = (
        [int(item) for item in re.findall(r"\d+", normalized)]
        if numeric_list
        else [int(item) for item in explicit_numeric]
    )
    if not values:
        raise ValueError(f"selection provider returned no candidate ids: {text!r}")
    if 0 not in values and all(1 <= item <= count for item in values):
        values = [item - 1 for item in values]
    if any(item < 0 or item >= count for item in values):
        raise ValueError(f"selection provider returned out-of-range candidate ids: {values}")
    return tuple(dict.fromkeys(values))
