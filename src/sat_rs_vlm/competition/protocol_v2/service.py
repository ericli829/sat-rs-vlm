"""One initialized runtime; a single lock spans execution and final resolution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from threading import Lock
import time
from typing import Any

from sat_rs_vlm.taskgraph.executor import TaskGraphExecutionError
from sat_rs_vlm.taskgraph.operators import RuntimeBudgetExceeded
from sat_rs_vlm.taskgraph.runtime import TaskGraphRuntime, runtime_from_config
from sat_rs_vlm.taskgraph.routing import ExecutionMode
from sat_rs_vlm.taskgraph.runtime_types import ImageRef
from sat_rs_vlm.taskgraph.semantic_decision import SemanticDecisionConfig, SemanticDecisionLayer

from .adapter import CompetitionRequestAdapter, CompetitionRoutingConfig
from .answers import AnswerResolver, serialize_response
from .errors import ProtocolValidationError, RuntimeInferenceError
from .image_package import ImagePackage
from .models import AnswerContract, PredictRequest


def validate_production_runtime(config: Mapping) -> None:
    def has_test_provider(value):
        if isinstance(value, Mapping):
            return any(
                (
                    key in {"kind", "provider", "base_provider"}
                    and isinstance(item, str)
                    and item.lower() in {"fake", "fixture", "mock"}
                )
                or has_test_provider(item)
                for key, item in value.items()
            )
        if isinstance(value, (tuple, list)):
            return any(has_test_provider(item) for item in value)
        return False

    providers = config.get("providers", {})
    if has_test_provider(providers):
        raise ValueError("Production competition runtime cannot contain test providers")
    for name in ("detection", "counting", "semantic_2b", "route_4b", "region_retriever", "planner"):
        provider = providers.get(name, {})
        if not isinstance(provider, Mapping) or provider.get("kind", "fake") in {"fake", "fixture"}:
            raise ValueError(f"Production competition runtime requires real {name}")
    choice = providers.get("choice", {})
    if (
        choice.get("reuse") not in {"semantic_2b", "route_4b"}
        and choice.get("kind", "fake") == "fake"
    ):
        raise ValueError("Production competition runtime requires a real choice provider")


class CompetitionService:
    def __init__(
        self, package: ImagePackage, runtime: TaskGraphRuntime, config: Mapping | None = None
    ):
        config = config or {}
        if config.get("bbox", {}).get("coordinate_format", "pixel_xyxy") != "pixel_xyxy":
            raise ValueError("Competition bbox coordinates must be pixel_xyxy")
        if not config.get("bbox", {}).get("validate_image_bounds", True):
            raise ValueError("Competition bbox bounds validation is required")
        if not config.get("logging", {}).get("request_metadata_only", True):
            raise ValueError("Competition request logs must contain only safe metadata")
        if runtime.choice_resolver.config.legacy_regex_fallback:
            raise ValueError("Competition choices require cached finite scoring")
        self.package = package
        self.runtime = runtime
        self._preflight_errors: list[str] = []
        self.adapter = CompetitionRequestAdapter(
            CompetitionRoutingConfig(**config.get("routing", {}))
        )
        enum_config = SemanticDecisionConfig.from_mapping(config.get("enum", {}))
        short_text_instruction = config.get("short_text", {}).get(
            "instruction", "Return only the shortest direct answer. No explanation."
        )
        bbox_model_coordinate_format = config.get("bbox", {}).get(
            "model_output_coordinate_format"
        )

        def make_resolver(provider, choice_provider):
            final_choice_resolver = type(runtime.choice_resolver)(
                choice_provider,
                runtime.composer,
                replace(
                    runtime.choice_resolver.config,
                    forbid_non_feature_options=False,
                ),
            )
            return AnswerResolver(
                provider,
                runtime.composer,
                final_choice_resolver,
                SemanticDecisionLayer(provider, enum_config),
                short_text_instruction,
                bbox_model_coordinate_format=bbox_model_coordinate_format,
            )

        self.resolver = make_resolver(
            runtime.providers.semantic_2b, runtime.choice_resolver.provider
        )
        self.temporal_change_resolver = make_resolver(
            runtime.providers.route_4b, runtime.providers.route_4b
        )
        retry_config = config.get("retry", {})
        self.graph_retry_limit = int(retry_config.get("taskgraph_graph_retries", 1))
        self.fallback_reserve_seconds = float(
            retry_config.get("fallback_reserve_seconds", 20)
        )
        self.fallback_minimum_seconds = float(
            retry_config.get("fallback_minimum_seconds", 6)
        )
        self.response_reserve_seconds = float(retry_config.get("response_reserve_seconds", 1))
        self.request_timeout_seconds = float(
            config.get("server", {}).get("timeout_seconds", 120)
        )
        if not 0 <= self.graph_retry_limit <= 1:
            raise ValueError("taskgraph_graph_retries must be between 0 and 1")
        if not (
            0 <= self.fallback_minimum_seconds <= self.fallback_reserve_seconds < 120
            and 0 <= self.response_reserve_seconds < 10
            and 0 < self.request_timeout_seconds <= 120
        ):
            raise ValueError("invalid Protocol 2.0 inference budget configuration")
        self.last_diagnostics: dict = {}
        self.last_trace = None
        self._lock = Lock()
        self._preflight_errors = self._check_local_model_paths()
        if self._preflight_errors:
            raise ValueError(
                "production model preflight failed: " + "; ".join(self._preflight_errors)
            )

    def _check_local_model_paths(self) -> list[str]:
        errors: list[str] = []
        for name in ("planner", "semantic_2b", "route_4b"):
            provider = getattr(self.runtime.providers, name, None)
            model_config = getattr(provider, "model_config", None)
            source = model_config if model_config is not None else provider
            if source is None or not getattr(source, "local_files_only", False):
                continue
            model_value = getattr(source, "model_id", None)
            adapter_value = getattr(source, "adapter_path", None)
            for label, value in (("base", model_value), ("adapter", adapter_value)):
                if not value:
                    if label == "base" or (name == "planner" and label == "adapter"):
                        errors.append(f"{name}_missing_base")
                    continue
                path = Path(str(value)).expanduser()
                if not path.is_absolute() and label == "adapter":
                    path = Path.cwd() / path
                if path.is_absolute() and not path.is_dir():
                    errors.append(f"{name}_missing_{label}")
                elif label == "adapter" and path.is_dir():
                    if not (path / "adapter_config.json").is_file() or not any(
                        (path / filename).is_file()
                        for filename in ("adapter_model.safetensors", "adapter_model.bin")
                    ):
                        errors.append(f"{name}_invalid_adapter")
        return errors

    def readiness(self) -> dict:
        def state(provider: object) -> str:
            if provider is None:
                return "not_configured"
            if hasattr(provider, "telemetry_model_loaded"):
                return "loaded" if bool(provider.telemetry_model_loaded) else "lazy"
            if hasattr(provider, "engine_identity"):
                return "loaded" if provider.engine_identity is not None else "lazy"
            return "configured"

        manager = self.runtime.providers.shared_model_manager
        planner_state = state(self.runtime.providers.planner)
        route_state = state(self.runtime.providers.route_4b)
        if manager is not None:
            shared_state = "loaded" if manager.loaded else "lazy"
            planner_state = route_state = shared_state
        return {
            "status": "ready" if not self._preflight_errors else "not_ready",
            "model_load_policy": "on_demand",
            "models_loaded": any(
                value == "loaded"
                for value in (
                    planner_state,
                    state(self.runtime.providers.semantic_2b),
                    route_state,
                )
            ),
            "providers": {
                "planner": planner_state,
                "semantic_2b": state(self.runtime.providers.semantic_2b),
                "route_4b": route_state,
                "detection": state(self.runtime.providers.detection),
                "counting": state(self.runtime.providers.counting),
                "region_retriever": state(self.runtime.providers.retriever),
            },
            "shared_4b_base": {
                "enabled": manager is not None,
                "loaded": bool(manager.loaded) if manager is not None else False,
                "adapter_switch_count": (
                    manager.metrics.get("adapter_switch_count") if manager is not None else 0
                ),
            },
            "preflight_errors": list(self._preflight_errors),
        }

    @classmethod
    def from_config(
        cls,
        package: ImagePackage,
        runtime_config: Mapping,
        config: Mapping | None = None,
        *,
        offline: bool = False,
    ):
        if type(offline) is not bool:
            raise ValueError("offline must be an explicit boolean")
        if not offline:
            validate_production_runtime(runtime_config)
        runtime = runtime_from_config(runtime_config)
        return cls(package, runtime, config)

    def predict(self, request: PredictRequest) -> dict:
        with self._lock:
            return self._predict(request)

    def _predict(self, request: PredictRequest) -> dict:
        self.last_trace = None
        self.last_diagnostics = {}
        started = time.monotonic()
        contract = AnswerContract.from_request(request)
        images = self.package.resolve_all(request.images)
        if contract.kind == "bbox" and not any(
            (image.width, image.height) == (contract.image_width, contract.image_height)
            for image in images
        ):
            raise ProtocolValidationError()
        deadline = started + self.request_timeout_seconds - self.response_reserve_seconds
        adapted = replace(
            self.adapter.adapt(request, images, contract),
            deadline_monotonic=deadline,
            minimum_remaining_seconds=self.fallback_reserve_seconds,
        )
        self.last_diagnostics = {
            "request_id": request.request_id,
            "execution_mode": adapted.execution_mode.value,
            "initial_route": adapted.execution_mode.value,
            "routing_reason": adapted.routing_reason,
            "constraint": contract.kind,
            "final_answer_contract": contract.kind,
            "image_count": len(images),
            "dimensions": [[x.width, x.height] for x in images],
            "memory_hit": False,
            "taskgraph_attempts": 0,
            "planner_retry_count": 0,
            "taskgraph_failure_stage": None,
            "taskgraph_failure_reason": None,
            "fallback_triggered": False,
            "fallback_model_role": None,
            "fallback_success": None,
        }
        try:
            originals = tuple(ImageRef(str(x.path), x.width, x.height) for x in images)
            result = None
            taskgraph_attempts = 0
            while result is None:
                if adapted.execution_mode is ExecutionMode.TASKGRAPH_UHR:
                    taskgraph_attempts += 1
                    self.last_diagnostics["taskgraph_attempts"] = taskgraph_attempts
                try:
                    result = self.runtime.run(adapted)
                except Exception as exc:
                    self.last_trace = getattr(exc, "execution_trace", self.last_trace)
                    route = dict(self.runtime.last_route_diagnostics)
                    self.last_diagnostics.update(
                        {
                            "execution_mode": route.get(
                                "execution_mode", adapted.execution_mode.value
                            ),
                            "memory_hit": bool(route.get("memory_hit", False)),
                            "routing_reason": route.get("routing_reason", adapted.routing_reason),
                        }
                    )
                    actual_taskgraph = (
                        route.get("execution_mode") == ExecutionMode.TASKGRAPH_UHR.value
                    )
                    if not actual_taskgraph:
                        raise RuntimeInferenceError() from exc
                    stage, reason = self._failure_stage(exc)
                    self.last_diagnostics["taskgraph_failure_stage"] = stage
                    self.last_diagnostics["taskgraph_failure_reason"] = reason
                    self.last_diagnostics["planner_retry_count"] = self._planner_retry_count(
                        exc, self.last_trace
                    )
                    if (
                        taskgraph_attempts <= self.graph_retry_limit
                        and self._retryable(exc)
                        and self._remaining(deadline) > self.fallback_reserve_seconds + 2
                    ):
                        self.last_diagnostics["taskgraph_retry_reason"] = reason
                        continue
                    return self._fallback(
                        request, contract, originals, deadline, cause=exc
                    )

            route = dict(self.runtime.last_route_diagnostics)
            self.last_diagnostics.update(
                {
                    "execution_mode": result.execution_mode.value,
                    "memory_hit": bool(route.get("memory_hit", False)),
                    "routing_reason": route.get("routing_reason", adapted.routing_reason),
                }
            )
            if result.execution_mode is not ExecutionMode.TASKGRAPH_UHR:
                self.last_diagnostics["taskgraph_attempts"] = 0
            self.last_trace = result.trace
            try:
                resolver = (
                    self.temporal_change_resolver
                    if adapted.task_category == "temporal_change" and len(images) == 2
                    else self.resolver
                )
                value = resolver.resolve(
                    result.output,
                    result.trace.final_question or request.question,
                    contract,
                    originals,
                )
                response = serialize_response(request, contract, value)
                if resolver.last_bbox_conversion is not None:
                    result.trace.telemetry["competition_bbox_conversion"] = (
                        resolver.last_bbox_conversion
                    )
                return response
            except Exception as exc:
                if result.execution_mode is not ExecutionMode.TASKGRAPH_UHR:
                    raise RuntimeInferenceError() from exc
                stage, reason = self._failure_stage(exc, answer_stage=True)
                self.last_diagnostics["taskgraph_failure_stage"] = stage
                self.last_diagnostics["taskgraph_failure_reason"] = reason
                self.last_diagnostics["planner_retry_count"] = self._planner_retry_count(
                    None, result.trace
                )
                return self._fallback(request, contract, originals, deadline, cause=exc)
        finally:
            self.last_diagnostics["total_elapsed_ms"] = round(
                (time.monotonic() - started) * 1000.0, 3
            )
            if self.last_trace is not None:
                self.last_trace.telemetry["competition_inference_control"] = dict(
                    self.last_diagnostics
                )
            self.runtime.composer.release_runtime_cache()
            self.runtime._purge_cached_sessions(self.runtime.providers)

    @staticmethod
    def _remaining(deadline: float) -> float:
        return deadline - time.monotonic()

    @staticmethod
    def _retryable(exc: Exception) -> bool:
        if isinstance(exc, RuntimeBudgetExceeded):
            return False
        if isinstance(exc, TaskGraphExecutionError):
            return bool(exc.details.get("retryable"))
        return isinstance(exc, (ConnectionError, TimeoutError))

    @staticmethod
    def _failure_stage(exc: Exception, *, answer_stage: bool = False) -> tuple[str, str]:
        if answer_stage:
            return "answer_contract", type(exc).__name__
        if isinstance(exc, RuntimeBudgetExceeded):
            return "budget", "fallback_reserve_reached"
        if isinstance(exc, TaskGraphExecutionError):
            return "operator", f"{exc.details.get('operator')}:{exc.details.get('provider')}"
        if hasattr(exc, "planner_metadata") or "planner" in type(exc).__name__.casefold():
            return "planner", type(exc).__name__
        return "taskgraph_runtime", type(exc).__name__

    @staticmethod
    def _planner_retry_count(exc: Exception | None, trace: Any) -> int:
        metadata = getattr(exc, "planner_metadata", None) if exc is not None else None
        if not isinstance(metadata, Mapping) and trace is not None:
            metadata = trace.telemetry.get("planner_metadata")
        attempts = metadata.get("attempts") if isinstance(metadata, Mapping) else None
        return max(0, len(attempts) - 1) if isinstance(attempts, list) else 0

    def _fallback(
        self,
        request: PredictRequest,
        contract: AnswerContract,
        originals: tuple[ImageRef, ...],
        deadline: float,
        *,
        cause: Exception,
    ) -> dict:
        self.last_diagnostics["fallback_triggered"] = True
        self.last_diagnostics["fallback_model_role"] = "semantic_2b"
        if self._remaining(deadline) < self.fallback_minimum_seconds:
            self.last_diagnostics["fallback_success"] = False
            self.last_diagnostics["fallback_skip_reason"] = "insufficient_remaining_budget"
            raise RuntimeInferenceError() from cause
        try:
            value = self.resolver.resolve(originals, request.question, contract, originals)
            response = serialize_response(request, contract, value)
        except Exception as fallback_exc:
            self.last_diagnostics["fallback_success"] = False
            self.last_diagnostics["fallback_failure_reason"] = type(fallback_exc).__name__
            raise RuntimeInferenceError() from fallback_exc
        self.last_diagnostics["fallback_success"] = True
        self.last_diagnostics["execution_mode"] = ExecutionMode.DIRECT_VLM.value
        if self.last_trace is not None:
            self.last_trace.execution_mode = ExecutionMode.DIRECT_VLM.value
            self.last_trace.telemetry["fallback_model_role"] = "semantic_2b"
            self.last_trace.telemetry["fallback_success"] = True
        return response

    def close(self) -> None:
        self.runtime.close()
