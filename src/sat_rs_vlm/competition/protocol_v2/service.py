"""One initialized runtime; a single lock spans execution and final resolution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from threading import Lock

from sat_rs_vlm.taskgraph.runtime import TaskGraphRuntime, runtime_from_config
from sat_rs_vlm.taskgraph.runtime_types import ImageRef
from sat_rs_vlm.taskgraph.semantic_decision import SemanticDecisionConfig, SemanticDecisionLayer

from .adapter import CompetitionRequestAdapter, CompetitionRoutingConfig
from .answers import AnswerResolver, serialize_response
from .errors import ProtocolError, ProtocolValidationError, RuntimeInferenceError
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
        # Competition legal values must all remain available, including absence options.
        choice_resolver = type(runtime.choice_resolver)(
            runtime.choice_resolver.provider,
            runtime.composer,
            replace(runtime.choice_resolver.config, forbid_non_feature_options=False),
        )
        self.package = package
        self.runtime = runtime
        self.adapter = CompetitionRequestAdapter(
            CompetitionRoutingConfig(**config.get("routing", {}))
        )
        self.resolver = AnswerResolver(
            runtime.providers.semantic_2b,
            runtime.composer,
            choice_resolver,
            SemanticDecisionLayer(
                runtime.providers.semantic_2b,
                SemanticDecisionConfig.from_mapping(config.get("enum", {})),
            ),
            config.get("short_text", {}).get(
                "instruction", "Return only the shortest direct answer. No explanation."
            ),
            bbox_model_coordinate_format=config.get("bbox", {}).get(
                "model_output_coordinate_format"
            ),
        )
        self.last_diagnostics: dict = {}
        self.last_trace = None
        self._lock = Lock()

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
        contract = AnswerContract.from_request(request)
        images = self.package.resolve_all(request.images)
        if contract.kind == "bbox" and not any(
            (image.width, image.height) == (contract.image_width, contract.image_height)
            for image in images
        ):
            raise ProtocolValidationError()
        adapted = self.adapter.adapt(request, images, contract)
        self.last_diagnostics = {
            "request_id": request.request_id,
            "execution_mode": adapted.execution_mode.value,
            "constraint": contract.kind,
            "image_count": len(images),
            "dimensions": [[x.width, x.height] for x in images],
        }
        try:
            result = self.runtime.run(adapted)
            self.last_diagnostics["execution_mode"] = result.execution_mode.value
            self.last_trace = result.trace
            originals = tuple(ImageRef(str(x.path), x.width, x.height) for x in images)
            value = self.resolver.resolve(
                result.output, result.trace.final_question or request.question, contract, originals
            )
            if self.resolver.last_bbox_conversion is not None:
                result.trace.telemetry["competition_bbox_conversion"] = (
                    self.resolver.last_bbox_conversion
                )
            return serialize_response(request, contract, value)
        except ProtocolError:
            raise
        except Exception as exc:
            self.last_trace = getattr(exc, "execution_trace", self.last_trace)
            raise RuntimeInferenceError() from exc
        finally:
            self.runtime.composer.release_runtime_cache()
            self.runtime._purge_cached_sessions(self.runtime.providers)

    def close(self) -> None:
        self.runtime.close()
