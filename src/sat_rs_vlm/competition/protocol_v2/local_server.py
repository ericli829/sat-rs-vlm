"""Diagnostic service for local integration, with fixtures or one real 2B model.

This entry point deliberately omits the production planner, detectors and memory.
Its results must not be used to assess the full system's accuracy or performance.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

from sat_rs_vlm.infrastructure.config import ModelConfig
from sat_rs_vlm.taskgraph.choice import ChoiceResolver
from sat_rs_vlm.taskgraph.choice_config import ChoiceSystemConfig
from sat_rs_vlm.taskgraph.input_composer import InputComposer
from sat_rs_vlm.taskgraph.providers import LazyQwenSemanticProvider, VLMRequest
from sat_rs_vlm.taskgraph.runtime_types import Answer, ImageRef
from sat_rs_vlm.taskgraph.semantic_decision import SemanticDecisionLayer

from .answers import AnswerResolver, serialize_response
from .errors import AnswerContractViolation, ProtocolError, ProtocolValidationError, RuntimeInferenceError
from .image_package import ImagePackage
from .models import AnswerContract, PredictRequest
from .server import create_app

logger = logging.getLogger("uvicorn.error")


class LocalProtocolService:
    def __init__(self, package: ImagePackage, *, mode: str, provider=None):
        if mode not in {"offline", "qwen2b"}:
            raise ValueError("Local test mode must be offline or qwen2b")
        if (mode == "qwen2b") != (provider is not None):
            raise ValueError("qwen2b requires exactly one configured provider")
        self.package = package
        self.mode = mode
        self.provider = provider
        self.last_diagnostics = {}
        self.composer = InputComposer(save_intermediate_artifacts=False, compact_trace=True)
        self.resolver = (
            AnswerResolver(
                provider,
                self.composer,
                ChoiceResolver(
                    provider, self.composer,
                    ChoiceSystemConfig(forbid_non_feature_options=False),
                ),
                SemanticDecisionLayer(provider),
                bbox_model_coordinate_format="normalized_0_1",
            )
            if provider is not None else None
        )

    def preload(self) -> None:
        if self.provider is not None:
            self.provider.preload()

    def predict(self, request: PredictRequest) -> dict:
        self.last_diagnostics = {}
        contract = AnswerContract.from_request(request)
        images = self.package.resolve_all(request.images)
        if contract.kind == "bbox" and not any(
            (image.width, image.height) == (contract.image_width, contract.image_height)
            for image in images
        ):
            raise ProtocolValidationError()
        self.last_diagnostics = {
            "request_id": request.request_id,
            "execution_mode": f"local_{self.mode}",
            "dimensions": [[image.width, image.height] for image in images],
        }
        originals = tuple(ImageRef(str(image.path), image.width, image.height) for image in images)
        try:
            if self.mode == "offline":
                value = self._fixture_answer(contract)
            else:
                value = self._model_answer(request, contract, originals)
            return serialize_response(request, contract, value)
        except ProtocolError:
            raise
        except Exception as exc:
            raise RuntimeInferenceError() from exc
        finally:
            self.composer.release_runtime_cache()
            if self.provider is not None:
                self.provider.release_idle_sessions()

    @staticmethod
    def _fixture_answer(contract: AnswerContract):
        if contract.kind in {"enum", "single_choice"}:
            return contract.values[0]
        if contract.kind == "integer":
            return max(0, contract.minimum or 0)
        if contract.kind == "bbox":
            return [
                contract.image_width * 0.1, contract.image_height * 0.1,
                contract.image_width * 0.5, contract.image_height * 0.5,
            ]
        return "local protocol test".ljust(max(1, contract.min_length), ".")

    def _model_answer(self, request, contract, originals):
        question = request.question
        if len(originals) == 2:
            question += "\nThe first image is BEFORE; the second image is AFTER."
        if contract.kind == "integer":
            minimum = f" Value must be >= {contract.minimum}." if contract.minimum is not None else ""
            model_input = self.resolver._compose(
                originals, question + "\nReturn only one decimal integer, with no explanation." + minimum
            )
            text = self.provider.infer(VLMRequest(model_input, "local_integer")).text.strip()
            if not re.fullmatch(r"-?(?:0|[1-9][0-9]*)", text):
                raise AnswerContractViolation("Local model did not return an integer")
            return int(text)
        if contract.kind == "bbox":
            matching = [
                index for index, image in enumerate(originals)
                if (image.width, image.height) == (contract.image_width, contract.image_height)
            ]
            if len(matching) != 1:
                raise AnswerContractViolation("Local bbox image is ambiguous")
            image = originals[matching[0]]
            model_input = self.composer.compose(
                [image], question=question + '\nReturn only JSON {"bbox":[x1,y1,x2,y2]}. '
                "Coordinates must be normalized to [0,1] relative to this complete image. "
                "Return exactly one non-empty box, with no explanation."
            )
            text = self.provider.infer(VLMRequest(model_input, "local_bbox")).text.strip()
            evidence = Answer(text, provenance={"image_index": matching[0]})
        else:
            evidence = originals
        return self.resolver.resolve(evidence, question, contract, originals)

    def close(self) -> None:
        self.composer.close()
        if self.provider is not None:
            self.provider.close()


def create_local_app():
    config = json.loads(Path(os.environ["COMPETITION_LOCAL_CONFIG"]).read_text(encoding="utf-8-sig"))
    package = ImagePackage(config["package"])
    mode = config["mode"]
    provider = None
    if mode == "qwen2b":
        model = ModelConfig.model_validate(config["model"])
        if not model.model_id or not Path(model.model_id).is_dir():
            raise ValueError("A local base model directory is required")
        if model.adapter_path is not None and (
            not model.adapter_path or not Path(model.adapter_path).is_dir()
        ):
            raise ValueError("The configured local adapter directory must exist")
        logger.info("Local model variant: %s", "base_only" if model.adapter_path is None else "with_lora")
        provider = LazyQwenSemanticProvider(model, role="local_qwen2b")
    service = LocalProtocolService(package, mode=mode, provider=provider)
    try:
        logger.info("Local diagnostic mode: %s; production planner/detectors/memory are omitted", mode)
        if config.get("preload", True):
            logger.info("Preloading local model before accepting requests")
            service.preload()
        return create_app(service=service, question_capture_path=config.get("question_capture_path"))
    except BaseException:
        service.close()
        raise
