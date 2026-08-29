"""Final ChoiceRequest contract and original-option resolver."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .input_composer import InputComposer
from .providers import SemanticVLMProvider, VLMRequest
from .runtime_types import STRUCTURED_AUTHORITATIVE_TYPES, ChoiceResult, RuntimeObject


@dataclass(frozen=True)
class ChoiceRequest:
    sources: tuple[RuntimeObject, ...]
    question: str
    options: tuple[str, ...]


class ChoiceResolver:
    def __init__(self, provider: SemanticVLMProvider, composer: InputComposer) -> None:
        self.provider = provider
        self.composer = composer
        self.last_model_input = None

    @staticmethod
    def _choice_id(text: str, options: tuple[str, ...]) -> str:
        valid = [chr(ord("A") + index) for index in range(len(options))]
        match = re.search(r"(?:^|[^A-Z])([A-Z])(?:[^A-Z]|$)", text.upper())
        if match and match.group(1) in valid:
            return match.group(1)
        normalized = text.strip().casefold()
        for index, option in enumerate(options):
            option_value = re.sub(r"^\s*[\(\[]?[A-Z][\)\].:]?\s*", "", option).strip()
            if normalized == option_value.casefold():
                return valid[index]
        raise ValueError(f"choice provider did not return a valid option id: {text!r}")

    def resolve(self, request: ChoiceRequest) -> ChoiceResult:
        if not request.options:
            raise ValueError("ChoiceRequest requires original dataset options")
        model_input = self.composer.compose(
            list(request.sources), question=request.question, options=request.options
        )
        # Safety invariant: an authoritative structured result never reintroduces an image.
        if (
            request.sources
            and all(
                isinstance(source, STRUCTURED_AUTHORITATIVE_TYPES) for source in request.sources
            )
            and model_input.visual_inputs
        ):
            raise AssertionError("structured authoritative choice unexpectedly contains visuals")
        self.last_model_input = model_input
        result = self.provider.infer(VLMRequest(model_input, output_contract="choice"))
        return ChoiceResult(
            choice_id=self._choice_id(result.text, request.options),
            raw_response=result.text,
            confidence=result.confidence,
            provenance={"provider": result.provider, **result.metadata},
        )
