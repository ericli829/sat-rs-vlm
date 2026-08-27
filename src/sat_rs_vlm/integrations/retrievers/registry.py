"""Lazy retriever registry; importing it never loads a model framework."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .config import expand_config_value
from .protocol import RetrievalError, RetrieverProvider

PROVIDER_NAMES = ("mock", "visrag")


def create_retriever_provider(
    name: str,
    config: Mapping[str, Any] | None = None,
) -> RetrieverProvider:
    provider_name = str(name).strip().lower()
    normalized_config = expand_config_value(config or {})
    if provider_name == "mock":
        from .mock import MockRetrieverProvider

        return MockRetrieverProvider(normalized_config)
    if provider_name == "visrag":
        from .visrag import VisRAGRetrieverProvider

        return VisRAGRetrieverProvider(normalized_config)
    raise RetrievalError(
        f"unsupported retriever provider {provider_name!r}; "
        f"choose one of {', '.join(PROVIDER_NAMES)}"
    )
