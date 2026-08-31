"""Replaceable region-retrieval integrations for UHR localization."""

from .clip import CLIPRetrieverProvider
from .openclip import OpenCLIPRetrieverProvider
from .protocol import RetrievalError, RetrievalResult, RetrieverProvider
from .registry import PROVIDER_NAMES, create_retriever_provider

__all__ = [
    "PROVIDER_NAMES",
    "RetrievalError",
    "RetrievalResult",
    "RetrieverProvider",
    "create_retriever_provider",
    "CLIPRetrieverProvider",
    "OpenCLIPRetrieverProvider",
]
