"""Lazy locator construction and provider wiring."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sat_rs_vlm.semantics import RuleBasedQueryParser
from sat_rs_vlm.semantics.ontology import load_ontology

from .config import selected_provider_config
from .protocol import LocatorProvider
from .router import TaskRouter
from .types import LocatorError

LOCATOR_NAMES = ("hierarchical",)


def create_locator(name: str, config: Mapping[str, Any]) -> LocatorProvider:
    locator_name = str(name).strip().lower()
    if locator_name not in LOCATOR_NAMES:
        raise LocatorError(
            f"unsupported locator {locator_name!r}; choose one of {', '.join(LOCATOR_NAMES)}"
        )
    parser_config = config.get("parser", {})
    if not isinstance(parser_config, Mapping):
        raise LocatorError("parser config must be a mapping")
    ontology_value = parser_config.get(
        "ontology_path", "configs/eval/semantic/remote_sensing_ontology.json"
    )
    ontology_path = Path(str(ontology_value)).expanduser().resolve()
    parser = RuleBasedQueryParser(load_ontology(ontology_path))
    router = TaskRouter(config.get("router", {}))

    detector_provider = None
    retriever_provider = None
    detector_section = config.get("detector", {})
    retriever_section = config.get("retriever", {})
    if not isinstance(detector_section, Mapping) or not isinstance(
        retriever_section, Mapping
    ):
        raise LocatorError("detector and retriever config sections must be mappings")
    try:
        if bool(detector_section.get("enabled", False)):
            from sat_rs_vlm.integrations.detectors.registry import create_proposal_provider

            detector_name = str(detector_section.get("provider", "mock"))
            detector_provider = create_proposal_provider(
                detector_name,
                selected_provider_config(
                    config,
                    kind="detector",
                    provider_name=detector_name,
                ),
            )
        if bool(retriever_section.get("enabled", True)):
            from sat_rs_vlm.integrations.retrievers.registry import (
                create_retriever_provider,
            )

            retriever_name = str(retriever_section.get("provider", "mock"))
            retriever_provider = create_retriever_provider(
                retriever_name,
                selected_provider_config(
                    config,
                    kind="retriever",
                    provider_name=retriever_name,
                ),
            )
        from .hierarchical import HierarchicalLocator

        return HierarchicalLocator(
            parser=parser,
            router=router,
            config=config,
            detector_provider=detector_provider,
            retriever_provider=retriever_provider,
        )
    except Exception:
        if detector_provider is not None:
            detector_provider.close()
        if retriever_provider is not None:
            retriever_provider.close()
        raise
