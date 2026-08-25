"""Lazy proposal-provider registry with no detector imports at module load."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .config import expand_config_value
from .protocol import ProposalError, ProposalProvider

PROVIDER_NAMES = (
    "mock",
    "grounding_dino",
    "lae_dino_lae1m",
    "lae_dino_dior",
    "lae_dino_dota",
)


def create_proposal_provider(name: str, config: Mapping[str, Any]) -> ProposalProvider:
    """Construct one provider; heavy detector dependencies stay lazy."""

    provider_name = str(name).strip().lower()
    normalized_config = expand_config_value(config)
    if provider_name == "mock":
        from .mock import MockProposalProvider

        return MockProposalProvider(normalized_config)
    if provider_name == "grounding_dino":
        from .grounding_dino import GroundingDinoProvider

        return GroundingDinoProvider(normalized_config)
    if provider_name in {"lae_dino_lae1m", "lae_dino_dior", "lae_dino_dota"}:
        from .lae_dino_sidecar import LAEDinoSidecarProvider

        return LAEDinoSidecarProvider(normalized_config, provider_name=provider_name)
    raise ProposalError(
        f"unsupported proposal provider {provider_name!r}; "
        f"choose one of {', '.join(PROVIDER_NAMES)}"
    )
