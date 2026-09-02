"""Lazy proposal-provider registry with no detector imports at module load."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .config import expand_config_value
from .protocol import ProposalError, ProposalProvider

PROVIDER_NAMES = (
    "mock",
    "tiled",
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
    if provider_name == "tiled":
        from .tiled import TiledProposalProvider

        base_provider_name = str(normalized_config.get("base_provider", "")).strip()
        if not base_provider_name or base_provider_name == "tiled":
            raise ProposalError("tiled detector requires a non-tiled base_provider")
        base_config = normalized_config.get("base_config", {})
        if not isinstance(base_config, Mapping):
            raise ProposalError("tiled detector base_config must be a mapping")
        base_config = dict(base_config)
        for key in (
            "parallel_workers",
            "parallel_max_workers",
            "parallel_worker_vram_gb",
            "parallel_vram_reserve_gb",
        ):
            if key in normalized_config:
                base_config.setdefault(key, normalized_config[key])
        base_provider = create_proposal_provider(base_provider_name, base_config)
        try:
            return TiledProposalProvider(
                base_provider,
                normalized_config,
                base_provider_name=base_provider_name,
            )
        except Exception:
            base_provider.close()
            raise
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
