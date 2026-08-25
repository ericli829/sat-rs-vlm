"""Proposal providers used to precompute evidence for VLM-FO1."""

from .protocol import (
    PROPOSAL_SCHEMA_VERSION,
    ProposalError,
    ProposalProvider,
    ProposalResult,
    canonicalize_proposals,
)
from .registry import create_proposal_provider

__all__ = [
    "PROPOSAL_SCHEMA_VERSION",
    "ProposalError",
    "ProposalProvider",
    "ProposalResult",
    "canonicalize_proposals",
    "create_proposal_provider",
]
