"""External evaluation integrations that stay isolated from model runtimes."""

from sat_rs_vlm.integrations.vlm_fo1 import (
    FO1_PROMPT_PROFILES,
    TargetPhraseResult,
    build_counting_prompt,
    extract_count_target_phrase,
    parse_region_indexes,
)

__all__ = [
    "FO1_PROMPT_PROFILES",
    "TargetPhraseResult",
    "build_counting_prompt",
    "extract_count_target_phrase",
    "parse_region_indexes",
]
