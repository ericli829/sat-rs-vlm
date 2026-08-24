"""Pure protocol helpers for the official VLM-FO1 sidecar.

This module deliberately has no import-time dependency on torch, transformers,
UPN, or the official ``vlm_fo1`` package.  The production worker loads those
packages lazily in the isolated ``vlm-fo1`` environment.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sat_rs_vlm.data.task_protocol import counting_json, parse_count

FO1_PROMPT_PROFILES = ("plain", "integer", "json", "official_fo1")
OFFICIAL_COUNTING_TEMPLATE = (
    "How many {target} are there in this image? Count each instance of the target object. "
    "Locate them with object indexes and then answer the question with the number of objects."
)
JSON_COUNTING_SUFFIX = (
    'Return ONLY a JSON object with this schema: {"count":<integer>}. '
    "Do not include any other text."
)
INTEGER_COUNTING_SUFFIX = "Answer with an integer only."

_UNSUPPORTED_PHRASE_RE = re.compile(
    r"\b(?:object|objects|thing|things|category|categories|class|classes|"
    r"lane|lanes|row|rows|unique|different|more\s+than|less\s+than|"
    r"multiple|at\s+least|at\s+most|ratio|percentage)\b",
    re.IGNORECASE,
)
_LEADING_ARTICLE_RE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")
_QUESTION_PREFIX_RE = re.compile(
    r"^\s*(?:how\s+many|number\s+of)\s+(?P<phrase>.+?)\s*$",
    re.IGNORECASE,
)
_TRAILING_CLAUSE_RE = re.compile(
    r"\s+(?:are|is|can|could|do|does|did|were|was|be|remain|"
    r"there|visible|shown|depicted|present|located|found|seen|observed|"
    r"identified|displayed|available|on|in|at|near|from|with|to)\b.*$",
    re.IGNORECASE,
)
_REGION_RE = re.compile(r"<region(?P<index>\d+)>", re.IGNORECASE)


@dataclass(frozen=True)
class TargetPhraseResult:
    """Deterministic extraction result; no reference answer is consulted."""

    phrase: str | None
    status: str
    reason: str | None = None

    @property
    def supported(self) -> bool:
        return self.status == "supported" and bool(self.phrase)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_phrase": self.phrase,
            "target_status": self.status,
            "target_reason": self.reason,
        }


def _normalize_question(question: str) -> str:
    return _WHITESPACE_RE.sub(" ", str(question or "").strip()).strip()


def extract_count_target_phrase(question: str) -> TargetPhraseResult:
    """Extract an open-vocabulary instance noun phrase from a count question.

    The resolver intentionally rejects comparative, category-level, and
    relationship questions.  It does not map phrases to the project's closed
    detection taxonomy and never reads a reference answer or sample id.
    """

    normalized = _normalize_question(question)
    if not normalized:
        return TargetPhraseResult(None, "unsupported", "empty_question")
    match = _QUESTION_PREFIX_RE.match(normalized.rstrip("?.!"))
    if match is None:
        return TargetPhraseResult(None, "unsupported", "not_an_instance_count_question")
    phrase = match.group("phrase").strip(" \t,;:?.!")
    phrase = _TRAILING_CLAUSE_RE.sub("", phrase).strip(" \t,;:?.!")
    phrase = _LEADING_ARTICLE_RE.sub("", phrase).strip()
    phrase = _WHITESPACE_RE.sub(" ", phrase)
    if not phrase:
        return TargetPhraseResult(None, "unsupported", "empty_target_phrase")
    if _UNSUPPORTED_PHRASE_RE.search(phrase):
        return TargetPhraseResult(None, "unsupported", "non_instance_or_comparative_target")
    if not re.search(r"[A-Za-z]", phrase):
        return TargetPhraseResult(None, "unsupported", "target_phrase_not_text")
    return TargetPhraseResult(phrase.lower(), "supported")


def build_counting_prompt(
    question: str,
    target_phrase: str | None,
    profile: str = "official_fo1",
) -> str:
    """Build one of the explicitly named counting prompt profiles."""

    if profile not in FO1_PROMPT_PROFILES:
        raise ValueError(f"unsupported FO1 prompt profile: {profile}")
    original = _normalize_question(question)
    if not original:
        raise ValueError("counting question must not be empty")
    phrase = _normalize_question(target_phrase or "")
    if profile == "plain":
        return original
    if profile == "integer":
        return f"{original} {INTEGER_COUNTING_SUFFIX}"
    if profile == "json":
        return f"{original} {JSON_COUNTING_SUFFIX}"
    if not phrase:
        raise ValueError("official_fo1 prompt requires a supported target phrase")
    return OFFICIAL_COUNTING_TEMPLATE.format(target=phrase)


def parse_region_indexes(
    output: str,
    *,
    proposal_count: int | None = None,
) -> dict[str, Any]:
    """Parse official ``<ground>...<objects><regionN>`` output safely.

    No count is guessed when the official region markup is absent.  Duplicate
    indexes are de-duplicated in first-seen order, while out-of-range indexes
    are reported separately and never used as evidence.
    """

    text = str(output or "")
    matches = list(_REGION_RE.finditer(text))
    indexes = [int(match.group("index")) for match in matches]
    unique: list[int] = []
    for index in indexes:
        if index not in unique:
            unique.append(index)
    invalid = (
        [
            index
            for index in unique
            if index < 0 or (proposal_count is not None and index >= proposal_count)
        ]
        if proposal_count is not None
        else []
    )
    valid = [index for index in unique if index not in invalid]
    grounding_blocks = re.findall(
        r"<ground>(?P<label>.*?)</ground>\s*<objects>(?P<objects>.*?)</objects>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return {
        # An empty <objects></objects> block is valid evidence for a zero-count
        # answer; absence of the official block is the parse failure.
        "parse_ok": bool(grounding_blocks) and not invalid,
        "selected_region_indexes": valid,
        "invalid_region_indexes": invalid,
        "raw_region_indexes": indexes,
        "grounding_blocks": [
            {"label": label.strip(), "objects": objects.strip()}
            for label, objects in grounding_blocks
        ],
        "parse_error": (
            "missing_official_region_markup"
            if not grounding_blocks
            else "region_index_out_of_range"
            if invalid
            else None
        ),
    }


def parse_fo1_count(output: str) -> tuple[int | None, str | None]:
    """Parse an integer/JSON answer without overriding official region evidence."""

    cleaned = re.sub(r"</?(?:ground|objects|region\d+|think)[^>]*>", " ", str(output), flags=re.I)
    result = parse_count(cleaned)
    return result.value, result.reason


def compact_proposal_evidence(
    boxes: Sequence[Sequence[float]],
    scores: Sequence[float],
    selected_indexes: Sequence[int],
) -> tuple[list[list[float]], list[float]]:
    """Return selected boxes/scores with strict index bounds."""

    selected_boxes: list[list[float]] = []
    selected_scores: list[float] = []
    for index in selected_indexes:
        if index < 0 or index >= len(boxes):
            raise IndexError(f"selected region index out of range: {index}")
        selected_boxes.append([float(value) for value in boxes[index]])
        selected_scores.append(float(scores[index]) if index < len(scores) else float("nan"))
    return selected_boxes, selected_scores


def request_has_reference_leak(request: Mapping[str, Any]) -> bool:
    """Guard the sidecar boundary against reference-answer leakage."""

    forbidden = {"reference", "answer", "ground_truth", "groundtruth", "label"}
    if any(str(key).lower() in forbidden for key in request):
        return True
    nested = request.get("metadata")
    if isinstance(nested, Mapping):
        return any(str(key).lower() in forbidden for key in nested)
    return False


def prediction_count_text(count: int | None) -> str:
    """Convert a parsed count to the standard prediction string."""

    return "" if count is None else str(int(count))


def counting_json_prediction(count: int | None) -> str:
    """Expose the repository's JSON counting protocol for tests and adapters."""

    return counting_json(count) or ""


def is_supported_prompt_profile(profile: str) -> bool:
    return profile in FO1_PROMPT_PROFILES


def protocol_error(message: str) -> dict[str, Any]:
    return {"status": "failed", "failure_stage": "protocol_guard", "error": message}


def json_dumps_compact(payload: Mapping[str, Any]) -> str:
    return json.dumps(dict(payload), ensure_ascii=False, separators=(",", ":"))
