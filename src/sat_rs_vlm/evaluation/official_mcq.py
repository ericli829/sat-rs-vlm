"""Dataset-native multiple-choice parsing for official evaluation profiles."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

MME_ANSWER_PREFIXES = (
    "The best answer is",
    "The correct answer is",
    "The answer is",
    "The answer",
    "The best option is",
    "The correct option is",
    "Best answer:",
    "Best option:",
)
XLRS_ANSWER_PREFIXES = (
    "The best answer is",
    "The correct answer is",
    "The answer is",
    "The answer",
    "The best option isThe correct option is",
    "Best answer:Best option:",
)


@dataclass(frozen=True)
class ChoiceParseResult:
    choices: tuple[str, ...]
    parse_ok: bool
    reason: str | None
    parser_profile: str


def _strip_prefixes(text: str, prefixes: Iterable[str]) -> str:
    value = text.strip()
    for prefix in prefixes:
        value = value.replace(prefix, "")
    return value.strip()


def parse_mme_realworld_choice(
    text: str,
    answer_choices: Iterable[str] = (),
) -> ChoiceParseResult:
    """Mirror the official MME-RealWorld first-uppercase-A-E parser."""

    candidate = _strip_prefixes(text, MME_ANSWER_PREFIXES)
    if len(candidate.split()) > 10 and re.search(r"[ABCDE]", candidate) is None:
        return ChoiceParseResult((), False, "no_official_choice", "mme_realworld_official")
    match = re.search(r"[ABCDE]", candidate)
    if match is not None:
        return ChoiceParseResult((match.group(0),), True, None, "mme_realworld_official")
    lowered = candidate.lower()
    for choice in answer_choices:
        option = str(choice)
        if lowered in option.lower():
            label = re.match(r"^\s*[（(]([A-Ea-e])[）)]", option)
            if label is not None:
                return ChoiceParseResult(
                    (label.group(1).upper(),),
                    True,
                    None,
                    "mme_realworld_official",
                )
    return ChoiceParseResult((), False, "no_official_choice", "mme_realworld_official")


def parse_xlrs_choices(text: str) -> ChoiceParseResult:
    """Mirror lmms-eval XLRS extraction while returning deterministic sets."""

    candidate = _strip_prefixes(text, XLRS_ANSWER_PREFIXES)
    if re.search(r"[ABCDEabcde]", candidate) is None:
        return ChoiceParseResult((), False, "no_official_choice", "xlrs_lmms_eval")
    matches = re.findall(r"\(([a-eA-E])\)", candidate)
    if not matches:
        matches = re.findall(r"(?:^|\s)?([a-eA-E])(?:$|[\s,.])?", candidate)
    if not matches:
        matches = re.findall(r"[a-eA-E]", candidate)
    choices = tuple(sorted({match.upper() for match in matches}))
    return ChoiceParseResult(
        choices,
        bool(choices),
        None if choices else "no_official_choice",
        "xlrs_lmms_eval",
    )


def parse_reference_choices(
    text: str,
    *,
    allowed: frozenset[str],
    single: bool,
) -> ChoiceParseResult:
    """Parse a dataset answer key without applying permissive prediction rules."""

    choices = tuple(sorted(set(re.findall(r"[A-Za-z]", text.upper())) & allowed))
    expected = 1 if single else None
    valid_count = len(choices) == expected if expected is not None else bool(choices)
    if not valid_count:
        reason = "reference_must_contain_one_choice" if single else "reference_has_no_choice"
        return ChoiceParseResult((), False, reason, "strict_reference")
    residue = re.sub(r"[\s,;|/()[\]{}]+", "", text.upper())
    if any(character not in allowed for character in residue):
        return ChoiceParseResult((), False, "reference_contains_invalid_choice", "strict_reference")
    return ChoiceParseResult(choices, True, None, "strict_reference")
