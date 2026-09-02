"""TargetSpec 与开放词汇 prompt profile。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from .paths import PROMPT_CONFIG, load_prompt_profiles

_WORD = re.compile(r"[a-zA-Z][a-zA-Z\- ]{1,40}")
_HOW_MANY = re.compile(
    r"how many\s+([a-zA-Z][a-zA-Z \-]+?)(?:\s+(?:are|is|can|do|does|in|on|at|near|next|beside|along|within|inside|around)\b|\s*[?.,]|$)",
    re.IGNORECASE,
)
_NUMBER_OF = re.compile(
    r"(?:number|count|quantity)\s+of\s+([a-zA-Z][a-zA-Z \-]+?)(?:\s+(?:are|is|in|on|at)\b|\s*[?.,]|$)",
    re.IGNORECASE,
)


@dataclass(slots=True)
class TargetSpec:
    """对齐 taskgraph.schema.TargetSpec：category + intrinsic attributes + phrase()。"""

    category: str
    attributes: dict[str, str | int | float | bool] = field(default_factory=dict)
    prompt: str = ""
    aliases: list[str] = field(default_factory=list)
    tiny: bool = False
    profile_name: str = ""

    @property
    def name(self) -> str:
        return self.category

    def phrase(self) -> str:
        prefix = " ".join(str(value) for value in self.attributes.values())
        return f"{prefix} {self.category}".strip()

    def texts(self) -> str:
        return self.prompt or self.phrase()

    def to_params(self) -> dict[str, Any]:
        return {"category": self.category, "attributes": dict(self.attributes)}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _singularize(name: str, book: dict[str, Any] | None = None) -> str:
    book = book or prompt_book()
    text = _normalize(name)
    if resolve_profile(text, book):
        return text
    candidates = []
    if text.endswith("ies") and len(text) > 4:
        candidates.append(text[:-3] + "y")
    if text.endswith("es") and len(text) > 3:
        candidates.append(text[:-2])
    if text.endswith("s") and len(text) > 2:
        candidates.append(text[:-1])
    for cand in candidates:
        if resolve_profile(cand, book):
            return cand
    return candidates[0] if candidates else text


@lru_cache(maxsize=1)
def prompt_book() -> dict[str, Any]:
    if not Path(PROMPT_CONFIG).exists():
        return {"profiles": {}, "tiny_classes": []}
    return load_prompt_profiles()


def is_tiny_class(name: str, book: dict[str, Any] | None = None) -> bool:
    book = book or prompt_book()
    key = _normalize(name)
    profile = book.get("profiles", {}).get(key.replace(" ", "_")) or book.get("profiles", {}).get(key)
    if isinstance(profile, dict) and "tiny" in profile:
        return bool(profile["tiny"])
    tiny = {_normalize(x) for x in book.get("tiny_classes") or []}
    return key in tiny or any(token in tiny for token in key.split())


def resolve_profile(name: str, book: dict[str, Any] | None = None) -> dict[str, Any]:
    book = book or prompt_book()
    key = _normalize(name)
    profiles: dict[str, Any] = book.get("profiles") or {}
    if key in profiles:
        return profiles[key]
    collapsed = key.replace(" ", "_")
    if collapsed in profiles:
        return profiles[collapsed]
    for alias, profile in profiles.items():
        variants = [_normalize(alias), _normalize(profile.get("canonical", ""))]
        variants.extend(_normalize(v) for v in profile.get("variants") or [])
        if key in variants:
            return profile
    return {}


def build_target(name: str, prompt: str | None = None) -> TargetSpec:
    profile = resolve_profile(name)
    canonical = _normalize(profile.get("canonical") or name)
    aliases = [_normalize(v) for v in profile.get("variants") or []]
    if canonical not in aliases:
        aliases.insert(0, canonical)
    return TargetSpec(
        category=canonical,
        prompt=prompt or profile.get("default") or canonical,
        aliases=aliases,
        tiny=bool(profile.get("tiny", is_tiny_class(canonical))),
        profile_name=canonical,
    )


def extract_target_from_question(question: str) -> TargetSpec:
    text = question.strip()
    match = _HOW_MANY.search(text) or _NUMBER_OF.search(text)
    raw = match.group(1) if match else ""
    raw = re.sub(r"\b(the|a|an|of|those|these|visible|shown)\b", " ", raw, flags=re.IGNORECASE)
    raw = _normalize(raw).strip(" .,:;")
    if not raw:
        words = [w.lower() for w in _WORD.findall(text)]
        book = prompt_book()
        for n in range(min(3, len(words)), 0, -1):
            for i in range(len(words) - n + 1):
                cand = " ".join(words[i : i + n])
                if resolve_profile(cand, book):
                    raw = cand
                    break
            if raw:
                break
    if not raw:
        raw = "object"
    raw = _singularize(raw)
    return build_target(raw)


def iter_prompt_variants(target: TargetSpec) -> Iterable[str]:
    profile = resolve_profile(target.name)
    variants = list(profile.get("variants") or [])
    if target.prompt and target.prompt not in variants:
        variants.insert(0, target.prompt)
    if not variants:
        variants = [target.texts()]
    seen: set[str] = set()
    for item in variants:
        key = _normalize(item)
        if key in seen:
            continue
        seen.add(key)
        yield item
