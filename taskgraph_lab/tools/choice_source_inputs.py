"""Normalize the explicitly paired HRSCD keys used in XLRS choice sources.

This narrow dataset-release helper does not interpret arbitrary colons, Windows
drive letters, URLs, or change production image handling.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

PAIRED_HRSCD = re.compile(
    r"^(HRSCD-\d+/[^:\r\n]+\.(?:tiff?|png|jpe?g)):(HRSCD-\d+/[^:\r\n]+\.(?:tiff?|png|jpe?g))$",
    re.I,
)


def normalize_hrscd_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(inputs)
    if set(normalized) != {"image0"}:
        return normalized
    image = normalized["image0"]
    if image.get("type") != "image" or not isinstance(image.get("uri_or_key"), str):
        return normalized
    pair = PAIRED_HRSCD.fullmatch(image["uri_or_key"])
    if pair is None:
        return normalized
    return {
        f"image{index}": {**image, "uri_or_key": uri} for index, uri in enumerate(pair.groups())
    }
