from __future__ import annotations

from PIL import Image


class FakeRetriever:
    name = "fake"

    def __init__(self, default: float = 1.0):
        self.default = default

    def score(self, image: Image.Image, text: str) -> float:
        _ = image, text
        return float(self.default)

    def close(self) -> None:
        return None
