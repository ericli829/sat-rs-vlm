from __future__ import annotations

from typing import Protocol

from PIL import Image


class Retriever(Protocol):
    name: str

    def score(self, image: Image.Image, text: str) -> float: ...

    def close(self) -> None: ...
