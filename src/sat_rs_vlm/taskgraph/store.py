"""Reference-addressed typed runtime store."""

from __future__ import annotations

from collections.abc import Mapping

from .runtime_types import RuntimeObject


class RuntimeStore:
    def __init__(self, initial: Mapping[str, RuntimeObject] | None = None) -> None:
        self._values: dict[str, RuntimeObject] = {}
        for ref, value in (initial or {}).items():
            self.put(ref, value)

    @staticmethod
    def canonical_ref(ref: str) -> str:
        if not ref.startswith("$"):
            ref = f"${ref}"
        return ref

    def put(self, ref: str, value: RuntimeObject) -> None:
        key = self.canonical_ref(ref)
        if key in self._values:
            raise KeyError(f"runtime ref already exists: {key}")
        self._values[key] = value

    def get(self, ref: str) -> RuntimeObject:
        key = self.canonical_ref(ref)
        try:
            return self._values[key]
        except KeyError as exc:
            raise KeyError(f"runtime ref is not available: {key}") from exc

    def resolve(self, value: str | list[str]) -> RuntimeObject | list[RuntimeObject]:
        if isinstance(value, list):
            return [self.get(ref) for ref in value]
        return self.get(value)

    def snapshot(self) -> dict[str, RuntimeObject]:
        return dict(self._values)
