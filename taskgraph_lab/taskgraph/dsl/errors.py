from __future__ import annotations


class DSLError(ValueError):
    """Base error for the TaskGraph DSL serialization boundary."""


class DSLParseError(DSLError):
    """Raised when DSL text cannot be lowered to a valid canonical TaskGraph."""


class DSLCompileError(DSLError):
    """Raised when a canonical TaskGraph cannot be serialized without loss."""
