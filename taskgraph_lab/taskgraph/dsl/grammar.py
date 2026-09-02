from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .errors import DSLParseError

EBNF = r"""
program       = [intent], node, {node}, final;
intent        = "INTENT", "(", enum, ")";
node          = node_id, "=", call;
final         = ("FINAL" | "FINAL_QUESTION"), "(", arguments, ")";
call          = identifier, "(", [arguments], ")";
arguments     = argument, {",", argument};
argument      = [identifier, "="], value;
value         = reference | string | number | boolean | "null" | enum | list | call;
list          = "[", [value, {",", value}], "]";
reference     = "$image", digit, {digit} | "$n", nonzero_digit, {digit};
node_id       = "n", nonzero_digit, {digit};
boolean       = "true" | "false";
string        = JSON-string;
""".strip()

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
_REFERENCE = re.compile(r"\$(?:image(?:0|[1-9][0-9]*)|n[1-9][0-9]*)")


@dataclass(frozen=True)
class Symbol:
    value: str


@dataclass(frozen=True)
class Reference:
    value: str


@dataclass(frozen=True)
class Call:
    name: str
    args: tuple[Any, ...]
    kwargs: tuple[tuple[str, Any], ...]

    def keyword_map(self) -> dict[str, Any]:
        return dict(self.kwargs)


@dataclass(frozen=True)
class NodeStatement:
    node_id: str
    call: Call


@dataclass(frozen=True)
class Program:
    intent: Call | None
    nodes: tuple[NodeStatement, ...]
    final: Call


@dataclass(frozen=True)
class Token:
    kind: str
    value: Any
    offset: int
    line: int
    column: int


def _location(text: str, offset: int) -> tuple[int, int]:
    line = text.count("\n", 0, offset) + 1
    previous = text.rfind("\n", 0, offset)
    column = offset - previous
    return line, column


def _fail(text: str, offset: int, message: str) -> DSLParseError:
    line, column = _location(text, offset)
    return DSLParseError(f"{message} at line {line}, column {column}")


def tokenize(text: str) -> list[Token]:
    tokens: list[Token] = []
    offset = 0
    length = len(text)
    decoder = json.JSONDecoder()
    while offset < length:
        char = text[offset]
        if char.isspace():
            offset += 1
            continue
        line, column = _location(text, offset)
        if char in "=(),[]":
            tokens.append(Token(char, char, offset, line, column))
            offset += 1
            continue
        if char == '"':
            try:
                value, consumed = decoder.raw_decode(text[offset:])
            except json.JSONDecodeError as exc:
                raise _fail(text, offset, f"malformed JSON string: {exc.msg}") from exc
            if not isinstance(value, str):
                raise _fail(text, offset, "expected a JSON string")
            tokens.append(Token("STRING", value, offset, line, column))
            offset += consumed
            continue
        if char == "$":
            match = _REFERENCE.match(text, offset)
            if match is None:
                raise _fail(text, offset, "invalid reference")
            end = match.end()
            if end < length and (text[end].isalnum() or text[end] == "_"):
                raise _fail(text, offset, "invalid reference")
            value = match.group(0)
            tokens.append(Token("REF", value, offset, line, column))
            offset = end
            continue
        number = _NUMBER.match(text, offset)
        if number is not None:
            end = number.end()
            if end < length and (text[end].isalnum() or text[end] in "._"):
                raise _fail(text, offset, "invalid number")
            raw = number.group(0)
            value: int | float = (
                float(raw) if any(marker in raw for marker in (".", "e", "E")) else int(raw)
            )
            tokens.append(Token("NUMBER", value, offset, line, column))
            offset = end
            continue
        identifier = _IDENTIFIER.match(text, offset)
        if identifier is not None:
            raw = identifier.group(0)
            keyword_values = {"true": True, "false": False, "null": None}
            if raw in keyword_values:
                tokens.append(Token("LITERAL", keyword_values[raw], offset, line, column))
            else:
                tokens.append(Token("IDENT", raw, offset, line, column))
            offset = identifier.end()
            continue
        raise _fail(text, offset, f"unexpected character {char!r}")
    line, column = _location(text, length)
    tokens.append(Token("EOF", None, length, line, column))
    return tokens


class SyntaxParser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.tokens = tokenize(text)
        self.index = 0

    @property
    def current(self) -> Token:
        return self.tokens[self.index]

    def _peek(self, distance: int = 1) -> Token:
        index = min(self.index + distance, len(self.tokens) - 1)
        return self.tokens[index]

    def _error(self, message: str, token: Token | None = None) -> DSLParseError:
        item = token or self.current
        return DSLParseError(f"{message} at line {item.line}, column {item.column}")

    def _consume(self, kind: str, value: str | None = None) -> Token:
        token = self.current
        if token.kind != kind or (value is not None and token.value != value):
            expected = value if value is not None else kind
            raise self._error(f"expected {expected}, got {token.value!r}", token)
        self.index += 1
        return token

    def parse(self) -> Program:
        if self.current.kind == "EOF":
            raise self._error("empty DSL program")
        intent: Call | None = None
        nodes: list[NodeStatement] = []
        final: Call | None = None
        if self.current.kind == "IDENT" and self.current.value == "INTENT":
            intent = self._parse_call()
        while self.current.kind != "EOF":
            if self.current.kind != "IDENT":
                raise self._error("expected a node assignment or FINAL")
            name = str(self.current.value)
            if name in {"FINAL", "FINAL_QUESTION"}:
                if final is not None:
                    raise self._error("multiple FINAL statements are forbidden")
                final = self._parse_call()
                if self.current.kind != "EOF":
                    raise self._error("no node or statement is allowed after FINAL")
                break
            if final is not None:
                raise self._error("no node is allowed after FINAL")
            node_id = self._consume("IDENT").value
            self._consume("=")
            call = self._parse_call()
            nodes.append(NodeStatement(str(node_id), call))
        if not nodes:
            raise self._error("DSL program requires at least one node")
        if final is None:
            raise self._error("DSL program requires exactly one FINAL statement")
        return Program(intent=intent, nodes=tuple(nodes), final=final)

    def _parse_call(self) -> Call:
        name = str(self._consume("IDENT").value)
        self._consume("(")
        args: list[Any] = []
        kwargs: list[tuple[str, Any]] = []
        seen_keywords: set[str] = set()
        if self.current.kind != ")":
            while True:
                if self.current.kind == "IDENT" and self._peek().kind == "=":
                    key = str(self._consume("IDENT").value)
                    self._consume("=")
                    if key in seen_keywords:
                        raise self._error(f"duplicate named argument {key!r}")
                    seen_keywords.add(key)
                    kwargs.append((key, self._parse_value()))
                else:
                    if kwargs:
                        raise self._error("positional arguments cannot follow named arguments")
                    args.append(self._parse_value())
                if self.current.kind != ",":
                    break
                self._consume(",")
                if self.current.kind == ")":
                    raise self._error("trailing commas are forbidden")
        self._consume(")")
        return Call(name=name, args=tuple(args), kwargs=tuple(kwargs))

    def _parse_value(self) -> Any:
        token = self.current
        if token.kind in {"STRING", "NUMBER", "LITERAL"}:
            self.index += 1
            return token.value
        if token.kind == "REF":
            self.index += 1
            return Reference(str(token.value))
        if token.kind == "IDENT":
            if self._peek().kind == "(":
                return self._parse_call()
            self.index += 1
            return Symbol(str(token.value))
        if token.kind == "[":
            return self._parse_list()
        raise self._error(f"expected a DSL value, got {token.value!r}")

    def _parse_list(self) -> list[Any]:
        self._consume("[")
        values: list[Any] = []
        if self.current.kind != "]":
            while True:
                values.append(self._parse_value())
                if self.current.kind != ",":
                    break
                self._consume(",")
                if self.current.kind == "]":
                    raise self._error("trailing commas are forbidden")
        self._consume("]")
        return values


def parse_syntax(text: str) -> Program:
    return SyntaxParser(text).parse()
