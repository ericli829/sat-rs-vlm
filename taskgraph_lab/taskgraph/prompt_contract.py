from __future__ import annotations

import re

from .enums import OperatorName
from .type_checker import SIGNATURES

_HEADING = re.compile(r"(?m)^([A-Z][A-Z_]*)\s*$")
_ROLE = re.compile(
    r"^\s+(?P<name>[a-z][a-z_]*)(?:\s+optional)?:\s+"
    r"(?P<types>[A-Za-z][A-Za-z0-9_]*(?:\s*\|\s*[A-Za-z][A-Za-z0-9_]*)*)\s*$"
)


def _operator_section(prompt: str, operator: OperatorName) -> str | None:
    matches = list(_HEADING.finditer(prompt))
    fallback: str | None = None
    for index, match in enumerate(matches):
        if match.group(1) != operator.value:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(prompt)
        section = prompt[match.end() : end]
        fallback = section
        if re.search(r"(?m)^Inputs(?: \([^\n]+\))?:", section) and re.search(
            r"(?m)^Params:", section
        ):
            return section
    return fallback


def prompt_operator_contract_issues(prompt: str) -> list[str]:
    """Compare prompt input role/type declarations with the runtime registry."""
    issues: list[str] = []
    for operator, signature in SIGNATURES.items():
        section = _operator_section(prompt, operator)
        if section is None:
            issues.append(f"{operator.value}: missing operator section")
            continue
        inputs_match = re.search(
            r"(?ms)^Inputs(?: \([^\n]+\))?:\s*\n(?P<body>.*?)(?=^Params:)", section
        )
        if inputs_match is None:
            issues.append(f"{operator.value}: missing explicit Inputs/Params contract")
            continue
        declared: dict[str, set[str]] = {}
        for line in inputs_match.group("body").splitlines():
            role_match = _ROLE.match(line)
            if role_match is None:
                continue
            declared[role_match.group("name")] = {
                value.strip() for value in role_match.group("types").split("|")
            }
        expected = {
            name: {runtime_type.value for runtime_type in runtime_types}
            for name, runtime_types in signature.items()
        }
        if set(declared) != set(expected):
            issues.append(
                f"{operator.value}: role names declared={sorted(declared)} "
                f"expected={sorted(expected)}"
            )
            continue
        for name in expected:
            if declared[name] != expected[name]:
                issues.append(
                    f"{operator.value}.{name}: types declared={sorted(declared[name])} "
                    f"expected={sorted(expected[name])}"
                )
    return issues
