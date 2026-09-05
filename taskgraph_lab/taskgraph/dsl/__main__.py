from __future__ import annotations

import argparse
import json
from pathlib import Path

from taskgraph_lab.taskgraph.canonicalize import canonicalize_target

from .compiler import compile_taskgraph_to_dsl
from .parser import parse_taskgraph_dsl


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile or parse the TaskGraph v1.1 DSL")
    subparsers = parser.add_subparsers(dest="command", required=True)
    compile_parser = subparsers.add_parser("compile", help="compile canonical JSON to DSL")
    compile_parser.add_argument("path", type=Path)
    parse_parser = subparsers.add_parser("parse", help="parse DSL to canonical JSON")
    parse_parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.command == "compile":
        graph = json.loads(args.path.read_text(encoding="utf-8"))
        print(compile_taskgraph_to_dsl(graph))
    else:
        target = parse_taskgraph_dsl(args.path.read_text(encoding="utf-8"))
        print(json.dumps(canonicalize_target(target), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
