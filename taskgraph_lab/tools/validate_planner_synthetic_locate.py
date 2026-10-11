"""Revalidate generated records independently; optionally exhaust all prefixes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from taskgraph_lab.synthetic_locate import (
    ROOT,
    generate_parameters,
    quality_statistics,
    sft_row,
    sha256_file,
    validate_sample,
)
from taskgraph_lab.taskgraph.dsl.constraint import CanonicalDSLPrefixGrammar


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--every-prefix", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    folder = args.dataset_dir
    config = yaml.safe_load((folder / "generator_config.yaml").read_text(encoding="utf-8"))
    manifest = json.loads((folder / "dataset_manifest.json").read_text(encoding="utf-8"))
    for name, details in manifest["files"].items():
        if sha256_file(folder / name) != details["sha256"]:
            raise ValueError(f"changed generated file: {name}")
    system = (
        (ROOT / "taskgraph_lab/prompts/planner_student_system_prompt.txt")
        .read_text(encoding="utf-8")
        .strip()
    )
    rows = [
        json.loads(line)
        for line in (folder / "synthetic_samples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    old_questions = (
        ROOT / "taskgraph_lab/data/questions/semantic_locate_v1_20261010/questions.jsonl"
    )
    excluded = (
        {
            json.loads(line)["question"].strip().casefold()
            for line in old_questions.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        if old_questions.exists()
        else set()
    )
    expected_params, expected_groups = generate_parameters(config, excluded)
    if (
        [r["generation"] for r in rows] != expected_params
        or manifest["split_categories"] != expected_groups
        or len(rows) != config["count"]
    ):
        raise ValueError("samples/configuration/seed/split assignment are not reproducible")
    grammar = CanonicalDSLPrefixGrammar(["image0"])
    prefixes = 0
    checked_prefixes: set[str] = set()
    for row in rows:
        checks = validate_sample(row, config, system, grammar=grammar)
        if checks != row["metadata"]["validation"]:
            raise ValueError("recorded validation disagrees with rerun")
        if args.every_prefix:
            prefixes += len(row["dsl"]) + 1
            # Stateless grammar + fixed image bindings: identical prefixes have
            # identical results. Cache only exact strings, never syntax classes.
            for index in range(len(row["dsl"]) + 1):
                prefix = row["dsl"][:index]
                if prefix not in checked_prefixes:
                    if not grammar.analyze(prefix).valid_prefix:
                        raise ValueError(f"{row['id']}: invalid prefix at {index}")
                    checked_prefixes.add(prefix)
    for split in ("train", "validation"):
        actual = [
            json.loads(line)
            for line in (folder / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        expected = [sft_row(row) for row in rows if row["generation"]["split"] == split]
        if actual != expected:
            raise ValueError(f"{split} differs from audited synthetic records")
    stats = quality_statistics(rows)
    for key in (
        "exact_text_duplicates",
        "cross_split_exact_text",
        "cross_split_categories",
        "cross_split_template_families",
        "cross_split_near_template_skeletons",
    ):
        if stats[key]:
            raise ValueError(f"split leakage or duplicate: {key}")
    if stats != manifest["statistics"]:
        # JSON object integer keys roundtrip to strings.
        if json.loads(json.dumps(stats)) != manifest["statistics"]:
            raise ValueError("manifest statistics differ from actual data")
    report = {
        "valid": True,
        "count": len(rows),
        "all_prefixes_checked": args.every_prefix,
        "prefixes_checked": prefixes,
        "unique_prefixes_checked": len(checked_prefixes),
        "checks": rows[0]["metadata"]["validation"],
        "statistics": stats,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"valid": True, "count": len(rows), "prefixes_checked": prefixes}, ensure_ascii=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
