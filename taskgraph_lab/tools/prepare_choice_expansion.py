"""Prepare an answer-free, source-backed Codex choice conversion queue."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from taskgraph_lab.datasets.mme_rs import iter_mme_rs
from taskgraph_lab.datasets.xlrs import iter_xlrs
from taskgraph_lab.tools.build_seed_set import structure_category
from taskgraph_lab.tools.choice_source_inputs import normalize_hrscd_inputs


def key(question: str) -> str:
    return " ".join(question.casefold().split())


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def prepare(
    *,
    base_train: Path,
    fixed_test: Path,
    mme_json: Path,
    xlrs_json: Path,
    output_dir: Path,
    target_count: int = 4500,
    seed: int = 20261010,
) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)
    base, test = read_rows(base_train), read_rows(fixed_test)
    if len({r["id"] for r in base}) != len(base):
        raise ValueError("base must contain independent rows, not curriculum exposure copies")
    needed = target_count - len(base)
    if needed < 1:
        raise ValueError("target must exceed the independent base population")
    excluded_ids = {r["id"] for r in base + test}
    excluded_questions = {
        key(json.loads(r["messages"][1]["content"])["question"]) for r in base + test
    }
    candidates = [
        s
        for s in [*iter_mme_rs(mme_json), *iter_xlrs(xlrs_json)]
        if s.choices
        and s.sample_id not in excluded_ids
        and key(s.question) not in excluded_questions
    ]
    semantic_hint = re.compile(
        r"\b(?:near|next to|beside|surrounded|adjacent|immediately|between|bordered|"
        r"along|around|connected|leading|parallel|with .* (?:right|left|above|below))\b",
        re.I,
    )
    rng = random.Random(seed)
    rng.shuffle(candidates)
    # Retrieval labels affect sampling only. Codex must independently decide the
    # logical graph and whether SEMANTIC, geometry, or an intrinsic target fits.
    groups = {}
    for sample in candidates:
        name = sample.metadata["dataset"]
        row = sample.model_dump(mode="json")
        normalized_inputs = normalize_hrscd_inputs(row["inputs"])
        if normalized_inputs != row["inputs"]:
            row["metadata"]["source_original_inputs"] = row["inputs"]
            row["metadata"]["input_normalization"] = "explicit_HRSCD_pair_to_image0_image1"
            row["inputs"] = normalized_inputs
        row["metadata"]["retrieval_semantic_hint"] = bool(semantic_hint.search(sample.question))
        row["metadata"]["sampling_structure"] = structure_category(sample)
        groups.setdefault(name, []).append(row)
    quotas = {"MME_RealWorld_RS": round(needed * 850 / 1685)}
    quotas["XLRS_Bench"] = needed - quotas["MME_RealWorld_RS"]
    selected = []
    seen_questions = set()
    for name, quota in quotas.items():
        pool = groups[name]
        # Prefer external-context questions, then non-bbox source diversity.
        # All queues are seeded; expensive complex examples are not excluded.
        pool.sort(
            key=lambda s: (
                not s["metadata"]["retrieval_semantic_hint"],
                s["metadata"]["sampling_structure"].startswith("bbox"),
            )
        )
        chosen = []
        for row in pool:
            question_key = key(row["question"])
            if question_key in seen_questions:
                continue
            chosen.append(row)
            seen_questions.add(question_key)
            if len(chosen) == quota:
                break
        if len(chosen) != quota:
            raise ValueError(f"insufficient unique {name} candidates: {len(chosen)}/{quota}")
        selected.extend(chosen)
    output_dir.mkdir(parents=True)
    (output_dir / "plans").mkdir()
    write_rows(output_dir / "seed.jsonl", selected)
    mme = [s for s in selected if s["metadata"]["dataset"] == "MME_RealWorld_RS"]
    xlrs = [s for s in selected if s["metadata"]["dataset"] == "XLRS_Bench"]
    packs = {
        "mme_a": mme[::2],
        "mme_b": mme[1::2],
        "xlrs": [s for s in xlrs if not s["metadata"]["sampling_structure"].startswith("bbox")],
        "xlrs_bbox": [s for s in xlrs if s["metadata"]["sampling_structure"].startswith("bbox")],
    }
    for name, rows in packs.items():
        write_rows(output_dir / f"{name}.jsonl", rows)
    manifest = {
        "version": "choice-expansion-codex-v1",
        "seed": seed,
        "target_training_count": target_count,
        "base_training_count": len(base),
        "selected_new_count": len(selected),
        "selected_sample_ids": [s["sample_id"] for s in selected],
        "fixed_test_count": len(test),
        "fixed_test_sha256": digest(fixed_test),
        "sources": {
            str(p.resolve()): digest(p) for p in (base_train, fixed_test, mme_json, xlrs_json)
        },
        "remaining_candidates_before_new_question_dedup": len(candidates),
        "selected_per_dataset": dict(Counter(s["metadata"]["dataset"] for s in selected)),
        "selected_structure_retrieval": dict(
            Counter(s["metadata"]["sampling_structure"] for s in selected)
        ),
        "semantic_hint_candidates": sum(s["metadata"]["retrieval_semantic_hint"] for s in selected),
        "packs": {name: len(rows) for name, rows in packs.items()},
        "exact_new_train_test_question_overlap": 0,
        "source_answers_in_teacher_input": False,
        "sampling_labels_are_not_gold": True,
        "paired_inputs_normalized": [
            s["sample_id"] for s in selected if "input_normalization" in s["metadata"]
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {k: v for k, v in manifest.items() if k not in {"selected_sample_ids", "sources"}},
            ensure_ascii=False,
            indent=2,
        )
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base-train", "fixed-test", "mme-json", "xlrs-json", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--target-count", type=int, default=4500)
    parser.add_argument("--seed", type=int, default=20261010)
    prepare(**vars(parser.parse_args()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
