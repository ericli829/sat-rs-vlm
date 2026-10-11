"""Resumable, bounded CPU-only cleaning of existing Planner datasets.

Run `audit --offset N --limit 1000` in short batches, then `release`. Source
files are immutable and their hashes are checked again before publication.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

from taskgraph_lab.quality.planner_cleaning import (
    VERSION,
    digest,
    identity_key,
    issue,
    question_key,
    repair,
    semantic_findings,
    unpack,
    validate,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "taskgraph_lab/data"
OUTPUT = DATA / "planner_quality_clean_v1"
REVIEWS = ROOT / "reports/taskgraph/planner_quality_clean_v1"
SOURCES = {
    "choice_train": DATA / "planner_choice_expanded_v1/train.jsonl",
    "choice_test": DATA / "planner_choice_expanded_v1/test.jsonl",
    "locate_train": DATA / "planner_synthetic_locate_v1/train.jsonl",
    "locate_validation": DATA / "planner_synthetic_locate_v1/validation.jsonl",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text(encoding="utf-8-sig").splitlines() if s.strip()]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )


def source_hashes() -> dict[str, str]:
    import hashlib

    return {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in SOURCES.values()
    }


def inventory() -> tuple[list[tuple[str, dict]], dict]:
    rows = [(split, r) for split, path in SOURCES.items() for r in read_jsonl(path)]
    canonical = {str(r["id"]) for _, r in rows}
    history = []
    for p in sorted(DATA.glob("*/*.jsonl")):
        if p.name not in {"train.jsonl", "validation.jsonl", "test.jsonl"} or p.parent == OUTPUT:
            continue
        old = read_jsonl(p)
        mapped = [str(r.get("metadata", {}).get("curriculum_source_id") or r["id"]) for r in old]
        history.append(
            {
                "path": str(p.relative_to(ROOT)),
                "rows": len(old),
                "unique_source_ids": len(set(mapped)),
                "canonical_sources_covered": len(set(mapped) & canonical),
                "outside_sources": sorted(set(mapped) - canonical),
            }
        )
    return rows, {
        "version": VERSION,
        "source_sha256": source_hashes(),
        "sources": {s: str(p.relative_to(ROOT)) for s, p in SOURCES.items()},
        "total_unique_source_rows": len(rows),
        "history": history,
    }


def audit(offset: int, limit: int) -> dict:
    rows, manifest = inventory()
    manifest_path = OUTPUT / "source_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf8"))
        if previous != manifest:
            raise ValueError("source files or historical inventory changed during audit")
    else:
        write_json(manifest_path, manifest)
    result = []
    for index, (split, row) in enumerate(rows[offset : offset + limit], offset):
        findings, changes = [], []
        cleaned = deepcopy(row)
        try:
            cleaned, changes = repair(row)
            user, dsl = unpack(cleaned)
            structural = validate(dsl, user, str(row["id"]))
            if not structural["valid"]:
                findings.append(issue("structural_invalid", structural["detail"]))
            findings += semantic_findings(user, dsl)
        except Exception as exc:
            structural = {"valid": False, "detail": f"{type(exc).__name__}: {exc}"}
            findings.append(issue("structural_invalid", structural["detail"]))
        result.append(
            {
                "index": index,
                "split": split,
                "id": row["id"],
                "source_row_sha256": digest(json.dumps(row, sort_keys=True)),
                "before_dsl": row["messages"][2]["content"],
                "row": cleaned,
                "changes": changes,
                "findings": findings,
                "validation": structural,
            }
        )
    path = OUTPUT / "shards" / f"audit_{offset:05d}_{len(result):04d}.jsonl"
    write_jsonl(path, result)
    summary = {
        "offset": offset,
        "rows": len(result),
        "path": str(path),
        "flagged_rows": sum(bool(r["findings"]) for r in result),
        "repaired_rows": sum(bool(r["changes"]) for r in result),
        "findings": dict(Counter(f["code"] for r in result for f in r["findings"])),
        "repairs": dict(Counter(f["code"] for r in result for f in r["changes"])),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def load_reviews() -> tuple[dict[str, list[dict]], dict]:
    reviews = defaultdict(list)
    coverage = {}
    for path in sorted(REVIEWS.glob("*_review.json")):
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        for f in payload.get("findings", []):
            reviews[str(f["id"])].append({**f, "review_file": str(path.relative_to(ROOT))})
        coverage[str(path.relative_to(ROOT))] = payload.get("covered_ids", [])
    resolutions = REVIEWS / "review_resolutions.json"
    if resolutions.exists():
        for resolution in json.loads(resolutions.read_text(encoding="utf8"))["decisions"]:
            sample_id = resolution["id"]
            for f in reviews[sample_id]:
                if digest(f["question"]) != resolution["question_sha256"]:
                    raise ValueError("manual review resolution/source mismatch")
            if resolution["action"] == "dismiss_manual_finding":
                reviews.pop(sample_id)
            elif resolution["action"] == "replace_manual_reason":
                for f in reviews[sample_id]:
                    f["reason"] = resolution["reason"]
            else:
                raise ValueError("unknown review resolution action")
    return reviews, coverage


def release() -> dict:
    rows, manifest = inventory()
    if manifest != json.loads((OUTPUT / "source_manifest.json").read_text(encoding="utf8")):
        raise ValueError("source files changed after audit")
    if (OUTPUT / "train.jsonl").exists():
        raise FileExistsError("immutable release already exists; use a new version directory")
    indexed = {}
    for p in sorted((OUTPUT / "shards").glob("audit_*.jsonl")):
        for a in read_jsonl(p):
            if a["index"] in indexed:
                raise ValueError("overlapping audit shards")
            indexed[a["index"]] = a
    if set(indexed) != set(range(len(rows))):
        raise ValueError("audit shards must cover every source row exactly once")
    reviews, coverage = load_reviews()
    known_ids = {r["id"] for _, r in rows}
    if set(reviews) - known_ids:
        raise ValueError(f"review contains foreign IDs: {sorted(set(reviews) - known_ids)}")
    system = (
        (ROOT / "taskgraph_lab/prompts/planner_student_system_prompt.txt")
        .read_text(encoding="utf-8")
        .strip()
    )
    retained, queue, decisions = defaultdict(list), [], []
    held_questions = {
        question_key(unpack(r)[0]["question"])
        for s, r in rows
        if s in {"choice_test", "locate_validation"}
    }
    duplicates = defaultdict(list)
    for index, (_, source) in enumerate(rows):
        a = indexed[index]
        if a["source_row_sha256"] != digest(json.dumps(source, sort_keys=True)):
            raise ValueError(f"audit/source mismatch: {source['id']}")
        # Re-evaluate cheap semantic rules at release so late-discovered runtime
        # contract issues are not missed in an earlier static audit shard.
        try:
            row, changes = repair(source)
            user, dsl = unpack(row)
            findings = semantic_findings(user, dsl)
            structural = a["validation"]
            if dsl != a["row"]["messages"][2]["content"]:
                structural = validate(dsl, user, str(source["id"]))
            if not structural["valid"]:
                findings.append(issue("structural_invalid", structural["detail"]))
        except Exception as exc:
            row, changes = deepcopy(source), []
            findings = [issue("structural_invalid", f"{type(exc).__name__}: {exc}")]
        user, dsl = unpack(row)
        manual = reviews.get(str(row["id"]), [])
        for f in manual:
            if f.get("question") and f["question"] != user["question"]:
                raise ValueError(f"review question/source mismatch: {row['id']}")
            expected_dsl = f.get("old_dsl", f.get("original_dsl"))
            if expected_dsl and expected_dsl != source["messages"][2]["content"]:
                raise ValueError(f"review DSL/source mismatch: {row['id']}")
        corrections = {f["corrected_dsl"] for f in manual if f.get("corrected_dsl")}
        if len(corrections) > 1:
            findings.append(
                issue("conflicting_reviews", "Different reviewers supplied different labels.")
            )
        elif corrections:
            corrected = corrections.pop()
            validation = validate(corrected, user, str(row["id"]))
            if not validation["valid"]:
                raise ValueError(f"reviewed correction invalid: {row['id']}: {validation}")
            row["messages"][2]["content"] = validation["dsl"]
            changes.append(
                issue(
                    "reviewed_semantic_repair", "Per-question reviewed correction.", reviews=manual
                )
            )
            findings = semantic_findings(user, validation["dsl"])
        for f in manual:
            if not f.get("corrected_dsl"):
                findings.append(
                    issue(
                        "manual_" + f.get("severity", "review"),
                        f["reason"],
                        review_file=f["review_file"],
                    )
                )
        if a["split"].endswith("train") and question_key(user["question"]) in held_questions:
            findings.append(
                issue(
                    "heldout_question_overlap",
                    "Normalized question also appears "
                    "in held-out test/validation; remove from training.",
                )
            )
        if row["messages"][0]["content"].strip() != system:
            row["messages"][0]["content"] = system
            changes.append(
                issue(
                    "system_prompt_alignment",
                    "Use one current system contract for old/new labels, including SELECT_SEM.",
                )
            )
        row.setdefault("metadata", {})["quality_cleaning"] = {
            "version": VERSION,
            "source_split": a["split"],
            "source_row_sha256": a["source_row_sha256"],
            "rules_only_not_visual_gold": True,
        }
        decision = {
            "id": row["id"],
            "split": a["split"],
            "changes": changes,
            "findings": findings,
            "before_dsl": a["before_dsl"],
            "after_dsl": row["messages"][2]["content"],
            "status": "quarantined" if findings else "retained",
        }
        decisions.append(decision)
        if findings:
            queue.append({"row": row, "decision": decision})
        else:
            duplicates[(a["split"], identity_key(user))].append((row, decision))
    for (split, _), group in duplicates.items():
        programs = {r["messages"][2]["content"] for r, _ in group}
        for i, (row, decision) in enumerate(group):
            if len(programs) > 1 or i > 0:
                code = "conflicting_duplicate_labels" if len(programs) > 1 else "duplicate_exposure"
                decision["status"] = "quarantined"
                decision["findings"].append(
                    issue(
                        code,
                        "Identical question, inputs and options have multiple supervision rows.",
                    )
                )
                queue.append({"row": row, "decision": decision})
            else:
                retained[split].append(row)
    for split, values in retained.items():
        write_jsonl(OUTPUT / (split + ".jsonl"), values)
    # Combined corpus has no implicit curriculum weighting; task-specific files
    # remain available for an explicit training mixture chosen later.
    train = retained["choice_train"] + retained["locate_train"]
    write_jsonl(OUTPUT / "train.jsonl", train)
    write_jsonl(OUTPUT / "test.jsonl", retained["choice_test"])
    write_jsonl(OUTPUT / "validation.jsonl", retained["locate_validation"])
    (OUTPUT / "original_test_reference.jsonl").write_bytes(SOURCES["choice_test"].read_bytes())
    write_jsonl(OUTPUT / "quarantine.jsonl", queue)
    write_jsonl(OUTPUT / "decisions.jsonl", decisions)
    summary = {
        "version": VERSION,
        "source_counts": dict(Counter(s for s, _ in rows)),
        "retained_counts": {s: len(retained[s]) for s in SOURCES},
        "quarantined_counts": dict(
            Counter(d["split"] for d in decisions if d["status"] == "quarantined")
        ),
        "combined_train_count": len(train),
        "finding_counts": dict(Counter(f["code"] for d in decisions for f in d["findings"])),
        "change_counts": dict(Counter(f["code"] for d in decisions for f in d["changes"])),
        "retained_changed_label_rows": sum(
            d["before_dsl"] != d["after_dsl"] for d in decisions if d["status"] == "retained"
        ),
        "review_coverage": {k: len(set(v)) for k, v in coverage.items()},
        "review_covered_ids": coverage,
        "semantic_accuracy_verified": False,
        "real_model_tested": False,
        "source_files_unchanged": source_hashes() == manifest["source_sha256"],
        "system_prompt_sha256": digest(system),
        "warning": "Risk filtering is not visual ground truth. Cleaned test is a different "
        "benchmark subset; keep the original test for historical comparisons. "
        "Combined train has no curriculum repetition or implicit 20% BBOX ratio.",
    }
    write_json(OUTPUT / "cleaning_summary.json", summary)
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "review_covered_ids"}, ensure_ascii=False
        )
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["audit", "release"])
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()
    if args.offset < 0 or args.limit < 1 or args.limit > 1000:
        parser.error("offset must be nonnegative; limit must be between 1 and 1000")
    if args.command == "audit":
        audit(args.offset, args.limit)
    else:
        release()


if __name__ == "__main__":
    main()
