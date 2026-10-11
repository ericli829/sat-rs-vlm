"""Offline diagnostics only; preserve the frozen D25 experiment and raw scores."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import textwrap
from collections import Counter
from pathlib import Path

from run_d25_spatial_understanding_qwen import LABELS, map_location_text


def prose_mapping(text: str) -> dict:
    """Diagnostic recovery of explicit cell phrases, without consulting labels.

    This intentionally does not treat a whole side/row as its central cell.
    Missing Location fields remain a format failure in the frozen experiment.
    """
    if re.search(r"\b(no (?:change|difference)|unchanged|not changed|unclear|unknown)\b", text, re.I):
        return {"status": "unresolved", "mapped_region": None,
                "reason": "negative_or_uncertain_change_statement"}
    if re.search(r"\bLocation\s*:", text, re.I):
        return map_location_text(text)
    return {**map_location_text("Location: " + text), "source": "whole_prose_diagnostic"}


def self_test() -> None:
    cases = {
        "A building appears in the upper-left corner.": "top_left",
        "Houses appear in the lower-left corner.": "bottom_left",
        "A road appears in the middle-right cell.": "right",
        "There is no change in the upper-left corner.": None,
        "A road on the right side.": None,
        "Buildings in the upper part of the scene.": None,
        "Houses at the bottom of the scene.": None,
        "Buildings in the upper-left and lower-right.": None,
        "Location: UNCLEAR": None,
    }
    for text, expected in cases.items():
        actual = prose_mapping(text)["mapped_region"]
        assert actual == expected, (text, actual, expected)


def diagnostics(result: dict) -> dict:
    clear = [row for row in result["predictions"] if row["reviewed_region"] in LABELS]
    spatial = {}
    for arm in ("A_original", "B_visual_grid"):
        row_hits = col_hits = 0
        for item in clear:
            gold = LABELS.index(item["reviewed_region"])
            pred = LABELS.index(item["arms"][arm]["prediction"])
            row_hits += gold // 3 == pred // 3
            col_hits += gold % 3 == pred % 3
        # Fixed geometric transforms are diagnostics, never a fitted label fix.
        transforms = {
            "identity": lambda r, c: (r, c),
            "horizontal_mirror": lambda r, c: (r, 2-c),
            "vertical_mirror": lambda r, c: (2-r, c),
            "rotate_180": lambda r, c: (2-r, 2-c),
            "rotate_90": lambda r, c: (c, 2-r),
            "rotate_270": lambda r, c: (2-c, r),
            "transpose": lambda r, c: (c, r),
            "anti_transpose": lambda r, c: (2-c, 2-r),
        }
        transform_hits = {}
        for name, transform in transforms.items():
            hits = 0
            for item in clear:
                index = LABELS.index(item["arms"][arm]["prediction"])
                r, c = transform(index // 3, index % 3)
                hits += LABELS[3*r+c] == item["reviewed_region"]
            transform_hits[name] = hits
        spatial[arm] = {"row_correct": row_hits, "column_correct": col_hits,
                        "count": len(clear), "fixed_transform_hits": transform_hits}
    prose = {}
    for arm in ("C_describe_map", "D_grid_describe_map"):
        entries = []
        reasons = Counter()
        for item in result["predictions"]:
            record = item["arms"][arm]
            mapped = prose_mapping(record["raw_output"])
            reasons[mapped["reason"]] += 1
            entries.append({"id": item["id"], "reviewed_region": item["reviewed_region"],
                            "raw_output": record["raw_output"], "mapping": mapped})
        resolved = [entry for entry in entries if entry["mapping"]["mapped_region"] is not None]
        hits = sum(entry["reviewed_region"] in LABELS and
                   entry["mapping"]["mapped_region"] == entry["reviewed_region"] for entry in entries)
        prose[arm] = {"note": "Post-run, label-independent diagnostic; does not replace frozen strict-field results.",
                      "clear_hits": hits, "clear_count": len(clear), "resolved_all25": len(resolved),
                      "reason_counts": dict(reasons), "entries": entries}
    improved = len(result["transitions_vs_A"]["B_visual_grid"]["wrong_to_correct"])
    worsened = len(result["transitions_vs_A"]["B_visual_grid"]["correct_to_wrong"])
    discordant = improved + worsened
    p = min(1., 2 * sum(math.comb(discordant, k) for k in range(min(improved, worsened)+1)) / 2**discordant)
    return {"spatial_diagnostics_clear23": spatial, "prose_recovery_diagnostic": prose,
            "A_B_exact_mcnemar": {"improved": improved, "worsened": worsened, "two_sided_p": p},
            "guardrails": "No model calls, no changed prompts, no altered predictions, no label-fitted mapping."}


def contact_sheets(manifest: dict, result: dict, output: Path) -> list[str]:
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
    by_id = {row["id"]: row for row in result["predictions"]}
    output.mkdir(exist_ok=True)
    paths = []
    for start in range(0, len(manifest["samples"]), 4):
        samples = manifest["samples"][start:start+4]
        canvas = Image.new("RGB", (1060, 300*len(samples)), "white")
        draw = ImageDraw.Draw(canvas)
        for index, sample in enumerate(samples):
            y = index*300
            item = by_id[sample["id"]]
            draw.text((8, y+3), sample["id"] + "  BEFORE                      AFTER", font=font, fill="black")
            for image_index, path in enumerate(sample["image_paths_original"]):
                with Image.open(path) as image:
                    canvas.paste(image.convert("RGB"), (8 + image_index*264, y+28))
            lines = ["Reference: " + item["reviewed_region"],
                     "A: " + str(item["arms"]["A_original"]["prediction"]),
                     "B: " + str(item["arms"]["B_visual_grid"]["prediction"])]
            for arm, prefix in (("C_describe_map", "C: "), ("D_grid_describe_map", "D: ")):
                lines += textwrap.wrap(prefix + item["arms"][arm]["raw_output"], width=57)
            draw.multiline_text((544, y+28), "\n".join(lines), font=font, spacing=5, fill="black")
            draw.line((0, y+299, 1060, y+299), fill="#aaaaaa")
        path = output / f"review_{start//4+1:02}.png"
        canvas.save(path)
        paths.append(str(path.resolve()))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    self_test()
    path = args.run_dir / "predictions.json"
    raw = path.read_bytes()
    result = json.loads(raw)
    manifest = json.loads((args.run_dir / "input_manifest.json").read_text(encoding="utf-8"))
    analysis = diagnostics(result)
    analysis["source_predictions_sha256"] = hashlib.sha256(raw).hexdigest()
    analysis["contact_sheets"] = contact_sheets(manifest, result, args.run_dir / "review_sheets")
    (args.run_dir / "offline_diagnostics.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    assert path.read_bytes() == raw
    print(json.dumps({key: value for key, value in analysis.items()
                      if key not in {"prose_recovery_diagnostic", "contact_sheets"}}, ensure_ascii=False, indent=2))
    print(json.dumps({arm: {key: value for key, value in metric.items() if key != "entries"}
                      for arm, metric in analysis["prose_recovery_diagnostic"].items()}, ensure_ascii=False, indent=2))
    print("diagnostics and seven review sheets saved; frozen predictions unchanged")


if __name__ == "__main__":
    main()
