#!/usr/bin/env python3
"""D25 spatial-understanding controls for Qwen3-VL-2B; isolated from production."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

LABELS = (
    "top_left", "top", "top_right", "left", "center", "right",
    "bottom_left", "bottom", "bottom_right",
)
OUTPUT_LABELS = (
    "bottom", "bottom_left", "bottom_right", "center", "left", "right",
    "top", "top_left", "top_right",
)
ARMS = ("A_original", "B_visual_grid", "C_describe_map", "D_grid_describe_map")
ANSWER_MAX_TOKENS = 2
DESCRIPTION_MAX_TOKENS = 96
EXPECTED_ROUTE_B_SHA256 = "91954847cea08e17a77a115ff13988ae82c4536b82c53cccad141626358ff2cf"

def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_reviews(path: Path) -> dict[str, str]:
    payload = read_json(path)
    return {
        str(item_id): str(value.get("region", "")).strip().lower()
        for item_id, value in payload.get("review", {}).items()
    }


def question_text(row: dict[str, Any]) -> str:
    text = str(row.get("metadata", {}).get("original_question") or row.get("instruction", ""))
    text = text.strip().splitlines()[0].strip()
    if not text:
        raise ValueError(f"{row.get('id')}: missing question")
    return text


def prompt_for(question: str, arm: str) -> str:
    common = [question, "The two images are ordered BEFORE then AFTER.",
              "Choose the single cell containing the primary change."]
    if arm == "A_original":
        return "\n".join(common + [
            "Answer with exactly one label from: " + ", ".join(OUTPUT_LABELS) + ".",
            "Answer:",
        ])
    if arm == "B_visual_grid":
        return "\n".join(common + [
            "Both images have the same thin 3x3 grid lines and small cell numbers: "
            "1=top_left, 2=top, 3=top_right; 4=left, 5=center, 6=right; "
            "7=bottom_left, 8=bottom, 9=bottom_right.",
            "Use the grid as a visual aid; do not let the key replace inspection of the image changes.",
            "Answer with exactly one label from: " + ", ".join(OUTPUT_LABELS) + ".",
            "Answer:",
        ])
    if arm in {"C_describe_map", "D_grid_describe_map"}:
        if arm == "D_grid_describe_map":
            common.append(
                "Both images have the same thin 3x3 grid lines and small cell numbers. "
                "Use them as a visual aid, but report the location in words, not as a number."
            )
        else:
            common.append(
                "Mentally divide each image into equal thirds by row and column; no grid is drawn on these images."
            )
        common.extend([
            "Describe the main visible change briefly, then give one specific location in natural language.",
            "Use this exact two-line format:\nChange: <what visibly changed>\nLocation: <one specific position such as upper-left, upper-center, upper-right, middle-left, center, middle-right, lower-left, lower-center, or lower-right>.",
            "If the location cannot be assigned to one specific cell, write Location: UNCLEAR. Do not output a final grid label.",
        ])
        return "\n".join(common)
    raise ValueError(f"unknown arm: {arm}")


def _font(size: int):
    from PIL import ImageFont

    for candidate in ("arial.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def make_grid_visual(source: Path, target: Path) -> dict[str, Any]:
    """Keep original dimensions; use thin translucent lines and unobtrusive digits."""

    from PIL import Image, ImageDraw

    with Image.open(source) as opened:
        photo = opened.convert("RGBA")
    width, height = photo.size
    overlay = Image.new("RGBA", photo.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    line_width = max(1, round(min(width, height) / 256))
    for x in (round(width / 3), round(2 * width / 3)):
        draw.line((x, 0, x, height), fill=(0, 230, 235, 145), width=line_width)
    for y in (round(height / 3), round(2 * height / 3)):
        draw.line((0, y, width, y), fill=(0, 230, 235, 145), width=line_width)
    digit_font = _font(max(10, round(min(width, height) / 23)))
    for row in range(3):
        for col in range(3):
            xy = (round(col * width / 3) + 3, round(row * height / 3) + 2)
            draw.text(xy, str(row * 3 + col + 1), font=digit_font,
                      fill=(255, 255, 255, 200), stroke_width=1,
                      stroke_fill=(0, 0, 0, 150))
    altered_pixels = sum(alpha > 0 for alpha in overlay.getchannel("A").getdata())
    canvas = Image.alpha_composite(photo, overlay).convert("RGB")
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, format="PNG", optimize=True)
    return {"source": str(source.resolve()), "grid_image": str(target.resolve()),
            "source_size": [width, height], "grid_size": [width, height],
            "grid_line_width_px": line_width, "digit_placement": "small upper-left cell corners",
            "overlay_touched_pixel_fraction": altered_pixels / (width * height)}


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_jsonl(args.requests)
    reviews = load_reviews(args.human_review)
    if len(rows) != 25 or len({str(row["id"]) for row in rows}) != 25:
        raise ValueError(f"expected 25 unique D-group rows, got {len(rows)}")
    if set(reviews) != {str(row["id"]) for row in rows}:
        raise ValueError("human-review IDs do not match D25 requests")
    if sum(region in LABELS for region in reviews.values()) != 23:
        raise ValueError("expected 23 explicit cell labels and 2 unclear records")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    visual_dir = args.output_dir / "visual_inputs"
    samples = []
    for row in rows:
        item_id = str(row["id"])
        image_refs = list(row.get("images", []))
        if len(image_refs) != 2:
            raise ValueError(f"{item_id}: expected BEFORE/AFTER image pair")
        resolved = [(args.image_root / image_ref).resolve() for image_ref in image_refs]
        for image_path in resolved:
            if not image_path.is_file():
                raise FileNotFoundError(f"{item_id}: missing input image {image_path}")
        grid_paths, grid_meta = [], []
        for index, image_path in enumerate(resolved):
            label = "before" if index == 0 else "after"
            grid_path = visual_dir / f"{item_id}_{label}_grid.png"
            grid_meta.append(make_grid_visual(image_path, grid_path))
            grid_paths.append(str(grid_path.resolve()))
        question = question_text(row)
        samples.append({
            "id": item_id,
            "question": question,
            "reviewed_region": reviews[item_id],
            "image_paths_original": [str(path) for path in resolved],
            "image_paths_grid": grid_paths,
            "grid_visual_metadata": grid_meta,
            "prompts": {arm: prompt_for(question, arm) for arm in ARMS},
        })
    manifest = {
        "experiment": "qwen3vl2b_d25_spatial_understanding_direction_bias",
        "arms": list(ARMS),
        "sample_count": len(samples),
        "input_images_per_sample": 2,
        "position_number_mapping": {
            "1": "top_left", "2": "top", "3": "top_right",
            "4": "left", "5": "center", "6": "right",
            "7": "bottom_left", "8": "bottom", "9": "bottom_right",
        },
        "grid_annotation": "Original dimensions are preserved. Thin translucent lines and small digits near cell corners minimize occlusion; no opaque badges or added image panels.",
        "image_root": str(args.image_root.resolve()),
        "samples": samples,
        "review_note": "23 explicit and 2 unclear human-reviewed reference labels; not official gold labels.",
    }
    path = args.output_dir / "input_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"manifest={path} rows={len(samples)} visuals={len(samples)*2}", flush=True)
    return manifest


def with_prompt_and_images(messages: list[dict[str, Any]], prompt: str,
                           image_paths: list[str] | None) -> list[dict[str, Any]]:
    copied = copy.deepcopy(messages)
    text_count = image_count = 0
    for message in copied:
        if message.get("role") != "user" or not isinstance(message.get("content"), list):
            continue
        image_index = 0
        for item in message["content"]:
            if item.get("type") == "text":
                item["text"] = prompt
                text_count += 1
            elif item.get("type") == "image":
                if image_paths is not None:
                    item["image"] = image_paths[image_index]
                image_index += 1
                image_count += 1
    if text_count != 1 or image_count != 2:
        raise ValueError(f"message shape changed: text={text_count}, images={image_count}")
    return copied


_PHRASES: dict[str, tuple[str, ...]] = {
    "top_left": ("top left", "upper left", "north west", "northwest", "northwestern"),
    "top": ("top center", "upper center", "top middle", "upper middle", "north central", "upper central"),
    "top_right": ("top right", "upper right", "north east", "northeast", "northeastern"),
    "left": ("middle left", "center left", "central left", "left center", "west central", "western center"),
    "center": ("center", "centre", "middle", "central area", "central region"),
    "right": ("middle right", "center right", "central right", "right center", "east central", "eastern center"),
    "bottom_left": ("bottom left", "lower left", "south west", "southwest", "southwestern"),
    "bottom": ("bottom center", "lower center", "bottom middle", "lower middle", "south central", "lower central"),
    "bottom_right": ("bottom right", "lower right", "south east", "southeast", "southeastern"),
}


def map_location_text(description: str) -> dict[str, Any]:
    field_text = description.replace("*", "").replace("`", "")
    fields = re.findall(r"(?i)\bLocation\s*:\s*([^\r\n]+)", field_text)
    if len(fields) != 1:
        return {"status": "unresolved", "location_text": None,
                "matched_phrases": [], "candidates": [], "mapped_region": None,
                "reason": "missing_location_field" if not fields else "multiple_location_fields"}
    location = fields[0].strip().strip(" .,:;\"'`")
    if not location or re.search(r"\b(unclear|unknown|cannot determine|can't determine|not sure)\b", location, re.I):
        return {"status": "unresolved", "location_text": location or None,
                "matched_phrases": [], "candidates": [], "mapped_region": None,
                "reason": "explicitly_unclear_or_empty"}
    normalized = re.sub(r"[_\-–—]+", " ", location.lower())
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    all_matches: list[dict[str, Any]] = []
    for region, aliases in _PHRASES.items():
        for alias in aliases:
            pattern = r"(?<![a-z])" + re.escape(alias) + r"(?![a-z])"
            for match in re.finditer(pattern, normalized):
                all_matches.append({"region": region, "phrase": alias,
                                    "start": match.start(), "end": match.end()})
    # Retain longest non-overlapping phrases: "top center" is one cell, and
    # its embedded "center" must not create a second cell. Separate phrases
    # such as "upper-left and center" remain conflicting.
    specific_matches: list[dict[str, Any]] = []
    for match in sorted(all_matches, key=lambda value: (-(value["end"] - value["start"]), value["start"])):
        if any(match["start"] < chosen["end"] and chosen["start"] < match["end"]
               for chosen in specific_matches):
            continue
        specific_matches.append(match)
    # A generic center/middle mention is not decisive when used as a relation, e.g. "left of center".
    if re.search(r"\b(?:left|right|above|below|near|north|south|east|west)\s+of\s+(?:the\s+)?(?:center|centre|middle)\b", normalized):
        specific_matches = [match for match in specific_matches if match["region"] != "center"]
    candidates = sorted({match["region"] for match in specific_matches})
    if candidates == ["center"] and re.search(r"\b(left|right|upper|lower|top|bottom|north|south|east|west)\b", normalized):
        return {"status": "ambiguous", "location_text": location,
                "matched_phrases": specific_matches, "candidates": candidates,
                "mapped_region": None, "reason": "center_with_unresolved_direction_qualifier"}
    if len(candidates) == 1:
        return {"status": "mapped", "location_text": location,
                "matched_phrases": specific_matches, "candidates": candidates,
                "mapped_region": candidates[0], "reason": "one_unique_cell_phrase"}
    if len(candidates) > 1:
        return {"status": "ambiguous", "location_text": location,
                "matched_phrases": specific_matches, "candidates": candidates,
                "mapped_region": None, "reason": "multiple_cell_phrases"}
    return {"status": "unresolved", "location_text": location,
            "matched_phrases": [], "candidates": [], "mapped_region": None,
            "reason": "no_specific_3x3_phrase"}


def _token_constraint(tokenizer: Any, torch: Any, LogitsProcessor: Any):
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise RuntimeError("tokenizer has no EOS token")
    tokens = {label: tokenizer.encode(label, add_special_tokens=False) for label in LABELS}
    if any(len(ids) not in (1, 2) for ids in tokens.values()):
        raise RuntimeError(f"unexpected token boundary for labels: {tokens}")
    paths = {label: ids if len(ids) == 2 else ids + [eos_id] for label, ids in tokens.items()}
    if any(len(path) != ANSWER_MAX_TOKENS for path in paths.values()):
        raise RuntimeError(f"canonical label paths are not all two tokens: {paths}")
    if len({tuple(path) for path in paths.values()}) != 9:
        raise RuntimeError("canonical labels do not produce nine unique paths")
    for label, ids in tokens.items():
        if tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False) != label:
            raise RuntimeError(f"tokenizer round trip failed for {label}: {ids}")
        if tokenizer.encode(" " + label, add_special_tokens=False) == ids:
            raise RuntimeError(f"leading-space path overlaps canonical path: {label}")
    first = {path[0] for path in paths.values()}
    followers: dict[int, set[int]] = {}
    for path in paths.values():
        followers.setdefault(path[0], set()).add(path[1])

    class Trie(LogitsProcessor):
        def __init__(self, prompt_width: int):
            self.prompt_width = prompt_width

        def __call__(self, input_ids, scores):
            masked = torch.full_like(scores, float("-inf"))
            if input_ids.shape[-1] == self.prompt_width:
                choices = [first] * input_ids.shape[0]
            elif input_ids.shape[-1] == self.prompt_width + 1:
                choices = [followers.get(int(prefix), set()) for prefix in input_ids[:, -1].tolist()]
            else:
                raise RuntimeError("label decoder exceeded its two-token path")
            for index, allowed in enumerate(choices):
                if not allowed:
                    raise RuntimeError("invalid answer prefix in constrained decoding")
                ids = torch.tensor(sorted(allowed), device=scores.device, dtype=torch.long)
                masked[index, ids] = scores[index, ids]
            return masked

    return tokens, paths, Trie


def validate_mapper() -> dict[str, Any]:
    cases = {
        "upper-left quadrant": "top_left", "top center area": "top",
        "upper right": "top_right", "middle-left": "left", "in the center": "center",
        "middle right": "right", "lower left": "bottom_left",
        "bottom-center cell": "bottom", "lower-right quadrant": "bottom_right",
        "left side": None, "upper portion": None, "left of center": None,
        "upper-left to lower-right": None, "upper-left and center": None, "upper half center": None,
    }
    results = {text: map_location_text("Change: visible change.\nLocation: " + text)["mapped_region"]
               for text in cases}
    failures = {text: {"expected": expected, "actual": results[text]}
                for text, expected in cases.items() if results[text] != expected}
    if failures:
        raise RuntimeError(f"deterministic mapper self-test failed: {failures}")
    if map_location_text("Change: A road. **Location:** upper-left.")["mapped_region"] != "top_left":
        raise RuntimeError("markdown/same-line Location field self-test failed")
    return {"case_count": len(cases), "passed": True, "cases": results,
            "ambiguous_directions_are_not_forced": True}


def _stage_adapter(args: argparse.Namespace) -> tuple[Path, str]:
    digest = sha256_file(args.route_b_adapter)
    if digest != EXPECTED_ROUTE_B_SHA256:
        raise RuntimeError(f"route-B adapter SHA mismatch: {digest}")
    if not args.route_a_adapter_config.is_file():
        raise FileNotFoundError(args.route_a_adapter_config)
    dest = args.output_dir / "route_b_stage2_adapter"
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.route_a_adapter_config, dest / "adapter_config.json")
    shutil.copy2(args.route_b_adapter, dest / "adapter_model.safetensors")
    return dest, digest


def _distribution(values: dict[str, str | None]) -> dict[str, int]:
    counts = Counter(value if value is not None else "unresolved" for value in values.values())
    return dict(sorted(counts.items()))


def _score(values: dict[str, str | None], reviews: dict[str, str]) -> dict[str, Any]:
    clear_ids = [key for key, truth in reviews.items() if truth in LABELS]
    all_ids = list(reviews)
    hits_clear = sum(values.get(key) == reviews[key] for key in clear_ids)
    hits_strict = sum(values.get(key) == reviews[key] for key in all_ids)
    correct_by_id = {key: values.get(key) == reviews[key] for key in clear_ids}
    return {
        "clear": {"hits": hits_clear, "count": len(clear_ids), "accuracy": hits_clear / len(clear_ids)},
        "strict_all25": {"hits": hits_strict, "count": len(all_ids), "accuracy": hits_strict / len(all_ids)},
        "format_or_mapping_resolved": sum(values.get(key) in LABELS for key in all_ids),
        "prediction_distribution": _distribution(values),
        "prediction_distribution_clear": _distribution({key: values.get(key) for key in clear_ids}),
        "per_direction_clear": {
            label: {
                "gold_count": sum(reviews[key] == label for key in clear_ids),
                "predicted_count": sum(values.get(key) == label for key in clear_ids),
                "true_positive": sum(values.get(key) == label and reviews[key] == label for key in clear_ids),
            } for label in LABELS
        },
        "clear_correct_by_id": correct_by_id,
    }


def _transitions(candidate: dict[str, str | None], baseline: dict[str, str | None], reviews: dict[str, str]) -> dict[str, Any]:
    clear = [key for key, truth in reviews.items() if truth in LABELS]
    wrong_to_right = [key for key in clear if candidate.get(key) == reviews[key] and baseline.get(key) != reviews[key]]
    right_to_wrong = [key for key in clear if candidate.get(key) != reviews[key] and baseline.get(key) == reviews[key]]
    changed = [key for key in clear if candidate.get(key) != baseline.get(key)]
    return {"wrong_to_correct": wrong_to_right, "correct_to_wrong": right_to_wrong,
            "changed_predictions": changed, "changed_count": len(changed)}


def _cell_distance(left: str, right: str) -> int | None:
    positions = {label: (i // 3, i % 3) for i, label in enumerate(LABELS)}
    if left not in positions or right not in positions:
        return None
    return max(abs(positions[left][0] - positions[right][0]), abs(positions[left][1] - positions[right][1]))


def _errors(values: dict[str, str | None], reviews: dict[str, str]) -> dict[str, Any]:
    matrix = {truth: {pred: 0 for pred in OUTPUT_LABELS} for truth in OUTPUT_LABELS}
    adjacent, far, unresolved = [], [], []
    for item_id, truth in reviews.items():
        pred = values.get(item_id)
        if truth not in LABELS:
            continue
        if pred not in LABELS:
            unresolved.append(item_id)
            continue
        matrix[truth][pred] += 1
        if pred != truth:
            if _cell_distance(truth, pred) == 1:
                adjacent.append(item_id)
            else:
                far.append(item_id)
    return {"confusion_matrix_gold_rows_predicted_columns": matrix,
            "off_by_one_adjacent": adjacent, "long_range_errors": far,
            "unresolved_clear_ids": unresolved}


def run_model(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    import torch
    import transformers
    from peft import PeftModel
    from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing CPU fallback")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 unavailable on selected GPU")
    torch.set_num_threads(max(1, min(args.cpu_threads, os.cpu_count() or 1)))
    torch.cuda.set_device(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    if free_bytes < args.min_free_gpu_mb * 1024 * 1024:
        raise RuntimeError(f"only {free_bytes/1024/1024:.0f} MiB free; require {args.min_free_gpu_mb} MiB")
    model_dir = args.base_model_dir
    if not (model_dir / "config.json").is_file() or not any(model_dir.glob("*.safetensors")):
        raise FileNotFoundError(f"incomplete model directory: {model_dir}")
    base_hashes = {path.name: sha256_file(path) for path in sorted(model_dir.glob("*.safetensors"))}
    adapter_dir, adapter_hash = _stage_adapter(args)

    sys.path.insert(0, str(args.repo / "src"))
    from sat_rs_vlm.data.qwen3vl_collator import Qwen3VLDataCollator
    from sat_rs_vlm.data.qwen3vl_dataset import Qwen3VLDataset
    from sat_rs_vlm.models.qwen3vl_loader import compatible_model_class

    dataset = list(Qwen3VLDataset(args.requests))
    rows = {str(row["id"]): row for row in dataset}
    samples = manifest["samples"]
    if len(rows) != 25 or set(rows) != {sample["id"] for sample in samples}:
        raise RuntimeError("normalized Qwen dataset differs from the D25 manifest")
    processor = transformers.AutoProcessor.from_pretrained(
        str(model_dir), trust_remote_code=True, local_files_only=True)
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "left"
    tokens, answer_paths, Trie = _token_constraint(tokenizer, torch, LogitsProcessor)
    labels_decode = {tuple(path): label for label, path in answer_paths.items()}

    # Unit-check the exact trie transitions before loading/generating.
    first_tokens = {path[0] for path in answer_paths.values()}
    followers: dict[int, set[int]] = {}
    for path in answer_paths.values():
        followers.setdefault(path[0], set()).add(path[1])
    vocab_max = max(max(first_tokens), max(max(v) for v in followers.values())) + 8
    dummy = torch.zeros((1, vocab_max), dtype=torch.float32)
    trie = Trie(5)
    if set(torch.isfinite(trie(torch.zeros((1, 5), dtype=torch.long), dummy)[0]).nonzero().flatten().tolist()) != first_tokens:
        raise RuntimeError("first-token trie self-test failed")
    for prefix, allowed in followers.items():
        ids = torch.tensor([[0, 0, 0, 0, 0, prefix]], dtype=torch.long)
        actual = set(torch.isfinite(trie(ids, dummy)[0]).nonzero().flatten().tolist())
        if actual != allowed:
            raise RuntimeError(f"second-token trie self-test failed at prefix {prefix}")
    mapper_validation = validate_mapper()

    model_started = time.perf_counter()
    model = compatible_model_class(transformers).from_pretrained(
        str(model_dir), trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": "cuda:0"})
    model = PeftModel.from_pretrained(model, str(adapter_dir), local_files_only=True)
    model.eval()
    collator = Qwen3VLDataCollator(processor, max_seq_length=1024,
                                   image_root=args.image_root, for_generation=True, truncation=False)

    class StopAfterLabel(StoppingCriteria):
        def __init__(self, prompt_width: int):
            self.prompt_width = prompt_width

        def __call__(self, input_ids, scores, **kwargs):
            done = input_ids.shape[-1] >= self.prompt_width + ANSWER_MAX_TOKENS
            return torch.full((input_ids.shape[0],), done, dtype=torch.bool, device=input_ids.device)
    model_load_seconds = time.perf_counter() - model_started
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(0)
    print(f"model_ready device={torch.cuda.get_device_name(0)} load_seconds={model_load_seconds:.2f} "
          f"free_mib={free_bytes/1024/1024:.0f} BF16", flush=True)

    by_arm: dict[str, dict[str, str | None]] = {arm: {} for arm in ARMS}
    runtimes: dict[str, list[float]] = {arm: [] for arm in ARMS}
    completed: list[dict[str, Any]] = []
    start_all = time.perf_counter()
    for index, sample in enumerate(samples):
        source = rows[sample["id"]]
        arm_order = list(ARMS[index % len(ARMS):]) + list(ARMS[:index % len(ARMS)])
        arm_records: dict[str, Any] = {}
        for arm in arm_order:
            grid_mode = arm in {"B_visual_grid", "D_grid_describe_map"}
            images = sample["image_paths_grid"] if grid_mode else None
            messages = with_prompt_and_images(source["messages"], sample["prompts"][arm], images)
            batch = collator([{"id": sample["id"], "messages": messages}])
            image_grid = batch.get("image_grid_thw")
            if image_grid is None or int(image_grid.shape[0]) != 2:
                raise RuntimeError(f"{sample['id']} {arm}: expected 2 processor image grids")
            # Verify the two image files and prompts were the intended inputs.
            actual_texts = [item["text"] for msg in messages if msg.get("role") == "user"
                            for item in msg.get("content", []) if item.get("type") == "text"]
            if actual_texts != [sample["prompts"][arm]]:
                raise RuntimeError(f"{sample['id']} {arm}: prompt did not reach collator")
            resolved_messages = collator._messages_with_resolved_images({"id": sample["id"], "messages": messages})
            actual_chat_prompt = str(processor.apply_chat_template(
                [message for message in resolved_messages if message.get("role") != "assistant"],
                tokenize=False, add_generation_prompt=True))
            if sample["prompts"][arm] not in actual_chat_prompt:
                raise RuntimeError(f"{sample['id']} {arm}: rendered chat prompt lost the instruction")
            prompt_width = int(batch["input_ids"].shape[-1])
            if prompt_width > 1024:
                raise RuntimeError(f"{sample['id']} {arm}: {prompt_width} input tokens exceed the shared 1024-token budget")
            device_batch = {key: value.to("cuda:0") if hasattr(value, "to") else value
                            for key, value in batch.items()}
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                if arm in {"A_original", "B_visual_grid"}:
                    processor_constraint = Trie(prompt_width)
                    generated = model.generate(
                        **device_batch, do_sample=False, num_beams=len(first_tokens),
                        num_return_sequences=1, max_new_tokens=DESCRIPTION_MAX_TOKENS,
                        length_penalty=0.0, early_stopping=False, use_cache=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        logits_processor=LogitsProcessorList([processor_constraint]),
                        stopping_criteria=StoppingCriteriaList([StopAfterLabel(prompt_width)]),
                    )
                else:
                    generated = model.generate(
                        **device_batch, do_sample=False, num_beams=len(first_tokens),
                        num_return_sequences=1, max_new_tokens=DESCRIPTION_MAX_TOKENS,
                        length_penalty=0.0, early_stopping=False, use_cache=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            ids = generated[0, prompt_width:].tolist()
            if arm in {"A_original", "B_visual_grid"}:
                decision_ids = ids[:ANSWER_MAX_TOKENS]
                pred = labels_decode.get(tuple(decision_ids))
                # Beam finalization may append an unscored EOS after the
                # two-token decision. It is not an extra candidate token.
                allowed_tail = {tokenizer.eos_token_id, tokenizer.pad_token_id}
                if pred is None or any(token not in allowed_tail for token in ids[ANSWER_MAX_TOKENS:]):
                    raise RuntimeError(f"{sample['id']} {arm}: invalid constrained tokens {ids}")
                decoded = tokenizer.decode(ids, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False).strip()
                if decoded != pred:
                    raise RuntimeError(f"{sample['id']} {arm}: decoded {decoded!r}, expected {pred}")
                mapping = {"status": "direct_label", "mapped_region": pred,
                           "location_text": None, "matched_phrases": [], "candidates": [pred],
                           "reason": "constrained_direct_label"}
            else:
                decoded = tokenizer.decode(ids, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False).strip()
                mapping = map_location_text(decoded)
                pred = mapping["mapped_region"]
            by_arm[arm][sample["id"]] = pred
            runtimes[arm].append(elapsed)
            arm_records[arm] = {
                "prediction": pred,
                "raw_output": decoded,
                "generated_token_ids": [int(token) for token in ids],
                "decision_token_ids": ids[:ANSWER_MAX_TOKENS] if arm in {"A_original", "B_visual_grid"} else None,
                "constraint_valid": arm in {"A_original", "B_visual_grid"},
                "mapping": mapping,
                "inference_seconds": elapsed,
                "input_images": ["BEFORE", "AFTER"],
                "input_image_count": 2,
                "visual_grid_used": grid_mode,
                "prompt_sha256": hashlib.sha256(sample["prompts"][arm].encode("utf-8")).hexdigest(),
                "prompt_text": sample["prompts"][arm],
                "actual_chat_prompt": actual_chat_prompt,
                "actual_chat_prompt_sha256": hashlib.sha256(actual_chat_prompt.encode("utf-8")).hexdigest(),
                "encoded_input_token_count": prompt_width,
                "processor_image_grid_thw": image_grid.tolist(),
            }
        completed.append({
            "id": sample["id"], "reviewed_region": sample["reviewed_region"],
            "arms": arm_records,
            "grid_visuals": sample["grid_visual_metadata"],
        })
        print(f"done={index+1}/25 id={sample['id']} " + " ".join(
            f"{arm}={arm_records[arm]['prediction'] or 'UNRESOLVED'}" for arm in ARMS), flush=True)
        (args.output_dir / "partial_predictions.json").write_text(
            json.dumps({"completed_samples": len(completed), "predictions": completed},
                       ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    torch.cuda.synchronize()
    total_inference_seconds = time.perf_counter() - start_all
    reviews = {sample["id"]: sample["reviewed_region"] for sample in samples}
    metrics = {arm: _score(by_arm[arm], reviews) for arm in ARMS}
    transitions = {arm: _transitions(by_arm[arm], by_arm["A_original"], reviews)
                   for arm in ARMS if arm != "A_original"}
    error_details = {arm: _errors(by_arm[arm], reviews) for arm in ARMS}
    timing = {arm: {"mean_seconds_per_sample": statistics.mean(runtimes[arm]),
                    "p50_seconds_per_sample": statistics.median(runtimes[arm]),
                    "total_generation_seconds": sum(runtimes[arm])}
               for arm in ARMS}
    positions = Counter(sample["reviewed_region"] for sample in samples if sample["reviewed_region"] in LABELS)
    metrics_extra = {
        "gold_distribution_clear": dict(sorted(positions.items())),
        "direction_bias": {
            arm: {"predicted_bottom_left": metrics[arm]["prediction_distribution_clear"].get("bottom_left", 0),
                  "gold_bottom_left_clear": positions.get("bottom_left", 0),
                  "bottom_left_overprediction": metrics[arm]["prediction_distribution_clear"].get("bottom_left", 0) - positions.get("bottom_left", 0)}
            for arm in ARMS
        },
    }
    result = {
        "experiment": manifest["experiment"],
        "model_id": "Qwen/Qwen3-VL-2B-Instruct",
        "base_model_path": str(model_dir.resolve()),
        "base_model_safetensors_sha256": base_hashes,
        "adapter": "route_b_stage2_batch8",
        "adapter_sha256": adapter_hash,
        "device": torch.cuda.get_device_name(0), "device_id": "cuda:0", "dtype": "bfloat16",
        "input_images_per_sample": 2, "sample_count": len(samples),
        "model_load_seconds": model_load_seconds,
        "inference_seconds_all_calls": total_inference_seconds,
        "peak_cuda_memory_allocated_mib": torch.cuda.max_memory_allocated(0) / 1024 / 1024,
        "gpu_free_before_load_mib": free_bytes / 1024 / 1024,
        "generation_parameters": {
            "do_sample": False, "num_beams": len(first_tokens), "length_penalty": 0.0,
            "use_cache": True, "max_new_tokens_all_arms": DESCRIPTION_MAX_TOKENS,
            "direct_label_stop_after_tokens": ANSWER_MAX_TOKENS,
            "note": "All arms share max_new_tokens=96 and deterministic beam settings. A/B additionally stop at the completed two-token constrained label; C/D generate descriptions up to the shared limit.",
        },
        "decoder_validation": {
            "label_token_ids": tokens, "equal_length_answer_paths": answer_paths,
            "all_label_paths_two_tokens": True, "short_labels_padded_with_eos": True,
            "leading_space_blocked": True, "prefix_trie_self_test": "passed",
            "location_mapper_self_test": mapper_validation,
            "runtime_direct_label_valid": sum(row["arms"][arm]["constraint_valid"]
                                               for row in completed for arm in ("A_original", "B_visual_grid")),
        },
        "metrics": metrics, "timing": timing, "transitions_vs_A": transitions,
        "error_analysis": error_details, "bias_analysis": metrics_extra,
        "predictions": completed,
        "review_note": "23 explicit and 2 unclear experimental human-review labels; not official gold labels.",
    }
    (args.output_dir / "predictions.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(args, result)
    print(json.dumps({key: result[key] for key in (
        "experiment", "device", "input_images_per_sample", "inference_seconds_all_calls",
        "metrics", "timing", "transitions_vs_A", "bias_analysis")
    }, ensure_ascii=False), flush=True)
    print(f"saved={args.output_dir / 'predictions.json'}", flush=True)


def _md(value: Any) -> str:
    if value is None:
        return "UNRESOLVED"
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def write_report(args: argparse.Namespace, result: dict[str, Any]) -> None:
    metrics, timing = result["metrics"], result["timing"]
    names = {"A_original": "A 原图直接九分类", "B_visual_grid": "B 九宫格视觉辅助",
             "C_describe_map": "C 自然语言描述＋确定性映射",
             "D_grid_describe_map": "D 网格辅助＋自然语言描述映射"}
    lines = [
        "# LEVIR-CC D 组九宫格变化定位：空间理解与方向偏置实验",
        "",
        "## 设计与有效性",
        "",
        "四组均使用同一 Qwen3-VL-2B-Instruct 基座、同一 route_b_stage2_batch8 LoRA、25 对 BEFORE/AFTER 图像及确定性 beam 解码。A/B 直接输出受 Trie 限制的九宫格标签；C/D 输出变化内容和位置描述，再由保守规则映射到九宫格，方向含糊或冲突时保留 unresolved/ambiguous，不猜测。D 是可选的组合组。",
        "",
        "A/B 的标签路径经 tokenizer 检查：九个标签皆是唯一的两 Token 决策路径；单 Token 标签以 EOS 补齐，不使用前导空格形式；约束 trie 两步自测通过。所有组共用 do_sample=False、beam=5、length_penalty=0、KV cache、max_new_tokens=96。A/B 的输出约束在两 Token 标签完成后停止；beam finalize 自动追加的 EOS 不参与决策。C/D 自然语言不使用标签约束，达到 EOS 或共同的 96 Token 上限时停止。",
        "",
        "B/D 在原图上叠加半透明窄线 3×3 网格和靠近各格左上角的小编号，无实心底色，不增加画布，保持 256×256。编号和细线仍可能覆盖少量像素，具体覆盖比例保存在输入清单中，作为实验局限。25 对网格输入均单独保存。每次验证两图均经过 processor，保存实际 chat prompt；不截断文本，超过共同的 1024 输入 Token 预算会报错。",
        "",
        f"设备：{result['device']}，BF16，GPU：cuda:0；权重 SHA256：基座见 JSON，route-B adapter `{result['adapter_sha256']}`。人工审核标签为 23 个明确格位、2 个 unclear；它们是实验参考标签，不是官方金标准，25 题严格计分把 unclear 计入分母。",
        "",
        "## 汇总",
        "",
        "| 方案 | 明确样本准确率 | 25 题严格计分 | 可映射/有效输出 | 平均生成耗时/题 |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        m = metrics[arm]
        lines.append(
            f"| {names[arm]} | {m['clear']['hits']}/23 ({m['clear']['accuracy']:.1%}) | "
            f"{m['strict_all25']['hits']}/25 ({m['strict_all25']['accuracy']:.1%}) | "
            f"{m['format_or_mapping_resolved']}/25 | {timing[arm]['mean_seconds_per_sample']:.2f}s |"
        )
    lines.extend(["", "## 方位分布和方向偏置", ""])
    lines.append("人工明确标签分布：`" + json.dumps(result["bias_analysis"]["gold_distribution_clear"], ensure_ascii=False) + "`。")
    for arm in ARMS:
        m = metrics[arm]
        bias = result["bias_analysis"]["direction_bias"][arm]
        lines.append(
            f"- {names[arm]}：25 题预测分布 `{json.dumps(m['prediction_distribution'], ensure_ascii=False)}`；"
            f"明确的 23 题中 bottom_left 预测 {bias['predicted_bottom_left']} 次，参考有 {bias['gold_bottom_left_clear']} 次，"
            f"预测－参考差 {bias['bottom_left_overprediction']:+d}。"
        )
    lines.extend(["", "## 相对 A 的变化", ""])
    for arm in ARMS[1:]:
        change = result["transitions_vs_A"][arm]
        lines.append(
            f"- {names[arm]}：错变对 {len(change['wrong_to_correct'])} 题 "
            f"({', '.join(change['wrong_to_correct']) or '无'})；对变错 {len(change['correct_to_wrong'])} 题 "
            f"({', '.join(change['correct_to_wrong']) or '无'})；明确样本预测改变 {change['changed_count']}/23。"
        )
    lines.extend(["", "## 边界/邻格错误", "", "这里只统计网格标签的相邻距离，是边界误差的待复核线索，不能据此断言真实变化目标位于格线附近。"])
    for arm in ARMS:
        errors = result["error_analysis"][arm]
        lines.append(
            f"- {names[arm]}：相邻格（Chebyshev 距离 1）{len(errors['off_by_one_adjacent'])} 题 "
            f"({', '.join(errors['off_by_one_adjacent']) or '无'})；跨两格以上 {len(errors['long_range_errors'])} 题 "
            f"({', '.join(errors['long_range_errors']) or '无'})；未映射明确样本 "
            f"{len(errors['unresolved_clear_ids'])} 题。"
        )
    lines.extend(["", "## 两个 unclear 样本", "", "| ID | A | B | C 映射 | D 映射 |", "|---|---|---|---|---|"])
    for row in result["predictions"]:
        if row["reviewed_region"] not in LABELS:
            lines.append("| " + row["id"] + " | " + " | ".join(_md(row["arms"][arm]["prediction"]) for arm in ARMS) + " |")
    lines.extend(["", "## 逐题对照", "", "C/D 描述只按 `Location:` 字段映射；明确但不在九宫格单元中的方位保持未决。"])
    lines.extend([
        "", "| ID | 参考 | A | B | C 描述 / 映射 | D 描述 / 映射 |",
        "|---|---|---|---|---|---|",
    ])
    for row in result["predictions"]:
        c = row["arms"]["C_describe_map"]
        d = row["arms"]["D_grid_describe_map"]
        lines.append(
            f"| {row['id']} | {row['reviewed_region']} | {row['arms']['A_original']['prediction']} | "
            f"{row['arms']['B_visual_grid']['prediction']} | "
            f"{_md(c['raw_output'])} → {_md(c['prediction'])} | "
            f"{_md(d['raw_output'])} → {_md(d['prediction'])} |"
        )
    lines.extend([
        "", "## 混淆矩阵", "", "每个矩阵的行是真值、列是预测；行列均按输出标签顺序。",
    ])
    for arm in ARMS:
        matrix = result["error_analysis"][arm]["confusion_matrix_gold_rows_predicted_columns"]
        lines.extend(["", f"### {names[arm]}", "", "| Gold \\ Pred | " + " | ".join(OUTPUT_LABELS) + " |",
                      "|---|" + "---|" * len(OUTPUT_LABELS)])
        for truth in OUTPUT_LABELS:
            lines.append("| " + truth + " | " + " | ".join(str(matrix[truth][pred]) for pred in OUTPUT_LABELS) + " |")
    lines.extend([
        "", "## 视觉样本及原始记录", "",
        "所有 B/D 组可视化输入保存在 `visual_inputs/`（每题 BEFORE/AFTER 各一张）。`predictions.json` 保存逐题全部原始输出、Token ID、自然语言 Location 提取、规则映射、提示词哈希、计时和混淆/转移统计；`input_manifest.json` 保存完整提示词和输入图映射。",
        "", "## 结论边界", "",
        "该 D25 小样本只支持描述观察到的行为，不能据此宣称稳定的性能提升。C/D 的未决率必须与准确率一并解读：保守映射拒绝模糊方位是预期行为，不可把未决重解释成方向错误。模型对变化内容的文字描述也不等同于内容识别正确，需结合保留的图像与逐题描述人工复核。",
        "",
    ])
    (args.output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def validate_preflight(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    """Exercise actual processor inputs and constraint masks without loading weights."""
    import torch
    from transformers import AutoProcessor, LogitsProcessor

    sys.path.insert(0, str(args.repo / "src"))
    from sat_rs_vlm.data.qwen3vl_collator import Qwen3VLDataCollator
    from sat_rs_vlm.data.qwen3vl_dataset import Qwen3VLDataset

    processor = AutoProcessor.from_pretrained(str(args.base_model_dir), trust_remote_code=True, local_files_only=True)
    tokens, paths, Trie = _token_constraint(processor.tokenizer, torch, LogitsProcessor)
    all_tokens = {token for path in paths.values() for token in path}
    dummy = torch.zeros((1, max(all_tokens) + 8))
    trie = Trie(5)
    expected_first = {path[0] for path in paths.values()}
    first = trie(torch.zeros((1, 5), dtype=torch.long), dummy)
    if set(torch.isfinite(first[0]).nonzero().flatten().tolist()) != expected_first:
        raise RuntimeError("preflight first-token constraint mismatch")
    for prefix in expected_first:
        next_ids = torch.tensor([[0, 0, 0, 0, 0, prefix]])
        masked = trie(next_ids, dummy)
        expected_next = {path[1] for path in paths.values() if path[0] == prefix}
        if set(torch.isfinite(masked[0]).nonzero().flatten().tolist()) != expected_next:
            raise RuntimeError("preflight second-token constraint mismatch")
    source = Qwen3VLDataset(args.requests)[0]
    sample = manifest["samples"][0]
    original_messages = copy.deepcopy(source["messages"])
    collator = Qwen3VLDataCollator(processor, max_seq_length=1024, image_root=args.image_root,
                                   for_generation=True, truncation=False)
    checks = {}
    for arm in ARMS:
        paths_override = sample["image_paths_grid"] if arm in {"B_visual_grid", "D_grid_describe_map"} else None
        messages = with_prompt_and_images(source["messages"], sample["prompts"][arm], paths_override)
        request = {"id": source["id"], "messages": messages}
        batch = collator([request])
        rendered = str(processor.apply_chat_template(
            [msg for msg in collator._messages_with_resolved_images(request) if msg.get("role") != "assistant"],
            tokenize=False, add_generation_prompt=True))
        width = int(batch["input_ids"].shape[-1])
        grid = batch.get("image_grid_thw")
        if grid is None or int(grid.shape[0]) != 2 or width > 1024:
            raise RuntimeError(f"preflight invalid image or token budget for {arm}")
        if sample["prompts"][arm] not in rendered:
            raise RuntimeError(f"preflight stale prompt for {arm}")
        checks[arm] = {"input_tokens": width, "image_grid_thw": grid.tolist(),
                       "actual_chat_prompt": rendered,
                       "actual_chat_prompt_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest()}
    if source["messages"] != original_messages:
        raise RuntimeError("preflight mutated the shared source messages")
    if len({value["actual_chat_prompt_sha256"] for value in checks.values()}) != len(ARMS):
        raise RuntimeError("preflight arm prompts are not distinct")
    return {
        "passed": True, "script_sha256": sha256_file(Path(__file__)),
        "mapper_self_test": validate_mapper(),
        "tokenizer_validation": {"label_token_ids": tokens, "equal_length_paths": paths,
                                 "prefix_mask_self_test": "passed", "leading_space_blocked": True},
        "actual_processor_inputs": checks,
        "visual_count": sum(len(sample["image_paths_grid"]) for sample in manifest["samples"]),
        "shared_source_messages_unchanged": True,
        "gpu": {"cuda_available": torch.cuda.is_available(),
                "free_mib": torch.cuda.mem_get_info(0)[0] / 1024 / 1024 if torch.cuda.is_available() else None},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--human-review", type=Path, required=True)
    parser.add_argument("--route-b-adapter", type=Path, required=True)
    parser.add_argument("--route-a-adapter-config", type=Path, required=True)
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--min-free-gpu-mb", type=int, default=5200)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run-model", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_manifest(args)
    if args.prepare_only:
        result = validate_preflight(args, manifest)
        (args.output_dir / "preflight.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.run_model:
        run_model(args, manifest)
    elif not args.prepare_only:
        raise ValueError("select --prepare-only or --run-model")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
