"""Unit coverage for the deterministic RS Object Adapter v0 experiment."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sat_rs_vlm.data.object_adapter_v0 import (
    BUILDER_VERSION,
    DataAuditBlocked,
    DetectionBox,
    build_object_adapter_dataset_from_rows,
    build_class_vocab,
    construct_object_pairs,
    deduplicate_detection_boxes,
    resolve_counting_class,
    resolve_prompt_class,
    stable_image_split,
    validate_data_manifest,
)
from sat_rs_vlm.models.reliability.checksum import file_sha256


def _row(
    sample_id: str,
    image: str,
    task: str,
    answer: str,
    prompt: str = "Find car objects.",
    **metadata: object,
) -> dict[str, object]:
    return {
        "id": sample_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
            {"role": "assistant", "content": answer},
        ],
        "task_type": task,
        "metadata": {"dataset": "VRSBench", **metadata},
    }


def test_class_resolution_prefers_metadata_and_rejects_ambiguous_alias() -> None:
    vocab = {
        "classes": ["car", "truck"],
        "class_to_id": {"car": 0, "truck": 1},
        "aliases": {"car": ["car"], "truck": ["truck", "vehicle"]},
    }
    metadata_row = _row("count-1", "a.png", "counting", "2", target_class="car")
    assert resolve_counting_class(metadata_row, vocab).class_name == "car"
    prompt_row = _row("count-2", "a.png", "counting", "2", prompt="Count the truck objects.")
    assert resolve_counting_class(prompt_row, vocab).class_name == "truck"
    detection_prompt = "Locate the car object and return its normalized bounding box."
    assert resolve_prompt_class(detection_prompt, vocab).class_name == "car"
    ambiguous_vocab = {
        "classes": ["car", "truck"],
        "class_to_id": {"car": 0, "truck": 1},
        "aliases": {"car": ["vehicle"], "truck": ["vehicle"]},
    }
    ambiguous = _row("count-3", "a.png", "counting", "2", prompt="Count each vehicle.")
    assert resolve_counting_class(ambiguous, ambiguous_vocab).status == "ambiguous"


def test_dedup_and_supervision_types() -> None:
    boxes = [
        DetectionBox("b", "a.png", "car", (0.0, 0.0, 0.5, 0.5)),
        DetectionBox("a", "a.png", "car", (0.0, 0.0, 0.5, 0.5)),
    ]
    retained, removed = deduplicate_detection_boxes(boxes)
    assert removed == 1
    assert [box.sample_id for box in retained] == ["a"]
    vocab = build_class_vocab(["car"])
    records = [
        {"kind": "detection", "sample_id": "d1", "image": "full.png", "class_name": "car", "bbox_xyxy": (0.0, 0.0, 0.2, 0.2)},
        {"kind": "counting", "sample_id": "c1", "image": "full.png", "class_name": "car", "count": 1},
        {"kind": "detection", "sample_id": "d2", "image": "partial.png", "class_name": "car", "bbox_xyxy": (0.0, 0.0, 0.2, 0.2)},
        {"kind": "counting", "sample_id": "c2", "image": "partial.png", "class_name": "car", "count": 2},
        {"kind": "counting", "sample_id": "count.png", "image": "count.png", "class_name": "car", "count": 3},
        {"kind": "detection", "sample_id": "d3", "image": "det.png", "class_name": "car", "bbox_xyxy": (0.0, 0.0, 0.2, 0.2)},
    ]
    pairs = construct_object_pairs(records, vocab)
    assert {row["supervision_type"] for row in pairs} == {
        "full_set",
        "partial_set",
        "count_only",
        "detection_only",
    }


def test_builder_removes_eval_images_and_split_is_reproducible(tmp_path: Path) -> None:
    train_rows = [
        _row("d1", "Images/Images_train/a.png", "detection", '{"label":"car","bbox":[0,0,0.2,0.2]}'),
        _row("c1", "Images/Images_train/a.png", "counting", "1", target_class="car"),
        _row("d2", "Images/Images_train/b.png", "detection", '{"label":"car","bbox":[0,0,0.2,0.2]}'),
        _row("c2", "Images/Images_train/b.png", "counting", "2", target_class="car"),
        _row("c3", "Images/Images_train/c.png", "counting", "3", target_class="car"),
        _row("d4", "Images/Images_train/d.png", "detection", '{"label":"car","bbox":[0,0,0.2,0.2]}'),
    ]
    eval_rows = [_row("eval", "Images/Images_train/d.png", "detection", '{"label":"car","bbox":[0,0,0.2,0.2]}')]
    manifest = build_object_adapter_dataset_from_rows(
        train_rows,
        eval_rows,
        output_dir=tmp_path,
        enforce_blockers=False,
    )
    assert manifest["final_image_overlap_count"] == 0
    train = [json.loads(line) for line in (tmp_path / "train.jsonl").read_text().splitlines()]
    validation = [json.loads(line) for line in (tmp_path / "val.jsonl").read_text().splitlines()]
    assert {row["image"] for row in train}.isdisjoint({row["image"] for row in validation})
    assert all(row["image"] != "Images/Images_train/d.png" for row in train + validation)
    first = stable_image_split(train + validation, seed=42, val_fraction=0.5)
    second = stable_image_split(train + validation, seed=42, val_fraction=0.5)
    assert first == second
    assert manifest["output_files"]["train.jsonl"]["sha256"] == file_sha256(tmp_path / "train.jsonl")


def test_manifest_sha_mismatch_fails_fast(tmp_path: Path) -> None:
    asset = tmp_path / "train.jsonl"
    asset.write_text("{}\n", encoding="utf-8")
    payload = {
        "builder_version": BUILDER_VERSION,
        "audit_status": "passed",
        "final_image_overlap_count": 0,
        "train_val_image_overlap": 0,
        "output_files": {"train.jsonl": {"path": str(asset), "sha256": "bad"}},
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA mismatch"):
        validate_data_manifest(manifest)


def test_blocked_audit_is_explicit(tmp_path: Path) -> None:
    row = _row("d", "a.png", "detection", '{"label":"car","bbox":[0,0,0.2,0.2]}')
    with pytest.raises(DataAuditBlocked):
        build_object_adapter_dataset_from_rows([row], [], output_dir=tmp_path, enforce_blockers=True)


def test_hungarian_permutation_and_loss_behaviour() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("scipy")
    from sat_rs_vlm.models.rs_object_adapter import RSObjectAdapter
    from sat_rs_vlm.training.object_adapter_v0 import compute_object_adapter_loss, hungarian_match

    pred_boxes = torch.tensor([[0.75, 0.75, 0.2, 0.2], [0.25, 0.25, 0.2, 0.2]] + [[0.5, 0.5, 0.1, 0.1]] * 62)
    target_boxes = torch.tensor([[0.0, 0.0, 0.2, 0.2], [0.65, 0.65, 0.85, 0.85]])
    rows, columns = hungarian_match(torch.zeros(64), pred_boxes, target_boxes)
    assert rows.numel() == columns.numel() == 2
    assert set(rows.tolist()) == {0, 1}
    outputs = {"object_logits": torch.zeros(4, 64, requires_grad=True), "boxes_cxcywh": pred_boxes.unsqueeze(0).repeat(4, 1, 1).requires_grad_()}
    targets = [
        {"supervision_type": "full_set", "boxes_xyxy": [[0.0, 0.0, 0.2, 0.2]], "count": 1},
        {"supervision_type": "partial_set", "boxes_xyxy": [[0.0, 0.0, 0.2, 0.2]], "count": 2},
        {"supervision_type": "count_only", "boxes_xyxy": [], "count": 3},
        {"supervision_type": "detection_only", "boxes_xyxy": [[0.0, 0.0, 0.2, 0.2]], "count": None},
    ]
    losses = compute_object_adapter_loss(outputs, targets)
    assert torch.isfinite(losses["loss_total"])
    assert float(losses["loss_binarization"]) > 0.0
    losses["loss_total"].backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in (outputs["object_logits"], outputs["boxes_cxcywh"])
    )

    target = {"supervision_type": "full_set", "boxes_xyxy": [[0.15, 0.15, 0.35, 0.35]], "count": None}
    exact_boxes = torch.tensor([[0.25, 0.25, 0.20, 0.20]] + [[0.8, 0.8, 0.1, 0.1]] * 63)
    offset_boxes = torch.tensor([[0.75, 0.75, 0.20, 0.20]] + [[0.8, 0.8, 0.1, 0.1]] * 63)
    exact_logits = torch.full((1, 64), -4.0)
    offset_logits = exact_logits.clone()
    exact_loss = compute_object_adapter_loss(
        {"object_logits": exact_logits, "boxes_cxcywh": exact_boxes.unsqueeze(0)}, [target]
    )
    offset_loss = compute_object_adapter_loss(
        {"object_logits": offset_logits, "boxes_cxcywh": offset_boxes.unsqueeze(0)}, [target]
    )
    assert float(exact_loss["loss_bbox_l1"]) < float(offset_loss["loss_bbox_l1"])
    assert float(exact_loss["loss_giou"]) < float(offset_loss["loss_giou"])

    full_negative_low = torch.full((1, 64), -8.0)
    full_negative_high = full_negative_low.clone()
    full_negative_low[0, 0] = 4.0
    full_negative_high[0, 0] = 4.0
    full_negative_low[0, 1] = -8.0
    full_negative_high[0, 1] = 8.0
    full_target = {"supervision_type": "full_set", "boxes_xyxy": [[0.15, 0.15, 0.35, 0.35]], "count": None}
    full_low = compute_object_adapter_loss(
        {"object_logits": full_negative_low, "boxes_cxcywh": exact_boxes.unsqueeze(0)}, [full_target]
    )
    full_high = compute_object_adapter_loss(
        {"object_logits": full_negative_high, "boxes_cxcywh": exact_boxes.unsqueeze(0)}, [full_target]
    )
    assert float(full_high["loss_objectness"]) > float(full_low["loss_objectness"])

    partial_low = compute_object_adapter_loss(
        {"object_logits": full_negative_low, "boxes_cxcywh": exact_boxes.unsqueeze(0)},
        [{"supervision_type": "partial_set", "boxes_xyxy": [[0.15, 0.15, 0.35, 0.35]], "count": None}],
    )
    partial_high = compute_object_adapter_loss(
        {"object_logits": full_negative_high, "boxes_cxcywh": exact_boxes.unsqueeze(0)},
        [{"supervision_type": "partial_set", "boxes_xyxy": [[0.15, 0.15, 0.35, 0.35]], "count": None}],
    )
    assert float(partial_low["loss_objectness"]) == pytest.approx(
        float(partial_high["loss_objectness"])
    )

    near_count_logits = torch.full((1, 64), -10.0)
    near_count_logits[0, :2] = 8.0
    far_count_logits = torch.full((1, 64), -10.0)
    near_count = compute_object_adapter_loss(
        {"object_logits": near_count_logits, "boxes_cxcywh": exact_boxes.unsqueeze(0)},
        [{"supervision_type": "count_only", "boxes_xyxy": [], "count": 2}],
    )
    far_count = compute_object_adapter_loss(
        {"object_logits": far_count_logits, "boxes_cxcywh": exact_boxes.unsqueeze(0)},
        [{"supervision_type": "count_only", "boxes_xyxy": [], "count": 2}],
    )
    assert float(near_count["loss_count"]) < float(far_count["loss_count"])

    adapter = RSObjectAdapter(2, vit_hidden_size=4, d_model=8, nhead=2, dim_feedforward=16)
    features = [torch.randn(2, 6, 4) for _ in range(4)]
    result = adapter(features, torch.rand(2, 6, 2), torch.tensor([0, 1]))
    assert tuple(result["object_logits"].shape) == (2, 64)
    assert tuple(result["boxes_cxcywh"].shape) == (2, 64, 4)
    visual = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3)
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert optimizer_ids == {id(parameter) for parameter in adapter.parameters()}
    assert not optimizer_ids.intersection({id(parameter) for parameter in visual.parameters()})
