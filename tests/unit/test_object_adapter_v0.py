"""Unit coverage for the deterministic RS Object Adapter v0 experiment."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from sat_rs_vlm.data.object_adapter_v0 import (
    BUILDER_VERSION,
    DataAuditBlocked,
    DetectionBox,
    build_class_vocab,
    build_object_adapter_dataset_from_rows,
    construct_object_pairs,
    deduplicate_detection_boxes,
    resolve_cardinality_prompt_class,
    resolve_counting_class,
    resolve_prompt_class,
    stable_image_split,
    validate_data_manifest,
)
from sat_rs_vlm.models.reliability.checksum import file_sha256


def _load_object_adapter_evaluator_module():
    evaluator_path = (
        Path(__file__).parents[2] / "scripts" / "evaluation" / "evaluate_object_adapter_v0.py"
    )
    spec = importlib.util.spec_from_file_location("object_adapter_evaluator_test", evaluator_path)
    assert spec is not None and spec.loader is not None
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    return evaluator


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


def _counting_vocab() -> dict[str, object]:
    return build_class_vocab(
        [
            "airplane",
            "baseball diamond",
            "bridge",
            "ground track field",
            "golffield",
            "helipad",
            "overpass",
            "ship",
            "soccer ball field",
            "trainstation",
            "vehicle",
        ]
    )


def test_class_resolution_prefers_metadata_and_keeps_detection_prompt_resolution() -> None:
    vocab = _counting_vocab()
    metadata_row = _row(
        "count-1",
        "a.png",
        "counting",
        '{"label":"ship"}',
        prompt="How many ships are visible?",
        target_class="vehicle",
    )
    assert resolve_counting_class(metadata_row, vocab).class_name == "vehicle"
    detection_prompt = "Locate the car object and return its normalized bounding box."
    detection_vocab = {
        "classes": ["car", "truck"],
        "class_to_id": {"car": 0, "truck": 1},
        "aliases": {"car": ["car"], "truck": ["truck", "vehicle"]},
    }
    assert resolve_prompt_class(detection_prompt, detection_vocab).class_name == "car"


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("How many ships are docked in the harbor?", "ship"),
        ("How many vehicles are visible on the overpass?", "vehicle"),
        ("How many vehicles are on the bridge?", "vehicle"),
        ("How many airplanes are present at the airport?", "airplane"),
        ("How many helipads are visible on the ship?", "helipad"),
        ("How many planes are visible?", "airplane"),
        ("How many unique planes are visible?", "airplane"),
        ("What is the total number of planes visible in the image?", "airplane"),
        ("What is the number of planes visible in the image?", "airplane"),
        ("How many baseball fields are visible?", "baseball diamond"),
        ("How many train stations are visible?", "trainstation"),
        ("How many soccer fields are visible?", "soccer ball field"),
        ("How many golf courses are visible?", "golffield"),
        ("How many ground track and field areas are visible?", "ground track field"),
        (
            "How many planes are visible? Return ONLY the integer. Do not include any other text.",
            "airplane",
        ),
    ],
)
def test_counting_cardinality_target_prefix_resolution(prompt: str, expected: str) -> None:
    row = _row("count", "a.png", "counting", "999", prompt=prompt)
    resolution = resolve_counting_class(row, _counting_vocab())
    assert resolution.status == "resolved"
    assert resolution.class_name == expected


def test_counting_plural_aliases_use_regular_rules() -> None:
    aliases = build_class_vocab(
        [
            "overpass",
            "ship",
            "vehicle",
            "baseball diamond",
            "tennis court",
            "expressway service area",
        ]
    )["aliases"]
    assert "overpasses" in aliases["overpass"]
    assert "overpasss" not in aliases["overpass"]
    assert "ships" in aliases["ship"]
    assert "vehicles" in aliases["vehicle"]
    assert "baseball diamonds" in aliases["baseball diamond"]
    assert "tennis courts" in aliases["tennis court"]
    assert "expressway service areas" in aliases["expressway service area"]


@pytest.mark.parametrize(
    "prompt",
    [
        "How many large planes are visible?",
        "How many small planes are visible?",
        "How many complete planes are visible?",
        "How many small vehicles are visible?",
        "How many large vehicles are visible?",
        "How many service vehicles are visible?",
        "How many cars are visible?",
        "How many trucks are visible?",
        "How many boats are visible?",
        "How many tanks are visible?",
        "How many jet bridges are visible?",
        "How many buildings are visible?",
        "How many aircraft are visible?",
        "How many unique objects are visible?",
    ],
)
def test_counting_attribute_and_unknown_targets_are_eligible_but_unsupported(prompt: str) -> None:
    row = _row("count", "a.png", "counting", "1", prompt=prompt)
    assert (
        resolve_cardinality_prompt_class(prompt, _counting_vocab()).status == "unsupported_target"
    )
    assert resolve_counting_class(row, _counting_vocab()).status == "unsupported_target"


@pytest.mark.parametrize(
    "prompt",
    [
        "Is there more than one plane visible?",
        "Are there multiple planes visible?",
        "Does the image contain multiple train stations?",
        "Are there more tennis courts than basketball courts?",
        "Are there more ships or small vehicles in the image?",
    ],
)
def test_counting_non_cardinality_questions_are_unsupported(prompt: str) -> None:
    row = _row("count", "a.png", "counting", "1", prompt=prompt)
    assert resolve_counting_class(row, _counting_vocab()).status == "unsupported_form"


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
        {
            "kind": "detection",
            "sample_id": "d1",
            "image": "full.png",
            "class_name": "car",
            "bbox_xyxy": (0.0, 0.0, 0.2, 0.2),
        },
        {
            "kind": "counting",
            "sample_id": "c1",
            "image": "full.png",
            "class_name": "car",
            "count": 1,
        },
        {
            "kind": "detection",
            "sample_id": "d2",
            "image": "partial.png",
            "class_name": "car",
            "bbox_xyxy": (0.0, 0.0, 0.2, 0.2),
        },
        {
            "kind": "counting",
            "sample_id": "c2",
            "image": "partial.png",
            "class_name": "car",
            "count": 2,
        },
        {
            "kind": "counting",
            "sample_id": "count.png",
            "image": "count.png",
            "class_name": "car",
            "count": 3,
        },
        {
            "kind": "detection",
            "sample_id": "d3",
            "image": "det.png",
            "class_name": "car",
            "bbox_xyxy": (0.0, 0.0, 0.2, 0.2),
        },
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
        _row(
            "d1", "Images/Images_train/a.png", "detection", '{"label":"car","bbox":[0,0,0.2,0.2]}'
        ),
        _row("c1", "Images/Images_train/a.png", "counting", "1", target_class="car"),
        _row(
            "d2", "Images/Images_train/b.png", "detection", '{"label":"car","bbox":[0,0,0.2,0.2]}'
        ),
        _row("c2", "Images/Images_train/b.png", "counting", "2", target_class="car"),
        _row("c3", "Images/Images_train/c.png", "counting", "3", target_class="car"),
        _row(
            "d4", "Images/Images_train/d.png", "detection", '{"label":"car","bbox":[0,0,0.2,0.2]}'
        ),
    ]
    eval_rows = [
        _row(
            "eval", "Images/Images_train/d.png", "detection", '{"label":"car","bbox":[0,0,0.2,0.2]}'
        )
    ]
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
    assert manifest["output_files"]["train.jsonl"]["sha256"] == file_sha256(
        tmp_path / "train.jsonl"
    )
    audit = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    assert audit["train_val_image_overlap"] == 0
    assert audit["train_images"] > 0
    assert audit["val_images"] > 0


def test_builder_counting_audit_distinguishes_target_coverage_from_correctness(
    tmp_path: Path,
) -> None:
    train_rows = [
        _row("d-ship", "ship.png", "detection", '{"label":"ship","bbox":[0,0,0.2,0.2]}'),
        _row("d-ships", "ships.png", "detection", '{"label":"ships","bbox":[0,0,0.2,0.2]}'),
        _row("resolved", "resolved.png", "counting", "1", prompt="How many ship are visible?"),
        _row("ambiguous", "ambiguous.png", "counting", "2", prompt="How many ships are visible?"),
        _row(
            "unresolved",
            "unresolved.png",
            "counting",
            "3",
            prompt="How many buildings are visible?",
        ),
        _row(
            "unsupported",
            "unsupported.png",
            "counting",
            "1",
            prompt="Is there more than one ship visible?",
        ),
    ]
    build_object_adapter_dataset_from_rows(
        train_rows,
        [],
        output_dir=tmp_path,
        enforce_blockers=False,
    )
    audit = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    assert audit["counting_total"] == 4
    assert audit["counting_exact_cardinality"] == 3
    assert audit["counting_cardinality_eligible"] == 3
    assert audit["counting_non_cardinality_excluded"] == 1
    assert audit["counting_class_resolved"] == 1
    assert audit["counting_supported_target"] == 1
    assert audit["counting_target_unsupported"] == 1
    assert audit["counting_class_unresolved"] == 1
    assert audit["counting_class_ambiguous"] == 1
    assert audit["counting_total"] == (
        audit["counting_non_cardinality_excluded"] + audit["counting_exact_cardinality"]
    )
    assert audit["counting_exact_cardinality"] == (
        audit["counting_class_resolved"]
        + audit["counting_target_unsupported"]
        + audit["counting_class_ambiguous"]
    )
    assert audit["counting_target_coverage"] == pytest.approx(1 / 3)
    assert audit["counting_class_resolution_rate"] == pytest.approx(1 / 3)
    assert audit["counting_resolution_status_distribution"] == {
        "ambiguous": 1,
        "resolved": 1,
        "unsupported_target": 1,
        "unsupported_form": 1,
    }
    assert len(audit["non_cardinality_examples"]) == 1
    assert len(audit["unsupported_target_examples"]) == 1
    assert len(audit["ambiguous_prompt_examples"]) == 1
    assert audit["unsupported_target_prefix_top50"] == [{"target_prefix": "buildings", "count": 1}]
    assert "counting_class_ambiguous=1 != 0" in audit["hard_blockers"]
    pairs = [
        json.loads(line)
        for line in (tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    pairs += [
        json.loads(line)
        for line in (tmp_path / "val.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all(pair["class_name"] != "buildings" for pair in pairs)


def test_low_target_coverage_is_not_a_hard_blocker(tmp_path: Path) -> None:
    rows = [
        _row("d-ship", "ship.png", "detection", '{"label":"ship","bbox":[0,0,0.2,0.2]}'),
        _row("resolved", "resolved.png", "counting", "1", prompt="How many ships are visible?"),
        _row(
            "unsupported", "building.png", "counting", "2", prompt="How many buildings are visible?"
        ),
    ]
    build_object_adapter_dataset_from_rows(rows, [], output_dir=tmp_path, enforce_blockers=False)
    audit = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    assert audit["counting_target_coverage"] == pytest.approx(0.5)
    assert all("counting_target_coverage" not in blocker for blocker in audit["hard_blockers"])
    assert all(
        "counting_class_resolution_rate" not in blocker for blocker in audit["hard_blockers"]
    )


def test_evaluator_skips_non_cardinality_counting_and_reports_eligible_support() -> None:
    evaluator_path = (
        Path(__file__).parents[2] / "scripts" / "evaluation" / "evaluate_object_adapter_v0.py"
    )
    spec = importlib.util.spec_from_file_location("object_adapter_evaluator_test", evaluator_path)
    assert spec is not None and spec.loader is not None
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    rows = [
        _row("supported", "ship.png", "counting", "1", prompt="How many ships are visible?"),
        _row("unknown", "building.png", "counting", "2", prompt="How many buildings are visible?"),
        _row(
            "binary", "plane.png", "counting", "1", prompt="Is there more than one plane visible?"
        ),
    ]
    prepared, skipped, counting_statistics = evaluator._prepare_rows(rows, _counting_vocab())
    assert [row["id"] for row in prepared] == ["supported"]
    assert skipped["counting_non_cardinality"] == 1
    assert skipped["counting_unsupported_target"] == 1
    assert skipped["class_unresolved"] == 0
    assert counting_statistics["counting_population_count"] == 3
    assert counting_statistics["counting_exact_cardinality_count"] == 2
    assert counting_statistics["counting_cardinality_eligible_count"] == 2
    assert counting_statistics["counting_supported_count"] == 1
    assert counting_statistics["counting_unsupported_target_count"] == 1
    assert counting_statistics["counting_non_cardinality_count"] == 1
    assert counting_statistics["counting_target_coverage"] == 0.5
    assert counting_statistics["counting_supported_prediction_count"] == 1
    assert isinstance(counting_statistics["counting_supported_prediction_ids_sha256"], str)


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


def test_manifest_from_builder_v1_1_is_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "builder_version": "rs-object-adapter-v0-1.1",
                "audit_status": "passed",
                "final_image_overlap_count": 0,
                "train_val_image_overlap": 0,
                "output_files": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsupported Object Adapter manifest builder"):
        validate_data_manifest(manifest)


def test_visual_processor_batch_disables_text_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """视觉特征路径不能截断 Qwen 图像占位 token。"""

    pytest.importorskip("torch")
    from sat_rs_vlm.training import object_adapter_v0 as training_module

    captured: dict[str, object] = {}

    class FakeCollator:
        def __init__(self, processor: object, **kwargs: object) -> None:
            captured["processor"] = processor
            captured.update(kwargs)

        def __call__(self, batch: list[dict[str, object]]) -> dict[str, object]:
            captured["batch"] = batch
            return {"pixel_values": object(), "image_grid_thw": object()}

    monkeypatch.setattr(training_module, "Qwen3VLDataCollator", FakeCollator)
    batch = training_module.visual_processor_batch(
        object(),
        [{"image": "image.png", "class_name": "ship"}],
        image_root=".",
    )

    assert captured["for_generation"] is True
    assert captured["truncation"] is False
    assert batch["pixel_values"] is not None


def test_blocked_audit_is_explicit(tmp_path: Path) -> None:
    row = _row("d", "a.png", "detection", '{"label":"car","bbox":[0,0,0.2,0.2]}')
    with pytest.raises(DataAuditBlocked) as exc_info:
        build_object_adapter_dataset_from_rows(
            [row], [], output_dir=tmp_path, enforce_blockers=True
        )
    assert all("counting_class_resolution_rate" not in item for item in exc_info.value.blockers)


def test_hungarian_permutation_and_loss_behaviour() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("scipy")
    from sat_rs_vlm.models.rs_object_adapter import RSObjectAdapter
    from sat_rs_vlm.training.object_adapter_v0 import compute_object_adapter_loss, hungarian_match

    pred_boxes = torch.tensor(
        [[0.75, 0.75, 0.2, 0.2], [0.25, 0.25, 0.2, 0.2]] + [[0.5, 0.5, 0.1, 0.1]] * 62
    )
    target_boxes = torch.tensor([[0.0, 0.0, 0.2, 0.2], [0.65, 0.65, 0.85, 0.85]])
    rows, columns = hungarian_match(torch.zeros(64), pred_boxes, target_boxes)
    assert rows.numel() == columns.numel() == 2
    assert set(rows.tolist()) == {0, 1}
    outputs = {
        "object_logits": torch.zeros(4, 64, requires_grad=True),
        "boxes_cxcywh": pred_boxes.unsqueeze(0).repeat(4, 1, 1).requires_grad_(),
    }
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

    target = {
        "supervision_type": "full_set",
        "boxes_xyxy": [[0.15, 0.15, 0.35, 0.35]],
        "count": None,
    }
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
    full_target = {
        "supervision_type": "full_set",
        "boxes_xyxy": [[0.15, 0.15, 0.35, 0.35]],
        "count": None,
    }
    full_low = compute_object_adapter_loss(
        {"object_logits": full_negative_low, "boxes_cxcywh": exact_boxes.unsqueeze(0)},
        [full_target],
    )
    full_high = compute_object_adapter_loss(
        {"object_logits": full_negative_high, "boxes_cxcywh": exact_boxes.unsqueeze(0)},
        [full_target],
    )
    assert float(full_high["loss_objectness"]) > float(full_low["loss_objectness"])

    partial_low = compute_object_adapter_loss(
        {"object_logits": full_negative_low, "boxes_cxcywh": exact_boxes.unsqueeze(0)},
        [
            {
                "supervision_type": "partial_set",
                "boxes_xyxy": [[0.15, 0.15, 0.35, 0.35]],
                "count": None,
            }
        ],
    )
    partial_high = compute_object_adapter_loss(
        {"object_logits": full_negative_high, "boxes_cxcywh": exact_boxes.unsqueeze(0)},
        [
            {
                "supervision_type": "partial_set",
                "boxes_xyxy": [[0.15, 0.15, 0.35, 0.35]],
                "count": None,
            }
        ],
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
    optimizer_ids = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }
    assert optimizer_ids == {id(parameter) for parameter in adapter.parameters()}
    assert not optimizer_ids.intersection({id(parameter) for parameter in visual.parameters()})


def test_aligned_multi_object_giou_uses_only_hungarian_pairs() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("scipy")
    from sat_rs_vlm.models.rs_object_adapter import xyxy_to_cxcywh
    from sat_rs_vlm.training.object_adapter_v0 import compute_object_adapter_loss

    target_boxes = torch.tensor(
        [[0.10, 0.10, 0.20, 0.20], [0.70, 0.70, 0.80, 0.80]],
        dtype=torch.float32,
    )
    predictions = torch.full((64, 4), 0.5, dtype=torch.float32)
    predictions[:, 2:] = 0.05
    predictions[:2] = xyxy_to_cxcywh(target_boxes)
    losses = compute_object_adapter_loss(
        {
            "object_logits": torch.zeros(1, 64),
            "boxes_cxcywh": predictions.unsqueeze(0),
        },
        [
            {
                "supervision_type": "detection_only",
                "boxes_xyxy": target_boxes.tolist(),
                "count": None,
            }
        ],
    )

    assert float(losses["loss_bbox_l1"]) < 1e-5
    assert float(losses["loss_giou"]) < 1e-5


def test_small_bbox_iou_and_giou_are_stable_for_bfloat16_inputs() -> None:
    torch = pytest.importorskip("torch")
    from sat_rs_vlm.models.rs_object_adapter import (
        generalized_iou_xyxy,
        pairwise_iou_xyxy,
    )

    box = torch.tensor(
        [[0.100, 0.100, 0.110, 0.110]],
        dtype=torch.bfloat16,
    )
    iou = pairwise_iou_xyxy(box, box)
    giou = generalized_iou_xyxy(box, box)

    assert iou.dtype == torch.float32
    assert giou.dtype == torch.float32
    assert float(iou.item()) == pytest.approx(1.0, abs=1e-5)
    assert float(giou.item()) == pytest.approx(1.0, abs=1e-5)


def test_hungarian_matches_swapped_small_boxes() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("scipy")
    from sat_rs_vlm.models.rs_object_adapter import xyxy_to_cxcywh
    from sat_rs_vlm.training.object_adapter_v0 import hungarian_match

    targets = torch.tensor(
        [[0.100, 0.100, 0.110, 0.110], [0.700, 0.700, 0.710, 0.710]],
        dtype=torch.bfloat16,
    )
    predictions = xyxy_to_cxcywh(targets.float()).flip(0).to(torch.bfloat16)
    rows, columns = hungarian_match(torch.zeros(2, dtype=torch.bfloat16), predictions, targets)

    assert rows.tolist() == [0, 1]
    assert columns.tolist() == [1, 0]


def test_object_adapter_loss_is_finite_for_bfloat16_predictions() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("scipy")
    from sat_rs_vlm.training.object_adapter_v0 import compute_object_adapter_loss

    boxes = torch.full((1, 64, 4), 0.5, dtype=torch.bfloat16)
    boxes[..., 2:] = 0.01
    losses = compute_object_adapter_loss(
        {
            "object_logits": torch.zeros(1, 64, dtype=torch.bfloat16),
            "boxes_cxcywh": boxes,
        },
        [
            {
                "supervision_type": "full_set",
                "boxes_xyxy": [[0.100, 0.100, 0.110, 0.110]],
                "count": 1,
            }
        ],
    )

    for name in ("loss_total", "loss_bbox_l1", "loss_giou", "loss_count"):
        assert losses[name].dtype == torch.float32
        assert bool(torch.isfinite(losses[name]).item())


def test_constant_after_warmup_uses_nonzero_first_optimizer_lr() -> None:
    torch = pytest.importorskip("torch")
    from sat_rs_vlm.training.object_adapter_v0 import _constant_after_warmup_lambda

    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.SGD([parameter], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _constant_after_warmup_lambda(step, 4),
    )
    used_lrs: list[float] = []
    for _ in range(6):
        parameter.grad = torch.ones_like(parameter)
        used_lrs.append(float(optimizer.param_groups[0]["lr"]))
        optimizer.step()
        scheduler.step()

    assert used_lrs[0] == pytest.approx(2.5e-4)
    assert used_lrs[3] == pytest.approx(1e-3)
    assert used_lrs[4] == pytest.approx(1e-3)


def test_accumulation_tail_window_uses_its_actual_size() -> None:
    torch = pytest.importorskip("torch")
    from sat_rs_vlm.training.object_adapter_v0 import _accumulation_window_size

    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    window_sizes: list[int] = []
    for batch_index in range(1, 4):
        window_size = _accumulation_window_size(batch_index, 3, 2)
        window_sizes.append(window_size)
        (parameter / window_size).backward()
        if batch_index % 2 == 0 or batch_index == 3:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    assert window_sizes == [2, 2, 1]
    assert float(parameter.item()) == pytest.approx(-2.0)


def test_frozen_visual_feature_extractor_rejects_reversed_selected_blocks() -> None:
    torch = pytest.importorskip("torch")
    from sat_rs_vlm.training.object_adapter_v0 import FrozenVisualFeatureExtractor

    class FakeVisual(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = torch.nn.ModuleList(torch.nn.Identity() for _ in range(24))

    with pytest.raises(ValueError, match="fixed"):
        FrozenVisualFeatureExtractor(
            FakeVisual(),
            selected_blocks=(23, 17, 11, 5),
        )


def test_epoch_checkpoint_is_saved_before_internal_validation() -> None:
    from sat_rs_vlm.training.object_adapter_v0 import _save_epoch_checkpoint_before_validation

    events: list[str] = []

    def save_checkpoint() -> Path:
        events.append("checkpoint")
        return Path("checkpoint_epoch_1")

    def validate() -> dict[str, str]:
        events.append("validation")
        return {"status": "ok"}

    checkpoint, validation = _save_epoch_checkpoint_before_validation(
        save_checkpoint,
        validate,
    )
    assert checkpoint == Path("checkpoint_epoch_1")
    assert validation == {"status": "ok"}
    assert events == ["checkpoint", "validation"]


def test_evaluator_adapter_forward_uses_cuda_bf16_autocast() -> None:
    evaluator = _load_object_adapter_evaluator_module()

    class FakeContext:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_args: object) -> None:
            return None

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

    class FakeTorch:
        bfloat16 = "bf16"
        cuda = FakeCuda()

        def __init__(self) -> None:
            self.autocast_calls: list[dict[str, object]] = []

        @staticmethod
        def no_grad() -> FakeContext:
            return FakeContext()

        def autocast(self, **kwargs: object) -> FakeContext:
            self.autocast_calls.append(kwargs)
            return FakeContext()

    class FakeTensor:
        def __init__(self) -> None:
            self.devices: list[object] = []

        def to(self, device: object) -> "FakeTensor":
            self.devices.append(device)
            return self

    class FakeAdapter:
        def __init__(self) -> None:
            self.called = False

        def __call__(self, *_args: object, **_kwargs: object) -> dict[str, str]:
            self.called = True
            return {"status": "ok"}

    class FakeDevice:
        type = "cuda"

    fake_torch = FakeTorch()
    adapter = FakeAdapter()
    positions = FakeTensor()
    padding_mask = FakeTensor()
    output = evaluator._run_object_adapter_forward(
        adapter,
        object(),
        positions,
        object(),
        padding_mask,
        torch_module=fake_torch,
        device=FakeDevice(),
        bf16_enabled=True,
    )

    assert output == {"status": "ok"}
    assert adapter.called
    assert fake_torch.autocast_calls == [
        {"device_type": "cuda", "dtype": "bf16", "enabled": True}
    ]
    assert len(positions.devices) == 1
    assert len(padding_mask.devices) == 1
