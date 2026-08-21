"""RS Object Adapter v0 的特征提取、Hungarian loss 和最小训练循环。

本模块只把已经加载的 Qwen3-VL R1 checkpoint 当作冻结视觉特征源。视觉 block
通过 forward hook 提取，forward 始终位于 ``torch.no_grad``；真正建立 autograd
graph 的只有 :class:`RSObjectAdapter`。训练 loss 按 ``full_set``、``partial_set``、
``detection_only`` 和 ``count_only`` 分开计算，并先逐样本归一化再做 batch mean。
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from sat_rs_vlm.data.object_adapter_v0 import count_bin, validate_data_manifest
from sat_rs_vlm.data.qwen3vl_collator import Qwen3VLDataCollator
from sat_rs_vlm.models.rs_object_adapter import (
    RSObjectAdapter,
    adapter_parameter_summary,
    cxcywh_to_xyxy,
    generalized_iou_xyxy,
    pairwise_iou_xyxy,
    xyxy_to_cxcywh,
)
from sat_rs_vlm.models.reliability.checksum import file_sha256
from sat_rs_vlm.training.utils import safe_import_model_dependencies, set_seed
from sat_rs_vlm.training.vision_tuning import load_visual_sidecar, resolve_visual_module
from sat_rs_vlm.utils.jsonl import read_jsonl


SELECTED_BLOCKS = (5, 11, 17, 23)


class ObjectAdapterDataset(Dataset[dict[str, Any]]):
    """读取 builder 产生的 pair JSONL；图片仍保持 portable relative path。"""

    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self.rows = [dict(row) for row in rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def object_adapter_collate(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """保留变长 pair 列表；视觉 token padding 在 hook 提取后按 batch 完成。"""

    return batch


def object_messages(row: Mapping[str, Any]) -> dict[str, Any]:
    """构造只服务于 processor 图像编码的短消息，不把文本送进 Qwen LLM。"""

    return {
        **dict(row),
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(row["image"])},
                    {"type": "text", "text": f"Find all {row['class_name']} objects in this image."},
                ],
            }
        ],
    }


def visual_processor_batch(
    processor: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    image_root: str | Path,
    max_seq_length: int = 128,
) -> dict[str, Any]:
    """复用正式 Qwen collator 的 image processor 路径，仅取 pixel/grid 张量。"""

    collator = Qwen3VLDataCollator(
        processor,
        max_seq_length=max_seq_length,
        image_root=image_root,
        for_generation=True,
    )
    batch = collator([object_messages(row) for row in rows])
    if "pixel_values" not in batch or "image_grid_thw" not in batch:
        raise ValueError("Qwen processor output must contain pixel_values and image_grid_thw")
    return batch


def _as_feature_tensor(value: Any, *, block_index: int) -> Tensor:
    """严格处理 hook 输出，禁止静默 reshape 或猜测 tuple 含义。"""

    if isinstance(value, Tensor):
        return value
    if isinstance(value, (tuple, list)):
        if not value or not isinstance(value[0], Tensor):
            raise TypeError(f"Visual block {block_index} tuple/list first item must be Tensor")
        return value[0]
    raise TypeError(f"Visual block {block_index} output must be Tensor or tuple/list[Tensor]")


def _grid_rows(image_grid_thw: Tensor, batch_size: int) -> list[tuple[int, int, int]]:
    """读取每张图的 t/h/w，不对 token 数做 shape 猜测。"""

    grid = image_grid_thw.detach().to("cpu").reshape(-1, 3)
    if int(grid.shape[0]) != batch_size:
        raise ValueError(
            "v0 requires exactly one image per pair; image_grid_thw rows do not match batch: "
            f"rows={int(grid.shape[0])}, batch={batch_size}"
        )
    rows: list[tuple[int, int, int]] = []
    for values in grid.tolist():
        t, height, width = (int(value) for value in values)
        if min(t, height, width) <= 0:
            raise ValueError(f"image_grid_thw must be positive, got {(t, height, width)}")
        rows.append((t, height, width))
    return rows


class FrozenVisualFeatureExtractor:
    """从 Qwen visual blocks[5,11,17,23] 提取并验证 patch features。"""

    def __init__(
        self,
        visual: Any,
        *,
        selected_blocks: Sequence[int] = SELECTED_BLOCKS,
        expected_num_blocks: int = 24,
        expected_hidden_size: int = 1024,
    ) -> None:
        self.visual = visual
        self.selected_blocks = tuple(int(index) for index in selected_blocks)
        self.expected_num_blocks = int(expected_num_blocks)
        self.expected_hidden_size = int(expected_hidden_size)
        blocks = list(getattr(visual, "blocks", []))
        if len(blocks) != self.expected_num_blocks:
            raise ValueError(
                f"Qwen visual blocks mismatch: expected {self.expected_num_blocks}, got {len(blocks)}"
            )
        if tuple(sorted(self.selected_blocks)) != SELECTED_BLOCKS:
            raise ValueError(f"v0 selected blocks are fixed to {SELECTED_BLOCKS}")
        self._captured: dict[int, Any] = {}
        self._handles = [
            blocks[index].register_forward_hook(self._make_hook(index))
            for index in self.selected_blocks
        ]

    def _make_hook(self, index: int):  # type: ignore[no-untyped-def]
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            self._captured[index] = _as_feature_tensor(output, block_index=index).detach()

        return hook

    def close(self) -> None:
        """移除四个 hook，避免重复训练/评测时累积引用。"""

        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown path
        try:
            self.close()
        except Exception:
            pass

    def extract(self, batch: Mapping[str, Any]) -> tuple[list[list[Tensor]], list[Tensor]]:
        """执行 frozen visual forward，返回每个样本的四层 feature 和 positions。"""

        pixel_values = batch["pixel_values"]
        image_grid_thw = batch["image_grid_thw"]
        if not isinstance(pixel_values, Tensor) or not isinstance(image_grid_thw, Tensor):
            raise TypeError("pixel_values and image_grid_thw must be torch tensors")
        batch_size = int(image_grid_thw.reshape(-1, 3).shape[0])
        grids = _grid_rows(image_grid_thw, batch_size)
        expected_counts = [t * height * width for t, height, width in grids]
        try:
            visual_parameter = next(self.visual.parameters())
        except StopIteration as exc:
            raise ValueError("Resolved Qwen visual module has no parameters") from exc
        device = visual_parameter.device
        self._captured.clear()
        with torch.no_grad():
            # Qwen3-VL 的视觉模块直接接收 patch_embed 的原始输入，参数名是
            # ``hidden_states``/``grid_thw``；顶层 Qwen 模型才把它们称为
            # ``pixel_values``/``image_grid_thw``。优先使用当前视觉模块的真实
            # 签名，同时保留旧版本 Transformers 的兼容尝试。
            visual_inputs = pixel_values.to(device=device, dtype=visual_parameter.dtype)
            visual_grid = image_grid_thw.to(device)
            try:
                self.visual(
                    hidden_states=visual_inputs,
                    grid_thw=visual_grid,
                )
            except TypeError as first_error:
                try:
                    self.visual(
                        pixel_values=visual_inputs,
                        image_grid_thw=visual_grid,
                    )
                except TypeError:
                    try:
                        self.visual(pixel_values=visual_inputs, grid_thw=visual_grid)
                    except TypeError:
                        raise first_error
        if set(self._captured) != set(self.selected_blocks):
            raise RuntimeError(
                f"Visual hooks did not capture all blocks: expected {self.selected_blocks}, "
                f"got {sorted(self._captured)}"
            )
        per_layer: dict[int, list[Tensor]] = {}
        for index in self.selected_blocks:
            output = self._captured[index]
            if output.ndim == 3:
                if int(output.shape[0]) != batch_size:
                    raise ValueError(
                        f"Block {index} 3D output first dimension must be batch, got {tuple(output.shape)}"
                    )
                if any(int(output.shape[1]) != count for count in expected_counts):
                    raise ValueError(
                        f"Block {index} 3D output requires equal per-sample token counts: "
                        f"expected={expected_counts}, actual={int(output.shape[1])}"
                    )
                chunks = [output[item, : expected_counts[item]] for item in range(batch_size)]
            elif output.ndim == 2:
                if int(output.shape[0]) != sum(expected_counts):
                    raise ValueError(
                        f"Block {index} token count mismatch: expected={sum(expected_counts)}, "
                        f"actual={int(output.shape[0])}"
                    )
                chunks = list(torch.split(output, expected_counts, dim=0))
            else:
                raise ValueError(f"Block {index} output must be 2D or 3D, got {tuple(output.shape)}")
            if int(output.shape[-1]) != self.expected_hidden_size:
                raise ValueError(
                    f"Block {index} hidden size mismatch: expected={self.expected_hidden_size}, "
                    f"actual={int(output.shape[-1])}"
                )
            per_layer[index] = [chunk.detach() for chunk in chunks]
        positions: list[Tensor] = []
        for t, height, width in grids:
            y, x = torch.meshgrid(
                (torch.arange(height, dtype=torch.float32) + 0.5) / height,
                (torch.arange(width, dtype=torch.float32) + 0.5) / width,
                indexing="ij",
            )
            centers = torch.stack((x.reshape(-1), y.reshape(-1)), dim=-1)
            positions.append(centers.repeat(t, 1))
        return [
            [per_layer[index][sample_index] for index in self.selected_blocks]
            for sample_index in range(batch_size)
        ], positions


def pad_visual_features(
    features: Sequence[Sequence[Tensor]], positions: Sequence[Tensor]
) -> tuple[list[Tensor], Tensor, Tensor]:
    """把不同 image grid 的 patch 序列 padding 成 adapter batch 输入。"""

    if len(features) != len(positions) or not features:
        raise ValueError("features and positions must be non-empty and have equal batch size")
    max_tokens = max(int(position.shape[0]) for position in positions)
    layer_batch: list[Tensor] = []
    for layer_index in range(4):
        hidden = int(features[0][layer_index].shape[-1])
        output = torch.zeros(
            (len(features), max_tokens, hidden), dtype=features[0][layer_index].dtype,
            device=features[0][layer_index].device,
        )
        for sample_index, sample_layers in enumerate(features):
            tokens = int(sample_layers[layer_index].shape[0])
            output[sample_index, :tokens] = sample_layers[layer_index]
        layer_batch.append(output)
    position_batch = torch.zeros(
        (len(positions), max_tokens, 2), dtype=positions[0].dtype, device=positions[0].device
    )
    padding_mask = torch.ones(
        (len(positions), max_tokens), dtype=torch.bool, device=positions[0].device
    )
    for index, position in enumerate(positions):
        tokens = int(position.shape[0])
        position_batch[index, :tokens] = position.to(position_batch.device)
        padding_mask[index, :tokens] = False
    return layer_batch, position_batch, padding_mask


def hungarian_match(
    pred_logits: Tensor,
    pred_boxes_cxcywh: Tensor,
    target_boxes_xyxy: Tensor,
) -> tuple[Tensor, Tensor]:
    """用指定 objectness/L1/GIoU cost 做 Hungarian assignment。

    assignment 只在 detached CPU numpy 上执行，不把离散匹配操作放入 autograd。
    """

    if target_boxes_xyxy.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=pred_logits.device)
        return empty, empty
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError("scipy>=1.10,<2 is required for Hungarian matching") from exc
    target_boxes = xyxy_to_cxcywh(target_boxes_xyxy)
    l1 = (pred_boxes_cxcywh[:, None, :] - target_boxes[None, :, :]).abs().mean(dim=-1)
    giou = generalized_iou_xyxy(
        cxcywh_to_xyxy(pred_boxes_cxcywh), target_boxes_xyxy
    )
    positive_bce = F.binary_cross_entropy_with_logits(
        pred_logits[:, None].expand(-1, target_boxes.shape[0]),
        torch.ones_like(pred_logits[:, None].expand(-1, target_boxes.shape[0])),
        reduction="none",
    )
    cost = 5.0 * l1 + 2.0 * (1.0 - giou) + positive_bce
    rows, columns = linear_sum_assignment(cost.detach().float().cpu().numpy())
    return (
        torch.as_tensor(rows, dtype=torch.long, device=pred_logits.device),
        torch.as_tensor(columns, dtype=torch.long, device=pred_logits.device),
    )


def _zero(reference: Tensor) -> Tensor:
    return reference.sum() * 0.0


def compute_object_adapter_loss(
    outputs: Mapping[str, Tensor],
    targets: Sequence[Mapping[str, Any]],
    *,
    objectness_weight: float = 1.0,
    bbox_l1_weight: float = 5.0,
    giou_weight: float = 2.0,
    count_weight: float = 1.0,
    binarization_weight: float = 0.01,
    negative_query_weight: float = 0.1,
    smooth_l1_beta: float = 1.0,
) -> dict[str, Tensor | float | int]:
    """按 supervision type 逐样本计算并归一化 v0 五项 loss。"""

    logits = outputs["object_logits"]
    boxes = outputs["boxes_cxcywh"]
    if int(logits.shape[0]) != len(targets):
        raise ValueError("outputs batch and targets length differ")
    components: dict[str, list[Tensor]] = {name: [] for name in (
        "loss_objectness", "loss_bbox_l1", "loss_giou", "loss_count", "loss_binarization"
    )}
    matched_objects = 0
    predicted_counts: list[Tensor] = []
    true_counts: list[float] = []
    count_errors: list[Tensor] = []
    for sample_index, target in enumerate(targets):
        sample_logits = logits[sample_index]
        sample_boxes = boxes[sample_index]
        supervision = str(target.get("supervision_type", "")).strip()
        raw_boxes = target.get("boxes_xyxy", [])
        target_boxes = torch.as_tensor(
            raw_boxes, dtype=sample_boxes.dtype, device=sample_boxes.device
        ).reshape(-1, 4)
        matched_rows, matched_columns = hungarian_match(sample_logits, sample_boxes, target_boxes)
        matched_objects += int(matched_rows.numel())
        if supervision == "full_set":
            object_target = torch.zeros_like(sample_logits)
            object_weight = torch.full_like(sample_logits, float(negative_query_weight))
            if matched_rows.numel():
                object_target[matched_rows] = 1.0
                object_weight[matched_rows] = 1.0
            components["loss_objectness"].append(
                F.binary_cross_entropy_with_logits(
                    sample_logits, object_target, weight=object_weight, reduction="sum"
                ) / object_weight.sum().clamp_min(torch.finfo(sample_logits.dtype).eps)
            )
        elif supervision in {"partial_set", "detection_only"}:
            if matched_rows.numel():
                components["loss_objectness"].append(
                    F.binary_cross_entropy_with_logits(
                        sample_logits[matched_rows], torch.ones_like(sample_logits[matched_rows])
                    )
                )
        elif supervision not in {"count_only"}:
            raise ValueError(f"Unsupported supervision_type: {supervision}")
        if supervision in {"full_set", "partial_set", "detection_only"} and matched_rows.numel():
            pred_matched = sample_boxes[matched_rows]
            gt_matched = xyxy_to_cxcywh(target_boxes[matched_columns])
            components["loss_bbox_l1"].append(F.l1_loss(pred_matched, gt_matched))
            giou = generalized_iou_xyxy(
                cxcywh_to_xyxy(pred_matched), target_boxes[matched_columns]
            )
            components["loss_giou"].append((1.0 - giou).mean())
        count_value = target.get("count")
        if count_value is not None:
            true_count = float(count_value)
            predicted_count = torch.sigmoid(sample_logits).sum()
            components["loss_count"].append(
                F.smooth_l1_loss(
                    predicted_count,
                    torch.as_tensor(true_count, dtype=predicted_count.dtype, device=predicted_count.device),
                    beta=smooth_l1_beta,
                )
            )
            predicted_counts.append(predicted_count.detach())
            true_counts.append(true_count)
            count_errors.append((predicted_count.detach() - true_count).abs())
        if supervision in {"partial_set", "count_only"}:
            probabilities = torch.sigmoid(sample_logits)
            components["loss_binarization"].append((probabilities * (1.0 - probabilities)).mean())
    means: dict[str, Tensor] = {}
    for name, values in components.items():
        means[name] = torch.stack(values).mean() if values else _zero(logits)
    means["loss_total"] = (
        objectness_weight * means["loss_objectness"]
        + bbox_l1_weight * means["loss_bbox_l1"]
        + giou_weight * means["loss_giou"]
        + count_weight * means["loss_count"]
        + binarization_weight * means["loss_binarization"]
    )
    means["matched_objects"] = int(matched_objects)
    means["mean_predicted_count"] = (
        torch.stack(predicted_counts).mean() if predicted_counts else _zero(logits)
    )
    means["mean_true_count"] = (
        float(sum(true_counts) / len(true_counts)) if true_counts else 0.0
    )
    means["mean_count_abs_error"] = (
        torch.stack(count_errors).mean() if count_errors else _zero(logits)
    )
    return means


def _metrics_from_count_pairs(predicted: list[float], truth: list[int]) -> dict[str, Any]:
    """聚合 internal/E2 counting 指标，rounded 使用 floor(x+0.5) 并 clip [0,64]。"""

    if not truth:
        return {"n": 0, "continuous_mae": None, "rounded_exact": None, "rounded_within_1": None}
    errors = [pred - target for pred, target in zip(predicted, truth)]
    rounded = [min(64, max(0, math.floor(pred + 0.5))) for pred in predicted]
    abs_errors = [abs(value) for value in errors]
    return {
        "n": len(truth),
        "continuous_mae": sum(abs_errors) / len(abs_errors),
        "rounded_exact": sum(int(pred == target) for pred, target in zip(rounded, truth)) / len(truth),
        "rounded_within_1": sum(int(abs(pred - target) <= 1) for pred, target in zip(rounded, truth)) / len(truth),
        "rounded_mae": sum(abs(pred - target) for pred, target in zip(rounded, truth)) / len(truth),
        "rmse": math.sqrt(sum(value * value for value in errors) / len(errors)),
        "bias": sum(errors) / len(errors),
        "by_count_bin": {
            bucket: _metrics_from_count_pairs(
                [pred for pred, target in zip(predicted, truth) if count_bin(target) == bucket],
                [target for target in truth if count_bin(target) == bucket],
            )
            for bucket in ("0-2", "3-5", "6-10", "11+")
            if any(count_bin(target) == bucket for target in truth)
        },
    }


def _box_metrics(predicted_boxes: list[Tensor], targets: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """计算 top-1、best-of-K、confident proposal coverage。"""

    top1: list[float] = []
    best: list[float] = []
    confident: list[float] = []
    by_area: dict[str, list[float]] = {"small": [], "medium": [], "large": []}
    for prediction, target in zip(predicted_boxes, targets):
        gt = torch.as_tensor(target.get("boxes_xyxy", []), dtype=prediction.dtype, device=prediction.device).reshape(-1, 4)
        if gt.numel() == 0:
            continue
        pred_xyxy = cxcywh_to_xyxy(prediction[:, 1:])
        probabilities = prediction[:, 0].sigmoid()
        top = int(probabilities.argmax().item())
        top_ious = pairwise_iou_xyxy(pred_xyxy[top], gt).reshape(-1)
        top1.extend(float(value) for value in top_ious)
        all_ious = pairwise_iou_xyxy(pred_xyxy, gt)
        best.extend(float(value) for value in all_ious.max(dim=0).values)
        selected = all_ious[probabilities >= 0.5]
        confident.extend(
            float(value) for value in (selected.max(dim=0).values if selected.numel() else torch.zeros(gt.shape[0], device=gt.device))
        )
        areas = ((gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])).tolist()
        best_values = all_ious.max(dim=0).values.tolist()
        for area, value in zip(areas, best_values):
            bucket = "small" if area < 0.01 else "medium" if area < 0.10 else "large"
            by_area[bucket].append(float(value))
    def summary(values: Sequence[float]) -> dict[str, Any]:
        return {
            "n": len(values),
            "mean_iou": sum(values) / len(values) if values else None,
            "recall_at_0_5": sum(value >= 0.5 for value in values) / len(values) if values else None,
            "recall_at_0_7": sum(value >= 0.7 for value in values) / len(values) if values else None,
        }
    return {"top1": summary(top1), "best_of_k": summary(best), "confident": summary(confident), "by_area": {key: summary(value) for key, value in by_area.items()}}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _resolve_project_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root / path


def _cast_features_for_adapter(layer_batch: Sequence[Tensor], adapter: RSObjectAdapter) -> list[Tensor]:
    """把 frozen visual features 转成 Adapter 权重 dtype，避免 bf16/fp32 混算报错。"""

    parameter = next(adapter.parameters(), None)
    if parameter is None:
        raise ValueError("RS Object Adapter has no parameters")
    return [feature.to(device=parameter.device, dtype=parameter.dtype) for feature in layer_batch]


def run_object_adapter_training(
    config: Mapping[str, Any],
    *,
    project_root: str | Path = ".",
    max_train_groups: int | None = None,
    max_steps: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """执行 v0 训练；真实模型加载和计算只发生在显式调用脚本时。"""

    root = Path(project_root).resolve()
    seed = int(config.get("experiment", {}).get("seed", 42))
    set_seed(seed)
    data_cfg = dict(config.get("data", {}))
    output_dir = _resolve_project_path(
        str(config.get("training", {}).get("output_dir", "outputs/experiments/rs_object_adapter_v0")), root
    )
    manifest_path = output_dir.parent / "data_manifest.json"
    configured_manifest = data_cfg.get("manifest") or "data/processed/rs_object_adapter_v0/manifest.json"
    manifest_path = _resolve_project_path(str(configured_manifest), root)
    data_manifest = validate_data_manifest(manifest_path)
    data_dir = manifest_path.parent
    train_rows = list(read_jsonl(data_dir / "train.jsonl"))
    val_rows = list(read_jsonl(data_dir / "val.jsonl"))
    class_vocab = json.loads((data_dir / "class_vocab.json").read_text(encoding="utf-8"))
    model_cfg = dict(config.get("model", {}))
    checkpoint = _resolve_project_path(str(model_cfg["checkpoint_dir"]), root)
    modules = safe_import_model_dependencies()
    torch_module = modules["torch"]
    model_cfg_for_loader = {
        "local_files_only": True,
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", True)),
        "torch_dtype": str(model_cfg.get("torch_dtype", "bfloat16")),
        "device_map": model_cfg.get("device_map", "auto"),
        "attn_implementation": model_cfg.get("attn_implementation", "sdpa"),
    }
    from sat_rs_vlm.evaluation.checkpoint_loader import load_finetuned_checkpoint

    model, processor, checkpoint_manifest = load_finetuned_checkpoint(
        checkpoint, model_cfg_for_loader, modules
    )
    sidecar_name = checkpoint_manifest.get("visual_sidecar")
    if not sidecar_name:
        for candidate in ("visual_trainable_weights.safetensors", "h1_visual_weights.safetensors"):
            if (checkpoint / candidate).is_file():
                sidecar_name = candidate
                break
    requires_sidecar = checkpoint_manifest.get("checkpoint_type") == "adapter_with_visual_sidecar"
    if requires_sidecar and not sidecar_name:
        raise ValueError("R1 strategy manifest does not declare visual_sidecar")
    if sidecar_name:
        load_visual_sidecar(model, checkpoint / str(sidecar_name))
    visual = resolve_visual_module(model)
    for parameter in visual.parameters():
        parameter.requires_grad = False
    visual.eval()
    visual_parameter_count = sum(int(parameter.numel()) for parameter in visual.parameters())
    extractor = FrozenVisualFeatureExtractor(
        visual,
        selected_blocks=tuple(model_cfg.get("selected_blocks", SELECTED_BLOCKS)),
        expected_num_blocks=int(model_cfg.get("expected_num_blocks", 24)),
        expected_hidden_size=int(model_cfg.get("expected_hidden_size", 1024)),
    )
    adapter_cfg = dict(config.get("adapter", {}))
    adapter = RSObjectAdapter(
        len(class_vocab["classes"]),
        vit_hidden_size=int(model_cfg.get("expected_hidden_size", 1024)),
        d_model=int(adapter_cfg.get("d_model", 256)),
        num_queries=int(adapter_cfg.get("num_queries", 64)),
        nhead=int(adapter_cfg.get("nhead", 8)),
        decoder_layers=int(adapter_cfg.get("decoder_layers", 2)),
        dim_feedforward=int(adapter_cfg.get("dim_feedforward", 1024)),
        dropout=float(adapter_cfg.get("dropout", 0.1)),
    )
    visual_device = next(visual.parameters()).device
    adapter.to(visual_device)
    summary = adapter_parameter_summary(adapter)
    print(f"visual_parameter_count={visual_parameter_count}")
    print(f"adapter_parameter_count={summary['parameter_count']}")
    print(f"trainable_parameter_count={summary['trainable_parameter_count']}")
    print("trainable_parameter_names=" + json.dumps(summary["names"], ensure_ascii=False))
    if any(parameter.requires_grad for parameter in visual.parameters()):
        raise AssertionError("Qwen visual trainable parameters must be zero")
    if dry_run:
        _write_json(output_dir / "dry_run_summary.json", {"data_manifest": data_manifest, "visual_parameter_count": visual_parameter_count, "adapter": summary})
        extractor.close()
        return {"status": "dry_run", "visual_parameter_count": visual_parameter_count, "adapter": summary}
    del model
    if bool(torch_module.cuda.is_available()):
        torch_module.cuda.empty_cache()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "trainable_parameters.json", {"visual": {"parameter_count": visual_parameter_count, "trainable": 0}, "adapter": summary, "total_trainable": summary["trainable_parameter_count"]})
    training_cfg = dict(config.get("training", {}))
    batch_size = int(training_cfg.get("batch_size", 4))
    accumulation = int(training_cfg.get("gradient_accumulation_steps", 4))
    if max_train_groups is not None:
        train_rows = train_rows[: int(max_train_groups) * batch_size]
    loader = DataLoader(
        ObjectAdapterDataset(train_rows),
        batch_size=batch_size,
        shuffle=True,
        generator=torch_module.Generator().manual_seed(seed),
        num_workers=int(training_cfg.get("num_workers", 0)),
        pin_memory=bool(training_cfg.get("pin_memory", False)),
        persistent_workers=bool(training_cfg.get("persistent_workers", False)) and int(training_cfg.get("num_workers", 0)) > 0,
        collate_fn=object_adapter_collate,
    )
    epochs = int(training_cfg.get("epochs", 2))
    configured_steps = max_steps if max_steps is not None else training_cfg.get("max_steps")
    total_steps = int(configured_steps) if configured_steps is not None else max(1, math.ceil(len(loader) / accumulation) * epochs)
    scheduler_name = str(training_cfg.get("scheduler", "constant_after_warmup"))
    if scheduler_name != "constant_after_warmup":
        raise ValueError(
            "RS Object Adapter v0 only supports scheduler='constant_after_warmup'"
        )
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=float(training_cfg.get("learning_rate", 2e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    warmup_steps = int(round(total_steps * float(training_cfg.get("warmup_ratio", 0.05))))
    warmup_steps = max(1, warmup_steps) if total_steps else 0
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, step / warmup_steps) if warmup_steps and step <= warmup_steps else 1.0
    )
    weights = dict(config.get("loss", {}))
    image_root = _resolve_project_path(str(data_cfg.get("image_root", os.environ.get("DATA_ROOT", "."))), root)
    log_path = output_dir / "train_metrics.jsonl"
    optimizer_steps = 0
    started = time.perf_counter()
    autocast_enabled = bool(training_cfg.get("bf16", True)) and bool(torch_module.cuda.is_available())
    autocast_dtype = torch_module.bfloat16 if autocast_enabled else torch_module.float32
    for epoch in range(1, epochs + 1):
        adapter.train()
        optimizer.zero_grad(set_to_none=True)
        for group_index, rows in enumerate(loader, 1):
            encoded = visual_processor_batch(processor, rows, image_root=image_root)
            features, positions = extractor.extract(encoded)
            layer_batch, position_batch, padding_mask = pad_visual_features(features, positions)
            layer_batch = _cast_features_for_adapter(layer_batch, adapter)
            class_ids = torch.as_tensor([int(row["class_id"]) for row in rows], dtype=torch.long, device=visual_device)
            with torch.autocast(device_type=visual_device.type, dtype=autocast_dtype, enabled=autocast_enabled):
                outputs = adapter(layer_batch, position_batch.to(visual_device), class_ids, memory_key_padding_mask=padding_mask.to(visual_device))
                losses = compute_object_adapter_loss(outputs, rows, **{key: float(weights[key]) for key in ("objectness_weight", "bbox_l1_weight", "giou_weight", "count_weight", "binarization_weight", "negative_query_weight", "smooth_l1_beta") if key in weights})
                loss = losses["loss_total"] / accumulation
            if not bool(torch_module.isfinite(loss).item()):
                raise FloatingPointError(f"Non-finite Object Adapter loss at epoch={epoch}, group={group_index}")
            loss.backward()
            should_step = group_index % accumulation == 0 or group_index == len(loader)
            if should_step:
                if any(
                    parameter.grad is not None
                    and not bool(torch_module.isfinite(parameter.grad).all().item())
                    for parameter in adapter.parameters()
                ):
                    raise FloatingPointError(
                        f"Non-finite Object Adapter gradient at epoch={epoch}, group={group_index}"
                    )
                grad_norm = float(torch.nn.utils.clip_grad_norm_(adapter.parameters(), float(training_cfg.get("max_grad_norm", 1.0))).item())
                if not math.isfinite(grad_norm):
                    raise FloatingPointError(
                        f"Non-finite Object Adapter grad norm at epoch={epoch}, group={group_index}"
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                optimizer_steps += 1
                metrics = {"epoch": epoch, "step": optimizer_steps, "lr": optimizer.param_groups[0]["lr"], "grad_norm": grad_norm, "elapsed_seconds": time.perf_counter() - started}
                for key, value in losses.items():
                    metrics[key] = float(value.detach().cpu().item()) if isinstance(value, Tensor) else value
                metrics["count_abs_error"] = metrics["mean_count_abs_error"]
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(metrics, ensure_ascii=False) + "\n")
                if optimizer_steps % 100 == 0 or optimizer_steps == 1:
                    print(json.dumps(metrics, ensure_ascii=False))
                if optimizer_steps >= total_steps:
                    break
        validation = evaluate_object_adapter_rows(adapter, extractor, processor, val_rows, image_root=image_root, device=visual_device, batch_size=batch_size)
        _write_json(output_dir / f"val_epoch_{epoch}.json", validation)
        save_object_adapter_checkpoint(
            adapter,
            class_vocab,
            output_dir / f"checkpoint_epoch_{epoch}",
            epoch=epoch,
            source_checkpoint=checkpoint,
            source_manifest=checkpoint / "strategy_manifest.json",
            data_manifest=manifest_path,
            selected_blocks=tuple(model_cfg.get("selected_blocks", SELECTED_BLOCKS)),
            seed=seed,
        )
        if optimizer_steps >= total_steps:
            break
    peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else None
    _write_json(output_dir / "e2_metrics.json", {"status": "not_run", "note": "Run evaluate_object_adapter_v0.py against the frozen E2 tier.", "peak_vram_mb": peak_vram_mb})
    extractor.close()
    return {"status": "completed", "optimizer_steps": optimizer_steps, "peak_vram_mb": peak_vram_mb, "output_dir": str(output_dir)}


def save_object_adapter_checkpoint(
    adapter: RSObjectAdapter,
    class_vocab: Mapping[str, Any],
    output_dir: str | Path,
    *,
    epoch: int,
    source_checkpoint: str | Path,
    source_manifest: str | Path,
    data_manifest: str | Path,
    selected_blocks: Sequence[int],
    seed: int,
) -> Path:
    """只保存 Adapter safetensors、manifest 和 class vocabulary。"""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - model dependency environment
        raise ImportError("safetensors is required for Object Adapter checkpoints") from exc
    weights_path = destination / "adapter_model.safetensors"
    save_file({name: value.detach().cpu().contiguous() for name, value in adapter.state_dict().items()}, str(weights_path))
    vocab_path = destination / "class_vocab.json"
    vocab_path.write_text(json.dumps(dict(class_vocab), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = adapter_parameter_summary(adapter)
    manifest = {
        "schema_version": "1.0",
        "experiment": "rs_object_adapter_v0",
        "architecture": {
            "d_model": adapter.d_model,
            "num_queries": adapter.num_queries,
            "num_classes": adapter.num_classes,
            "vit_hidden_size": adapter.vit_hidden_size,
            "nhead": adapter.decoder.layers[0].self_attn.num_heads,
            "decoder_layers": len(adapter.decoder.layers),
            "dim_feedforward": adapter.decoder.layers[0].linear1.out_features,
            "dropout": float(adapter.decoder.layers[0].dropout.p),
        },
        "source_r1_checkpoint": str(source_checkpoint),
        "source_r1_manifest": str(source_manifest),
        "source_r1_manifest_sha256": file_sha256(source_manifest) if Path(source_manifest).is_file() else None,
        "training_data_manifest": str(data_manifest),
        "training_data_manifest_sha256": file_sha256(data_manifest),
        "selected_vit_blocks": list(selected_blocks),
        "trainable_parameter_count": summary["trainable_parameter_count"],
        "epoch": int(epoch),
        "seed": int(seed),
        "weights": weights_path.name,
        "weights_sha256": file_sha256(weights_path),
    }
    (destination / "adapter_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destination


def evaluate_object_adapter_rows(
    adapter: RSObjectAdapter,
    extractor: FrozenVisualFeatureExtractor,
    processor: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    image_root: str | Path,
    device: torch.device,
    batch_size: int = 4,
) -> dict[str, Any]:
    """在 internal val/E2 上输出 count 与 proposal coverage 指标。"""

    adapter.eval()
    predicted_counts: list[float] = []
    true_counts: list[int] = []
    predicted_boxes: list[Tensor] = []
    detection_targets: list[Mapping[str, Any]] = []
    for start in range(0, len(rows), batch_size):
        group = list(rows[start : start + batch_size])
        encoded = visual_processor_batch(processor, group, image_root=image_root)
        features, positions = extractor.extract(encoded)
        layer_batch, position_batch, padding_mask = pad_visual_features(features, positions)
        layer_batch = _cast_features_for_adapter(layer_batch, adapter)
        class_ids = torch.as_tensor([int(row["class_id"]) for row in group], dtype=torch.long, device=device)
        with torch.no_grad():
            outputs = adapter(layer_batch, position_batch.to(device), class_ids, memory_key_padding_mask=padding_mask.to(device))
        for index, row in enumerate(group):
            logits = outputs["object_logits"][index]
            predicted_count = float(torch.sigmoid(logits).sum().item())
            if row.get("count") is not None:
                predicted_counts.append(predicted_count)
                true_counts.append(int(row["count"]))
            predicted_boxes.append(torch.cat((logits[:, None], outputs["boxes_cxcywh"][index]), dim=-1).detach().cpu())
            detection_targets.append(row)
    count_metrics = _metrics_from_count_pairs(predicted_counts, true_counts)
    box_metrics = _box_metrics(predicted_boxes, detection_targets)
    return {"counting": count_metrics, "detection_proposals": box_metrics, "sample_count": len(rows)}
