"""RS Object Adapter v0 的纯 PyTorch模型和几何工具。

模型只接收冻结 Qwen ViT 的四层 patch hidden states、二维 patch center 和一个
class id。四层先独立 ``LayerNorm + Linear(1024 -> 256)``，再用可学习 softmax
权重融合；随后加入二维位置投影，使用 64 个 class-conditioned query 经过两层
Transformer Decoder 输出 objectness 和 normalized ``cxcywh`` bbox。

这里没有 Qwen LLM、文本 embedding、router 或额外视觉 backbone，便于把实验结论
限定为“显式 object-centric representation 是否有效”。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn


def xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    """将 ``[..., x_min,y_min,x_max,y_max]`` 转为 ``[..., cx,cy,w,h]``。"""

    x_min, y_min, x_max, y_max = boxes.unbind(dim=-1)
    return torch.stack(
        ((x_min + x_max) / 2, (y_min + y_max) / 2, x_max - x_min, y_max - y_min),
        dim=-1,
    )


def cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    """将 ``[..., cx,cy,w,h]`` 转为 ``[..., x_min,y_min,x_max,y_max]``。"""

    cx, cy, width, height = boxes.unbind(dim=-1)
    return torch.stack((cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2), dim=-1)


def pairwise_iou_xyxy(first: Tensor, second: Tensor) -> Tensor:
    """以 FP32 计算两个 ``[...,4]`` 集合的两两 IoU，返回 ``[...,N,M]``。"""

    first = first.float()
    second = second.float()
    first = first.unsqueeze(-2)
    second = second.unsqueeze(-3)
    left_top = torch.maximum(first[..., :2], second[..., :2])
    right_bottom = torch.minimum(first[..., 2:], second[..., 2:])
    intersection = (right_bottom - left_top).clamp_min(0).prod(dim=-1)
    first_area = (first[..., 2:] - first[..., :2]).clamp_min(0).prod(dim=-1)
    second_area = (second[..., 2:] - second[..., :2]).clamp_min(0).prod(dim=-1)
    union = first_area + second_area - intersection
    return intersection / union.clamp_min(1e-7)


def generalized_iou_xyxy(first: Tensor, second: Tensor) -> Tensor:
    """以 FP32 计算两组 xyxy 框的两两 Generalized IoU。"""

    first = first.float()
    second = second.float()
    first = first.unsqueeze(-2)
    second = second.unsqueeze(-3)
    left_top = torch.maximum(first[..., :2], second[..., :2])
    right_bottom = torch.minimum(first[..., 2:], second[..., 2:])
    intersection = (right_bottom - left_top).clamp_min(0).prod(dim=-1)
    first_area = (first[..., 2:] - first[..., :2]).clamp_min(0).prod(dim=-1)
    second_area = (second[..., 2:] - second[..., :2]).clamp_min(0).prod(dim=-1)
    union = first_area + second_area - intersection
    iou = intersection / union.clamp_min(1e-7)
    enclosing_left_top = torch.minimum(first[..., :2], second[..., :2])
    enclosing_right_bottom = torch.maximum(first[..., 2:], second[..., 2:])
    enclosing_area = (enclosing_right_bottom - enclosing_left_top).clamp_min(0).prod(dim=-1)
    return iou - (enclosing_area - union) / enclosing_area.clamp_min(1e-7)


class RSObjectAdapter(nn.Module):
    """冻结视觉特征上的 class-conditioned object proposal adapter。"""

    def __init__(
        self,
        num_classes: int,
        *,
        vit_hidden_size: int = 1024,
        d_model: int = 256,
        num_queries: int = 64,
        nhead: int = 8,
        decoder_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be positive")
        if num_queries != 64:
            raise ValueError("RS Object Adapter v0 requires num_queries=64")
        self.num_classes = int(num_classes)
        self.vit_hidden_size = int(vit_hidden_size)
        self.d_model = int(d_model)
        self.num_queries = int(num_queries)
        self.layer_norms = nn.ModuleList(nn.LayerNorm(vit_hidden_size) for _ in range(4))
        self.layer_projections = nn.ModuleList(
            nn.Linear(vit_hidden_size, d_model) for _ in range(4)
        )
        self.layer_fusion_logits = nn.Parameter(torch.zeros(4))
        self.position_projection = nn.Linear(2, d_model)
        self.class_embedding = nn.Embedding(num_classes, d_model)
        self.query_embeddings = nn.Parameter(torch.randn(num_queries, d_model) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.objectness_head = nn.Linear(d_model, 1)
        self.bbox_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 4),
            nn.Sigmoid(),
        )

    def forward(
        self,
        features: Mapping[int, Tensor] | Sequence[Tensor],
        positions: Tensor,
        class_ids: Tensor,
        *,
        memory_key_padding_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """前向计算。

        参数：
            features：四层 ``[B,T,1024]`` hidden states，键顺序必须对应选中层。
            positions：``[B,T,2]`` 的 normalized patch centers。
            class_ids：``[B]`` 的 class vocabulary id。
            memory_key_padding_mask：``[B,T]``，True 表示 padding patch。

        返回：
            ``object_logits [B,64]``、``boxes_cxcywh [B,64,4]`` 和 decoder tokens。
        """

        if isinstance(features, Mapping):
            values = list(features.values())
        else:
            values = list(features)
        if len(values) != 4:
            raise ValueError(f"RS Object Adapter expects four visual layers, got {len(values)}")
        if positions.ndim != 3 or positions.shape[-1] != 2:
            raise ValueError(f"positions must have shape [B,T,2], got {tuple(positions.shape)}")
        projected: list[Tensor] = []
        for index, feature in enumerate(values):
            if feature.ndim != 3 or feature.shape[-1] != self.vit_hidden_size:
                raise ValueError(
                    "visual feature must have shape [B,T,vit_hidden_size], "
                    f"got layer={index} shape={tuple(feature.shape)}"
                )
            if feature.shape[:2] != positions.shape[:2]:
                raise ValueError("All visual layers and positions must share [B,T]")
            projected.append(self.layer_projections[index](self.layer_norms[index](feature)))
        alpha = torch.softmax(self.layer_fusion_logits, dim=0)
        memory = sum(weight * value for weight, value in zip(alpha, projected, strict=False))
        memory = memory + self.position_projection(
            positions.to(device=memory.device, dtype=memory.dtype)
        )
        batch_size = memory.shape[0]
        query = self.query_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        query = query + self.class_embedding(class_ids.to(device=query.device))[:, None, :]
        if memory_key_padding_mask is not None:
            memory_key_padding_mask = memory_key_padding_mask.to(device=memory.device)
        decoded = self.decoder(
            query,
            memory,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return {
            "object_tokens": decoded,
            "object_logits": self.objectness_head(decoded).squeeze(-1),
            "boxes_cxcywh": self.bbox_head(decoded),
        }


def adapter_parameter_summary(adapter: nn.Module) -> dict[str, Any]:
    """返回可审计的 Adapter 参数名、数量和比例。"""

    named = list(adapter.named_parameters())
    total = sum(int(parameter.numel()) for _, parameter in named)
    trainable = sum(int(parameter.numel()) for _, parameter in named if parameter.requires_grad)
    return {
        "parameter_count": total,
        "trainable_parameter_count": trainable,
        "trainable_ratio": trainable / total if total else 0.0,
        "names": [name for name, parameter in named if parameter.requires_grad],
    }
