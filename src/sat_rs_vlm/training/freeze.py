"""模型冻结策略。"""

from typing import Any


def _freeze_by_name(model: Any, keywords: tuple[str, ...], label: str) -> int:
    """按参数名关键词冻结模块。"""

    frozen = 0
    for name, parameter in model.named_parameters():
        lowered = name.lower()
        if any(keyword in lowered for keyword in keywords):
            parameter.requires_grad = False
            frozen += int(parameter.numel())
    if frozen == 0:
        print(f"WARNING: no parameters matched for {label}; please inspect model module names.")
    else:
        print(f"Frozen {frozen} parameters for {label}.")
    return frozen


def freeze_vision_encoder(model: Any) -> int:
    """冻结视觉编码器。

    通过 vision、visual、vision_tower、image_tower 等关键词匹配参数名。
    """

    return _freeze_by_name(
        model,
        ("vision", "visual", "vision_tower", "image_tower"),
        "vision encoder",
    )


def freeze_projector(model: Any) -> int:
    """冻结视觉-语言投影层。"""

    return _freeze_by_name(
        model,
        ("projector", "mm_projector", "multi_modal_projector", "vision_proj"),
        "multimodal projector",
    )
