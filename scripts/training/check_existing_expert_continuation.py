"""Load real C2/C3 sidecars into an architecture-shaped lightweight harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn

from sat_rs_vlm.models.rs_merger_expert import RSMergerExpertController
from sat_rs_vlm.training.rs_merger_expert import (
    inspect_expert_checkpoint,
    load_expert_checkpoint,
    validate_expert_checkpoint_compatibility,
)


class _Projection(nn.Module):
    def __init__(self, input_width: int, output_width: int) -> None:
        super().__init__()
        self.in_features = input_width
        self.out_features = output_width
        self.marker = nn.Parameter(torch.zeros(1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values


class _Attention(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.q_proj = _Projection(width, 4096)
        self.k_proj = _Projection(width, 1024)
        self.v_proj = _Projection(width, 1024)
        self.o_proj = _Projection(4096, width)


class _Layer(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.self_attn = _Attention(width)


class _Language(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(_Layer(width) for _ in range(4))


class _Merger(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.marker = nn.Parameter(torch.zeros(1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values


class _VisionConfig:
    hidden_size = 1024
    out_hidden_size = 2560
    spatial_merge_size = 2
    deepstack_visual_indexes = [5, 11, 17]


class _Vision(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _VisionConfig()
        self.spatial_merge_size = 2
        self.patch_embed = nn.Identity()
        self.deepstack_visual_indexes = [5, 11, 17]
        self.blocks = nn.ModuleList(nn.Identity() for _ in range(24))
        self.deepstack_merger_list = nn.ModuleList(_Merger() for _ in range(3))
        self.merger = _Merger()


class _Core(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = _Vision()
        self.language_model = _Language(2560)


class _Holder(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model


class _Harness(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_model = _Holder(_Holder(_Core()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument(
        "--output", default="reports/rs_merger_expert/checkpoint_continuation_parity.json"
    )
    return parser.parse_args()


def check(path: str) -> dict[str, object]:
    resume = inspect_expert_checkpoint(path)
    manifest = resume.manifest
    lora = dict(manifest["interface_lora"])
    model = _Harness()
    controller = RSMergerExpertController(
        model,
        variant="rs_detail",
        detail_hidden_size=int(manifest.get("detail_hidden_size", 512)),
        local_depth=int(manifest.get("local_depth", 1)),
        interface_lora_enabled=bool(lora["enabled"]),
        lora_rank=int(lora["r"]),
        lora_alpha=float(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
    )
    validation = validate_expert_checkpoint_compatibility(
        resume,
        expected_variant=str(manifest["variant"]),
        expected_expert_variant=str(manifest.get("expert_variant", "rs_detail")),
        expected_detail_hidden_size=int(manifest.get("detail_hidden_size", 512)),
        expected_local_depth=int(manifest.get("local_depth", 1)),
        expected_interface_lora=lora,
        architecture={
            "deepstack_visual_indexes": [5, 11, 17],
            "vision_block_count": 24,
            "spatial_merge_size": 2,
            "vision_hidden_size": 1024,
            "llm_hidden_size": 2560,
        },
        architecture_audit_sha256=str(manifest["architecture_audit_sha256"]),
        source_r1_manifest_sha256=str(manifest["source_r1_manifest_sha256"]),
        source_visual_sidecar_sha256=str(manifest["source_visual_sidecar_sha256"]),
        source_r1_checkpoint=str(manifest["source_r1_checkpoint"]),
        r1_integration=str(manifest["r1_integration"]),
    )
    load_expert_checkpoint(controller, resume)
    source = load_file(str(resume.weights), device="cpu")
    restored = controller.expert_state_dict()
    exact = set(source) == set(restored) and all(
        torch.equal(source[name], restored[name].cpu()) for name in source
    )
    if not exact:
        raise AssertionError("Loaded expert state is not tensor-exact")
    audit = controller.freeze_base_and_enable_expert()
    controller.close()
    return {
        "checkpoint": str(Path(path).resolve()),
        "validation": validation,
        "tensor_key_count": len(source),
        "tensor_exact_parity": exact,
        "trainable_parameter_count": audit["total_trainable_parameter_count"],
    }


def main() -> int:
    args = parse_args()
    report = {"schema_version": "1.0", "checkpoints": [check(path) for path in args.checkpoint]}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
