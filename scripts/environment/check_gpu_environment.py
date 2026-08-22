"""Fail-closed CUDA/attention environment audit for Counting Expert runs."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attention-backend", choices=("auto", "sdpa", "flash_attention_2"), default="auto"
    )
    parser.add_argument("--strict-5090", action="store_true")
    parser.add_argument("--output", default="reports/rs_merger_expert/gpu_environment.json")
    return parser.parse_args()


def _version(module_name: str) -> str | None:
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None
    return str(getattr(module, "__version__", "unknown"))


def _major_minor(value: str | None) -> tuple[int, int]:
    try:
        pieces = str(value).split("+", 1)[0].split(".")
        return int(pieces[0]), int(pieces[1])
    except (IndexError, TypeError, ValueError):
        return (0, 0)


def _driver_version() -> str | None:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return completed.stdout.strip().splitlines()[0]
    except (FileNotFoundError, IndexError, subprocess.SubprocessError):
        return None


def _flash_smoke(torch: Any) -> dict[str, Any]:
    report: dict[str, Any] = {"importable": False, "finite_forward_backward": False}
    try:
        flash_attn = importlib.import_module("flash_attn")
        function = flash_attn.flash_attn_func
        report["importable"] = True
        qkv = [
            torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            for _ in range(3)
        ]
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = function(*qkv, causal=False)
        output.float().square().mean().backward()
        torch.cuda.synchronize()
        report["elapsed_seconds"] = time.perf_counter() - start
        report["finite_forward_backward"] = bool(
            torch.isfinite(output).all()
            and all(item.grad is not None and torch.isfinite(item.grad).all() for item in qkv)
        )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def audit_environment(attention_backend: str, *, strict_5090: bool) -> dict[str, Any]:
    torch = importlib.import_module("torch")
    cuda_available = bool(torch.cuda.is_available())
    device_name = torch.cuda.get_device_name(0) if cuda_available else None
    capability = list(torch.cuda.get_device_capability(0)) if cuda_available else None
    arch_list = list(torch.cuda.get_arch_list()) if cuda_available else []
    bf16 = bool(torch.cuda.is_bf16_supported()) if cuda_available else False
    sdpa = bool(hasattr(torch.nn.functional, "scaled_dot_product_attention"))
    flash = (
        _flash_smoke(torch) if cuda_available and attention_backend != "sdpa" else {"tested": False}
    )
    blockers: list[str] = []
    is_5090 = bool(device_name and "5090" in device_name)
    if strict_5090 and not is_5090:
        blockers.append(f"strict 5090 gate expected RTX 5090, got {device_name!r}")
    if is_5090 and capability != [12, 0]:
        blockers.append(f"RTX 5090 must report compute capability 12.0, got {capability}")
    if is_5090 and "sm_120" not in arch_list:
        blockers.append(f"PyTorch binary lacks sm_120: {arch_list}")
    if is_5090 and _major_minor(str(torch.__version__)) < (2, 7):
        blockers.append(f"RTX 5090 requires torch >=2.7, got {torch.__version__}")
    if is_5090 and _major_minor(str(torch.version.cuda)) < (12, 8):
        blockers.append(f"RTX 5090 requires CUDA >=12.8, got {torch.version.cuda}")
    if not bf16:
        blockers.append("BF16 is unavailable")
    if not sdpa:
        blockers.append("PyTorch SDPA is unavailable")
    selected = "sdpa"
    if attention_backend == "flash_attention_2":
        if not flash.get("finite_forward_backward"):
            blockers.append("flash_attention_2 was forced but its CUDA smoke failed")
        else:
            selected = "flash_attention_2"
    elif attention_backend == "auto" and flash.get("finite_forward_backward"):
        selected = "flash_attention_2"
    report = {
        "schema_version": "1.0",
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "driver": _driver_version() if cuda_available else None,
        "cuda_available": cuda_available,
        "device_name": device_name,
        "compute_capability": capability,
        "compiled_arch_list": arch_list,
        "bf16_supported": bf16,
        "sdpa_available": sdpa,
        "transformers": _version("transformers"),
        "peft": _version("peft"),
        "flash_attn": _version("flash_attn"),
        "flash_attention_smoke": flash,
        "requested_attention_backend": attention_backend,
        "selected_attention_backend": selected,
        "target_environment": "torch 2.12 + CUDA 13, or torch >=2.7 + CUDA >=12.8 with sm_120",
        "strict_5090": strict_5090,
        "status": "pass" if not blockers else "blocked",
        "blockers": blockers,
    }
    return report


def main() -> int:
    args = parse_args()
    report = audit_environment(args.attention_backend, strict_5090=args.strict_5090)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
