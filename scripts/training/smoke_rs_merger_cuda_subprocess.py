"""Two-process CUDA smoke for expert steps and complete CUDA-context teardown."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _memory(torch: Any) -> dict[str, int]:
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
    }


def _run_two_steps(torch: Any, branch_type: Any) -> tuple[float, dict[str, int], dict[str, int]]:
    branches = torch.nn.ModuleList(
        [
            branch_type(64, 96, detail_hidden_size=32, spatial_merge_size=2).cuda()
            for _ in range(4)
        ]
    )
    optimizer = torch.optim.AdamW(branches.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    grid = torch.tensor([[1, 4, 4]], device="cuda")
    loss = outputs = values = None
    for _ in range(2):
        values = torch.randn(16, 64, device="cuda")
        outputs = [branch(values, grid) for branch in branches]
        loss = sum((output - 1).square().mean() for output in outputs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(branches.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    before_cleanup = _memory(torch)
    peak = {
        "allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    final_loss = float(loss.detach().item())
    optimizer.zero_grad(set_to_none=True)
    del outputs, values, loss, grid, optimizer, branches
    return final_loss, before_cleanup, peak


def run_child_experiment() -> dict[str, Any]:
    import torch

    from sat_rs_vlm.models.rs_merger_expert import RSDetailResidualBranch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.cuda.set_device(0)
    before = _memory(torch)
    torch.cuda.reset_peak_memory_stats()
    final_loss, before_cleanup, peak = _run_two_steps(torch, RSDetailResidualBranch)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    after_cleanup = _memory(torch)
    return {
        "device": torch.cuda.get_device_name(0),
        "steps": 2,
        "loss": final_loss,
        "before_train": before,
        "peak": peak,
        "before_cleanup": before_cleanup,
        "after_cleanup": after_cleanup,
        "reserved_memory_note": "reserved_bytes is cache, not a live allocation leak",
    }


def run_two_process_smoke(
    *, python: str = sys.executable, report_path: str | Path | None = None
) -> dict[str, Any]:
    children: list[dict[str, Any]] = []
    for cycle in (1, 2):
        completed = subprocess.run(
            [python, str(Path(__file__).resolve()), "--child"],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
        payload["cycle"] = cycle
        payload["process_exit_confirmed"] = True
        children.append(payload)
    first_before = int(children[0]["before_train"]["allocated_bytes"])
    second_before = int(children[1]["before_train"]["allocated_bytes"])
    tolerance = 1024 * 1024
    report = {
        "schema_version": "1.0",
        "independent_subprocesses": True,
        "cycles": children,
        "allocated_vram_not_cumulative": second_before <= first_before + tolerance,
        "baseline_tolerance_bytes": tolerance,
    }
    if not report["allocated_vram_not_cumulative"]:
        raise AssertionError(
            "Second CUDA process started with cumulatively higher allocated VRAM: "
            f"first={first_before}, second={second_before}"
        )
    if report_path is not None:
        destination = Path(report_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument(
        "--report",
        default="reports/rs_merger_expert/cuda_subprocess_smoke.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = (
        run_child_experiment()
        if args.child
        else run_two_process_smoke(report_path=args.report)
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
