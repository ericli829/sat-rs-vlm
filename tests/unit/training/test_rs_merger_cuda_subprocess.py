from __future__ import annotations

import sys

import pytest

torch = pytest.importorskip("torch")

from scripts.training.smoke_rs_merger_cuda_subprocess import (  # noqa: E402
    run_two_process_smoke,
)


@pytest.mark.gpu
def test_two_independent_expert_processes_do_not_accumulate_allocated_vram(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    report = run_two_process_smoke(
        python=sys.executable, report_path=tmp_path / "cuda_smoke.json"
    )
    assert report["independent_subprocesses"] is True
    assert report["allocated_vram_not_cumulative"] is True
    assert len(report["cycles"]) == 2
    assert all(cycle["steps"] == 2 for cycle in report["cycles"])
    assert all(
        cycle["after_cleanup"]["allocated_bytes"]
        <= cycle["before_cleanup"]["allocated_bytes"]
        for cycle in report["cycles"]
    )
