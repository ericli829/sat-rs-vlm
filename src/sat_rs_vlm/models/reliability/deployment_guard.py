"""Preflight guard that blocks inference on an untrusted LoRA adapter."""

from __future__ import annotations

from pathlib import Path

from sat_rs_vlm.models.reliability.adapter_redundancy import AdapterScrubResult, scrub_adapter_replicas


def guard_adapter_before_inference(
    working_adapter: str | Path,
    *,
    warm_adapter: str | Path,
    golden_adapter: str | Path,
    manifest: str | Path,
) -> AdapterScrubResult:
    """Scrub the deployed adapter and fail closed when no trusted recovery source exists."""

    result = scrub_adapter_replicas(
        working_adapter,
        warm_root=warm_adapter,
        golden_root=golden_adapter,
        manifest=manifest,
    )
    if not result.success:
        raise RuntimeError(
            "Adapter deployment guard blocked inference: no trusted working adapter; "
            f"errors={result.errors}, unresolved={result.unresolved_files}"
        )
    return result


class PeriodicAdapterGuard:
    """Run a deployment-file scrub every configured inference batches.

    This guard protects the on-disk adapter chain for subsequent model loads.  It
    does not repair parameters already resident in model memory.
    """

    def __init__(
        self,
        working_adapter: str | Path,
        *,
        warm_adapter: str | Path,
        golden_adapter: str | Path,
        manifest: str | Path,
        interval_batches: int,
    ) -> None:
        if interval_batches < 1:
            raise ValueError("interval_batches must be positive")
        self.working_adapter = Path(working_adapter)
        self.warm_adapter = Path(warm_adapter)
        self.golden_adapter = Path(golden_adapter)
        self.manifest = Path(manifest)
        self.interval_batches = interval_batches
        self.completed_batches = 0
        self.events: list[dict[str, object]] = []

    def after_batch(self) -> AdapterScrubResult | None:
        """Scrub at the configured interval; fail closed if trusted recovery fails."""

        self.completed_batches += 1
        if self.completed_batches % self.interval_batches:
            return None
        result = guard_adapter_before_inference(
            self.working_adapter,
            warm_adapter=self.warm_adapter,
            golden_adapter=self.golden_adapter,
            manifest=self.manifest,
        )
        self.events.append({
            "batch": self.completed_batches,
            "success": result.success,
            "action": "restored" if result.restored_from_warm or result.restored_from_golden else "verified",
            "restored_from_warm": result.restored_from_warm,
            "restored_from_golden": result.restored_from_golden,
        })
        return result

    def report(self) -> dict[str, object]:
        return {
            "schema_version": "1.0",
            "scope": "on_disk_adapter_chain_only",
            "does_not_protect": ["loaded_model_memory", "activations", "kv_cache"],
            "interval_batches": self.interval_batches,
            "completed_batches": self.completed_batches,
            "events": self.events,
        }
