"""从标准可靠性 metrics 生成静态图表。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sat_rs_vlm.evaluation.reliability.plotting import plot_reliability_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sensitivity-root",
        type=Path,
        help="v15 sensitivity run directory containing condition_plan.json and conditions/",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sensitivity_root:
        generated = plot_sensitivity_results(args.sensitivity_root, args.output)
    elif args.input:
        generated = plot_reliability_results(args.input, args.output)
    else:
        raise SystemExit("provide --input or --sensitivity-root")
    print(json.dumps({"generated": [str(path) for path in generated]}, indent=2))
    return 0


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, dict):
        nested = value.get("value")
        if isinstance(nested, (int, float)):
            return float(nested)
    return None


def _condition_metric(payload: dict[str, Any], names: tuple[str, ...]) -> float | None:
    for name in names:
        value = _number(payload.get(name))
        if value is not None:
            return value
    for key in ("overall", "metrics", "summary"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            result = _condition_metric(nested, names)
            if result is not None:
                return result
    return None


def _load_condition(directory: Path, condition: dict[str, Any]) -> dict[str, Any]:
    comparison = directory / "comparison" / "comparison.json"
    injection = directory / "fault_injection_summary.json"
    if not comparison.is_file():
        return {**condition, "status": "missing"}
    payload = json.loads(comparison.read_text(encoding="utf-8"))
    record = json.loads(injection.read_text(encoding="utf-8")) if injection.is_file() else {}
    return {
        **condition,
        "status": "complete",
        "changed_rate": _condition_metric(payload, ("changed_rate", "changed_rate_mean", "prediction_changed_rate")),
        "invalid_rate": _condition_metric(payload, ("invalid_rate", "invalid_rate_mean")),
        "exact_match_drop": _condition_metric(payload, ("exact_match_drop", "exact_match_drop_mean")),
        "records": record.get("records", []),
    }


def plot_sensitivity_results(input_dir: str | Path, output_dir: str | Path) -> list[Path]:
    """Create plots from a completed or partially completed v15 scan.

    This function only reads condition artifacts; it never launches inference.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise ImportError("Sensitivity plotting requires matplotlib and numpy") from exc

    root = Path(input_dir).expanduser().resolve()
    plan = json.loads((root / "condition_plan.json").read_text(encoding="utf-8"))
    rows = []
    for condition in plan.get("conditions", []):
        condition_dir = root / "conditions" / condition["id"]
        rows.append(_load_condition(condition_dir, condition))
    complete = [row for row in rows if row["status"] == "complete"]
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    (destination / "sensitivity_rows.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not complete:
        return generated

    def save(fig: Any, name: str) -> None:
        path = destination / name
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        generated.append(path)

    # Coverage tells the operator whether a partial run is safe to interpret.
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["target"]] = counts.get(row["target"], 0) + int(row["status"] == "complete")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(list(counts), list(counts.values()), color="#2563eb")
    ax.set_title("Sensitivity scan coverage")
    ax.set_ylabel("Completed conditions")
    ax.tick_params(axis="x", rotation=25)
    save(fig, "scan_coverage.png")

    # Task-level quality deltas when comparison reports expose by-task rows.
    task_values: dict[str, list[float]] = {}
    for row in complete:
        comparison_path = root / "conditions" / row["id"] / "comparison" / "comparison.json"
        if not comparison_path.is_file():
            continue
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        by_task = comparison.get("by_task", comparison.get("task_metrics", {}))
        if isinstance(by_task, dict):
            for task, metrics in by_task.items():
                if isinstance(metrics, dict):
                    delta = _condition_metric(metrics, ("exact_match_drop", "quality_drop", "metric_drop", "changed_rate"))
                    if delta is not None:
                        task_values.setdefault(str(task), []).append(delta)
    if task_values:
        names = sorted(task_values)
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.bar(names, [float(np.mean(task_values[name])) for name in names], color="#7c3aed")
        ax.set_ylabel("Quality change/drop")
        ax.set_title("Task-level sensitivity")
        ax.tick_params(axis="x", rotation=25)
        save(fig, "task_quality_sensitivity.png")

    values = [row for row in complete if row.get("changed_rate") is not None]
    if not values:
        return generated
    # Aggregate repeated conditions before plotting heatmaps.
    groups: dict[tuple[str, int | None, str], list[float]] = {}
    for row in values:
        layer = row["layers"][0] if row.get("layers") else None
        key = (row["target"], layer, row["bit_plane"])
        groups.setdefault(key, []).append(float(row["changed_rate"]))
    targets = sorted({key[0] for key in groups})
    planes = ["sign", "exponent", "mantissa", "all"]
    layers = sorted({key[1] for key in groups if key[1] is not None})
    if layers:
        matrix = np.full((len(targets) * len(planes), len(layers)), np.nan)
        labels = []
        for target_index, target in enumerate(targets):
            for plane_index, plane in enumerate(planes):
                row_index = target_index * len(planes) + plane_index
                labels.append(f"{target}:{plane}")
                for layer_index, layer in enumerate(layers):
                    samples = groups.get((target, layer, plane), [])
                    if samples:
                        matrix[row_index, layer_index] = float(np.mean(samples))
        fig, ax = plt.subplots(figsize=(max(12, len(layers) * 0.45), max(5, len(labels) * 0.28)))
        image = ax.imshow(matrix, aspect="auto", cmap="magma", vmin=0, vmax=1)
        ax.set_title("Changed rate by target, layer and bit-plane")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Target : bit-plane")
        ax.set_xticks(range(len(layers)), layers)
        ax.set_yticks(range(len(labels)), labels, fontsize=7)
        fig.colorbar(image, ax=ax, label="Changed rate")
        save(fig, "layer_bitplane_changed_heatmap.png")

    # Bit-plane aggregate with uncertainty bars when repeats exist.
    plane_values: dict[str, list[float]] = {}
    for row in values:
        plane_values.setdefault(row["bit_plane"], []).append(float(row["changed_rate"]))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = [name for name in planes if name in plane_values]
    means = [float(np.mean(plane_values[name])) for name in names]
    errors = [float(np.std(plane_values[name])) if len(plane_values[name]) > 1 else 0.0 for name in names]
    ax.bar(names, means, yerr=errors, capsize=4, color="#dc2626")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Changed rate")
    ax.set_title("Output sensitivity by bit-plane")
    save(fig, "bitplane_changed_rate.png")

    # Parameter projection and exact bit-index coverage from injection records.
    projection_values: dict[str, list[float]] = {}
    bit_index_values: dict[int, list[float]] = {}
    for row in values:
        for record in row.get("records", []):
            name = str(record.get("target_name", "")).lower()
            projection = next(
                (token for token in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj") if token in name),
                "other",
            )
            projection_values.setdefault(projection, []).append(float(row["changed_rate"]))
            bit_index = record.get("bit_index")
            if isinstance(bit_index, int):
                bit_index_values.setdefault(bit_index, []).append(float(row["changed_rate"]))
    if projection_values:
        names = sorted(projection_values)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.bar(names, [float(np.mean(projection_values[name])) for name in names], color="#0891b2")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Changed rate")
        ax.set_title("Sensitivity by parameter projection")
        ax.tick_params(axis="x", rotation=25)
        save(fig, "projection_changed_rate.png")
    if bit_index_values:
        indices = sorted(bit_index_values)
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(indices, [float(np.mean(bit_index_values[index])) for index in indices], "o-")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Exact bit index")
        ax.set_ylabel("Changed rate")
        ax.set_title("Sensitivity by exact injected bit index")
        save(fig, "bit_index_changed_rate.png")

    # Optional protection-cost plot. The plot is intentionally schema-tolerant.
    protection = root / "protection" / "strategy_results.json"
    if protection.is_file():
        payload = json.loads(protection.read_text(encoding="utf-8"))
        strategies = payload.get("strategies", payload.get("results", payload))
        if isinstance(strategies, dict):
            strategies = [{"strategy": key, **(value if isinstance(value, dict) else {})} for key, value in strategies.items()]
        if isinstance(strategies, list):
            rows_protection = [item for item in strategies if isinstance(item, dict)]
            names = [str(item.get("strategy", item.get("name", "unknown"))) for item in rows_protection]
            recovery = [_number(item.get("recovery_rate", item.get("recovered_rate", item.get("success_rate")))) or 0.0 for item in rows_protection]
            cost = [_number(item.get("latency_overhead", item.get("extra_latency_ms", item.get("cost")))) or 0.0 for item in rows_protection]
            if names:
                fig, axis = plt.subplots(figsize=(9, 4.5))
                axis.scatter(cost, recovery, s=70)
                for x, y, name in zip(cost, recovery, names):
                    axis.annotate(name, (x, y), fontsize=8)
                axis.set_xlabel("Protection cost (reported units)")
                axis.set_ylabel("Recovery/success rate")
                axis.set_ylim(0, 1.05)
                axis.set_title("Protection benefit versus reported cost")
                save(fig, "protection_benefit_cost.png")
    return generated


if __name__ == "__main__":
    raise SystemExit(main())
