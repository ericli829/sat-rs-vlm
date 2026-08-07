"""单粒子翻转可靠性实验应用服务。

服务负责运行目录、clean/fault/recovery 状态流转和报告，不实现模型加载、bit flip、
校验或指标算法。真实模式通过 `EvaluationRunner` 复用现有评测入口；Mock 模式必须由配置
显式选择，任何真实模式错误都不会触发自动降级。
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Protocol

from sat_rs_vlm.configuration.layered import write_resolved_config
from sat_rs_vlm.evaluation.checkpoint_loader import (
    read_strategy_manifest,
    validate_checkpoint_files,
)
from sat_rs_vlm.evaluation.reliability.metrics import (
    build_prediction_pairs,
    summarize_reliability,
)
from sat_rs_vlm.evaluation.reliability.reports import (
    ReliabilityRunLayout,
    create_reliability_run_layout,
    open_reliability_run_layout,
    write_metric_reports,
    write_run_metadata,
)
from sat_rs_vlm.models.reliability.bitflip import flip_random_tensor_bits
from sat_rs_vlm.models.reliability.checksum import (
    file_sha256,
    verify_checksum_manifest,
    write_checksum_manifest,
)
from sat_rs_vlm.models.reliability.fault_injector import (
    ParameterSelector,
    inject_safetensors_adapter,
    inject_state_dict_bitflips,
    load_safetensors_state,
    save_safetensors_state,
    summarize_parameter_changes,
)
from sat_rs_vlm.models.reliability.protection import (
    clamp_state_dict,
    no_protection,
    output_guard_vote,
)
from sat_rs_vlm.models.reliability.recovery import recover_file_from_backup
from sat_rs_vlm.models.reliability.schemas import AdapterInjectionReport, BitFlipRecord
from sat_rs_vlm.training.experiment import environment_snapshot, git_commit, write_json
from sat_rs_vlm.utils.jsonl import read_jsonl, write_jsonl

ExecutionMode = Literal["smoke_mock", "real_inference"]
ExperimentMode = Literal["baseline", "inject", "compare", "protect", "recover", "full"]
SmokeCase = Literal[
    "tensor",
    "state-dict",
    "adapter-file",
    "output-guard",
    "recovery",
    "weight-clamp",
    "all",
]


class EvaluationRunner(Protocol):
    """真实 checkpoint 评测适配接口。"""

    def run(
        self,
        checkpoint: Path,
        config_path: Path,
        output_dir: Path,
        log_path: Path,
    ) -> Path:
        """执行评测并返回 predictions.jsonl。"""


class SubprocessEvaluationRunner:
    """调用项目现有 `evaluate_rs_vlm.py` 的真实评测适配器。"""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()

    def run(
        self,
        checkpoint: Path,
        config_path: Path,
        output_dir: Path,
        log_path: Path,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(self.project_root / "scripts/evaluate_rs_vlm.py"),
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output_dir),
        ]
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=self.project_root,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Existing evaluation runner failed with exit code {completed.returncode}; "
                f"see {log_path}"
            )
        predictions = output_dir / "predictions.jsonl"
        if not predictions.is_file():
            raise FileNotFoundError(f"Evaluation runner did not create predictions: {predictions}")
        return predictions


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _required_path(value: Any, *, name: str, base_dir: Path) -> Path:
    if value is None or not str(value).strip():
        raise ValueError(f"Required reliability path is not configured: {name}")
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def _selector_from_config(config: Mapping[str, Any]) -> ParameterSelector:
    fault = _as_dict(config.get("fault"))
    scope = str(fault.get("bit_scope", "all")).lower()
    if scope not in {"all", "a", "b"}:
        raise ValueError("fault.bit_scope must be one of: all, a, b")
    modules = fault.get("modules", [])
    layers = fault.get("layers", [])
    return ParameterSelector(
        name_regex=str(fault["parameter_pattern"]) if fault.get("parameter_pattern") else None,
        module_names=tuple(str(item) for item in modules),
        layer_indices=tuple(int(item) for item in layers),
        lora_scope=scope,  # type: ignore[arg-type]
    )


def _smoke_prediction_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    samples = [
        ("caption-1", "captioning", "港口内有船舶和仓储区。", ""),
        ("vqa-1", "vqa", "yes", "no"),
        ("count-1", "counting", "2", "many objects"),
        (
            "detect-1",
            "detection",
            '{"label":"ship","bbox":[0.1,0.2,0.3,0.4]}',
            '{"label":"ship","bbox":[0.8,0.2,0.2,0.4]}',
        ),
        ("scene-1", "scene_classification", "airport", "harbor"),
    ]
    clean: list[dict[str, Any]] = []
    fault: list[dict[str, Any]] = []
    for sample_id, task, reference, fault_prediction in samples:
        common = {"id": sample_id, "task_type": task, "reference": reference, "metadata": {}}
        clean.append({**common, "prediction": reference})
        fault.append({**common, "prediction": fault_prediction})
    return clean, fault


class ReliabilityExperimentService:
    """统一执行本地 Mock 或云端真实可靠性实验。"""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        project_root: str | Path,
        output_root: str | Path,
        command: str,
    ) -> None:
        self.config = config
        self.project_root = Path(project_root).resolve()
        self.output_root = Path(output_root).resolve()
        self.command = command

    def run(
        self,
        *,
        mode: ExperimentMode = "full",
        smoke_case: SmokeCase = "all",
        run_id: str | None = None,
        overwrite: bool = False,
        resume: bool = False,
        runner: EvaluationRunner | None = None,
    ) -> ReliabilityRunLayout:
        """执行实验并返回标准目录；失败时保留机器可读失败报告。"""

        experiment = _as_dict(self.config.get("experiment"))
        name = str(experiment.get("name", "reliability_experiment"))
        execution_mode = str(experiment.get("execution_mode", ""))
        if execution_mode not in {"smoke_mock", "real_inference"}:
            raise ValueError("experiment.execution_mode must be smoke_mock or real_inference")
        if resume:
            if run_id is None:
                raise ValueError("resume requires an explicit run_id")
            layout = open_reliability_run_layout(
                self.output_root,
                experiment_name=name,
                run_id=run_id,
            )
        else:
            layout = create_reliability_run_layout(
                self.output_root,
                experiment_name=name,
                run_id=run_id,
                overwrite=overwrite,
            )
        write_run_metadata(
            layout,
            resolved_config=self.config,
            command=self.command,
            environment=environment_snapshot(),
            git_commit=git_commit(self.project_root),
        )
        try:
            if execution_mode == "smoke_mock":
                self._run_smoke(layout, smoke_case)
            else:
                self._run_real(
                    layout,
                    mode,
                    runner or SubprocessEvaluationRunner(self.project_root),
                    resume=resume,
                )
            write_json(
                layout.root / "run_report.json",
                {
                    "schema_version": "1.0",
                    "success": True,
                    "execution_mode": execution_mode,
                    "experiment_name": name,
                    "run_id": layout.root.name,
                    "mode": mode,
                },
            )
            return layout
        except Exception as exc:
            write_json(
                layout.root / "run_report.json",
                {
                    "schema_version": "1.0",
                    "success": False,
                    "execution_mode": execution_mode,
                    "experiment_name": name,
                    "run_id": layout.root.name,
                    "mode": mode,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            raise

    def _create_smoke_adapter(self, directory: Path) -> dict[str, Any]:
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "Adapter smoke requires torch from the optional 'model' extra"
            ) from exc
        directory.mkdir(parents=True, exist_ok=False)
        state = {
            "model.layers.0.self_attn.q_proj.lora_A.default.weight": torch.arange(
                16, dtype=torch.float32
            ).reshape(4, 4),
            "model.layers.0.self_attn.q_proj.lora_B.default.weight": torch.ones(
                (4, 4), dtype=torch.float16
            ),
        }
        save_safetensors_state(state, directory / "adapter_model.safetensors", {"format": "pt"})
        write_json(directory / "adapter_config.json", {"peft_type": "LORA", "r": 4})
        write_json(
            directory / "strategy_manifest.json",
            {
                "strategy": "lora",
                "adapter_based": True,
                "quantized_base": False,
                "model_dir": "smoke_mock",
            },
        )
        (directory / "processor").mkdir()
        write_json(directory / "processor/mock.json", {"execution_mode": "smoke_mock"})
        return state

    def _run_smoke(self, layout: ReliabilityRunLayout, case: SmokeCase) -> None:
        allowed = {
            "tensor",
            "state-dict",
            "adapter-file",
            "output-guard",
            "recovery",
            "weight-clamp",
            "all",
        }
        if case not in allowed:
            raise ValueError(
                f"Unknown smoke case '{case}'; available: {', '.join(sorted(allowed))}"
            )
        selected = allowed.difference({"all"}) if case == "all" else {case}
        seed = int(_as_dict(self.config.get("experiment")).get("seed", 2026))
        records: list[BitFlipRecord] = []
        protection_results: dict[str, Any] = {}
        clean_rows, fault_rows = _smoke_prediction_rows()

        if "tensor" in selected:
            try:
                import torch
            except ImportError as exc:
                raise ImportError("Tensor smoke requires torch from the 'model' extra") from exc
            _, tensor_records = flip_random_tensor_bits(
                torch.arange(8, dtype=torch.float32),
                num_bits=3,
                seed=seed,
                target_name="smoke.tensor",
            )
            records.extend(tensor_records)

        clean_state: dict[str, Any] | None = None
        fault_state: dict[str, Any] | None = None
        if selected.intersection({"state-dict", "adapter-file", "weight-clamp"}):
            clean_adapter = layout.artifacts / "clean_adapter"
            clean_state = self._create_smoke_adapter(clean_adapter)
            fault_state, state_records = inject_state_dict_bitflips(
                clean_state,
                num_bits=2,
                seed=seed,
                selector=ParameterSelector(lora_scope="all"),
            )
            if "state-dict" in selected:
                records.extend(state_records)
                protection_results["state_dict_changes"] = summarize_parameter_changes(
                    clean_state, fault_state
                )
            if "adapter-file" in selected:
                report = inject_safetensors_adapter(
                    clean_adapter,
                    layout.fault_adapters / "smoke_adapter",
                    num_bits=2,
                    seed=seed,
                    selector=ParameterSelector(lora_scope="all"),
                )
                records.extend(report.records)
                manifest_path = layout.artifacts / "clean_adapter_checksums.json"
                write_checksum_manifest(clean_adapter, manifest_path)
                fault_verification = verify_checksum_manifest(
                    manifest_path,
                    root=layout.fault_adapters / "smoke_adapter",
                )
                protection_results["adapter_file"] = report.model_dump(mode="json")
                protection_results["checksum_detection"] = fault_verification.model_dump(
                    mode="json"
                )
            if "weight-clamp" in selected and clean_state is not None and fault_state is not None:
                protected, clamp_report = clamp_state_dict(clean_state, fault_state)
                protection_results["weight_clamp"] = {
                    "report": clamp_report.model_dump(mode="json"),
                    "changes_after": summarize_parameter_changes(clean_state, protected),
                }

        recovered_rows: list[dict[str, Any]] | None = None
        if "output-guard" in selected:
            recovered_rows = []
            votes: list[dict[str, Any]] = []
            for clean, fault in zip(clean_rows, fault_rows, strict=True):
                vote = output_guard_vote(
                    str(clean["task_type"]),
                    [str(fault["prediction"]), str(clean["prediction"]), str(clean["prediction"])],
                    fallback=str(clean["prediction"]),
                )
                recovered_rows.append({**clean, "prediction": vote.selected})
                votes.append({"id": clean["id"], **vote.model_dump(mode="json")})
            protection_results["output_guard_vote"] = votes

        if "recovery" in selected:
            backup = layout.artifacts / "clean_backup.bin"
            deployed = layout.artifacts / "deployed_adapter.bin"
            backup.write_bytes(b"clean-adapter-smoke")
            deployed.write_bytes(b"fault-adapter-smoke")
            recovery = recover_file_from_backup(
                deployed,
                backup,
                expected_sha256=file_sha256(backup),
            )
            protection_results["checksum_recovery"] = recovery.model_dump(mode="json")

        write_jsonl(
            layout.faults / "injection_records.jsonl",
            (record.model_dump(mode="json") for record in records),
        )
        write_jsonl(layout.clean / "predictions.jsonl", clean_rows)
        write_jsonl(layout.faults / "predictions.jsonl", fault_rows)
        pairs = build_prediction_pairs(clean_rows, fault_rows, recovered_rows)
        write_jsonl(layout.predictions / "clean_fault_pairs.jsonl", pairs)
        summary = summarize_reliability(
            pairs,
            execution_mode="smoke_mock",
            experiment_name=str(_as_dict(self.config.get("experiment")).get("name", "local_smoke")),
            run_id=layout.root.name,
        )
        write_metric_reports(layout, summary)
        protection_results["no_protection"] = [
            no_protection(str(clean["prediction"]), str(fault["prediction"]))
            for clean, fault in zip(clean_rows, fault_rows, strict=True)
        ]
        write_json(layout.protection / "strategy_results.json", protection_results)
        write_json(
            layout.root / "smoke_report.json",
            {
                "schema_version": "1.0",
                "execution_mode": "smoke_mock",
                "case": case,
                "seed": seed,
                "num_fault_records": len(records),
                "mock_results_are_real_model_metrics": False,
            },
        )

    def _assert_cuda(self) -> dict[str, Any]:
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "real_inference requires torch and the project's model extra"
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("real_inference requires CUDA; refusing to fall back to smoke_mock")
        return {
            "type": "cuda",
            "name": torch.cuda.get_device_name(0),
            "count": torch.cuda.device_count(),
        }

    def _write_eval_config(self, layout: ReliabilityRunLayout) -> Path:
        model = _as_dict(self.config.get("model"))
        data = _as_dict(self.config.get("data"))
        eval_manifest = _required_path(
            data.get("eval_manifest"), name="data.eval_manifest", base_dir=self.project_root
        )
        dataset_root = _required_path(
            data.get("dataset_root"), name="data.dataset_root", base_dir=self.project_root
        )
        if not eval_manifest.is_file():
            raise FileNotFoundError(f"Reliability eval manifest does not exist: {eval_manifest}")
        if not dataset_root.is_dir():
            raise NotADirectoryError(f"Reliability dataset root does not exist: {dataset_root}")
        config = {
            "model": {
                "local_files_only": bool(model.get("local_files_only", True)),
                "trust_remote_code": bool(model.get("trust_remote_code", True)),
                "device_map": model.get("device_map", "auto"),
                "torch_dtype": model.get("torch_dtype", "auto"),
                "attn_implementation": model.get("attn_implementation", "sdpa"),
            },
            "data": {
                "eval_file": str(eval_manifest),
                "image_root": str(dataset_root),
                "max_eval_samples": data.get("max_eval_samples"),
                "max_seq_length": int(data.get("max_seq_length", 1024)),
                "eval_batch_size": int(data.get("eval_batch_size", 1)),
                "group_by_task": bool(data.get("group_by_task", True)),
                "log_every_samples": int(data.get("log_every_samples", 100)),
            },
            "generation": _as_dict(self.config.get("generation")),
            "output": {
                "summary_file": str(layout.artifacts / "unused_summary.json"),
                "predictions_file": str(layout.artifacts / "unused_predictions.jsonl"),
            },
        }
        path = layout.artifacts / "evaluation_config.yaml"
        write_resolved_config(config, path)
        return path

    def _comparison_from_files(self, layout: ReliabilityRunLayout, *, protect: bool) -> None:
        inputs = _as_dict(self.config.get("inputs"))
        clean_path = _required_path(
            inputs.get("clean_predictions"),
            name="inputs.clean_predictions",
            base_dir=self.project_root,
        )
        fault_path = _required_path(
            inputs.get("fault_predictions"),
            name="inputs.fault_predictions",
            base_dir=self.project_root,
        )
        clean_rows = list(read_jsonl(clean_path))
        fault_rows = list(read_jsonl(fault_path))
        recovered: list[dict[str, Any]] | None = None
        protection_results: dict[str, Any] = {}
        if protect:
            recovered = []
            votes = []
            for clean, fault in zip(clean_rows, fault_rows, strict=True):
                vote = output_guard_vote(
                    str(clean.get("task_type", "unknown")),
                    [str(fault.get("prediction", ""))],
                    fallback=str(clean.get("prediction", "")),
                )
                recovered.append({**clean, "prediction": vote.selected})
                votes.append({"id": clean.get("id"), **vote.model_dump(mode="json")})
            protection_results["output_guard_vote"] = votes
        pairs = build_prediction_pairs(clean_rows, fault_rows, recovered)
        self._write_real_outputs(layout, pairs, protection_results)

    def _write_real_outputs(
        self,
        layout: ReliabilityRunLayout,
        pairs: list[dict[str, Any]],
        protection_results: dict[str, Any],
    ) -> None:
        write_jsonl(layout.predictions / "clean_fault_pairs.jsonl", pairs)
        summary = summarize_reliability(
            pairs,
            execution_mode="real_inference",
            experiment_name=str(_as_dict(self.config.get("experiment")).get("name", "reliability")),
            run_id=layout.root.name,
        )
        write_metric_reports(layout, summary)
        write_json(layout.protection / "strategy_results.json", protection_results)

    def _run_real(
        self,
        layout: ReliabilityRunLayout,
        mode: ExperimentMode,
        runner: EvaluationRunner,
        *,
        resume: bool,
    ) -> None:
        if mode in {"compare", "protect"}:
            self._comparison_from_files(layout, protect=mode == "protect")
            return

        model = _as_dict(self.config.get("model"))
        adapter = _required_path(
            model.get("adapter_path"), name="model.adapter_path", base_dir=self.project_root
        )
        manifest = read_strategy_manifest(adapter)
        validate_checkpoint_files(adapter, manifest)
        device = self._assert_cuda() if mode in {"baseline", "full"} else {"type": "not_used"}
        eval_config = self._write_eval_config(layout) if mode in {"baseline", "full"} else None
        seed = int(_as_dict(self.config.get("experiment")).get("seed", 2026))
        fault_config = _as_dict(self.config.get("fault"))
        counts = [int(value) for value in fault_config.get("bit_flip_counts", [1])]
        repeats = int(fault_config.get("repeats", 1))
        if not counts or any(value <= 0 for value in counts) or repeats <= 0:
            raise ValueError("fault.bit_flip_counts and fault.repeats must be positive")
        selector = _selector_from_config(self.config)
        strategies = [
            str(value) for value in _as_dict(self.config.get("protection")).get("strategies", [])
        ]

        clean_rows: list[dict[str, Any]] = []
        if mode in {"baseline", "full"}:
            assert eval_config is not None
            clean_predictions = layout.clean / "predictions.jsonl"
            if not (resume and clean_predictions.is_file()):
                clean_predictions = runner.run(
                    adapter,
                    eval_config,
                    layout.clean,
                    layout.logs / "clean_evaluation.log",
                )
            clean_rows = list(read_jsonl(clean_predictions))

        all_records: list[BitFlipRecord] = []
        fault_rows_by_condition: dict[str, list[dict[str, Any]]] = {}
        recovered_rows_by_condition: dict[str, list[dict[str, Any]]] = {}
        recovery_reports: list[dict[str, Any]] = []
        adapter_reports: list[dict[str, Any]] = []
        cleanup_temporary_adapters = mode == "full" and not bool(
            fault_config.get("retain_temporary_adapters", False)
        )
        for count in counts:
            for repeat in range(repeats):
                condition = f"bits_{count}_repeat_{repeat}"
                fault_adapter = layout.fault_adapters / condition
                report_path = fault_adapter / "fault_record.json"
                fault_weights = fault_adapter / "adapter_model.safetensors"
                if resume and report_path.is_file() and fault_weights.is_file():
                    report = AdapterInjectionReport.model_validate_json(
                        report_path.read_text(encoding="utf-8")
                    )
                else:
                    report = inject_safetensors_adapter(
                        adapter,
                        fault_adapter,
                        num_bits=count,
                        seed=seed + repeat,
                        selector=selector,
                        overwrite=resume,
                    )
                condition_dir = layout.faults / condition
                condition_dir.mkdir(parents=True, exist_ok=True)
                write_json(
                    condition_dir / "fault_record.json",
                    report.model_dump(mode="json"),
                )
                write_jsonl(
                    condition_dir / "fault_records.jsonl",
                    (record.model_dump(mode="json") for record in report.records),
                )
                adapter_reports.append({"condition": condition, **report.model_dump(mode="json")})
                all_records.extend(report.records)
                if mode == "full":
                    assert eval_config is not None
                    evaluation_dir = condition_dir / "evaluation"
                    predictions = evaluation_dir / "predictions.jsonl"
                    if not (resume and predictions.is_file()):
                        predictions = runner.run(
                            fault_adapter,
                            eval_config,
                            evaluation_dir,
                            layout.logs / f"fault_{condition}.log",
                        )
                    fault_rows_by_condition[condition] = list(read_jsonl(predictions))

                if mode in {"recover", "full"} and "checksum_recovery" in strategies:
                    recovered_adapter = layout.faults / "recovered" / condition
                    if recovered_adapter.exists():
                        shutil.rmtree(recovered_adapter)
                    shutil.copytree(fault_adapter, recovered_adapter)
                    recovery = recover_file_from_backup(
                        recovered_adapter / "adapter_model.safetensors",
                        adapter / "adapter_model.safetensors",
                        expected_sha256=file_sha256(adapter / "adapter_model.safetensors"),
                    )
                    recovery_reports.append(
                        {"condition": condition, **recovery.model_dump(mode="json")}
                    )
                    if mode == "full" and recovery.success:
                        assert eval_config is not None
                        recovered_output = layout.predictions / "recovered" / condition
                        recovered_predictions = runner.run(
                            recovered_adapter,
                            eval_config,
                            recovered_output,
                            layout.logs / f"recovered_{condition}.log",
                        )
                        recovered_rows_by_condition[condition] = list(
                            read_jsonl(recovered_predictions)
                        )

                if mode == "full" and "weight_clamp" in strategies:
                    clean_state, metadata = load_safetensors_state(
                        adapter / "adapter_model.safetensors"
                    )
                    fault_state, _ = load_safetensors_state(
                        fault_adapter / "adapter_model.safetensors"
                    )
                    protected_state, clamp_report = clamp_state_dict(
                        clean_state, fault_state, selector=selector
                    )
                    protected_adapter = layout.faults / "weight_clamp" / condition
                    if protected_adapter.exists():
                        shutil.rmtree(protected_adapter)
                    shutil.copytree(fault_adapter, protected_adapter)
                    save_safetensors_state(
                        protected_state,
                        protected_adapter / "adapter_model.safetensors",
                        metadata,
                    )
                    adapter_reports[-1]["weight_clamp"] = clamp_report.model_dump(mode="json")

                if cleanup_temporary_adapters:
                    temporary_adapters = [fault_adapter]
                    if mode in {"recover", "full"} and "checksum_recovery" in strategies:
                        temporary_adapters.append(layout.faults / "recovered" / condition)
                    if mode == "full" and "weight_clamp" in strategies:
                        temporary_adapters.append(layout.faults / "weight_clamp" / condition)
                    for temporary_adapter in temporary_adapters:
                        if temporary_adapter.exists():
                            shutil.rmtree(temporary_adapter)

        write_jsonl(
            layout.faults / "injection_records.jsonl",
            (record.model_dump(mode="json") for record in all_records),
        )
        write_json(layout.faults / "adapter_reports.json", adapter_reports)
        protection_results: dict[str, Any] = {
            "no_protection": {"enabled": "no_protection" in strategies},
            "checksum_recovery": recovery_reports,
        }
        pairs: list[dict[str, Any]] = []
        if mode == "baseline":
            pairs = build_prediction_pairs(clean_rows, clean_rows)
        elif mode == "full":
            guarded_by_count: dict[int, dict[str, dict[str, Any]]] = {}
            if "output_guard_vote" in strategies:
                clean_index = {str(row["id"]): row for row in clean_rows}
                grouped_faults: dict[int, dict[str, list[str]]] = defaultdict(
                    lambda: defaultdict(list)
                )
                for condition, rows in fault_rows_by_condition.items():
                    count = int(condition.split("_")[1])
                    for row in rows:
                        grouped_faults[count][str(row["id"])].append(str(row["prediction"]))
                vote_reports: list[dict[str, Any]] = []
                for count, samples in grouped_faults.items():
                    guarded_by_count[count] = {}
                    for sample_id, candidates in samples.items():
                        clean = clean_index[sample_id]
                        vote = output_guard_vote(
                            str(clean["task_type"]),
                            candidates,
                            fallback=str(clean["prediction"]),
                        )
                        guarded_by_count[count][sample_id] = {**clean, "prediction": vote.selected}
                        vote_reports.append(
                            {
                                "bit_flip_count": count,
                                "id": sample_id,
                                **vote.model_dump(mode="json"),
                            }
                        )
                protection_results["output_guard_vote"] = vote_reports

            aggregate_fault_rows: list[dict[str, Any]] = []
            for condition, fault_rows in fault_rows_by_condition.items():
                count = int(condition.split("_")[1])
                recovered_rows = recovered_rows_by_condition.get(condition)
                if recovered_rows is None and count in guarded_by_count:
                    recovered_rows = [
                        guarded_by_count[count][str(clean["id"])] for clean in clean_rows
                    ]
                condition_pairs = build_prediction_pairs(clean_rows, fault_rows, recovered_rows)
                for pair in condition_pairs:
                    pair["fault_condition"] = condition
                pairs.extend(condition_pairs)
                aggregate_fault_rows.extend(
                    {**row, "fault_condition": condition} for row in fault_rows
                )
            write_jsonl(layout.faults / "predictions.jsonl", aggregate_fault_rows)
        self._write_real_outputs(layout, pairs, protection_results)
        write_json(
            layout.root / "real_inference_manifest.json",
            {
                "schema_version": "1.0",
                "execution_mode": "real_inference",
                "model_path": manifest.get("model_dir"),
                "adapter_path": str(adapter),
                "dataset_manifest": _as_dict(self.config.get("data")).get("dataset_manifest"),
                "eval_split": _as_dict(self.config.get("data")).get("eval_split"),
                "device": device,
                "dtype": _as_dict(self.config.get("model")).get("torch_dtype", "auto"),
                "seed": seed,
                "fault_config": fault_config,
                "protection_config": _as_dict(self.config.get("protection")),
            },
        )
