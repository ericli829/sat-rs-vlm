import json

import yaml
from scripts.training.run_rs_merger_experiments import (
    DEFAULT_CONFIGS,
    _find_completed_run,
    build_experiment_command,
)

from sat_rs_vlm.models.reliability.checksum import file_sha256


def test_default_merger_matrix_uses_continuations_and_wide_control():
    assert [config.split("/")[-1] for config in DEFAULT_CONFIGS] == [
        "rs_count_merger_c2_cont_4090.yaml",
        "rs_count_merger_c3_cont_4090.yaml",
        "rs_count_merger_c4_wide_4090.yaml",
    ]


def test_launcher_builds_one_child_command_per_variant():
    command = build_experiment_command(
        python="python",
        config=DEFAULT_CONFIGS[0],
        output_root="out",
        max_train_samples=4,
        max_steps=2,
    )
    assert command[:2] == ["python", "scripts/training/train_rs_merger_expert.py"]
    assert command.count("--config") == 1
    assert command[command.index("--max-steps") + 1] == "2"


def test_launcher_completed_detection_requires_weights_and_source_provenance(tmp_path):
    r1 = tmp_path / "r1"
    r1.mkdir()
    (r1 / "strategy_manifest.json").write_text("{}", encoding="utf-8")
    visual = tmp_path / "visual.safetensors"
    visual.write_bytes(b"visual")
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "experiment": "C2_LM_4E",
                "model": {"r1_checkpoint": str(r1), "visual_sidecar": str(visual)},
                "output": {"root": str(tmp_path / "outputs")},
            }
        ),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "outputs" / "C2_LM_4E_1" / "checkpoint"
    checkpoint.mkdir(parents=True)
    weights = checkpoint / "expert_model.safetensors"
    weights.write_bytes(b"weights")
    manifest = {
        "expert_weights_sha256": file_sha256(weights),
        "source_r1_manifest_sha256": file_sha256(r1 / "strategy_manifest.json"),
        "source_visual_sidecar_sha256": file_sha256(visual),
    }
    (checkpoint / "expert_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (checkpoint / "config_resolved.yaml").write_text("{}\n", encoding="utf-8")
    assert _find_completed_run(config, output_root=None, environ={}) == checkpoint
