# LAE-DINO sidecar environment

LAE-DINO is deliberately isolated from the main `sat-rs-vlm` environment.  Its
MMDetection/MMCV/PyTorch versions must come from the downloaded LAE-DINO source
README and requirements, not from the main project requirements. Installation
occurs only after the explicit LAE flag and only inside the dedicated LAE env;
the main environment is never modified by the LAE orchestrator.

`install.sh` is the isolated-environment orchestrator. It creates or reuses a
dedicated Conda environment, installs only source-provided requirement files,
installs the source checkout's `mmdetection_lae` package, and then runs
`check_environment.py`. A new environment is cloned from an explicit base env;
all subsequent package changes occur only inside the clone.

The config must exactly match the selected checkpoint. Models, checkpoints and
BERT files must already exist locally:

```bash
LAE_DINO_REQUIREMENTS=/absolute/LAE-DINO/mmdetection_lae/requirements.txt \
LAE_DINO_SOURCE_ROOT=/absolute/LAE-DINO \
LAE_DINO_CONFIG=/absolute/LAE-DINO/mmdetection_lae/configs/exact_config.py \
LAE_DINO_CHECKPOINT=/absolute/checkpoints/exact_checkpoint.pth \
LAE_DINO_BERT_ROOT=/absolute/weights/bert-base-uncased \
bash environments/lae_dino/install.sh \
  --env-name rs-vlm-lae \
  --base-env rs-vlm \
  --env-script /root/autodl_env.sh
```

When `LAE_DINO_REQUIREMENTS` is omitted, the orchestrator uses the checkout's
`mmdetection_lae/requirements.txt`. It also consumes the checkout's
`requirements/mminstall.txt` through OpenMIM and its
`requirements/multimodal.txt` when those files exist. No MMDetection/MMCV/LAE
version is declared by `sat-rs-vlm`.

The AutoDL wrapper invokes the same orchestrator:

```bash
bash scripts/environment/setup_autodl.sh \
  --install-model \
  --install-retriever \
  --install-lae \
  --lae-source-root /absolute/LAE-DINO \
  --lae-config /absolute/LAE-DINO/mmdetection_lae/configs/exact_config.py \
  --lae-checkpoint /absolute/checkpoints/exact_checkpoint.pth \
  --lae-bert-root /absolute/weights/bert-base-uncased
```

Use `--dry-run` on either installer to inspect the plan. Re-running either
command reuses the named Conda environment and refreshes the editable source
installation. After a successful check, the AutoDL environment script records:

- `LAE_DINO_PYTHON`
- `LAE_DINO_SOURCE_ROOT`
- `LAE_DINO_CONFIG`
- `LAE_DINO_CHECKPOINT`
- `LAE_DINO_BERT_ROOT`

The current LAE-1M config aliases are written as well so existing TaskGraph
configuration remains compatible.

Validate an already-created environment with:

```bash
python environments/lae_dino/check_environment.py \
  --source-root /root/autodl-fs/rs_detectors/lae_dino/source/LAE-DINO \
  --config /absolute/path/to/the/exact/lae_dino_config.py \
  --checkpoint /root/autodl-fs/rs_detectors/lae_dino/checkpoints/lae_dino_swint_lae1m-28ca3a15.pth \
  --bert-root /root/autodl-fs/rs_detectors/lae_dino/weights/bert-base-uncased
```

Use `--discover` first to list candidate config files.  The sidecar requires an
explicit config path; it never guesses a config and never downloads weights.

Run `scripts/integrations/lae_dino_worker.py` only with `LAE_DINO_PYTHON` from
this isolated environment. The main `rs-vlm` process communicates with it
through JSONL and never imports MMDetection.
