# LAE-DINO sidecar environment

LAE-DINO is deliberately isolated from the main `sat-rs-vlm` environment.  Its
MMDetection/MMCV/PyTorch versions must come from the downloaded LAE-DINO source
README and requirements, not from the main project requirements.  This
repository does not install or downgrade either environment automatically.

If the LAE-DINO checkout provides a requirements file, the explicit installer
can be run from that isolated environment (all four variables are required;
the config must match the selected checkpoint):

```bash
LAE_DINO_REQUIREMENTS=/absolute/LAE-DINO/requirements.txt \
LAE_DINO_SOURCE_ROOT=/absolute/LAE-DINO \
LAE_DINO_CONFIG=/absolute/LAE-DINO/configs/exact_config.py \
LAE_DINO_CHECKPOINT=/absolute/checkpoints/exact_checkpoint.pth \
LAE_DINO_BERT_ROOT=/absolute/weights/bert-base-uncased \
bash environments/lae_dino/install.sh
```

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

Run `scripts/integrations/lae_dino_worker.py` only with the Python executable
from this isolated environment.  The main `rs-vlm` process communicates with it
through JSONL and never imports MMDetection.
