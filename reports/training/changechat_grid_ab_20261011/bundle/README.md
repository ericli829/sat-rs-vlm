# ChangeChat 2B grid-first A/B cloud bundle

This is a CPU-only preparation bundle. It does not launch training or load a model.

- `route_a`: the 2026-08-04 VRSBench-only Qwen3-VL-2B adapter, before LEVIR-CC replay.
- `route_b`: the existing replay round-2 adapter, already present on the cloud host and copied into the run folder during staging.
- `prepare_changechat_grid_ab.py`: creates the pair-grouped grid train/dev split, quarantines global-flag contradictions, samples single-turn support rows, adds a stratified VRSBench replay subset, and verifies image paths.
- Route configs use batch 1 × gradient accumulation 16, BF16 LoRA, frozen vision/projector, 70/20/10 exposure mix, and persistent output storage.

After extraction, CPU preparation command (run only after confirming the destination is the dedicated run folder):

```bash
source /root/autodl_env.sh
conda run -n rs-vlm python scripts/prepare_changechat_grid_ab.py \
  --changechat-train inputs/changechat_105k_train.json.gz \
  --vrs-train /root/autodl-tmp/sat-rs-vlm/data/processed/qwen3vl_train.jsonl \
  --image-root /root/autodl-tmp/datasets \
  --output-dir .
```

This CPU step produces `data/train.jsonl`, `data/grid_dev.jsonl`, `data/grid_review.jsonl`, and a SHA256 manifest. Then run both route configs with `--dry-run`; do not start training until the existing 4B GPU job has finished and its result has been checked.
