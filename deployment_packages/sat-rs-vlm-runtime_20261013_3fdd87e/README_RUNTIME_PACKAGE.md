# sat-rs-vlm TaskGraph runtime package

This is a source-only snapshot for the cloud TaskGraph competition service. It contains the committed runtime code, configuration templates, planner prompts, deployment launcher, and relevant tests.

- Package source commit: `3fdd87e668aed92f9a986fe84531b18830a53c82`
- Runtime implementation commit: `90be53c95745bf99ba79f115190ec998aef584b0`
- The working tree had unrelated uncommitted planner-cleaning changes; they are not included.
- Model checkpoints, image datasets, API credentials, and evaluation result payloads are not included.
- The selected Oct 11 Planner LoRA remains external at `/root/autodl-fs/data/sat-rs-vlm-training/planner_4b_clean_20261011/output/training`; `adapter_model.safetensors` SHA256: `d7169cce768dcd01e86f1c2fd2847f7fddf85e45f9d61db3e23180a97211a4e6`.

## Cloud dependencies

The deployed service uses external paths for Qwen3-VL 4B and 2B base weights, the Oct 11 Planner LoRA, ChangeChat and 2B semantic adapters, GeoRSCLIP, LAE-DINO source/config/checkpoint/BERT weights, and `/root/autodl-fs/dev100-images`. Their locations are recorded in `RUN_PACKAGE_MANIFEST.json`; these artifacts are deliberately not copied into this source package. The frozen-model SHA256 tables remain alongside this package under `D:\Desktop\SHA256\模型`.

## Launcher

On the cloud host, after placing the source tree and supplying the external model paths plus `MODEL_API_KEY` through the host environment, run `bash scripts/competition/start_cloud_protocol_v2.sh 8000`. The launcher reads `/root/autodl_env.sh` when present, requires `QWEN3VL_4B_PLANNER_LORA`, and starts the existing protocol-v2 service. Do not put API keys in this package.

The verified 25-request UHR replay report is stored in the project workspace at `reports/experiments/cloud_taskgraph_deployment_20261013/UHR25_REPORT.md`; reports and request images are intentionally omitted from this runtime source archive.
