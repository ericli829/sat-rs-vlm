# 本地 Qwen3-VL-2B Smoke 训练清单

```text
[ ] 使用同一个 Python 环境安装了项目：python -m pip install -e ".[model]"
[ ] python scripts/check_env.py 能正确显示解释器、torch 与 CUDA 状态
[ ] 本地模型存在：<model-parent>\Qwen3-VL-2B-Instruct
[ ] VRSBench 存在：<data-parent>\VRSBench
[ ] data/processed/qwen3vl_train.jsonl 与 qwen3vl_val.jsonl 已生成
[ ] validate_training_assets.py 通过
[ ] dry-run 通过
[ ] forward-only 通过
[ ] max_steps=2 的 smoke 训练通过
[ ] checkpoints/smoke/qwen3vl-local-smoke/ 包含 adapter 文件
[ ] reports/training/smoke_train_report.json 已生成
```

PowerShell 推荐命令：

```powershell
$env:LOCAL_MODEL_DIR="<model-parent>\Qwen3-VL-2B-Instruct"
$env:DATA_ROOT="<data-parent>\VRSBench"
$env:TRAIN_JSONL="data\processed\qwen3vl_train.jsonl"
$env:VAL_JSONL="data\processed\qwen3vl_val.jsonl"

python scripts/check_env.py
python scripts/validate_training_assets.py --config configs/train/qwen3vl_local_smoke.yaml

python scripts/train_qwen3vl_lora.py --config configs/train/qwen3vl_local_smoke.yaml --dry-run

python scripts/train_qwen3vl_lora.py `
  --config configs/train/qwen3vl_local_smoke.yaml `
  --model-dir "$env:LOCAL_MODEL_DIR" `
  --train-file "$env:TRAIN_JSONL" `
  --val-file "$env:VAL_JSONL" `
  --image-root "$env:DATA_ROOT" `
  --forward-only

python scripts/run_local_smoke_train.py `
  --model-dir "$env:LOCAL_MODEL_DIR" `
  --train-file "$env:TRAIN_JSONL" `
  --val-file "$env:VAL_JSONL" `
  --image-root "$env:DATA_ROOT"
```

完成 smoke 后再进行全量训练。若任一步失败，保留完整命令、解释器路径和 traceback，先修复环境或输入资产，不要直接提高训练步数。
