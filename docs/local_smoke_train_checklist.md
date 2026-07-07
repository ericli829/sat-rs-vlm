# 本地 Qwen3-VL Smoke Train Checklist

```text
[ ] 已创建 .venv
[ ] 已安装 pip install -e ".[model]"
[ ] 本地模型目录存在
[ ] 数据 JSONL 存在
[ ] 图片路径可解析
[ ] validate_training_assets.py 通过
[ ] dry-run 通过
[ ] forward-only 通过
[ ] max_steps=2 训练通过
[ ] checkpoint 已保存
[ ] smoke_train_report.json 已生成
```

推荐命令：

```bash
python scripts/bootstrap_env.py --with-model
python scripts/check_env.py
python scripts/validate_training_assets.py --config configs/train/qwen3vl_local_smoke.yaml
python scripts/run_local_smoke_train.py \
  --model-dir /path/to/qwen3vl \
  --train-file /path/to/train.jsonl \
  --val-file /path/to/val.jsonl \
  --image-root /path/to/data
```
