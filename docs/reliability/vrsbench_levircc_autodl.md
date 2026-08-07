# VRSBench + LEVIR-CC 单粒子翻转测评

正式配置 `configs/reliability/experiments/lora_bitflip.yaml` 使用统一的
`/root/autodl-tmp/datasets` 图片根目录，评测内容为：

- VRSBench：captioning、VQA、counting、detection、scene classification 各 20 条。
- LEVIR-CC：change detection 20 条，按前后图像对去重抽样。
- 合计 120 条。所有任务使用同一个固定 seed，clean、fault 和 recovery 阶段复用同一清单。

## 1. 环境与数据检查

```bash
cd /root/autodl-tmp/sat-rs-vlm
source /root/autodl_env.sh
source scripts/environment/activate_autodl_python.sh
activate_autodl_python rs-vlm

test -f /root/autodl-tmp/datasets/VRSBench/project_metadata/dataset_manifest.json
test -f /root/autodl-tmp/datasets/LEVIR-CC/annotations/levircc_val.jsonl
test -d /root/autodl-tmp/datasets/LEVIR-CC/images/val/A
test -d /root/autodl-tmp/datasets/LEVIR-CC/images/val/B
```

## 2. 构建固定多数据源评测清单

```bash
"$AUTODL_PYTHON" scripts/data/build_reliability_eval_manifest.py \
  --config configs/reliability/experiments/lora_bitflip.yaml \
  --environment autodl \
  --overwrite

cat /root/autodl-tmp/datasets/project_metadata/reliability/vrsbench_levircc_eval.stats.json
```

期望 `num_samples` 为 120，`source_distribution` 为 VRSBench 100、LEVIR-CC 20，
并且 `task_distribution` 包含 `change_detection`。

## 3. 后台运行完整测评

```bash
ADAPTER_PATH=/root/autodl-tmp/outputs/vrsbench_levircc_replay_formal/round_2_adapter
RUN_ID=vrsbench-levircc-bitflip-$(date +%Y%m%d-%H%M%S)
mkdir -p /root/autodl-tmp/outputs/reliability
LOG=/root/autodl-tmp/outputs/reliability/${RUN_ID}.log

screen -dmS rs-bitflip env \
  ADAPTER_PATH="$ADAPTER_PATH" RUN_ID="$RUN_ID" LOG="$LOG" \
  bash -lc '
set -Eeuo pipefail
cd /root/autodl-tmp/sat-rs-vlm
source /root/autodl_env.sh
source scripts/environment/activate_autodl_python.sh
activate_autodl_python rs-vlm

"$AUTODL_PYTHON" scripts/reliability/run_experiment.py \
  --config configs/reliability/experiments/lora_bitflip.yaml \
  --mode full \
  --environment autodl \
  --adapter-path "$ADAPTER_PATH" \
  --run-id "$RUN_ID" \
  2>&1 | tee "$LOG"
'
```

查看状态与日志：

```bash
screen -ls
tail -f "$LOG"
```

停止任务：

```bash
screen -S rs-bitflip -X quit
```

## 4. 查看结果

```bash
RUN_DIR=/root/autodl-tmp/outputs/reliability/qwen3vl_lora_bitflip/$RUN_ID

cat "$RUN_DIR/metrics/summary.json"
cat "$RUN_DIR/run_report.json"
find "$RUN_DIR" -maxdepth 3 -type f | sort
```

`metrics/summary.json` 同时包含 `overall`、`by_task` 和 `by_dataset`。
`change_detection` 额外报告 clean/fault 的 BLEU-1、BLEU-4、ROUGE-L 及下降量。
