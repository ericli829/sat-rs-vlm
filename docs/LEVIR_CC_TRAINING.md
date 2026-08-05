# LEVIR-CC 多任务回放训练

## 方案概览

生产模型使用同一个 LoRA adapter，同时学习 VRSBench 的检测、计数、场景分类、描述和 VQA，以及 LEVIR-CC 的双时相变化描述。训练从已经完成的 VRSBench LoRA 开始，但创建新的优化器和学习率调度器；这不是恢复旧 Trainer 状态，也不是叠加两个 adapter。

每个回放轮次使用以下训练数据：

| 数据源 | 选取数量 | 方法 |
| --- | ---: | --- |
| VRSBench | 40,920 | 按任务配额、按图像组均衡抽样 |
| LEVIR-CC | 13,630 | 每个图像对轮换选择 2 条描述 |
| 合计 | 54,550 | VRSBench 约 75%，LEVIR-CC 约 25% |

VRSBench 配额如下：

```text
detection             12,000
counting               8,000
captioning             6,000
vqa                    13,720
scene_classification   1,200
```

训练 batch 固定为 8，梯度累积为 2。采样器输出来源同质的固定 batch，并按以下顺序循环：

```text
VRSBench -> VRSBench -> VRSBench -> LEVIR-CC
```

第一轮使用 LEVIR-CC 每个图像对排序后的第 0、1 条描述；第二轮使用第 2、3 条描述。VRSBench 在第二轮使用不同随机种子重新分层抽样。

## AutoDL 数据目录

```text
/root/autodl-tmp/datasets/
├── VRSBench/
└── LEVIR-CC/
    ├── annotations/
    │   ├── levircc_train.jsonl
    │   ├── levircc_val.jsonl
    │   └── levircc_test.jsonl
    └── images/
        ├── train/{A,B}/
        ├── val/{A,B}/
        └── test/{A,B}/
```

构建器会把 LEVIR-CC 标注中的旧 Windows 绝对路径转换为相对 `/root/autodl-tmp/datasets` 的路径，并逐张检查图像。

## 确定已有 Adapter

必须显式指定原 VRSBench adapter。路径中应直接包含 `adapter_config.json` 和 `strategy_manifest.json`：

```bash
INITIAL_ADAPTER="/root/autodl-tmp/outputs/autodl_lora_4090_bs16_gc_20260804_095914/checkpoints"

test -f "$INITIAL_ADAPTER/adapter_config.json"
test -f "$INITIAL_ADAPTER/strategy_manifest.json"
```

不要把 `checkpoint-<step>` 用作新任务初始化目录，除非明确要恢复同一个中断实验。新任务回放训练使用 `--initial-adapter`，断点续训才使用 `--resume-from-checkpoint`。

## 数据准备检查

```bash
cd /root/autodl-tmp/sat-rs-vlm

bash scripts/training/run_autodl_levircc_train.sh \
  --mode joint \
  --round-index 0 \
  --prepare-only

cat data/processed/multisource/vrsbench_levircc_report.json
```

报告应接近：

```text
VRSBench selected: 40920
LEVIR-CC selected: 13630
validation: VRSBench 1024 + LEVIR-CC 1333
```

## 真实模型冒烟测试

```bash
INITIAL_ADAPTER="/root/autodl-tmp/outputs/autodl_lora_4090_bs16_gc_20260804_095914/checkpoints"

bash scripts/training/run_autodl_levircc_train.sh \
  --mode joint \
  --initial-adapter "$INITIAL_ADAPTER" \
  --round-index 0 \
  --learning-rate 0.00002 \
  --num-train-epochs 1 \
  --max-steps 10 \
  --max-train-samples 256 \
  --max-eval-samples 64 \
  --output-dir /root/autodl-tmp/outputs/smoke/vrsbench_levircc_replay
```

冒烟结束后检查：

```bash
cat /root/autodl-tmp/outputs/smoke/vrsbench_levircc_replay/smoke_train_report.json
```

## 正式两轮训练

第一轮从原 VRSBench adapter 开始，学习率为 `2e-5`；第二轮从第一轮 adapter 开始，学习率为 `1e-5`。两轮各训练 1 epoch。

```bash
cd /root/autodl-tmp/sat-rs-vlm

INITIAL_ADAPTER="/root/autodl-tmp/outputs/autodl_lora_4090_bs16_gc_20260804_095914/checkpoints"
RUN_ROOT="/root/autodl-tmp/outputs/vrsbench_levircc_replay_formal"

screen -dmS rs-replay-train bash -lc "
cd /root/autodl-tmp/sat-rs-vlm
bash scripts/training/run_autodl_levircc_replay.sh \
  --initial-adapter '$INITIAL_ADAPTER' \
  --run-root '$RUN_ROOT'
"
```

查看 screen：

```bash
screen -ls
screen -r rs-replay-train
```

退出 screen 但保持训练：按 `Ctrl+A`，再按 `D`。

查看最新训练日志：

```bash
tail -f "$(ls -t /root/autodl-tmp/outputs/logs/joint_levircc_*.log | head -1)"
```

第一轮和最终 adapter 分别位于：

```text
/root/autodl-tmp/outputs/vrsbench_levircc_replay_formal/round_1_adapter
/root/autodl-tmp/outputs/vrsbench_levircc_replay_formal/round_2_adapter
```

如果第一轮完成后实例中断，可以跳过第一轮继续：

```bash
bash scripts/training/run_autodl_levircc_replay.sh \
  --initial-adapter "$INITIAL_ADAPTER" \
  --run-root "$RUN_ROOT" \
  --skip-round-1
```

## 分数据集评测

VRSBench 使用原完整验证 JSONL；LEVIR-CC 使用全部 1,333 个去重验证图像对。两套结果分别保存，不能只使用合并平均分选模。

```bash
cd /root/autodl-tmp/sat-rs-vlm

FINAL_ADAPTER="/root/autodl-tmp/outputs/vrsbench_levircc_replay_formal/round_2_adapter"
EVAL_ROOT="/root/autodl-tmp/eval/vrsbench_levircc_replay_formal"

screen -dmS rs-replay-eval bash -lc "
cd /root/autodl-tmp/sat-rs-vlm
bash scripts/evaluation/run_autodl_replay_eval.sh \
  --adapter-dir '$FINAL_ADAPTER' \
  --eval-root '$EVAL_ROOT'
"
```

查看评测进度：

```bash
screen -r rs-replay-eval
tail -f "$(ls -t /root/autodl-tmp/outputs/logs/replay_eval_*.log | head -1)"
```

查看结果：

```bash
cat "$EVAL_ROOT/vrsbench/summary.json"
cat "$EVAL_ROOT/levircc/summary.json"
cat "$EVAL_ROOT/evaluation_index.json"
```

## 选模约束

最终 adapter 至少应满足：

```text
VRSBench detection valid_json_rate 相对原 LoRA 下降不超过 1%
VRSBench counting parsable_rate 相对原 LoRA 下降不超过 1%
VRSBench detection mean_iou 不应明显下降
LEVIR-CC BLEU-1、BLEU-4、ROUGE-L 相对原 LoRA 明显提升
```

如果第二轮出现格式退化，应优先使用第一轮 adapter，而不是继续增加 LEVIR-CC 训练轮数。

## 本地测试解释器

本地测试统一使用：

```powershell
& "C:\Users\Ericoneabc\AppData\Local\Microsoft\WindowsApps\python.exe" -m pytest -q
```

后续增加数据集时，在 `configs/data/autodl_vrsbench_levircc.yaml` 中增加数据源，并为该数据源配置训练配额、图像分组策略和验证抽样数。
