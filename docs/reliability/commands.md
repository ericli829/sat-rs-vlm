# 可靠性命令

所有脚本不再修改 `sys.path`。新环境先安装项目：

```bash
python -m pip install -e ".[dev,model]"
```

## 本地 smoke

```bash
python scripts/reliability/run_smoke.py --case all
```

- GPU/真实模型：不需要；tensor、state dict 和 Adapter 文件案例需要 `model` extra。
- 输入：`configs/reliability/local_smoke.yaml` 和内置固定预测。
- 输出：`${OUTPUT_ROOT}/reliability/local_reliability_smoke/<run_id>`。
- 源 Adapter：使用临时小型 fake Adapter，不修改真实文件。
- seed：`--seed 2026`；指定目录可用 `--output-root` 和 `--run-id`。
- 单案例：`tensor`、`state-dict`、`adapter-file`、`output-guard`、`recovery`、`weight-clamp`。
- 覆盖：同名运行只有传入 `--overwrite` 才会替换。

## Checksum manifest

```bash
python scripts/reliability/checksum_manifest.py build \
  --root <adapter-or-model-directory> \
  --output <manifest.json>

python scripts/reliability/checksum_manifest.py verify \
  --manifest <manifest.json>
```

- GPU/真实模型：不需要。
- build 分块读取目录内文件，写相对路径、大小和 SHA-256；不修改输入。
- verify 检查缺失、大小变化和 hash 不匹配；manifest 位于其他目录时可传 `--root`。

## 固定评测样本

```bash
python scripts/data/build_reliability_eval_manifest.py \
  --config configs/reliability/experiments/lora_bitflip.yaml \
  --environment autodl
```

- GPU/真实模型：不需要，但必须能访问 DatasetManifest、processed JSONL 和图片。
- 输入：当前 VRSBench manifest 与配置中的 `eval_split`。
- 输出：`project_metadata/reliability/eval.jsonl` 及 `.stats.json`。
- seed/抽样数：`--seed`、`--samples-per-task`；默认拒绝覆盖，重建时显式 `--overwrite`。
- 图片路径始终相对数据集根目录。

## AutoDL 完整实验

```bash
source /root/autodl_env.sh
conda activate rs-vlm

python scripts/reliability/run_experiment.py \
  --config configs/reliability/experiments/lora_bitflip.yaml \
  --mode full \
  --environment autodl \
  --adapter-path "${OUTPUT_ROOT}/training/<experiment>/best_adapter"
```

- GPU/真实模型：`baseline` 和 `full` 必须有 CUDA，并加载真实 Qwen3-VL。
- 输入：标准 checkpoint，必须包含 strategy manifest、Adapter 权重/配置和 Processor；还需固定
  评测 JSONL 与数据根目录。
- 输出：`${OUTPUT_ROOT}/reliability/qwen3vl_lora_bitflip/<run_id>`。
- 源 Adapter：始终只读；每个 fault count/repeat 写入独立目录。
- seed：配置或 `--seed`。路径可用 `--adapter-path`、`--dataset-root`、`--eval-manifest` 覆盖。
- 恢复：`--resume --run-id <existing-run>`；新运行默认不覆盖，覆盖必须显式 `--overwrite`。
- 当前状态：真实入口已实现，但本地没有执行 Qwen3-VL bit flip 实验，需在 AutoDL 验证。

仅注入或恢复文件、不运行模型：

```bash
python scripts/reliability/run_experiment.py --config <config> --mode inject
python scripts/reliability/run_experiment.py --config <config> --mode recover
```

## 绘图

```bash
python -m pip install -e ".[reliability-plot]"
python scripts/reliability/plot_results.py \
  --input <run-directory>/metrics \
  --output <run-directory>/figures
```

- GPU/真实模型：不需要。
- 只读取标准 `summary.json`，不会重新推理或故障注入。

