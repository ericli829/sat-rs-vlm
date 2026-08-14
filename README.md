# sat-rs-vlm

`sat-rs-vlm` 是面向卫星平台受限算力部署的多模态遥感大模型工程框架。项目提供
遥感检测、分类、分割、变化检测、计数、描述和 VQA 的统一领域接口，并包含
Mock 推理、FastAPI/CLI、本地 Qwen3-VL、VRSBench 转换、LoRA 训练与评估链路。

当前稳定基线是已经实际跑通的 Qwen3-VL-2B LoRA。此次版本增加本地与 AutoDL
共用的配置、数据 manifest、训练包装、环境检查和结果备份，但保留
`scripts/train_qwen3vl_lora.py` 原命令及原配置兼容性。

当前版本还整合了 single-event-upset/bit flip 可靠性工具。旧实验脚本已重构为统一核心、
应用服务和少量薄入口，保留当前云端配置、数据和 LoRA 评测架构。

本次选择性整合还加入 SHC 结构化训练协议和统一 INT8 benchmark：assistant-only loss、
detection/counting JSON、E0/E1/E1b/E1d 实验、CPU dynamic INT8 与 bitsandbytes INT8。
来源分支中的完整派生数据、本地报告、绝对路径和未经复现的性能数字没有合入。

## 环境边界

- 本地 Windows/Linux/macOS：代码检查、微型数据验证、Mock 训练和可选真实模型 smoke。
- AutoDL/Linux GPU：真实 smoke、正式训练、断点恢复和结果备份。
- 业务 Python 代码不包含 AutoDL 绝对路径；云路径集中在 `configs/cloud/` 和 shell 脚本。
- 默认 pytest 不联网、不下载模型、不读取完整 VRSBench，也不要求 GPU。

## 五分钟本地启动

```powershell
python scripts/environment/bootstrap_local.py --with-dev
.venv\Scripts\Activate.ps1
python scripts/environment/check_environment.py
pytest -q
python scripts/data/validate_dataset.py --dataset-root tests/fixtures/miniature_dataset
python scripts/training/run_smoke_train.py --config configs/local/train_lora_smoke.yaml
```

本仓库当前开发机验证约定使用系统默认的 Windows Store Python 3.11，不使用依赖
不完整的现有 `.venv`。上面的 bootstrap 命令是提供给新环境或其他开发者的标准流程。

## 五分钟 AutoDL 初始化

```bash
cd /root/autodl-tmp/sat-rs-vlm
bash scripts/environment/setup_autodl.sh \
  --env-name rs-vlm --clone-current --install-dev --install-model
source /root/autodl_env.sh
conda activate rs-vlm
python scripts/environment/check_environment.py --require-model --require-gpu
```

初始化脚本不会替换镜像中已经匹配 CUDA 的 PyTorch。模型依赖通过
`environments/requirements-model.txt` 单独安装；云端 TensorBoard 依赖会自动安装。
需要 QLoRA 或 `bnb_int8` 时追加 `--install-qlora`，bitsandbytes 不会进入基础 LoRA 环境。

## 配置分层

优先级从高到低：

```text
CLI > 环境变量 > 实验配置 > local/cloud 配置 > base 配置 > 代码默认值
```

基础模型、数据与 LoRA 超参数位于 `configs/base/`；本地覆盖位于
`configs/local/`；AutoDL 路径和运行参数位于 `configs/cloud/`；可比较实验位于
`configs/experiments/`。支持 `PROJECT_ROOT`、`DATA_ROOT`、`MODEL_ROOT`、
`OUTPUT_ROOT`、各类缓存和备份环境变量。

## 数据组织

VRSBench 原始图片和标注保持不变，只新增：

```text
VRSBench/project_metadata/
├── dataset_manifest.json
├── train.jsonl
├── validation.jsonl
├── test.jsonl
├── smoke.jsonl
├── statistics.json
└── splits/
```

JSONL 中保存相对图片路径，运行时使用 `dataset_root / relative_path`。校验：

```bash
python scripts/data/validate_dataset.py \
  --dataset-root /path/to/VRSBench \
  --manifest-name project_metadata/dataset_manifest.json
```

详见 [数据目录说明](docs/dataset_layout.md) 和 [data/README.md](data/README.md)。

## 训练

本地 Mock smoke：

```bash
python scripts/training/run_smoke_train.py \
  --config configs/local/train_lora_smoke.yaml
```

本地真实模型前向、短 LoRA 训练和 CLI 推理：

```powershell
$env:LOCAL_MODEL_DIR="<model-parent>\\Qwen3-VL-2B-Instruct"
$env:DATA_ROOT="<data-parent>\\VRSBench"
$env:TRAIN_JSONL="$PWD\\data\\processed\\qwen3vl_train.jsonl"
$env:VAL_JSONL="$PWD\\data\\processed\\qwen3vl_val.jsonl"
python scripts/train_qwen3vl_lora.py --config configs/train/qwen3vl_local_smoke.yaml --forward-only
python scripts/train_qwen3vl_lora.py --config configs/train/qwen3vl_local_smoke.yaml --max-steps 2
python -m sat_rs_vlm.interfaces.cli infer --config configs/local/qwen3vl_real_infer.yaml --image "<image>" --prompt "请描述主要地物。"
```

外部微调插件必须显式指定根目录；先执行 `--check-only`，再选择策略的 smoke 配置运行。
完整命令见 [训练命令清单](docs/training_commands.md)。

统一真实训练：

```bash
python scripts/training/run_train.py \
  --config configs/experiments/lora_baseline.yaml \
  --environment local
```

AutoDL 正式训练：

```bash
bash scripts/training/run_autodl_train.sh \
  --config configs/cloud/train_lora_autodl.yaml
```

训练包装入口会把结构化数据组成、均匀/加权采样配置完整传给稳定 LoRA 脚本。真实训练完成后，
Adapter 根目录包含 `strategy_manifest.json` 和 `processor/`，可直接交给统一评估与可靠性入口；
不要把 Trainer 的中间 `checkpoint-*` 子目录当作最终 Adapter 根目录。

断点恢复：

```bash
python scripts/training/resume_train.py \
  --config configs/experiments/lora_baseline.yaml \
  --environment local \
  --output-dir outputs/qwen3_vl_2b_lora/<run> \
  --latest
```

旧命令继续可用：

```bash
python scripts/train_qwen3vl_lora.py \
  --config configs/train/qwen3vl_local_smoke.yaml
```

统一入口只负责分层配置、预检、快照和恢复点解析；真实训练仍委托上述稳定脚本。

### 结构化微调

```bash
python scripts/prepare_e1_datasets.py
python scripts/train_qwen3vl_lora.py --config configs/train/qwen3vl_local_e1_balanced.yaml
python scripts/evaluate_rs_vlm.py --config configs/eval/qwen3vl_eval_e0_zeroshot.yaml
```

训练 labels 只保留 assistant 答案；截断后没有监督 token 会带 sample id 失败。E1d 已拆为
data、sampler 和 combined 三种配置，避免把配额倾斜和加权采样混为同一变量。详见
[结构化微调实验](docs/finetune_playbook.md) 和 [数据协议](docs/data_format.md)。

AutoDL RTX 4090 正式 LoRA 配置使用环境变量路径：

```bash
python scripts/train_qwen3vl_lora.py \
  --config configs/train/qwen3vl_autodl_4090.yaml
```

### H1 困难样本视觉适配

H1 从已经完成 Stage A/Stage B 的 LoRA adapter 继续训练，不从 base model 重启。它使用
Evaluation v1.5 在训练 mining split 上产生的逐样本指标构造 70% hard examples + 30%
regular replay，并只额外解冻 Qwen3-VL 最后 2 个视觉 block 和主 visual merger。bbox 仍为
`{"label":"class","bbox":[x1,y1,x2,y2]}` 与 `normalized_0_1`。

当前训练默认 loss 为 `task_weighted`：先对每条样本的 assistant token CE 求均值，再按
`loss.task_weights` 聚合样本；`token_mean` 仅用于历史兼容对照。H1 的 70/30 组成保持为
legacy experimental config，不会成为普通 LoRA 的默认数据策略。

```bash
python scripts/training/analyze_training_data.py \
  --config configs/train/qwen3vl_hard_visual_adaptation.yaml

python scripts/training/build_hard_example_dataset.py \
  --config configs/train/qwen3vl_hard_visual_adaptation.yaml

python scripts/train_qwen3vl_lora.py \
  --config configs/train/qwen3vl_hard_visual_adaptation.yaml \
  --dry-run
```

正式 refinement 配置使用 `target_effective_epochs` 计算训练步数，并以
`max_effective_epochs` 阻止固定 `max_steps` 导致意外重复曝光；显式越界只有设置
`allow_overtrain: true` 才能继续。H1 使用 LoRA、merger、ViT 三组独立学习率，训练前生成
完整参数审计。新 checkpoint 保存标准 PEFT adapter、`visual_trainable_weights.safetensors`
和校验 manifest；加载器仍兼容历史 `h1_visual_weights.safetensors`。

完整流程、泄漏保护和 AutoDL 命令见
[H1 Hard Example Visual Adaptation](docs/training/hard_example_visual_adaptation.md)。

## Evaluation v1.5

`src/sat_rs_vlm/evaluation/` 是唯一正式评估内核，统一负责协议解析、任务指标、语义诊断、
配对比较和绘图。`evaluate_rs_vlm.py` 只负责真实模型生成，生成的 predictions 会自动进入
v1.5 runner；Keyword Hit 仅保留在 repository compatibility 诊断中，不作为主要指标。

```bash
# 真实模型生成 + 统一评估
python scripts/evaluate_rs_vlm.py --config configs/eval/qwen3vl_eval.yaml

# 对已有 predictions 做离线评估、比较和绘图
python scripts/evaluation/evaluate_predictions.py \
  --config configs/eval/evaluation_v1_5.yaml
python scripts/evaluation/compare_evaluations.py \
  --config configs/eval/evaluation_v1_5.yaml
python scripts/evaluation/plot_evaluation_results.py \
  --config configs/eval/evaluation_v1_5.yaml
```

标准评估目录包含 `metrics.json`、`evaluated_predictions.jsonl` 和
`evaluation_manifest.json`；`summary.json` 是兼容别名。详见
[Evaluation Framework](docs/evaluation/README.md)。

## INT8 量化

```bash
python scripts/quantize_rs_vlm.py \
  --config configs/quantization/qwen3vl_torch_dynamic_int8.yaml --dry-run
```

`torch_dynamic_int8` 是 CPU Linear 动态量化；`bnb_int8` 是 CUDA bitsandbytes 8-bit 加载，
二者不会混称。统一 benchmark 固定 messages 样本、Processor 和生成参数，baseline 与
quantized predictions 分别进入 Evaluation v1.5，再生成配对 `comparison.json`。顶层实现位于
`src/sat_rs_vlm/quantization/`，旧 `compression.quantization` 仅为导入兼容层。

```bash
# FP32 merged LoRA -> dynamic INT8 -> 统一评估 -> 配对比较
python scripts/quantize_rs_vlm.py \
  --config configs/quantization/quantization_eval.yaml

# 配置/资产 smoke，不加载真实模型
python scripts/quantization_sensitivity_test.py \
  --config configs/quantization/quantization_sensitivity_smoke.yaml --dry-run

# 真实 GPU 层敏感度实验并绘图
python scripts/quantization_sensitivity_test.py \
  --config configs/quantization/sensitivity_layer_autodl.yaml --plot
```

当前默认测试验证 dry-run、toy Linear 和 fake module tree；真实 Qwen3-VL GPU 结果仍需在
AutoDL 显式运行。详见 [量化说明](docs/quantization.md) 和
[敏感度分析](docs/quantization_sensitivity.md)。

## 实验输出

```text
outputs/<group>/<timestamp>_<experiment>_seed<seed>/
├── config_resolved.yaml
├── command.txt
├── environment.json
├── git_commit.txt
├── preflight.json
├── logs/
├── checkpoints/
├── predictions/
├── metrics/
└── artifacts/
```

统一评估与量化对比默认写入：

```text
reports/evaluation/<experiment>/
├── baseline/
│   ├── metrics.json
│   └── evaluated_predictions.jsonl
├── quantized/
│   ├── metrics.json
│   └── evaluated_predictions.jsonl
├── comparison/
│   └── comparison.json
├── figures/
└── benchmark_report.json
```

备份只复制关键文件和最新若干 checkpoint：

```bash
bash scripts/storage/backup_results.sh \
  --experiment-dir /root/autodl-tmp/outputs/<experiment> \
  --backup-root /root/autodl-fs/experiments
```

## Bit Flip 可靠性

| 能力 | 状态 |
|---|---|
| bytes/int/bytearray bit flip | 本地已测试 |
| float16/bfloat16/float32/int8/uint8 tensor bit flip | 本地 CPU 已测试 |
| state dict 与 LoRA A/B/正则/层筛选 | 本地 CPU 已测试 |
| LoRA safetensors 独立故障 Adapter | 小型 Adapter 本地已测试 |
| checksum manifest 与原子备份恢复 | 本地已测试 |
| counting/detection/VQA 输出验证与 output guard vote | 本地已测试 |
| 实验性 weight clamp | 本地 state dict smoke 已测试 |
| 本地统一 reliability smoke | 本地已测试，明确标记 `smoke_mock` |
| 云端 Qwen3-VL clean/fault/recovery 入口 | 代码已实现，等待 AutoDL 真实验证 |
| region/layer/sign-exponent-mantissa sensitivity | 单元测试通过，等待 RTX 4090 真机实验 |
| Evaluation v1.5 paired Bootstrap + E1/E2/E3 | 已接入，tier/hash 不同会拒绝比较 |
| activation guard research/deployment 模式 | 单元测试通过 |
| sensitivity groups → task-metric risk policy | 单元测试通过 |
| warm/golden adapter replicas 与 scrub | 单元测试通过 |

本地全量 smoke，不加载 Qwen3-VL：

```bash
python scripts/reliability/run_smoke.py --case all
```

AutoDL 真实入口复用已有模型、Processor、LoRA checkpoint loader、数据 Collator 和评测脚本：

```bash
python scripts/reliability/run_experiment.py \
  --config configs/reliability/experiments/lora_bitflip.yaml \
  --mode full \
  --environment autodl
```

Evaluation v1.5 SEU sensitivity 默认使用 E1；E1 高风险条件再显式升到 E2/E3：

```bash
python scripts/reliability/run_v15_sensitivity.py \
  --config configs/reliability/experiments/v15_sensitivity.yaml \
  --environment autodl --preflight

python scripts/reliability/run_v15_sensitivity.py \
  --config configs/reliability/experiments/v15_sensitivity.yaml \
  --environment autodl --run-id seu-e1-pilot \
  --activation-guard --activation-guard-mode research
```

真实模式缺少 CUDA、依赖、模型、数据或 Adapter 时会直接失败，不会回退为 Mock。本地 smoke
结果只证明工程流程可运行，不代表 Qwen3-VL 或星载硬件的实际抗辐射能力。详见
[可靠性模块](docs/reliability/README.md)、[实验流程](docs/reliability/experiment_workflow.md) 和
[命令清单](docs/reliability/commands.md)。

## 质量检查

```bash
ruff check src tests scripts
ruff format --check src tests scripts
mypy src
pytest -q
```

真实模型测试必须显式设置 `RUN_REAL_MODEL_TESTS=1`，并使用 `real_model`、`gpu`、
`slow` 或 `cloud` marker。当前默认环境缺失哪些依赖可通过
`scripts/environment/check_environment.py` 查看。

## 文档

- [本地环境](docs/local_setup.md)
- [AutoDL 环境](docs/autodl_setup.md)
- [数据布局](docs/dataset_layout.md)
- [结构化微调](docs/finetune_playbook.md)
- [Evaluation Framework](docs/evaluation/README.md)
- [Fixed Evaluation Tiers (E1/E2/E3)](docs/evaluation/evaluation_tiers.md)
- [INT8 量化](docs/quantization.md)
- [量化敏感度分析](docs/quantization_sensitivity.md)
- [三分支架构合并说明](docs/architecture_merge_v1_5.md)
- [完整命令清单](docs/training_commands.md)
- [环境版本](docs/environment_versions.md)
- [实验工作流](docs/experiment_workflow.md)
- [故障排查](docs/troubleshooting.md)
- [原 LoRA 训练说明](docs/training_qwen3vl.md)
- [外部实验插件](docs/external_plugins.md)
- [Bit Flip 可靠性](docs/reliability/README.md)
- [可靠性命令](docs/reliability/commands.md)
- [SEU v1.5 sensitivity 与保护策略](docs/reliability/seu_v15_operation.md)
- [Bit Flip 迁移映射](docs/reliability/bitflip_migration_plan.md)
