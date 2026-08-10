# sat-rs-vlm

`sat-rs-vlm` 是面向卫星平台受限算力部署的遥感视觉语言模型工程框架。第一版提供统一任务领域模型、Mock/HuggingFace 推理、CLI/HTTP 接口、VRSBench 数据转换、本地 Qwen3-VL LoRA 训练与评估，以及 bit flip、checksum 等可靠性基础接口。

主仓库将已经跑通的 LoRA 作为稳定基线。QLoRA、DoRA、AdaLoRA、IA3、Partial Unfreeze、Full SFT 和 Prompt Tuning 属于实验性方法，保存在 Git 忽略的本地插件包中，只有用户显式指定插件根目录时才会被发现和加载。

## 当前能力

| 能力 | 状态 | 说明 |
|---|---|---|
| Mock CLI / HTTP 推理 | 可用 | 无 GPU、无真实模型也可运行 |
| 本地 Qwen3-VL 推理 | 可用 | 使用 HuggingFace 本地模型目录 |
| VRSBench 转换 | 可用 | caption、检测、计数、场景分类、VQA，框坐标裁剪到 `[0,1]` |
| Qwen3-VL LoRA 训练 | ✅ 已验证 | 50-step CPU: loss 24→6.68, 12.6 min, 0% empty predictions, valid_json 100% |
| Qwen3-VL LoRA 完整训练 | ✅ 已验证 | 17800-step GPU (RTX 4090): loss 2.59→0.72, eval_loss 0.55, 6.7h, 142k 样本 |
| Qwen3-VL LoRA 评估 | ✅ 已验证 | 评估链路完整，分任务指标可用 |
| 外部实验插件 API | 可用 | 显式发现、manifest、依赖检查、路径隔离、无静默回退 |
| INT8 量化 | ✅ 已验证 | PyTorch 原生动态 INT8 量化，CPU 推理加速 1.43× |
| 星载压缩与容错 | 部分实现 | 量化已实现，蒸馏/剪枝/bit flip/checksum 接口预留 |

### 初步训练验证 (2026-07-16)

**硬件**：Intel Core Ultra 7 155H, 32 GB RAM, CPU-only（无 NVIDIA GPU）

**数据**：50 张 VRSBench 图片 → 318 训练 / 328 验证样本（6 种遥感任务）

**方法**：LoRA (r=16, alpha=32), max_steps=50, seq_len=1024, bs=1, adamw + cosine lr

| 阶段 | 关键结果 |
|---|---|
| 训练 | 757.7s (≈12.6 min), loss 24.01 → **6.68** (↓72%), trainable 17.4M / 2.14B (0.81%) |
| 评估 | empty_prediction_rate **0%**, detection valid_json_rate **100%**, keyword_hit 100% (caption/detection/scene), 25% (VQA) |

> **结论**：50-step CPU 训练即可让模型对遥感图像产生有意义的回答，完整链路已验证。正式训练需 NVIDIA GPU（RTX 4060 8 GB 预计 2.5-5 h/epoch），CPU 完整训练 ≈58 天不切实际。

### 完整训练验证 (2026-08-04)

**硬件**：NVIDIA GeForce RTX 4090, 24 GB VRAM, AutoDL 云服务器

**数据**：VRSBench 完整数据集 → 142,390 训练 / 62,918 验证样本（6 种遥感任务）

**方法**：LoRA (r=16, alpha=32), num_train_epochs=2, bs=16, seq_len=1024, adamw + cosine lr

| 阶段 | 关键结果 |
|---|---|
| 训练 | 24,110.9s (≈6.7h), loss 2.59 → **0.72** (↓72%), eval_loss **0.55**, trainable 17.4M / 2.14B (0.81%) |
| 资源 | 峰值显存 18,214 MB, 训练样本/秒 11.811 |

> **结论**：完整 2-epoch 训练使模型在遥感任务上达到 eval_loss 0.55，验证了大规模数据训练的可行性。详细对比见 [training_comparison.md](docs/training_comparison.md)。

### 模型评估验证 (2026-08-10)

**评估数据**: VRSBench 验证集 (50条样本)

**评估环境**: Windows 11, CPU-only (Intel Core Ultra7 155H)

| 任务类型 | 样本数 | 精确匹配率 | 关键词命中率 | 空预测率 | 质量评级 |
|----------|--------|------------|--------------|----------|----------|
| 图像描述 (Captioning) | 8 | 0.0% | 100% | 0.0% | ✅ 优秀 |
| 目标检测 (Detection) | 12 | 0.0% | 100% | 0.0% | ❌ 格式问题 |
| 视觉问答 (VQA) | 26 | 84.6% | 84.6% | 0.0% | ✅ 良好 |
| 场景分类 (Scene) | 2 | 100% | 100% | 0.0% | ✅ 优秀 |
| 目标计数 (Counting) | 2 | 0.0% | 0.0% | 0.0% | ⚠️ 需改进 |

**关键发现**:
- ✅ 无空预测，模型稳定性好
- ✅ 图像描述质量高，符合遥感领域特点
- ❌ Detection任务输出格式错误，只输出label没有输出bbox
- ⚠️ Counting任务输出英文数字而不是阿拉伯数字

**详细评估报告**: [evaluation_report.md](docs/evaluation_report.md)

### INT8 量化验证 (2026-07-27)

**硬件**：Intel Core Ultra 7 155H, 32 GB RAM, CPU-only（无 NVIDIA GPU）

**方法**：PyTorch 原生动态 INT8 量化（`torch.quantization.quantize_dynamic`）

| 指标 | 基线 (BF16) | INT8 量化 | 变化 |
|---|---|---|---|
| 模型大小 | 3.97 GB | — | — |
| 参数量 | 2,127,532,032 | 315,346,944 | -85% |
| 推理速度 (CPU) | 20,576.8 ms | 14,361.7 ms | **1.43× 加速** |
| 准确率 | 37.5% | 25.0% | 66.67% 保持率 |

> **结论**：INT8 量化在 CPU 上获得 1.43× 推理加速，但精度保持率仅 66.67%。Qwen3-VL 的视觉编码器和跨模态层对量化敏感，后续需尝试 INT4 GPTQ/AWQ 或知识蒸馏方案。

## 安装

Python 3.10 或更高版本：

```powershell
python scripts/bootstrap_env.py
.\.venv\Scripts\Activate.ps1
python scripts/check_env.py
```

真实模型训练与评估：

```powershell
python scripts/bootstrap_env.py --with-model
python scripts/check_env.py --require-model
```

也可直接安装：

```powershell
python -m pip install -e ".[dev,model]"
```

实验插件拥有各自的 `requirements.txt`。主项目不会因 QLoRA 自动安装 bitsandbytes。

## Mock 推理

```powershell
python -m sat_rs_vlm.interfaces.cli infer `
  --backend mock `
  --image data/samples/demo_image.png `
  --prompt "请描述这张遥感图像中的主要地物。"
```

```powershell
uvicorn sat_rs_vlm.interfaces.http.app:app --reload --host 127.0.0.1 --port 8000
```

`GET /health` 返回 `{"status":"ok"}`，`POST /infer` 返回统一 `InferenceResult`。

## VRSBench 数据

> **注意**：以下路径为本地开发环境约定，请根据实际情况修改。

当前本地约定：

```text
模型：D:\Models\Qwen3-VL-2B-Instruct
数据：d:\project\database\VRSBench
量化模型：d:\project\sat-rs-vlm\checkpoints\quantized\int8_cpu
```

转换命令：

```powershell
python scripts/prepare_rs_instruction_data.py --config configs/data/remote_sensing_data.yaml
python scripts/convert_to_qwen3vl_format.py --config configs/data/remote_sensing_data.yaml
```

输出位于 `data/processed/`。VRSBench 没有独立 test 标注，因此 test JSONL 为空不是转换失败。详细字段和坐标规则见 [data_format.md](docs/data_format.md)。

## 稳定 LoRA 基线

> **注意**：以下路径为本地开发环境约定，请根据实际情况修改。

设置本地资产：

```powershell
$env:LOCAL_MODEL_DIR="D:\Models\Qwen3-VL-2B-Instruct"
$env:DATA_ROOT="d:\project\database\VRSBench"
$env:TRAIN_JSONL="$PWD\data\processed\qwen3vl_train.jsonl"
$env:VAL_JSONL="$PWD\data\processed\qwen3vl_val.jsonl"
```

按顺序检查：

```powershell
python scripts/validate_training_assets.py --config configs/train/qwen3vl_local_smoke.yaml
python scripts/train_qwen3vl_lora.py --config configs/train/qwen3vl_local_smoke.yaml --dry-run
python scripts/train_qwen3vl_lora.py --config configs/train/qwen3vl_local_smoke.yaml --forward-only
python scripts/train_qwen3vl_lora.py --config configs/train/qwen3vl_local_smoke.yaml
```

正式训练使用 `configs/train/qwen3vl_local.yaml`。原 LoRA 命令、配置、checkpoint 和评估方式保持不变。

评估：

```powershell
python scripts/evaluate_rs_vlm.py --config configs/eval/qwen3vl_eval.yaml
```

评估 AutoDL 4090 完整训练模型：

```powershell
python scripts/evaluate_rs_vlm.py --config configs/eval/qwen3vl_autodl_4090_eval.yaml
```

## INT8 量化

> **注意**：以下路径为本地开发环境约定，请根据实际情况修改。

CPU 环境下使用 PyTorch 原生动态 INT8 量化：

```powershell
python scripts/quantize_int8_cpu.py `
  --model-dir "D:\Models\Qwen3-VL-2B-Instruct" `
  --output-dir "checkpoints\quantized\int8_cpu" `
  --val-jsonl "data\processed\qwen3vl_val.jsonl" `
  --image-root "d:\project\database\VRSBench" `
  --num-samples 20 `
  --warmup-samples 2
```

NVIDIA GPU 环境下使用 bitsandbytes INT8 量化（需先安装 bitsandbytes）：

```powershell
python scripts/quantize_int8.py `
  --model-dir "D:\Models\Qwen3-VL-2B-Instruct" `
  --output-dir "checkpoints\quantized\int8_cuda" `
  --val-jsonl "data\processed\qwen3vl_val.jsonl" `
  --image-root "d:\project\database\VRSBench"
```

## 外部实验插件

默认本地插件包位于：

```text
.local_plugins/sat-rs-vlm-local-plugins/
```

该目录已加入 `.gitignore`，不会被主项目导入或自动扫描。显式列出和验证：

```powershell
python scripts/list_external_plugins.py `
  --plugin-root .local_plugins/sat-rs-vlm-local-plugins `
  --validate

python scripts/validate_external_plugin.py `
  --plugin-root .local_plugins/sat-rs-vlm-local-plugins `
  --strategy qlora
```

先做依赖检查，再做 dry-run：

```powershell
python scripts/run_external_strategy.py `
  --plugin-root .local_plugins/sat-rs-vlm-local-plugins `
  --strategy qlora `
  --check-only

python scripts/run_external_strategy.py `
  --plugin-root .local_plugins/sat-rs-vlm-local-plugins `
  --strategy dora `
  --config .local_plugins/sat-rs-vlm-local-plugins/plugins/dora/configs/smoke.yaml `
  --dry-run
```

默认不安装依赖、不联网。`--install-missing` 只安装当前插件缺失且不冲突的依赖；离线安装还需提供 wheel 目录。未知策略、API 不兼容、模块未匹配或依赖冲突都会明确失败，绝不回退到 LoRA。

完整说明见 [external_plugins.md](docs/external_plugins.md)。本地插件包中的 `MIGRATION_REPORT.md` 记录了原实验文件与新位置的映射。

## 测试与质量

```powershell
python -m pytest -q
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
python -m mypy src
```

默认测试不联网、不加载真实 Qwen3-VL、不要求 GPU，也不扫描本地插件目录。

## 项目边界

- `domain/`：任务、输入和统一结果模型。
- `application/`：推理用例编排。
- `models/`：Mock/HuggingFace 引擎与可靠性模块。
- `data/`：遥感数据集、VRSBench 转换和 Qwen3-VL collator。
- `training/`：稳定 LoRA 需要的配置、冻结和通用工具。
- `plugins/`：外部插件公开 API、显式发现、依赖和运行服务。
- `interfaces/`：CLI 与 HTTP 协议适配。
- `scripts/`：训练、评估、量化等脚本（含 `quantize_int8_cpu.py`）。

## 文档

- [使用说明](docs/usage_guide.md)
- [LoRA 训练说明](docs/training_qwen3vl.md)
- [外部插件说明](docs/external_plugins.md)
- [数据格式](docs/data_format.md)
- [本地 smoke 清单](docs/local_smoke_train_checklist.md)
- [实验记录](docs/experiment_log.md)
