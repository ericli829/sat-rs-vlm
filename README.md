# sat-rs-vlm

`sat-rs-vlm` 是面向卫星平台受限算力部署的遥感视觉语言模型工程框架。第一版提供统一任务领域模型、Mock/HuggingFace 推理、CLI/HTTP 接口、VRSBench 数据转换、本地 Qwen3-VL LoRA 训练与评估，以及 bit flip、checksum 等可靠性基础接口。

主仓库将已经跑通的 LoRA 作为稳定基线。QLoRA、DoRA、AdaLoRA、IA3、Partial Unfreeze、Full SFT 和 Prompt Tuning 属于实验性方法，保存在 Git 忽略的本地插件包中，只有用户显式指定插件根目录时才会被发现和加载。

## 当前能力

| 能力 | 状态 | 说明 |
|---|---|---|
| Mock CLI / HTTP 推理 | 可用 | 无 GPU、无真实模型也可运行 |
| 本地 Qwen3-VL 推理 | 可用 | 使用 HuggingFace 本地模型目录 |
| VRSBench 转换 | 可用 | caption、检测、计数、场景分类、VQA，框坐标裁剪到 `[0,1]` |
| Qwen3-VL LoRA | 已验证 | 本地数据、forward-only、2-step smoke、checkpoint 和评估链路 |
| 外部实验插件 API | 可用 | 显式发现、manifest、依赖检查、路径隔离、无静默回退 |
| 星载压缩与容错 | 接口预留 | 蒸馏、剪枝、量化、bit flip、checksum 和恢复仍需继续实现 |

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

当前本地约定：

```text
模型：D:\Desktop\tzb-2026\Qwen3-VL-2B-Instruct
数据：F:\VIT-data\VRSBench
```

转换命令：

```powershell
python scripts/prepare_rs_instruction_data.py --config configs/data/remote_sensing_data.yaml
python scripts/convert_to_qwen3vl_format.py --config configs/data/remote_sensing_data.yaml
```

输出位于 `data/processed/`。VRSBench 没有独立 test 标注，因此 test JSONL 为空不是转换失败。详细字段和坐标规则见 [data_format.md](docs/data_format.md)。

## 稳定 LoRA 基线

设置本地资产：

```powershell
$env:LOCAL_MODEL_DIR="D:\Desktop\tzb-2026\Qwen3-VL-2B-Instruct"
$env:DATA_ROOT="F:\VIT-data\VRSBench"
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

## 文档

- [使用说明](docs/usage_guide.md)
- [LoRA 训练说明](docs/training_qwen3vl.md)
- [外部插件说明](docs/external_plugins.md)
- [数据格式](docs/data_format.md)
- [本地 smoke 清单](docs/local_smoke_train_checklist.md)
- [实验记录](docs/experiment_log.md)
