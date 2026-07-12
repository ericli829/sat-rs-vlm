# sat-rs-vlm

`sat-rs-vlm` 第一版是一个面向受限星载算力场景的遥感视觉语言模型工程框架。它把
自然语言任务路由、统一推理结果、VRSBench 数据转换、本地 Qwen3-VL LoRA 微调、
评估和可靠性接口放在同一套可测试的 Python 工程中。

第一版的目标是建立一条可复现的本地研发链路，而不是宣称已经完成星载部署或获得
遥感 benchmark 的最终精度。

## 第一版状态

| 能力 | 当前状态 | 说明 |
| --- | --- | --- |
| Mock CLI / HTTP 推理 | 可用 | 支持任务路由和统一 JSON 结果，不读取真实图像内容。 |
| HuggingFace 推理 | 可用 | 支持 Qwen3-VL 多模态聊天模板和本地模型加载。 |
| VRSBench 转换 | 可用 | 展开 caption、referring/detection、counting、scene classification、VQA。 |
| 坐标处理 | 可用 | VRSBench 检测框裁剪并规范到 `[0,1]`，保留原始坐标 metadata。 |
| 本地 Qwen3-VL-2B LoRA smoke | 已验证 | 使用 `Qwen3-VL-2B-Instruct`、4 个训练样本、2 个 step。 |
| 本地 adapter 评估链路 | 可用 | 使用 generation prompt，不把标准答案送入模型。 |
| 完整 VRSBench 训练与正式指标 | 待执行 | 当前 smoke adapter 只用于验证链路，不代表模型精度。 |
| 量化、剪枝、蒸馏、故障恢复 | 接口预留 | 尚未形成完整训练或部署流程。 |

## 当前资产

本工作区已经配置为使用：

```text
本地模型：D:\Desktop\tzb-2026\Qwen3-VL-2B-Instruct
VRSBench：F:\VIT-data\VRSBench
```

VRSBench 转换后的内部数据写入 `data/processed/`：

```text
rs_train.jsonl              142,390 条内部指令样本
rs_val.jsonl                 62,918 条内部指令样本
qwen3vl_train.jsonl         142,390 条 Qwen3-VL messages 样本
qwen3vl_val.jsonl            62,918 条 Qwen3-VL messages 样本
```

VRSBench 没有独立 test 标注，因此 test JSONL 为空；不要把它误当作转换失败。

## 架构

```text
CLI / HTTP
    -> InferenceService
        -> TaskRouter
        -> BaseVLMEngine
            -> MockVLMEngine / HuggingFaceVLMEngine

VRSBench 原始标注
    -> rs_*.jsonl
    -> qwen3vl_*.jsonl
    -> Qwen3VLDataCollator
    -> LoRA 训练 / generation 评估
```

业务逻辑集中在 `application/` 与 `domain/`；模型加载在 `models/`；数据适配在 `data/`；
CLI 与 HTTP 只承担协议适配。

## 安装

要求 Python 3.10 或更高版本。基础开发环境：

```powershell
python scripts/bootstrap_env.py
.\.venv\Scripts\Activate.ps1
python scripts/check_env.py
```

真实模型、训练和评估需要可选依赖：

```powershell
python scripts/bootstrap_env.py --with-model
python scripts/check_env.py --require-model
```

Windows NVIDIA GPU 建议从 PyTorch 官方 CUDA wheel index 安装 Torch；完整命令与
`c10.dll` 排障见 [使用说明](docs/usage_guide.md)。

## 快速开始

Mock 推理不需要 GPU：

```powershell
python -m sat_rs_vlm.interfaces.cli infer `
  --backend mock `
  --image data/samples/demo_image.png `
  --prompt "请描述这张遥感图像中的主要地物。"
```

启动 HTTP 服务：

```powershell
uvicorn sat_rs_vlm.interfaces.http.app:app --reload --host 127.0.0.1 --port 8000
```

健康检查返回：

```json
{"status":"ok"}
```

## VRSBench 数据准备

默认配置 [remote_sensing_data.yaml](configs/data/remote_sensing_data.yaml) 指向当前
VRSBench 路径。完整转换：

```powershell
python scripts/prepare_rs_instruction_data.py `
  --config configs/data/remote_sensing_data.yaml

python scripts/convert_to_qwen3vl_format.py `
  --config configs/data/remote_sensing_data.yaml
```

只验证少量图像：

```powershell
python scripts/prepare_rs_instruction_data.py `
  --config configs/data/remote_sensing_data.yaml `
  --max-images-per-split 2
```

这个 smoke 命令会覆盖 `data/processed/` 中同名真实数据。示例数据使用独立配置
`configs/data/sample_data.yaml`，输出到 `data/processed/sample/`，不会覆盖 VRSBench。

## 本地 Qwen3-VL-2B 训练

先声明本地训练路径：

```powershell
$env:LOCAL_MODEL_DIR="D:\Desktop\tzb-2026\Qwen3-VL-2B-Instruct"
$env:DATA_ROOT="F:\VIT-data\VRSBench"
$env:TRAIN_JSONL="D:\Desktop\tzb-2026\sat-rs-vlm\data\processed\qwen3vl_train.jsonl"
$env:VAL_JSONL="D:\Desktop\tzb-2026\sat-rs-vlm\data\processed\qwen3vl_val.jsonl"
```

推荐按这个顺序运行：

```powershell
python scripts/validate_training_assets.py `
  --config configs/train/qwen3vl_local_smoke.yaml

python scripts/train_qwen3vl_lora.py `
  --config configs/train/qwen3vl_local_smoke.yaml `
  --dry-run

python scripts/train_qwen3vl_lora.py `
  --config configs/train/qwen3vl_local_smoke.yaml `
  --forward-only

python scripts/run_local_smoke_train.py `
  --model-dir "$env:LOCAL_MODEL_DIR" `
  --train-file "$env:TRAIN_JSONL" `
  --val-file "$env:VAL_JSONL" `
  --image-root "$env:DATA_ROOT"
```

正式本地 LoRA 使用 `configs/train/qwen3vl_local.yaml`。`configs/train/qwen3vl_lora*.yaml`
保留为远程 8B QLoRA 模板，需要网络、较大显存和额外验证，不是当前默认路径。

## 评估

默认评估配置使用已生成的两步 smoke adapter，因此只用于验证评估流程：

```powershell
python scripts/evaluate_rs_vlm.py --config configs/eval/qwen3vl_eval.yaml
```

结果写入：

```text
reports/eval/qwen3vl_eval_summary.json
reports/eval/qwen3vl_predictions.jsonl
```

评估会校验 adapter 文件、使用不含 assistant 标准答案的 generation prompt，并报告
`empty_prediction_rate`。正式训练得到 adapter 后，将 `adapter_path` 改为其输出目录。

## 测试与质量

```powershell
python -m pytest -q
python -m ruff check src tests scripts
python -m mypy src
```

当前测试不下载模型、不执行完整 VRSBench 训练，也不替换真实 processed 数据。

## 文档

- [使用说明](docs/usage_guide.md)：Windows 环境、CLI、HTTP、训练、评估和排障。
- [训练说明](docs/training_qwen3vl.md)：本地 Qwen3-VL-2B LoRA 工作流和产物说明。
- [数据格式](docs/data_format.md)：内部 JSONL、Qwen messages、VRSBench 映射和坐标规则。
- [Smoke Checklist](docs/local_smoke_train_checklist.md)：最小模型验证清单。
- [实验记录](docs/experiment_log.md)：当前第一版验证记录和待完成事项。

## 后续范围

第一版后的重点是正式 VRSBench LoRA 实验、任务专用指标（mAP、CIDEr、BLEU/ROUGE 等）、
量化与蒸馏、模型校验和、bit flip 注入，以及面向星载的故障检测与恢复策略。
