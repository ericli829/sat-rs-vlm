# sat-rs-vlm

`sat-rs-vlm` 是一个面向受限星载算力场景的多模态遥感视觉语言模型工程框架，支持推理、数据准备、训练、评估和环境自动化。项目当前包含可运行的第一阶段骨架、第二阶段环境脚手架、模型工厂接入、可选的 HuggingFace 模型集成，以及轻量级推理性能分析。

完整的中文操作流程、Windows 命令、本地 Qwen3-VL 训练步骤和故障排查见
[使用说明](docs/usage_guide.md)。

## 功能

- 面向检测、场景分类、分割、变化检测、计数、描述和 VQA 的自然语言任务路由。
- 清晰分层的接口层、应用层、领域层、模型层、数据层和基础设施层。
- 使用 YAML 配置并映射到 Pydantic 配置模型。
- 提供 Typer CLI 和 FastAPI HTTP API。
- 提供比特翻转模拟和校验和验证等可靠性扩展点。
- 为 LoRA 微调、蒸馏、剪枝、量化和星载故障恢复预留接口。
- 默认依赖保持轻量，真实模型依赖放在 `[model]` 可选扩展中，便于本地开发和 CI 使用。

## 目录结构

```text
configs/                 YAML 配置
examples/                提示词示例和演示输入
src/sat_rs_vlm/
  interfaces/            CLI 和 HTTP 适配层
  application/           用例服务
  domain/                任务、实体、结果和路由模型
  models/                VLM 引擎接口与实现
  data/                  数据集抽象和注册表
  infrastructure/        配置、日志、设备和随机种子工具
  utils/                 通用辅助工具
tests/                   单元测试和集成测试
```

## 安装

```bash
python scripts/bootstrap_env.py
pip install -e ".[dev]"
```

如果你需要 HuggingFace 后端，再安装可选的模型依赖：

```bash
python scripts/bootstrap_env.py --with-model
pip install -e ".[model]"
pip install -e ".[dev,model]"
```

真实模型栈包含 `torch`、`transformers`、`peft`、`accelerate` 等包，因此不会默认安装。这样可以让 mock 推理、API 测试和本地 CI 在没有 GPU 或大模型运行时的机器上正常工作。

检查当前环境：

```bash
python scripts/check_env.py
make check-env
```

## 运行

```bash
python -m sat_rs_vlm.interfaces.cli config
python -m sat_rs_vlm.interfaces.cli infer --image examples/demo_image.jpg --prompt "请描述这张遥感图像中的主要地物。"
python -m sat_rs_vlm.interfaces.cli infer --backend mock --image examples/demo_image.jpg --prompt "请检测图像中的飞机。"
uvicorn sat_rs_vlm.interfaces.http.app:app --reload --host 127.0.0.1 --port 8000
```

后端选择由 `configs/default.yaml` 控制：

```yaml
model:
  backend: mock
  model_id: ""
```

如果要使用真实的 HuggingFace 兼容 VLM，先安装 `[model]`，然后设置 `model.backend: huggingface`，填写 `model.model_id`，也可以在 CLI 中直接覆盖：

```bash
python -m sat_rs_vlm.interfaces.cli infer \
  --backend huggingface \
  --model-id your-org/your-vlm \
  --image examples/demo_image.jpg \
  --prompt "请描述这张遥感图像中的主要地物。"
```

HTTP 示例：

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/infer \
  -H "Content-Type: application/json" \
  -d '{"image_path":"examples/demo_image.jpg","prompt":"请检测图像中的机场跑道。"}'
```

## 测试与质量

```bash
pytest -q
make test
make lint
make format
```

## 第二阶段

第二阶段增加了可复现的 `.venv` 引导、环境诊断、模型工厂、延迟加载的 `HuggingFaceVLMEngine`，以及 `InferenceProfiler`。CLI 和 HTTP 层现在从配置构建 `InferenceService`，而不是直接实例化具体模型类，这样业务层只依赖 `BaseVLMEngine` 抽象。

## 第三阶段：Qwen3-VL 遥感指令微调

第三阶段加入了配置驱动的 Qwen3-VL LoRA/QLoRA 训练流水线。当前临时基座模型是 `Qwen/Qwen3-VL-8B-Instruct`，默认训练使用 QLoRA/LoRA，并冻结视觉编码器，以降低显存压力并减少视觉骨干遗忘。

准备样例或真实数据转换后的内部 JSONL：

```bash
python scripts/prepare_rs_instruction_data.py \
  --config configs/data/remote_sensing_data.yaml
```

将内部 `rs_*.jsonl` 转换为 Qwen3-VL 的聊天消息格式：

```bash
python scripts/convert_to_qwen3vl_format.py \
  --config configs/data/remote_sensing_data.yaml
```

安装模型依赖并运行一个 smoke training：

```bash
python scripts/bootstrap_env.py --with-model
python scripts/train_qwen3vl_lora.py \
  --config configs/train/qwen3vl_lora_smoke.yaml
```

运行完整基线配置：

```bash
python scripts/train_qwen3vl_lora.py \
  --config configs/train/qwen3vl_lora.yaml
```

多卡启动：

```bash
torchrun --nproc_per_node=4 scripts/train_qwen3vl_lora.py \
  --config configs/train/qwen3vl_lora.yaml
```

评估和合并 LoRA：

```bash
python scripts/evaluate_rs_vlm.py \
  --config configs/eval/qwen3vl_eval.yaml

python scripts/merge_lora.py \
  --base-model Qwen/Qwen3-VL-8B-Instruct \
  --adapter checkpoints/qwen3vl-rs-lora/best \
  --output checkpoints/qwen3vl-rs-merged
```

默认的 `pytest -q` 不会下载 Qwen3-VL，不需要 GPU，也不会执行真实训练。真实模型测试应通过 `RUN_MODEL_TRAINING_TESTS=1` 之类的环境变量显式开启。

## 扩展方向

真实模型集成应在 `src/sat_rs_vlm/models/base.py` 中实现 `BaseVLMEngine`，把模型加载和设备放置放到模型层内部，同时保持 CLI 和 HTTP 适配层不变。第三阶段后续将重点推进遥感数据集接入、LoRA/QLoRA 微调、评估流水线、量化部署、剪枝、蒸馏，以及比特翻转注入、校验和、看门狗恢复和降级模式等星载可靠性能力。
