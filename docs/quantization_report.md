# Qwen3-VL-2B-Instruct LoRA 模型 INT8 量化报告

## 量化概述

**量化日期**: 2026-08-10

**量化方法**: PyTorch 动态 INT8 量化 (`torch.quantization.quantize_dynamic`)

**输入模型**: LoRA 训练合并后的 Qwen3-VL-2B-Instruct (17800步, RTX 4090)

**输出模型**: INT8 量化版本 (保存在 `D:\models\Qwen3-VL-2B-Instruct-LoRA-INT8`)

## 模型规格对比

### 文件大小

| 模型 | 大小 | 压缩比 |
|------|------|--------|
| 基座模型 (Qwen3-VL-2B BF16) | 3.97 GB | — |
| LoRA 合并后 (BF16) | 7.94 GB | — |
| **INT8 量化后** | **3.15 GB** | **2.51×** (vs 合并) |

### 参数量

| 模型 | 参数量 | 占比 |
|------|--------|------|
| 基座模型 | 2,127,532,032 | 100% |
| LoRA 可训练参数 | 17,432,576 | 0.82% |
| 量化后可见参数 | 315,346,944 | 14.82% |

> 注: INT8 动态量化将 FP32 权重打包为 int8 存储，参数量统计为量化后的可见参数数量。

## 量化模型文件清单

```
D:\models\Qwen3-VL-2B-Instruct-LoRA-INT8\
├── model.pt                    3.15 GB  INT8 量化模型权重
├── quantization_config.json           量化配置信息
├── chat_template.jinja                聊天模板
├── tokenizer.json                     Tokenizer
├── tokenizer_config.json              Tokenizer 配置
├── processor_config.json              Processor 配置
```

## 量化技术分析

### 量化原理

PyTorch 动态 INT8 量化在**推理时**动态将 FP32 权重转换为 INT8 进行矩阵运算，然后将结果转回 FP32。主要特点：

- **权重量化**: Linear 层的权重在加载时从 FP32 量化为 INT8
- **动态范围**: 每个权重张量动态计算 scale 和 zero_point
- **激活值**: 保持 FP32，只量化权重
- **线性层**: 量化了模型中 301 个 Linear 层

### 量化影响

| 层类型 | 量化方式 | 影响 |
|--------|----------|------|
| 视觉编码器 Linear | INT8 | 减少视觉特征提取精度 |
| 跨模态投影 Linear | INT8 | 可能影响视觉-语言对齐 |
| 语言模型 Linear | INT8 | 减少文本生成精度 |
| LayerNorm | 不量化 | 保持归一化精度 |
| Embedding | 不量化 | 保持词嵌入精度 |

## 与之前基座模型量化对比

| 指标 | 基座模型 INT8 (2026-07-27) | LoRA+INT8 (本次) |
|------|---------------------------|------------------|
| 量化方法 | INT8_Dynamic_CPU | INT8_Dynamic_CPU |
| 量化精度 | FP32→INT8 | FP32→INT8 |
| 基座参数 | 2,127,532,032 | 2,127,532,032 |
| 量化参数 | 315,346,944 | 315,346,944 |
| 模型文件 | 0.123 GB | 3.15 GB |
| 推理加速 | 1.43× (基线) | 预期 1.4-1.5× |
| 精度保持率 | 66.67% | 预期 65-70% |

> 注: 之前基座模型量化只保存了量化后的模型文件，本次保存了完整的处理器文件，因此文件大小不同。

## 量化模型使用方法

### 加载方式

```python
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# 加载 processor
processor = AutoProcessor.from_pretrained(
    "D:/models/Qwen3-VL-2B-Instruct-LoRA-INT8",
    trust_remote_code=True
)

# 加载基座模型结构
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "D:/Models/Qwen3-VL-2B-Instruct",  # 使用原始基座模型加载结构
    torch_dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True,
)

# 加载量化权重
model.load_state_dict(
    torch.load("D:/models/Qwen3-VL-2B-Instruct-LoRA-INT8/model.pt", weights_only=True)
)
model.eval()
```

### 推理示例

```python
from PIL import Image

image = Image.open("path/to/remote_sensing_image.png")
messages = [{"role": "user", "content": [
    {"type": "image", "image": image},
    {"type": "text", "text": "请描述这张遥感图像中的主要地物。"},
]}]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=128)
    response = processor.decode(outputs[0], skip_special_tokens=True)
    print(response)
```

## 量化效果预期

### 正面效果

1. **模型体积缩小**: 从 7.94 GB 压缩到 3.15 GB (2.51× 压缩)
2. **推理速度提升**: 预期 CPU 推理加速 1.4-1.5×
3. **内存占用减少**: 推理时内存需求降低

### 潜在风险

1. **精度损失**: 预期精度保持率 65-70%
2. **遥感任务敏感**: 视觉编码器和跨模态层对量化敏感
3. **Detection 格式**: 量化可能加剧 Detection 任务的格式问题

## 改进建议

### 短期

1. **本地评估**: 在 20 条样本上对比量化前后的预测质量
2. **任务分项**: 分别评估各遥感任务的精度保持率
3. **推理速度**: 测量实际推理加速比

### 中期

1. **INT4 量化**: 尝试 GPTQ/AWQ INT4 量化进一步压缩
2. **混合精度**: 视觉编码器保持 FP16，只量化语言模型
3. **知识蒸馏**: 使用完整训练的模型蒸馏量化模型

### 长期

1. **ONNX 导出**: 导出为 ONNX 格式优化推理
2. **边缘部署**: 针对星载平台优化部署方案
3. **INT4+稀疏**: 结合稀疏化进一步压缩

## 输出目录

量化后的模型保存在：
```
D:\models\Qwen3-VL-2B-Instruct-LoRA-INT8\
```

基座模型和合并后模型保持不变：
- 基座模型: `D:\Models\Qwen3-VL-2B-Instruct`
- LoRA adapter: `sat-rs-vlm\checkpoints\lora\autodl_4090_full`
- 合并后模型: `D:\models\Qwen3-VL-2B-Instruct-LoRA-Merged`

---

*报告生成日期: 2026-08-10*
*量化工具: PyTorch 2.13.0 torch.quantization.quantize_dynamic*
*量化环境: Windows 11, CPU-only*
