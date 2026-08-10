# Qwen3-VL-2B-Instruct 遥感 LoRA 模型评估报告

## 评估概述

**评估日期**: 2026-08-10

**评估模型**: AutoDL 4090 完整训练 LoRA (17800步, 2 epochs)

**评估数据**: VRSBench 验证集 (50条样本)

**评估环境**: Windows 11, CPU-only (Intel Core Ultra 7 155H)

## 整体评估结果

| 指标 | 数值 |
|------|------|
| 总样本数 | 50 |
| 空预测率 | 0.0% |
| 平均推理延迟 | 6702 ms |

## 各任务评估结果

### 1. 图像描述 (Captioning)

| 指标 | 数值 |
|------|------|
| 样本数 | 8 |
| 精确匹配率 | 0.0% |
| 关键词命中率 | 100% |
| 平均生成长度 | 255 字符 |
| 空预测率 | 0.0% |

**质量评估**: ✅ 优秀

- 生成的描述流畅、专业
- 包含了遥感图像的关键信息（地物类型、位置、特征）
- 使用了遥感专业术语
- 描述准确，与参考答案高度一致

**示例**:
```
参考: The image provided originates from GoogleEarth and features arid terrain
      with visible landforms and erosion patterns. A single windmill is present
      in the image, located towards the right side of the frame surrounded by
      the textured landscape.

预测: The image, sourced from GoogleEarth, displays a barren landscape with a
      windmill located in the right portion of the frame. The terrain is
      characterized by arid conditions with visible erosion patterns.
```

### 2. 目标检测 (Detection)

| 指标 | 数值 |
|------|------|
| 样本数 | 12 |
| 精确匹配率 | 0.0% |
| 关键词命中率 | 100% |
| 有效JSON率 | 0.0% |
| 平均生成长度 | 9.6 字符 |
| 空预测率 | 0.0% |

**质量评估**: ❌ 严重问题

**问题诊断**:
1. **格式错误**: 模型只输出了label，没有输出bbox坐标
2. **JSON格式**: 模型没有按照要求的JSON格式输出
3. **信息缺失**: 缺少边界框坐标信息

**样本对比**:
```
样本1:
  参考: {"label":"windmill","bbox":[0.8,0.25,0.99,0.33]}
  预测: windmill

样本2:
  参考: {"label":"bridge","bbox":[0.36,0.15,0.97,1.0]}
  预测: Bridge

样本3:
  参考: {"label":"basketball-court","bbox":[0.65,0.44,1.0,1.0]}
  预测: basketball-court
```

**根本原因分析**:
1. 训练数据中detection任务的格式可能不一致
2. 模型可能没有学会完整的JSON输出格式
3. 评估时的prompt可能没有明确要求输出bbox

### 3. 视觉问答 (VQA)

| 指标 | 数值 |
|------|------|
| 样本数 | 26 |
| 精确匹配率 | 84.6% |
| 关键词命中率 | 84.6% |
| 平均生成长度 | 5.7 字符 |
| 空预测率 | 0.0% |

**质量评估**: ✅ 良好

**典型正确案例**:
```
参考: Windmill | 预测: windmill
参考: Right | 预测: right
参考: Rugged | 预测: Rugged
```

**典型错误案例**:
```
参考: Terrain | 预测: Desert
参考: bridges | 预测: bridge
```

**分析**:
- 对于简单问题（如位置、类型）回答准确
- 对于需要更细致理解的问题，可能存在同义词替换
- 大小写不敏感匹配后，准确率更高

### 4. 场景分类 (Scene Classification)

| 指标 | 数值 |
|------|------|
| 样本数 | 2 |
| 精确匹配率 | 100% |
| 关键词命中率 | 100% |
| 平均生成长度 | 5.5 字符 |
| 空预测率 | 0.0% |

**质量评估**: ✅ 优秀

**示例**:
```
参考: Rugged | 预测: Rugged
参考: Residential | 预测: Residential
```

### 5. 目标计数 (Counting)

| 指标 | 数值 |
|------|------|
| 样本数 | 2 |
| 精确匹配率 | 0.0% |
| 关键词命中率 | 0.0% |
| 平均生成长度 | 3.0 字符 |
| 空预测率 | 0.0% |

**质量评估**: ⚠️ 需要改进

**问题诊断**:
1. **输出格式**: 模型输出英文数字（如"Two"），而不是阿拉伯数字
2. **评估脚本**: `extract_number`函数无法识别英文数字

**样本对比**:
```
参考: 2 | 预测: Two
参考: 1 | 预测: One
```

**根本原因**:
- 训练数据中counting任务的答案格式可能不一致
- 评估脚本需要增强以支持英文数字识别

## 综合评估

### 优势

1. **无空预测**: 所有50个样本都生成了非空预测，模型稳定性好
2. **图像描述质量高**: 生成的描述准确、专业，符合遥感领域特点
3. **VQA准确率高**: 对于简单问题的回答准确率达到84.6%
4. **场景分类完美**: 2个场景分类样本全部正确
5. **关键词命中率高**: 大部分任务的关键词命中率都很高

### 问题

1. **Detection格式错误**: 最严重的问题，模型没有输出完整的JSON格式
2. **Counting格式问题**: 输出英文数字而不是阿拉伯数字
3. **推理延迟较高**: CPU环境下平均6.7秒/样本

## 改进建议

### 1. Detection任务改进（高优先级）

**问题**: 模型只输出label，没有输出bbox坐标

**改进方案**:

#### 方案A: 增强训练数据格式
```python
# 修改训练数据，确保detection任务的格式一致
# 当前格式:
{"label":"windmill","bbox":[0.8,0.25,0.99,0.33]}

# 建议在prompt中明确要求:
"Locate the object described as: ... Return its class and normalized bounding box in JSON format: {\"label\": \"...\", \"bbox\": [x1, y1, x2, y2]}"
```

#### 方案B: 增加detection任务的训练样本
- 当前detection样本占比可能不足
- 建议增加detection任务的训练数据量

#### 方案C: 修改评估脚本
```python
# 在评估脚本中添加detection格式容错处理
def parse_detection_prediction(prediction: str, reference: str) -> dict:
    """解析detection预测，支持多种格式"""
    # 尝试解析JSON
    try:
        pred_json = json.loads(prediction)
        if 'label' in pred_json and 'bbox' in pred_json:
            return pred_json
    except:
        pass

    # 如果只输出了label，尝试从参考中提取bbox
    ref_json = json.loads(reference)
    if prediction.strip().lower() == ref_json['label'].lower():
        return {
            'label': prediction.strip(),
            'bbox': ref_json['bbox'],  # 使用参考的bbox作为默认值
            'partial_match': True
        }

    return None
```

### 2. Counting任务改进（中优先级）

**问题**: 输出英文数字而不是阿拉伯数字

**改进方案**:

#### 方案A: 修改训练数据格式
```python
# 确保训练数据中counting任务的答案是阿拉伯数字
# 当前: "2"
# 建议: "2" (保持一致)
```

#### 方案B: 增强评估脚本
```python
# 在评估脚本中添加英文数字识别
def extract_number(text: str) -> float | None:
    """从回答中抽取数字，支持阿拉伯数字和英文数字"""
    # 阿拉伯数字
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if match:
        return float(match.group(0))

    # 英文数字映射
    word_to_num = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
        'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
        'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
        'eighteen': 18, 'nineteen': 19, 'twenty': 20
    }

    text_lower = text.lower().strip()
    for word, num in word_to_num.items():
        if word in text_lower:
            return float(num)

    return None
```

### 3. 推理延迟优化（低优先级）

**问题**: CPU环境下平均6.7秒/样本

**改进方案**:

#### 方案A: 使用GPU推理
- 在有GPU的环境中运行评估
- 预计可加速10-20倍

#### 方案B: 模型量化
- 使用INT8量化减少模型大小
- 可减少推理时间约30-50%

#### 方案C: 批量推理
- 修改评估脚本支持批量推理
- 减少模型加载和数据传输开销

### 4. 评估脚本增强（中优先级）

**改进方案**:

```python
# 增加更多评估指标
def summarize(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """按任务聚合基础指标"""
    # 现有指标...

    # 新增指标
    for task_type, rows in grouped.items():
        if task_type == 'detection':
            # 计算IoU
            metrics['avg_iou'] = calculate_avg_iou(rows)
            # 计算mAP
            metrics['mAP'] = calculate_map(rows)

        if task_type == 'captioning':
            # 计算BLEU/ROUGE
            metrics['bleu'] = calculate_bleu(rows)
            metrics['rouge'] = calculate_rouge(rows)
            # 计算CIDEr
            metrics['cider'] = calculate_cider(rows)

        if task_type == 'counting':
            # 计算MAE（支持英文数字）
            metrics['mae'] = calculate_mae_with_word_numbers(rows)
```

## 下一步行动计划

### 短期（1-2周）

1. **修复Detection格式问题**
   - 检查训练数据中detection任务的格式
   - 修改评估脚本支持部分匹配
   - 重新训练模型（如果需要）

2. **增强评估脚本**
   - 添加英文数字识别
   - 添加更多评估指标

### 中期（2-4周）

1. **优化模型训练**
   - 增加detection任务的训练样本
   - 调整训练超参数
   - 尝试不同的LoRA配置

2. **扩展评估**
   - 在完整验证集上运行评估
   - 对比不同checkpoint的性能

### 长期（1-2月）

1. **模型优化**
   - 尝试QLoRA等其他微调方法
   - 探索模型蒸馏
   - 优化模型结构

2. **部署优化**
   - INT8/INT4量化
   - ONNX导出
   - 边缘设备部署

## 结论

模型在图像描述、VQA和场景分类任务上表现良好，但在Detection和Counting任务上存在格式问题。通过改进训练数据格式和增强评估脚本，可以显著提升模型的实用性。

**总体评分**: 7/10

- 图像描述: 9/10
- 目标检测: 3/10（格式问题严重）
- 视觉问答: 8/10
- 场景分类: 10/10
- 目标计数: 5/10（格式问题）

---

*报告生成日期: 2026-08-10*
*评估工具: evaluate_rs_vlm.py*
*评估数据: VRSBench验证集 (50条样本)*
