# 量化敏感度测试报告

**测试日期**: 2026-08-10 14:44:40

**测试模型**: D:/Models/Qwen3-VL-2B-Instruct

**测试样本数**: 5

**测试方法**: component_wise

## 基线结果

| 指标 | 数值 |
|------|------|
| Keyword Hit Rate | 40.00% |
| Exact Match Rate | 0.00% |
| Latency | 12630 ms |

## 各层敏感度结果

| 层名称 | 类型 | 参数量 | Keyword Hit Δ | 敏感度 |
|--------|------|--------|---------------|--------|
| full_model... | Component | 2,031,739,904 | -20.00% | 0.1400 |
| language_model_only... | Component | 1,720,574,976 | +0.00% | 0.0228 |

## 高敏感层 (Top 10)

1. `full_model`
2. `language_model_only`

## 建议

- 发现 1 个高敏感层，建议保持 FP32 精度
