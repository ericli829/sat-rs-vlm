# 模型评测与绘图第一阶段结果

评测契约：v1.5。以下指标均为内部评测结果。

## 数据范围

| 对象 | 样本数 |
| --- | ---: |
| 原始VRSBench基线 | 62,918 |
| 加入LEVIR-CC回放后的VRSBench结果 | 62,918 |
| 回放后LEVIR-CC结果 | 1,333 |

两份VRSBench结果已完成ID、任务类型和参考答案的逐样本一致性检查。

## VRSBench结果摘要

| 任务 | 指标 | 基线 | 回放后 | 差值 |
| --- | --- | ---: | ---: | ---: |
| Grounding | Mean IoU | 0.650716 | 0.654847 | +0.004131 |
| Grounding | Mean GIoU | 0.568744 | 0.574045 | +0.005300 |
| Grounding | Acc@0.5 | 0.749087 | 0.755340 | +0.006253 |
| Counting | Exact Accuracy | 0.617354 | 0.621432 | +0.004078 |
| Counting | MAE | 0.625836 | 0.631382 | +0.005546 |
| VQA | Normalized Accuracy | 0.739394 | 0.738033 | -0.001361 |
| Caption | BLEU-1 Approx | 0.530951 | 0.536370 | +0.005419 |
| Caption | ROUGE-L Approx | 0.384993 | 0.386768 | +0.001775 |
| Caption | chrF Approx | 0.462214 | 0.459758 | -0.002455 |
| Caption | 单参考CIDEr-D Approx | 0.524119 | 0.510352 | -0.013767 |

配对Bootstrap结果表明Grounding获得稳定小幅改善；Counting、VQA和场景分类的主要区间跨0；Caption指标出现分化，不能表述为描述能力全面提升。

## LEVIR-CC结果摘要

| 指标 | 数值 |
| --- | ---: |
| Accuracy | 0.895724 |
| Balanced Accuracy | 0.895785 |
| Change Precision | 0.973118 |
| Change Recall | 0.814093 |
| Change F1 | 0.886531 |
| MCC | 0.802272 |
| Cohen's Kappa | 0.791473 |
| TP/TN/FP/FN | 543 / 651 / 15 / 124 |

主要问题是变化召回率：误报仅15条，但漏掉124条真实变化样本。

## 当前限制

- 缺少回放训练前模型在同一LEVIR-CC验证集上的预测，暂不能给出变化能力的严格增量。
- LEVIR-CC当前每条预测只有一个参考描述，不能与五参考论文指标直接比较。
- Grounding缺少 `is_unique` 时不能进行官方Unique/Non-Unique分组。
- VQA缺少原始问题文本时不能实现问题感知语义Judge。
- 当前高级语义指标不是图像级事实正确率。

## 下一阶段

1. 分析LEVIR-CC的124个False Negative并建立错误类型清单。
2. 补跑回放训练前模型的同集LEVIR-CC预测并进行配对比较。
3. 接入压缩组INT8/INT4结果，绘制质量、显存和速度Pareto图。
4. 接入容错组fault/recovered结果，统一比较质量下降和恢复效果。
5. 在具备图像路径和完整标注后建立图像级事实正确率协议。
