# 赛事 D 组 25 条变化区域 LoRA 测试结果

## 评测摘要

- 模型：Qwen3-VL-2B-Instruct + route_b_stage2_batch8 LoRA。
- 环境：云端 NVIDIA GeForce RTX 4080 SUPER，BF16，batch size 4。
- 25/25 输出均符合九宫格标签格式；模型加载 5.84 秒，推理 2.95 秒。
- 人工审核明确标签：10/23 精确区域命中（43.5%）；将 2 条 unclear 按未命中计入则为 10/25（40.0%）。
- 行匹配 14/23（60.9%），列匹配 17/23（73.9%），仅作粗粒度诊断。
- 限制：审核文件采用等分九宫格和“主要结构变化簇中心”的实验约定；赛题官方没有定义区域边界和主变化并列规则，故不是官方准确率。

## 逐题对照

| 题目 ID | 人工审核区域 | LoRA 预测 | 结果 |
|---|---|---|---|
| RSU_D_0033 | right | bottom_right | 未命中 |
| RSU_D_0058 | right | right | 命中 |
| RSU_D_0064 | top_left | top_left | 命中 |
| RSU_D_0071 | top_left | top_left | 命中 |
| RSU_D_0073 | left | bottom_left | 未命中 |
| RSU_D_0097 | right | bottom_left | 未命中 |
| RSU_D_0131 | bottom | bottom | 命中 |
| RSU_D_0139 | top_left | top_left | 命中 |
| RSU_D_0141 | top_right | top_right | 命中 |
| RSU_D_0142 | left | bottom_left | 未命中 |
| RSU_D_0145 | unclear | bottom_right | 审核不明确；不计入明确样本准确率 |
| RSU_D_0147 | unclear | bottom_right | 审核不明确；不计入明确样本准确率 |
| RSU_D_0148 | right | top_left | 未命中 |
| RSU_D_0175 | left | bottom_left | 未命中 |
| RSU_D_0179 | top_right | top_right | 命中 |
| RSU_D_0206 | left | bottom_left | 未命中 |
| RSU_D_0224 | bottom | bottom_left | 未命中 |
| RSU_D_0233 | top | top_left | 未命中 |
| RSU_D_0252 | top | top_left | 未命中 |
| RSU_D_0255 | top_right | top_right | 命中 |
| RSU_D_0277 | left | bottom_left | 未命中 |
| RSU_D_0279 | left | bottom_left | 未命中 |
| RSU_D_0282 | top | top_left | 未命中 |
| RSU_D_0329 | bottom_left | bottom_left | 命中 |
| RSU_D_0362 | bottom | bottom | 命中 |

## 预测分布

- bottom：2
- bottom_left：9
- bottom_right：3
- right：1
- top_left：7
- top_right：3

预测原始文件 SHA256：f5ecb12c6c45e50e58c4c76a7ea45775dd0d4d092f1c9acdb92e0cf6316c1506
人工审核源文件 SHA256：76507bccf9b5ac7808ec0472ea3dd51db2eb9c90d3047b02000b3cd7e46d4424
