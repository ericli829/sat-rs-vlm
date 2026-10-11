# LEVIR-CC D25 空间理解实验：预先固定的设计

本设计在读取本轮模型输出之前固定。仅使用现有 D25 双图、route_b_stage2_batch8 adapter 和本地 Qwen3-VL-2B-Instruct。人工标签仅用于结果计分，不参与提示词、图像标记或方位映射规则。没有使用 MCI。

## 四组

| 组 | 输入 | 模型输出 | 最终结果 |
|---|---|---|---|
| A | 原始 BEFORE/AFTER 双图 | 受约束的九标签之一 | 直接标签 |
| B | 同尺寸双图，叠加薄网格和小编号 | 同一九标签约束 | 直接标签 |
| C | 原始双图 | Change 与 Location 自然语言字段 | 固定保守规则映射 |
| D | 网格编号双图 | 与 C 相同的字段 | 相同映射规则 |

A 的提示词与上一轮有效的无 MCI 控制组一致。B 的编号是按行从左上至右下的 1–9；在两张 256×256 图中使用相同位置。编号没有实心底色，细线半透明，不扩大画布或改变图像分辨率；视觉检查样本为 RSU_D_0033。标记仍会改变少量图像像素，覆盖比例在 input_manifest.json 中记录。

## 生成与输入检查

四组共用 BF16、CUDA、SDPA、do_sample=False、num_beams=5、length_penalty=0、use_cache=True、max_new_tokens=96。A/B 有两 Token 标签停止条件；C/D 在 EOS 或共同上限时停止。九个规范标签的决策路径均长两 Token，单 Token 标签补 EOS；beam finalize 若附加 EOS，单独验证并排除出决策路径。禁止前导空格路径。

每次都替换实际 messages 中的用户文本，并保存完整 chat prompt；同时断言 processor 编码了两张图。输入不截断，超过共同 1024 Token 预算时报错。真实 processor 预检的 A/B/C/D 输入长度分别为 200/282/281/295 Token，均为两张图。权重 SHA256 在运行输出中保存。

## 映射规则

仅消费显式 Location 字段，支持常见英语位置短语、连字符、下划线与 Markdown 字段。最长短语优先，避免 top center 被误分解为 top 和 center。多个冲突格位、left side 等无法确定行位置的描述、缺失字段和 UNCLEAR 均保留为未决，不猜测。编号本身不替代自然语言方位。规则先通过 15 个固定方位用例，运行后不按人工标签调参。

## 评估与后续复核

固定报告 23 道明确样本准确率、两个 unclear 的独立输出及 25 题严格计分（unclear 计入分母）。同时保存原始输出、Token ID、Location 提取和映射依据、提示词、计时、九方向分布、混淆矩阵、相对 A 的错变对/对变错。相邻标签距离只作为边界错误线索，不等同于变化目标确实位于格线附近。

模型输出完成后，结合原始图像和 C/D 描述复核变化内容识别、方位理解及映射错误。可计算整体水平/垂直翻转等方向变换作为诊断，但不修改正式计分，也不把事后最优映射作为部署规则。微小样本差异不作为稳定收益证据。

## 预检记录

- Python：C:\Users\Ericoneabc\AppData\Local\Microsoft\WindowsApps\python.exe。
- unittest 命令：`python -m unittest discover -s scripts/experiments -p test_d25_spatial_understanding.py -v`，8 项通过。
- Python 编译检查及 PowerShell 启动器解析检查通过。
- 真实 processor、标签 Token 路径与映射检查记录在 preflight.json，脚本 SHA256 已冻结；启动器拒绝在脚本与预检不一致时启动。
- 启动：`& .\scripts\experiments\launch_d25_spatial_understanding.ps1`。
- 启动前预检：`& .\scripts\experiments\launch_d25_spatial_understanding.ps1 -PrepareOnly`。

这是独立实验，不修改生产推理路径或模型权重。自动报告完成后，仍需结合样本图像做内容识别与空间误差的定性复核。
