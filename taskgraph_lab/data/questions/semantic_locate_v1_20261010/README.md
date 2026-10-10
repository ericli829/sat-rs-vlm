# 语义定位问题试验集：纯英文修订版

按现有 Planner 英文训练语料的使用方式，使用本地英文模板生成问题池。共 80 条不重复的英文问题，未调用模型或下载图像。此修订版取代此前误生成的中英双语版本；中文翻译不再输出，也不计入样本量。

## 文件

- `questions_en.txt`：80 条英文问题，每行一条。
- `questions.jsonl`：全部问题，字段为 `id`、`question`、`language`、`family`，其中 `language` 全部为 `en`。
- `manifest.json`：生成参数、数量统计、来源说明和 SHA256。
- `validation.json`：实际落盘数据的一致性校验结果。

`family` 仅表示生成模板类别，不是正确算子或正确解题路径标签。

## 覆盖范围

每类 10 条英文问题。

| 模板类别 | 英文例子 |
| --- | --- |
| 简单目标属性定位 | Find the white square building in the image. |
| 双背景空间条件 | Find the building with a row of trees to its right and grass below it in the image. |
| 目标属性与背景条件 | Find the white square building with a row of trees to its right and grass below it in the image. |
| 局部区域内语义定位 | Within the lower-right part of the image, find the building with a row of trees to its right and grass below it. |
| 明确单体参照物与背景条件 | Using the topmost warehouse in the image as the reference, find the building below it with a row of trees to the target's right. |
| 多目标语义筛选 | Find all buildings with a row of trees to their right and grass below them in the image. |
| 确定性极值对照 | Find the topmost building in the image. |
| 明确排序轴的序数对照 | Order all buildings in the image from left to right and find the third one. |

目标词表覆盖建筑、仓库、储罐、游泳池、网球场、停车场、运动场、温室、船只、车辆。背景词表包括树木、道路、河流、绿地、草地、小路、树林、铺装空地。建筑例子有意保持条件一致，便于跨模板比较。

## 生成方式

在仓库根目录运行以下 PowerShell 命令，可生成相同内容的新副本（指定一个尚未生成过的新目录）：

```powershell
& 'C:\Users\Ericoneabc\AppData\Local\Microsoft\WindowsApps\python.exe' -m taskgraph_lab.tools.generate_semantic_locate_questions --output-dir taskgraph_lab/data/questions/semantic_locate_v1_copy --seed 20261010 --per-family 10
```

生成器为 `taskgraph_lab/tools/generate_semantic_locate_questions.py`，仅依赖 Python 标准库。固定种子可复现；已有问题文件不会被覆盖。试验词表限定 `--per-family` 为 1–10，每个模板类别最多生成 10 条英文问题。更换随机种子会改变部分条件和顺序，但不保证不同批次之间没有重复；合并时仍需去重。

## 使用边界

这批数据是待标注的问题池，没有图像、答案、bbox、TaskGraph 或 DSL。文本检查不能证明某个真实图像中存在目标，也不能证明单数问题一定有唯一匹配。

当前文件不是训练加载器要求的 `NormalizedSample`，不能直接当作完整 Planner SFT 数据。后续若用于监督训练，需要另行提供经过审核的 TaskGraph/DSL 标签及适配流程。此轮未修改训练数据集或启动训练，未验证真实模型准确率。

该试验集的句式和类别仍较有限。后续划分训练集、验证集时，应将同场景变体放在同一分组，并按模板族或改写来源控制泄漏，避免把模板记忆误当作泛化能力。
