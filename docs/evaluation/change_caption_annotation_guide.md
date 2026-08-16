# LEVIR-CC 变化描述语义盲审标注规范

## 1. 标注目的

判断模型生成的英文 Caption **本身表达了什么**，用于验证自动文本判定器是否正确理解 Caption。该标注不评价图像预测是否正确，也不查看原图、参考答案或 `changeflag`。

## 2. 标签定义

| 标签 | 含义 | 典型情况 |
|---|---|---|
| `0` | Caption 没有表达建筑或永久结构的实质变化 | 明确无变化；只有车辆、光照、阴影、色彩、清晰度、裁剪、视角、天气等临时或成像差异 |
| `1` | Caption 表达了建筑、道路或其他永久结构的实质变化 | 新建、拆除、扩建、缩小、替换、结构改变、道路出现或消失 |
| `U` | 仅凭 Caption 无法确定 | 语义矛盾、残缺，或明确表示证据不确定；不要因句子较长或语法一般就使用 `U` |

LEVIR-CC 的目标口径是建筑/永久结构变化。单独出现或消失的车辆不算目标变化；单纯亮度、阴影、季节、分辨率或拍摄角度变化也不算目标变化。

## 3. 判定原则

1. 只读 `caption`，不要打开图片，也不要搜索样本 ID。
2. Caption 只要明确提到一处永久结构变化，即标 `1`，即使其余区域保持不变。
3. “a house is built”表示两时相之间发生新建，标 `1`，不是静态地说画面里有房屋。
4. “mostly similar”“almost identical”后若没有具体永久结构变化，标 `0`；若后面接“but a building was demolished”，标 `1`。
5. 不根据常识补充 Caption 没有表达的内容。
6. 两位标注者独立完成，不讨论答案；分歧由第三次裁决处理。

## 4. 示例

| Caption | 标签 | 原因 |
|---|---:|---|
| `No change has occurred.` | 0 | 明确无变化 |
| `Only a vehicle is now visible on the road.` | 0 | 只有临时车辆差异 |
| `The lighting is brighter but the structures remain the same.` | 0 | 只有成像差异 |
| `A villa is built in the forest.` | 1 | 新建永久结构 |
| `No buildings changed, but a new road appeared.` | 1 | 道路属于永久结构变化 |
| `The scenes are similar, although two houses were demolished.` | 1 | 明确拆除房屋 |
| `It is unclear whether the structure changed.` | U | Caption 明确表示无法判断 |

## 5. 文件填写

优先填写 `annotator_a.csv` 或 `annotator_b.csv` 中的：

- `human_caption_semantic_label`：只能填 `0`、`1` 或 `U`；
- `human_note`：可选，仅记录产生歧义的短原因。

CSV 使用 UTF-8 BOM，Windows Excel 可直接打开。完成后保留原文件名。答案键 `answer_key.json` 由评测人员保管，在两位标注者完成之前不得打开。
