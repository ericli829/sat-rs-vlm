# LEVIR-CC D 组九宫格变化定位：空间理解与方向偏置实验

## 复核结论

本轮四组共 100 次 GPU 推理已完成。原始九分类 A 为 **11/23（47.8%）**，网格辅助 B 为 **12/23（52.2%）**，只净增 1 题：纠正 4 题，同时破坏 3 题。小样本配对精确 McNemar 检验 p=1.0，没有稳定提升证据，暂不建议替换生产基线。

C/D 下表的 **0/23 是固定字段格式与映射整条流水线的结果，不能解释为模型空间能力为零**。两组全部未按要求输出 `Change:` / `Location:` 字段，初版固定映射器拒绝了全部自由句子。这暴露了评估器过于依赖字段的限制，也暴露了当前模型的指令跟随问题。为区分这两点，另行保存了不依赖人工标签的离线诊断：沿用原有方位词表，从完整句子提取唯一、明确的格位短语；拒绝“没有变化”、整行/整列方位及冲突描述，不重新推理、不改原始预测或正式分数。C 可映射 2/25，答对 **0/23**；D 可映射 6/25，答对 **4/23（17.4%）**。这项运行后补充诊断单独列出，不能作为预注册方案成绩。

最突出的问题是 **变化漏识别、主要变化目标选择和行方向偏差同时存在**。C 有 21/25、D 有 15/25 回答“没有变化/没有差异”，其中明确样本分别为 19/23、13/23。少数描述能够正确指出变化及方位，但不足以支持“模型已理解变化，只是最后标签映射错误”的结论。没有使用 MCI，没有修改模型或生产推理路径。

## 设计与有效性

四组均使用同一 Qwen3-VL-2B-Instruct 基座、同一 route_b_stage2_batch8 LoRA、25 对 BEFORE/AFTER 图像及确定性 beam 解码。A/B 直接输出受 Trie 限制的九宫格标签；C/D 输出变化内容和位置描述，再由保守规则映射到九宫格，方向含糊或冲突时保留 unresolved/ambiguous，不猜测。D 是可选的组合组。

A/B 的标签路径经 tokenizer 检查：九个标签皆是唯一的两 Token 决策路径；单 Token 标签以 EOS 补齐，不使用前导空格形式；约束 trie 两步自测通过。所有组共用 do_sample=False、beam=5、length_penalty=0、KV cache、max_new_tokens=96。A/B 的输出约束在两 Token 标签完成后停止；beam finalize 自动追加的 EOS 不参与决策。C/D 自然语言不使用标签约束，达到 EOS 或共同的 96 Token 上限时停止。

B/D 在原图上叠加半透明窄线 3×3 网格和靠近各格左上角的小编号，无实心底色，不增加画布，保持 256×256。编号和细线仍可能覆盖少量像素，具体覆盖比例保存在输入清单中，作为实验局限。25 对网格输入均单独保存。每次验证两图均经过 processor，保存实际 chat prompt；不截断文本，超过共同的 1024 输入 Token 预算会报错。

设备：NVIDIA GeForce RTX 4060 Laptop GPU，BF16，GPU：cuda:0；权重 SHA256：基座见 JSON，route-B adapter `91954847cea08e17a77a115ff13988ae82c4536b82c53cccad141626358ff2cf`。人工审核标签为 23 个明确格位、2 个 unclear；它们是实验参考标签，不是官方金标准，25 题严格计分把 unclear 计入分母。

## 汇总

| 方案 | 明确样本准确率 | 25 题严格计分 | 可映射/有效输出 | 平均生成耗时/题 |
|---|---:|---:|---:|---:|
| A 原图直接九分类 | 11/23 (47.8%) | 11/25 (44.0%) | 25/25 | 0.53s |
| B 九宫格视觉辅助 | 12/23 (52.2%) | 12/25 (48.0%) | 25/25 | 0.61s |
| C 自然语言描述＋确定性映射 | 0/23 (0.0%) | 0/25 (0.0%) | 0/25 | 2.09s |
| D 网格辅助＋自然语言描述映射 | 0/23 (0.0%) | 0/25 (0.0%) | 0/25 | 2.14s |

## 方位分布和方向偏置

人工明确标签分布：`{"bottom": 3, "bottom_left": 1, "left": 6, "right": 4, "top": 3, "top_left": 3, "top_right": 3}`。
- A 原图直接九分类：25 题预测分布 `{"bottom": 3, "bottom_left": 8, "bottom_right": 3, "right": 1, "top": 2, "top_left": 5, "top_right": 3}`；明确的 23 题中 bottom_left 预测 8 次，参考有 1 次，预测－参考差 +7。
- B 九宫格视觉辅助：25 题预测分布 `{"bottom": 1, "bottom_left": 8, "bottom_right": 1, "left": 1, "right": 2, "top": 1, "top_left": 8, "top_right": 3}`；明确的 23 题中 bottom_left 预测 7 次，参考有 1 次，预测－参考差 +6。
- C 自然语言描述＋确定性映射：25 题预测分布 `{"unresolved": 25}`；明确的 23 题中 bottom_left 预测 0 次，参考有 1 次，预测－参考差 -1。
- D 网格辅助＋自然语言描述映射：25 题预测分布 `{"unresolved": 25}`；明确的 23 题中 bottom_left 预测 0 次，参考有 1 次，预测－参考差 -1。

## 相对 A 的变化

- B 九宫格视觉辅助：错变对 4 题 (RSU_D_0033, RSU_D_0139, RSU_D_0233, RSU_D_0277)；对变错 3 题 (RSU_D_0224, RSU_D_0282, RSU_D_0362)；明确样本预测改变 9/23。
- C 自然语言描述＋确定性映射：错变对 0 题 (无)；对变错 11 题 (RSU_D_0058, RSU_D_0064, RSU_D_0071, RSU_D_0131, RSU_D_0141, RSU_D_0179, RSU_D_0224, RSU_D_0255, RSU_D_0282, RSU_D_0329, RSU_D_0362)；明确样本预测改变 23/23。
- D 网格辅助＋自然语言描述映射：错变对 0 题 (无)；对变错 11 题 (RSU_D_0058, RSU_D_0064, RSU_D_0071, RSU_D_0131, RSU_D_0141, RSU_D_0179, RSU_D_0224, RSU_D_0255, RSU_D_0282, RSU_D_0329, RSU_D_0362)；明确样本预测改变 23/23。

## 边界/邻格错误

这里只统计网格标签的相邻距离，是边界误差的待复核线索，不能据此断言真实变化目标位于格线附近。
- A 原图直接九分类：相邻格（Chebyshev 距离 1）10 题 (RSU_D_0033, RSU_D_0073, RSU_D_0139, RSU_D_0142, RSU_D_0175, RSU_D_0206, RSU_D_0233, RSU_D_0252, RSU_D_0277, RSU_D_0279)；跨两格以上 2 题 (RSU_D_0097, RSU_D_0148)；未映射明确样本 0 题。
- B 九宫格视觉辅助：相邻格（Chebyshev 距离 1）9 题 (RSU_D_0073, RSU_D_0142, RSU_D_0175, RSU_D_0206, RSU_D_0224, RSU_D_0252, RSU_D_0279, RSU_D_0282, RSU_D_0362)；跨两格以上 2 题 (RSU_D_0097, RSU_D_0148)；未映射明确样本 0 题。
- C 自然语言描述＋确定性映射：相邻格（Chebyshev 距离 1）0 题 (无)；跨两格以上 0 题 (无)；未映射明确样本 23 题。
- D 网格辅助＋自然语言描述映射：相邻格（Chebyshev 距离 1）0 题 (无)；跨两格以上 0 题 (无)；未映射明确样本 23 题。

## 两个 unclear 样本

| ID | A | B | C 映射 | D 映射 |
|---|---|---|---|---|
| RSU_D_0145 | bottom_right | bottom_right | UNRESOLVED | UNRESOLVED |
| RSU_D_0147 | bottom_right | bottom_left | UNRESOLVED | UNRESOLVED |

## 逐题对照

C/D 描述只按 `Location:` 字段映射；明确但不在九宫格单元中的方位保持未决。

| ID | 参考 | A | B | C 描述 / 映射 | D 描述 / 映射 |
|---|---|---|---|---|---|
| RSU_D_0033 | right | bottom_right | right | There is no change in the scene. → UNRESOLVED | There is no change. → UNRESOLVED |
| RSU_D_0058 | right | right | right | There is no change. → UNRESOLVED | A parking lot and a building appear on the right side of the scene. → UNRESOLVED |
| RSU_D_0064 | top_left | top_left | top_left | There is no change. → UNRESOLVED | A building appears in the upper-left corner of the scene. → UNRESOLVED |
| RSU_D_0071 | top_left | top_left | top_left | There is no change. → UNRESOLVED | A road appears in the upper-right corner of the scene. → UNRESOLVED |
| RSU_D_0073 | left | bottom_left | bottom_left | Houses appear in the lower-left corner. → UNRESOLVED | There is no change. → UNRESOLVED |
| RSU_D_0097 | right | bottom_left | top_left | Buildings appear in the upper-left corner. → UNRESOLVED | New buildings appear in the upper-left corner of the scene. → UNRESOLVED |
| RSU_D_0131 | bottom | bottom | bottom | Houses and a road appear in the lower part of the scene. → UNRESOLVED | Houses and a road appear in the lower part of the scene. → UNRESOLVED |
| RSU_D_0139 | top_left | top | top_left | There is no change in the scene. → UNRESOLVED | A building appears in the upper-left corner of the scene. → UNRESOLVED |
| RSU_D_0141 | top_right | top_right | top_right | There is no change in the scene. → UNRESOLVED | There is no change. → UNRESOLVED |
| RSU_D_0142 | left | bottom_left | top_left | There is no change in the scene. → UNRESOLVED | There is no change. → UNRESOLVED |
| RSU_D_0145 | unclear | bottom_right | bottom_right | There is no change. → UNRESOLVED | There is no difference. → UNRESOLVED |
| RSU_D_0147 | unclear | bottom_right | bottom_left | There is no change in the scene. → UNRESOLVED | There is no change. → UNRESOLVED |
| RSU_D_0148 | right | top_left | top_left | There is no change in the scene. → UNRESOLVED | A row of houses appears in the upper part of the scene. → UNRESOLVED |
| RSU_D_0175 | left | bottom_left | bottom_left | There is no change. → UNRESOLVED | There is no change. → UNRESOLVED |
| RSU_D_0179 | top_right | top_right | top_right | There is no difference. → UNRESOLVED | There is no difference. → UNRESOLVED |
| RSU_D_0206 | left | bottom_left | bottom_left | There is no change. → UNRESOLVED | There is no change. → UNRESOLVED |
| RSU_D_0224 | bottom | bottom | bottom_left | There is no change. → UNRESOLVED | There is no difference. → UNRESOLVED |
| RSU_D_0233 | top | top_left | top | There is no change in the scene. → UNRESOLVED | There is no change. → UNRESOLVED |
| RSU_D_0252 | top | top_left | top_left | There is no change. → UNRESOLVED | There is no change. → UNRESOLVED |
| RSU_D_0255 | top_right | top_right | top_right | There is no change. → UNRESOLVED | A building appears in the upper-right corner of the scene. → UNRESOLVED |
| RSU_D_0277 | left | bottom_left | left | There is no change in the scene. → UNRESOLVED | There is no difference. → UNRESOLVED |
| RSU_D_0279 | left | bottom_left | bottom_left | There is no difference. → UNRESOLVED | There is no difference. → UNRESOLVED |
| RSU_D_0282 | top | top | top_left | There is no change in the scene. → UNRESOLVED | There is no change. → UNRESOLVED |
| RSU_D_0329 | bottom_left | bottom_left | bottom_left | There is no change. → UNRESOLVED | A road appears in the lower-left corner of the scene. → UNRESOLVED |
| RSU_D_0362 | bottom | bottom | bottom_left | Houses appear at the bottom of the scene. → UNRESOLVED | Houses are built at the bottom of the scene. → UNRESOLVED |

## 混淆矩阵

每个矩阵的行是真值、列是预测；行列均按输出标签顺序。

### A 原图直接九分类

| Gold \ Pred | bottom | bottom_left | bottom_right | center | left | right | top | top_left | top_right |
|---|---|---|---|---|---|---|---|---|---|
| bottom | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| bottom_left | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| bottom_right | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| center | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| left | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| right | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 1 | 0 |
| top | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 2 | 0 |
| top_left | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 2 | 0 |
| top_right | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |

### B 九宫格视觉辅助

| Gold \ Pred | bottom | bottom_left | bottom_right | center | left | right | top | top_left | top_right |
|---|---|---|---|---|---|---|---|---|---|
| bottom | 1 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| bottom_left | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| bottom_right | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| center | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| left | 0 | 4 | 0 | 0 | 1 | 0 | 0 | 1 | 0 |
| right | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 2 | 0 |
| top | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 2 | 0 |
| top_left | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 |
| top_right | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |

### C 自然语言描述＋确定性映射

| Gold \ Pred | bottom | bottom_left | bottom_right | center | left | right | top | top_left | top_right |
|---|---|---|---|---|---|---|---|---|---|
| bottom | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| bottom_left | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| bottom_right | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| center | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| left | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| right | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| top | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| top_left | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| top_right | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### D 网格辅助＋自然语言描述映射

| Gold \ Pred | bottom | bottom_left | bottom_right | center | left | right | top | top_left | top_right |
|---|---|---|---|---|---|---|---|---|---|
| bottom | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| bottom_left | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| bottom_right | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| center | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| left | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| right | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| top | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| top_left | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| top_right | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## 视觉样本及原始记录

所有 B/D 组可视化输入保存在 `visual_inputs/`（每题 BEFORE/AFTER 各一张）。`predictions.json` 保存逐题全部原始输出、Token ID、自然语言 Location 提取、规则映射、提示词哈希、计时和混淆/转移统计；`input_manifest.json` 保存完整提示词和输入图映射。

## 结论边界

该 D25 小样本只支持描述观察到的行为，不能据此宣称稳定的性能提升。C/D 的未决率必须与准确率一并解读：保守映射拒绝模糊方位是预期行为，不可把未决重解释成方向错误。模型对变化内容的文字描述也不等同于内容识别正确，需结合保留的图像与逐题描述人工复核。

## 完整句子离线映射诊断

原始 `predictions.json` 及以上固定字段分数保持不变；其 SHA256 为 `5ced13c882cfca98a0ada5da9923e0c7a021f07620cfa9eeeec0a49c6b9aff62`。补充记录在 `offline_diagnostics.json`，每题保留完整原文、提取原因、候选与映射。

| 描述组 | 没有变化/差异 | 明确格位可提取 | 其余方位含糊 | 明确23题命中 | 25题严格命中 |
|---|---:|---:|---:|---:|---:|
| C 原图描述 | 21/25 | 2/25 | 2/25 | 0/23 | 0/25 |
| D 网格描述 | 15/25 | 6/25 | 4/25 | 4/23 | 4/25 |

C 的明确短语来自 0073（lower-left）与 0097（upper-left），均与参考的主要变化格位不符。D 的六个明确短语来自 0064、0071、0097、0139、0255、0329，答对 0064、0139、0255、0329。`right side` 只确定列，`lower part` / `at the bottom` 只确定行，均不擅自映射到中间格。0131、0362 虽描述了可见的底部新增建筑，仍因此未决，不能算作方向判错，也不能据参考标签补齐另一维。

全部 C/D 自由输出仅为 6–16 个生成 Token，未触及 96 Token 上限。保存的完整 chat prompt 包含描述指令和两张图；没有消息替换遗漏或输入截断。这支持“模型没有遵循字段要求”的判断，但无法区分训练后的回答模板偏好与 length_penalty=0 的短输出偏好各占多少。为保持对照一致，本轮未修改生成参数重跑。

## 空间方向、标签映射与视觉复核

明确23题的 A 行位置正确 14/23，列位置正确 18/23；B 行位置正确 16/23，列位置正确 17/23。A 的六道参考 `left` 全被输出为 `bottom_left`；B 其中四道仍为 `bottom_left`，一道变为 `top_left`，只答对一道。说明本样本中上下行判断比左右列更弱。A 的 `bottom_left` 为 8 次、B 为 7 次，而参考仅 1 次；B 的 `top_left` 又达到 8 次，参考为 3 次，网格没有消除角落偏好。样本未包含明确 `center`、`bottom_right` 标签，无法评价全部九类的召回率或作总体类别偏置结论。

固定方向变换也未找到统一标签颠倒：A 恒等映射命中 11，水平镜像 4、垂直镜像 1，其余旋转/转置至多 4；B 恒等映射 12，其余至多 4。此项仅作诊断，不重映射答案。现有证据不支持统一左右翻转、上下翻转或行列颠倒是主要原因；它不能排除更细粒度的模型标签偏好。

已查看全部25题原始双图与逐题输出，复核图见 `review_sheets/review_01.png` 至 `review_07.png`。下列观察是定性复核，没有重新标注或更改参考答案，也没有像素级变化掩膜，不能量化真实变化面积或边界距离。

| 样本 | 观察与输出 | 主要问题 |
|---|---|---|
| 0058 | 右边新增建筑及停车区域明显；A/B 为 right，D 正确描述 parking lot and a building on the right side；C 却说无变化 | 同一模型在不同提问下识别不稳定；D 描述有用，但只确定一列 |
| 0064、0139、0255 | 图中对应角落新增建筑，D 描述及方位吻合参考，A/B 也答对（0139 的 A 为 top） | 模型确有部分局部变化与方位理解能力，不能因字段失败判为零 |
| 0071 | 左上新增建筑与上边道路均可见；D 选择了右上道路，A/B 选左上 | 主要变化对象选择不同，不能单归因于标签映射 |
| 0097 | 原先左上已有建筑，右侧/中右停车及建筑区发生显著变化；C/D 却都强调左上建筑 | 对显著静态/次要对象的注意可能盖过真正主要变化；描述本身已有错误 |
| 0073、0175、0279 | 左侧新增建筑或道路沿纵向延伸，可能横跨多行；A/B 大多落到 bottom_left | 变化对象范围与主要格位汇总问题；属于跨格或邻格误差线索，不能断言只是格线附近像素误差 |
| 0131、0362 | C/D 正确描述底部新增房屋（0131 还有道路），但未确定列；0362 的 B 从 bottom 退化为 bottom_left | 内容识别可以正确而格位仍不完整；网格也可能引入列偏差 |
| 0148 | 上沿房屋、右侧道路及建设区域均有变化；D 只描述上部房屋，未能给出主要变化格位 | 多区域变化下选择哪一处作为“主要”仍是瓶颈 |
| 0145、0147 | 变化分散或参考标为 unclear，A/B 仍被九分类强制选格；C/D 说无变化 | unclear 不等于无变化；原九分类不提供弃答，必须单独报告 |

A 的12道错误中10道、B 的11道错误中9道与参考相邻。这只能提示邻格/跨格问题，不能单凭标签相邻就判断变化识别正确。图像普遍存在季节、植被、亮度和阴影差异，与新增建筑/道路并存；没有逐像素标注，无法将每题错误严格归因到这些因素。

## 对部署问题的回答

1. **理解了变化却定位失败吗？** 部分样本是，如0131/0362，但大量自由描述漏报变化，0097等还选错变化内容。不能把总体错误解释成纯定位或映射问题。
2. **视觉网格有效吗？** 本轮仅净增1题，纠正4题、破坏3题，配对检验p=1.0；同时存在 bottom_left/top_left 偏好，没有稳定收益证据。
3. **自然语言方位比直接分类可靠吗？** 本配置下更不可靠。格式失败、无变化模板和不完整方位共同降低可用率；修复离线字段依赖后依然低于 A/B。
4. **最终标签映射是否有系统性错位？** Token边界、前导空格、EOS及50次A/B约束输出均通过检查；固定镜像/旋转也不能改善。未发现统一标签转换错误，但不能凭本实验排除模型自身的类别偏好。
5. **哪种有部署价值？** 现阶段保留 A 为基线；B可保留为研究对照，不替换生产；C/D适合诊断解释，不作为自动答案路径。后续如继续，应在独立、方向均衡且明确“主要变化”标准的数据上验证变化识别与行方向，避免在这25题上反复改提示词、映射或规则。

## 耗时、验证与复现

使用 NVIDIA GeForce RTX 4060 Laptop GPU、BF16。模型加载5.51s，100次推理循环总计137.75s，峰值CUDA已分配显存4478MiB（约4.37GiB；不是整个GPU或预留显存峰值）。A/B平均生成0.53/0.61s，网格约增加0.08s；C/D约2.09/2.14s。逐题生成计时包含GPU同步，仅覆盖 model.generate，不含读图、processor、加载和落盘，不能视为服务端到端延迟。

原始8项单测、真实processor预检及标签Trie检查通过；结果复核新增2项离线自然语言映射测试，共10项通过。离线分析内另有9条固定断言检查明确格位、否定、整行/列与冲突，全部通过。模型运行日志无异常退出；PEFT提示忽略额外的空配置字段，不影响本轮完成。修改仅涉及独立实验脚本、测试与报告。

在仓库根目录执行短时离线检查与报告诊断：

```powershell
& 'C:\Users\Ericoneabc\AppData\Local\Microsoft\WindowsApps\python.exe' -m unittest discover -s scripts/experiments -p test_d25_spatial_understanding.py -v
& 'C:\Users\Ericoneabc\AppData\Local\Microsoft\WindowsApps\python.exe' scripts/experiments/analyze_d25_spatial_results.py reports/experiments/d25_spatial_understanding_20261011/run_v1
```

本轮推理启动命令为 `& .\scripts\experiments\launch_d25_spatial_understanding.ps1`；完成结果受防覆盖保护，重复实验须另设输出目录并重新预检。启动器在后台保存PID和stdout/stderr，日志位于 `logs/spatial_experiment.stdout.log` 与 `logs/spatial_experiment.stderr.log`。所有原始双图路径、50张网格输入、完整提示词、Token、描述和计时均已保存，可复核。
