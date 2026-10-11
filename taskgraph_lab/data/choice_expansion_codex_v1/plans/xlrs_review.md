# XLRS Codex 转换复核记录

本分片为 449 条原选择题，输入不含 GT。工具只适用于已经审阅的本分片，问题、选项或次序变化会触发 SHA256 指纹拒绝复用规则。没有训练，没有 API 调用，没有视觉答案预测。

## 审核范围

- 完整阅读全部 449 条英文问题；空间关系题的 subject/reference 对逐条检查。
- 方向及计数选项的有限取值完整检查；复杂推理题选项按原顺序阅读，用于确定单选或不定项合同，未读取 GT。
- 修正 square building 被错误解析成 square 的名词头问题；颜色、形状、大小、物体部件和状态进入 TargetSpec。
- 修正先全图求 largest 再筛区域的逻辑次序：先对完整候选筛域，再用 bbox_area 排序。
- 纯区域计数使用 REGION；语义外部条件使用既有候选的 SELECT_SEM MULTI，再接权威 COUNT。
- 两条 HRSCD 复合 URI 是独立双时相图片，不假设左右拼图；展开为两个图像输入后，分别 FIND_MARKER、COUNT，再 ABS_DIFF。
- 13 条空间题在描述中已经给出方位或背景关系，使用视觉 FINAL_QUESTION 保留原问题，避免把已给条件当成待求关系。
- 竖排棚舍、温室群使用 GROUP；明确的最终空间比较使用 RELATION。

## 验证结果

449/449 通过 parser、原输入/问题/题型 Validator、canonical JSON/DSL 往返及完整约束语法；pending 为 0。另有 8 项语义断言验证双图引用、square 属性、筛域后排序、绝对位置残余问题、small 属性和 COUNT 证据。

| 项目 | 数量 |
|---|---:|
| CHOICE_SINGLE | 365 |
| CHOICE_MULTI | 84 |
| 含 SELECT_SEM 的样本 | 242 |
| SELECT_SEM SINGLE 节点 | 230 |
| SELECT_SEM MULTI 节点 | 94 |
| COUNT 节点 | 64 |
| RELATION 节点 | 214 |
| SELECT_RANK 节点 | 71 |
| SELECT_EXTREME 节点 | 1 |
| GROUP 节点 | 2 |
| ABS_DIFF 节点 | 2 |

plans/xlrs.jsonl SHA256：73fa7241ff61d07797d830b85dbe42185361dcb35116111f8e8678836ed79fa8。

生成命令：

~~~powershell
& 'C:\Users\Ericoneabc\AppData\Local\Microsoft\WindowsApps\python.exe' taskgraph_lab\tools\codex_choice_xlrs.py
~~~

## 未验证与语义风险

这些验证仅证明规划表达、类型及已审阅规则一致，不能证明图像中存在描述的目标、视觉模型能筛出目标或最终选项正确。未加载模型权重或运行图像推理。

部分 XLRS 空间题的给定方位、参考物和选项含义存在源文本歧义；保留完整 query 或原静态 residual question，而不预测关系值。largest roof 使用现有 bbox_area 排序，是对象包围框面积对屋顶面积的近似；区域、森林、自然地物等候选生成仍取决于执行后端能力。复杂原因、建设适宜性和环境影响依赖 VLM 的视觉与知识判断；问题语义决定的 CHOICE_MULTI 允许最终仅选择一个选项。

旧数据、固定测试、生产代码均未由本转换脚本修改。
