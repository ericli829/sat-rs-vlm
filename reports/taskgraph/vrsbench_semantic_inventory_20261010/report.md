# VRSBench 对新 Planner 数据集的可用性初查

日期：2026-10-10。范围：查找现有 VRSBench 中与 LOCATE / SELECT_SEM、集合操作、区域参照相关的问题；没有生成正式 Planner SFT 标签、重新训练模型或修改运行时。

## 1. 已推送的修改

LOCATE / SELECT v2 重构已推送至 `origin/codex/competition-protocol-v2`。

- Commit：[`8ee1c788ba9c4ebcb17d943bd956f9a455dfc209`](https://github.com/ericli829/sat-rs-vlm/commit/8ee1c788ba9c4ebcb17d943bd956f9a455dfc209)。
- 提交说明：`feat(taskgraph): route LOCATE by dataflow and add semantic SELECT`。
- 本地 HEAD 与远端跟踪分支指向同一提交，push 命令已成功完成。
- 提交前对暂存内容导出的独立目录重新运行新增路由与语义筛选测试：56 passed。见 [推送验证记录](../locate_select_v2_validation/push_verification.md)。更早的完整回归统计见原实施报告；没有把重复测试累加为新覆盖。
- 工作区中原先其他 HTTP/API、HF 引擎、哈希记忆等未提交修改保留在工作区；没有混入这次重构提交。本目录是推送后形成的本地分析产物，尚未提交。

## 2. 数据源与完整扫描结果

原始数据在 `F:\VIT-data\VRSBench`，含 `official`、`Annotations/Annotations_train`、`Annotations/Annotations_val`、`Images/Images_train`、`Images/Images_val`。

本次完整扫描仓库已有的 `data/processed/rs_train.jsonl` 和 `rs_val.jsonl`，并将抽出的 122 个代表样本逐条核对至原始单图标注，检查原始问题/描述、object_id/question_id、图片是否存在及图片尺寸。122 个样本均为 512×512，训练/验证各 61 个。这只是被抽中样本的尺寸统计，不代表已经检查全库图片尺寸。

| 已有转换数据 | train | val |
| --- | ---: | ---: |
| 总行数 | 142,390 | 62,918 |
| 不同图片文件名 | 20,262 | 9,350 |
| 指代表达定位 | 36,313 | 16,159 |
| counting 类记录 | 15,428 | 6,374 |
| captioning | 20,264 | 9,350 |
| scene_classification | 6,639 | 3,197 |
| 其他 vqa | 63,746 | 27,838 |

训练 caption 行数比不同图片数多 2：`02725_0000.png`、`P2708_0016.png` 各出现两条 caption。构建新数据时应按源图片分组、保留原始 ID，再检查重复记录，不能用行数代替图片数。

完整统计和源 JSONL 的 SHA256 保存在 [summary.json](summary.json)。本次统计针对当前已有转换文件，不重新生成这些文件，也不假设它们与当前 converter 的输出版本完全相同。

## 3. 确实有类似问题，但关键词初筛不等于 gold

以下数量是独立的、可重叠的正则标签计数，**不能相加为独立样本数**，也不是经人工确认的任务分类。`near/above/below` 可能出现在提问选项中；`green` 可能描述背景；`second` 可能缺少排序方向。完整正则在 inventory.py 和 summary.json 中。

| 训练集初筛类型 | referring | QA |
| --- | ---: | ---: |
| 背景词与关系词同时出现 | 1,823 | 1,523 |
| 背景关系 + 多关系或至少两类属性/位置线索 | 826 | 16 |
| 至少两次关系词命中 | 548 | 35 |
| 极值词命中 | 6,824 | 4,370 |
| 序数词命中 | 95 | 128 |
| counting 记录同时命中关系词 | 不适用 | 166 |

有 building 字样的训练记录为 4,554 条，有 `parking area/lot` 字样的为 1,640 条，均排除 caption。这些计数涵盖目标、背景、参照物和问句选项，**不是这两类目标的 bbox 标注数量**。26 种 referring 类别中没有普通 `building` 和 `parking area` 类；有 `expressway-service-area`、`trainstation`、`harbor` 等设施类。

因此，VRS 对“完整自然语言条件怎样进入 SELECT.query”“集合极值不能变成单体猜测”“区域作为单体参照物”等规划能力有帮助；对普通建筑与停车区域的精确定位监督，需要其他数据或人工补标。

## 4. 真实样本及建议用途

以下均来自 **train**。这里给的是待审核的逻辑拆分建议，不是已经编译、验证和运行过的 gold TaskGraph。完整原问题、答案、源行号、图片/标注路径在 [curated_examples.jsonl](curated_examples.jsonl) 中。

### A. 复合候选语义筛选

`vrsbench_train_P0806_0004_det_001`，object_id=1：

> The tennis court that is adjacent to the topmost court on its bottom right side is surrounded by a large green area and trees.

原始 bbox：`[0.36, 0.00, 0.63, 0.38]`。已查看原图。

适合研究：`LOCATE(tennis court 候选集合) → SELECT_SEM(完整描述, SINGLE)`，或由 Planner 显式产生参照场地再执行相应关系筛选。不能因为出现 topmost 就只挑最上方场地；这里 topmost 修饰的是参照物。树木与绿地条件可留在 query 中。图右边缘有裁断场地，集合完整性与计数口径仍需复核。

`vrsbench_train_P2002_0009_det_001`，object_id=1：

> Adjacent to the right-most bus, towards the center of the parking area, is another large vehicle with white and black coloring.

原始 bbox：`[0.18, 0.47, 0.46, 0.66]`。已查看原图，附近存在多个外观相似的公交车。

适合研究：车辆候选集合上的复合条件筛选、同类参照物、属性和背景上下文。可作为 SELECT_SEM 示例，也可与显式参照物的几何方案作对照；应由 Planner 标注规则确定允许的图，不能只按算子数量选择 gold。原单图 objects 仅有 3 个车辆描述，而同图 QA “How many large vehicles are present in the image?” 的答案为 5。**这 3 个 objects 不能充当完整候选池。**

`vrsbench_train_P2228_0008_det_001`：

> The small vehicle positioned near the top-most part of the curved road in the middle-left area is bordered by trees on three sides.

适合研究：弯曲道路、背景方位、三面树木等条件进入完整 SELECT.query；`top-most part` 描述道路局部，不能直接等同于全图车辆 TOPMOST。已核对标注原文，尚未逐图验证目标正确性。

`vrsbench_train_P1481_0022_det_000`：

> The small vehicle located at the center right-most edge of the terminal building, near the feeder road, has a white roof.

适合研究：终端建筑边缘、道路邻近和白色车顶共同筛选车辆。不应因 query 中出现建筑/道路就自动强制新增 LOCATE。已核对标注原文，尚未逐图验证目标正确性。

### B. 条件计数、区域参照及输入角色

`vrsbench_train_P0050_0007_qa_002`：

> How many small vehicles are visible near the road?

原 QA 答案：`2`。适合研究车辆全集 → 条件筛选 → COUNT，覆盖 DINO 集合需求；道路条件可由 Planner 用现有 SELECT 模式表达。不能把车辆 LOCATE 变成单框。尚未逐图复核计数。

`vrsbench_train_P1084_0005_qa_004`：

> Is the parking area located above or below the tennis court?

答案：`below`。已查看原图，停车区确实位于网球场下方。适合补区域目标和两个单体之间的 RELATION，不应该为了新增 SEM 把这种简单关系也全部改成 SELECT_SEM。原 objects 没有停车区 bbox，不能假定已有该区域的定位监督。

`vrsbench_train_00212_0000_qa_002`：

> Are there vehicles near the building with a large white rooftop?

答案：`Yes`。适合覆盖同图的不同输入角色：白屋顶建筑是单体参照物，车辆是待筛选的集合。现有 schema 没有显式 EXISTS 算子，后续正式 gold 需选用已支持的布尔判断/答案路径，不能凭空写 EXISTS；也没有建筑 bbox 监督。已核对原 QA，尚未逐图验证答案。

### C. 保留旧算子和清洗负例

`vrsbench_train_P0578_0009_qa_001`：

> Is the topmost small vehicle near the body of water?

答案：`Yes`。应保留候选集合 → `EXTREME(TOPMOST)` → 单体关系判断这类控制样本，让 Planner 学会确定性极值与语义筛选的边界。尚未逐图验证答案。

`vrsbench_train_P2611_0017_qa_003`：

> What color is the second large vehicle?

答案：`White`。没有给出序数排序方向，不得自行补成“从左向右第二个”。建议先人工审核上下文，否则排除正式 gold。

`vrsbench_train_00082_0000_det_000`：

> The tennis court located in the top-left quarter of the image, closest to the orange-roofed building and surrounded by trees to its left.

bbox：`[0.23, 0.46, 0.55, 0.92]`。已查看原图：该 bbox 位于中下部，不在 top-left quarter；描述中的橙色屋顶建筑在当前裁图中不明显。同图还有超出边界的目标框。**这个例子保留为清洗案例，暂不作为正确规划标签。** 需要确认描述是否沿用了裁剪前图像语境。

## 5. 现有 Planner 数据缺口

完整扫描 `taskgraph_lab/data/planner_sft*/train.jsonl` 下当前四个文件：

| 数据目录 | train 行数 | 数据来源 | SELECT_SEM 目标数 | system 提及 SELECT_SEM 的行数 |
| --- | ---: | --- | ---: | ---: |
| planner_sft_v1 | 193 | MME + XLRS | 0 | 0 |
| planner_sft_new2701 | 2,701 | MME + XLRS | 0 | 0 |
| planner_sft_hard_augmented_v1_20260901_173936 | 2,815 | MME + XLRS | 0 | 0 |
| planner_sft_hard_curriculum_v1 | 3,934 | MME + XLRS | 0 | 0 |

这四版彼此可能重复/重采样，不能合计成独立训练样本。它们都没有 VRS 来源记录。当前代码和外置提示词支持 SELECT_SEM，并不代表旧 Planner 权重学过该 DSL；旧 SFT 文件还内嵌旧 system prompt。后续构建新版本时需要同步更新整套训练消息中的契约，并检查保留旧算子的任务分布。

## 6. 数据清洗与协议约束

1. **候选不完整。** VRS objects 是被描述的目标，不保证穷尽同类物体。`is_unique` 表示类别内唯一性；false 也不意味着 objects 已标齐其他同类。不能直接用它生成“其他候选均为负例”、MULTI 完整集合、NONE 或精确全量计数的监督。
2. **坐标与裁剪。** 当前转换文件 train 有 7,393 条 referring 记录发生 bbox 裁剪，val 有 4,399 条；转换后仍有 20 / 6 条非正面积 bbox。裁剪本身不一定是错误，但需检查描述是否与可见图像一致；非正面积框应排除定位监督。官方 `<...>` bbox 与单图 obj_coord 的量纲和舍入也不同，不能混用。
3. **QA 类型不是 gold 任务。** 旧转换文件的 counting 类记录中包含 `Yes`、`No`、`Multiple`、`Several` 等答案；例如 train 的 `Multiple` 有 170 条。应结合真实问题和答案重新审核意图，不能因为 qa_type=object quantity 就一律输出 INTEGER/COUNT。
4. **终止协议。** 当前 `AnswerType` 只有 CHOICE_SINGLE、CHOICE_MULTI、INTEGER、BOOLEAN、LABEL、LABEL_SET、TEXT，没有 BBOX。原始 referring 的直接 bbox 答案不能原样冒充现有 Planner FINAL 标签。优先把现有 VQA 转成完整合法 TaskGraph；referring 用作描述来源与中间目标监督。如果派生新属性/关系问句，必须标明 derived、保留父记录并人工确认答案，不能伪称原始问题。
5. **划分。** 本次 train/val 图片文件名交集为 0，文件名前缀交集为 0；前缀只是相关切片的启发式，尚未进行图像哈希、地理范围或近重复核验。VRS 官方 val/eval 不混入新训练集；训练内划分至少以源图片/可识别母图分组。
6. **分辨率。** 已检查代表样本为 512×512，适合补问题到程序的表达与算子选择；不能据此证明 UHR 上的检测召回、局部裁剪、VLM 定位或最终准确率改善。

## 7. 建议的新数据集构建顺序

建议先从 **VRS train** 做约 300–500 条人工审核的试点，而不是立刻自动转换全部 36,313 条 referring：

1. 优先选真实 VQA：复杂候选描述后问属性、条件计数、区域/单体参照物；referring 为描述与目标核验提供辅助证据。记录完整来源和标注可信度。
2. 将复合筛选与简单几何控制样本配对，覆盖 SEM_SINGLE、已有 REL/EXTREME/RANK/ORD/SUBREGION、直接单体 LOCATE、关系计数；避免把新数据变成“所有问题都 SELECT_SEM”。
3. 给每条审核程序记录预期候选集合和参照物角色，以及各 LOCATE 的预期 SET_REQUIRED/SINGLE_TARGET 路由。Planner 的图决定逻辑，Executor 不猜题、不补图。
4. 严格区分 Planner 程序标签与 Executor 候选选择标签。SEM_MULTI、NONE、AMBIGUOUS、UNRESOLVED 的执行监督需要完整候选或额外验证，不能仅由 VRS 正例 bbox 自动推出。
5. 同步当前 schema/DSL/system prompt，做编译、类型/输入角色、query 原文保留、最终答案类型及路由 trace 检查，再人工检查程序与题意一致。允许语义等价的程序，不能只以 DSL 字符串完全相同为规划准确率。
6. 与旧 MME/XLRS 训练样本混合，并保持按图分组的独立验证；先报告规划正确率、非法 DSL 和算子/路由错误，再用独立 UHR 原图测试端到端准确率与资源开销。

本轮停留在发现和分析阶段，上述试点尚未生成，模型权重没有变化。

## 8. 产物与复核

- [inventory.py](inventory.py)：完整扫描与原标注抽样核对脚本。
- [summary.json](summary.json)：任务、启发式标记、数据质量、Planner 覆盖及源文件 SHA256。
- [review_candidates.jsonl](review_candidates.jsonl)：122 个抽样候选，含源行号、原始标注 SHA256、图片尺寸；不是全量候选清单或训练 gold。
- [targeted_keyword_examples.json](targeted_keyword_examples.json)：building、parking、white/square 关键词的补充扫描示例；同样不代表类别标注或 gold。
- [curated_examples.jsonl](curated_examples.jsonl)：本报告重点讨论的 10 条记录及审核备注，没有 assistant/TaskGraph 训练标签。

复核命令（在仓库根目录执行；本机运行已成功完成）：

```powershell
& 'C:\Users\Ericoneabc\AppData\Local\Microsoft\WindowsApps\python.exe' reports/taskgraph/vrsbench_semantic_inventory_20261010/inventory.py
```

脚本对源数据只读；会覆盖本目录的 summary.json / review_candidates.jsonl。人工讨论清单和补充关键词文件不会被该命令重写。没有执行 Planner 训练、真实模型推理或 UHR 准确率实验。
